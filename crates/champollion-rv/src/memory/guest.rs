//! The guest's address space, backed by one large host reservation.
//!
//! Guest address `g` lives at host address `base + g` for every `g < GUEST_SIZE`. The whole
//! range is reserved `PROT_NONE` up front, which costs no physical memory; pages are
//! committed as the guest maps them. [`GUARD_SIZE`] bytes on either side stay `PROT_NONE`
//! forever, so the JIT may add a small offset to an in-range address without masking the
//! sum: the access either lands in the guest space or faults. Below the lower guard lies a
//! read/write control area for the runtime: one interrupt flag per thread slot, which
//! generated code checks at block entry (see [`GuestMemory::interrupt_flag`]).
//!
//! ```text
//! base - CONTROL_OFFSET   interrupt flags (INTERRUPT_SLOTS x u32)
//! base - GUARD_SIZE       guard
//! base                    guest address 0
//! base + GUEST_SIZE       guard
//! ```
//!
//! Guest pages are 4 KiB (what Linux on RISC-V reports in `AT_PAGESZ`), but macOS on arm64
//! uses 16 KiB host pages, so host protections cannot represent guest permissions exactly.
//! Instead, every host page that overlaps a guest mapping is simply read/write, and guest
//! permissions are tracked and checked in software at 4 KiB granularity.
//!
//! Invariant: every guest byte that is not mapped reads as zero on the host side (either its
//! host page is decommitted, or it was zeroed on unmap), so a new mapping never needs to be
//! cleared and never touches physical memory until the guest uses it.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::{MemFault, Memory};

/// Size of the guest address space: 2^38 bytes (256 GiB), the user half of RISC-V Sv39.
pub const GUEST_SIZE: u64 = 1 << 38;
pub const GUEST_PAGE_SIZE: u64 = 4096;
/// Inaccessible host memory reserved below and above the guest space.
pub const GUARD_SIZE: u64 = 64 << 10;
/// Size of the control area below the lower guard.
const CONTROL_SIZE: u64 = 16 << 10;
/// Distance from the guest base down to the control area.
pub const CONTROL_OFFSET: u64 = GUARD_SIZE + CONTROL_SIZE;
/// Number of interrupt flags (thread slots) in the control area.
pub const INTERRUPT_SLOTS: usize = (CONTROL_SIZE / 4) as usize;

/// Guest page permissions, with Linux `PROT_*` bit values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Prot(pub u32);

impl Prot {
    pub const NONE: Prot = Prot(0);
    pub const READ: Prot = Prot(1);
    pub const WRITE: Prot = Prot(2);
    pub const EXEC: Prot = Prot(4);
    pub const RW: Prot = Prot(3);

    pub fn contains(self, other: Prot) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for Prot {
    type Output = Prot;
    fn bitor(self, rhs: Prot) -> Prot {
        Prot(self.0 | rhs.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapError {
    /// Address or length not page aligned, or the range leaves the guest address space.
    Invalid,
    /// `protect` on a range that is not fully mapped.
    NotMapped,
    /// No free range of the requested size.
    NoSpace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mapping {
    end: u64,
    prot: Prot,
}

/// A validated host view of guest memory, for handing to the host kernel. Layout-compatible
/// with `struct iovec`.
///
/// Guest memory is shared with other guest threads and can be modified or even unmapped
/// concurrently, so it is deliberately not a Rust slice. Pass it to system calls, which
/// report `EFAULT` rather than crash if the memory disappears; only touch it from Rust when
/// no other guest thread can exist (e.g. in single-threaded tests).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestBuf {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Non-overlapping guest mappings, keyed by start address. Adjacent mappings are not
/// merged; nothing depends on that and it keeps splitting simple.
#[derive(Default)]
struct Maps(BTreeMap<u64, Mapping>);

impl Maps {
    fn is_any_mapped(&self, r: Range<u64>) -> bool {
        self.0.range(..r.end).next_back().is_some_and(|(_, m)| m.end > r.start)
    }

    /// Check that `[addr, addr + len)` is mapped with at least `need` permissions.
    fn covering(&self, addr: u64, len: u64, need: Prot) -> Result<(), MemFault> {
        let end = addr.checked_add(len).filter(|&e| e <= GUEST_SIZE).ok_or(MemFault { addr })?;
        let mut at = addr;
        while at < end {
            match self.0.range(..=at).next_back() {
                Some((_, m)) if m.end > at && m.prot.contains(need) => at = m.end,
                _ => return Err(MemFault { addr: at }),
            }
        }
        Ok(())
    }

    /// Split mappings so that `r`'s boundaries fall on mapping boundaries, then return the
    /// mappings inside `r`.
    fn split_out(&mut self, r: Range<u64>) -> Vec<(u64, Mapping)> {
        for at in [r.start, r.end] {
            if let Some((&start, &m)) = self.0.range(..at).next_back()
                && m.end > at
            {
                self.0.insert(start, Mapping { end: at, prot: m.prot });
                self.0.insert(at, m);
            }
        }
        self.0.range(r.start..r.end).map(|(&s, &m)| (s, m)).collect()
    }

    /// The highest free range of `len` bytes that ends at or below `limit`.
    fn find_free(&self, len: u64, limit: u64) -> Result<u64, MapError> {
        let mut hi = limit.min(GUEST_SIZE) & !(GUEST_PAGE_SIZE - 1);
        // Walk mappings from the top down, looking for a gap of `len` below `hi`.
        for (&start, m) in self.0.range(..hi).rev() {
            if m.end <= hi && hi - m.end >= len {
                return Ok(hi - len);
            }
            hi = hi.min(start);
        }
        if hi >= len { Ok(hi - len) } else { Err(MapError::NoSpace) }
    }
}

/// The guest address space. Shared by all guest threads: every method takes `&self`.
///
/// Concurrency: the mapping table sits behind an `RwLock`. Data accesses hold the read lock
/// for the duration of the access, so a concurrent `munmap` cannot decommit memory under
/// them, and touch memory only through atomics (plain `ldr`/`str` on arm64 when aligned),
/// so racing guest threads never cause undefined behavior on the host side. Mapping changes
/// take the write lock. [`GuestBuf`]s are validated under the lock but used after it is
/// released, so blocking host I/O never holds it.
///
/// Caching: every mapping change bumps [`generation`](Self::generation), and changes that
/// involve executable pages also bump [`code_generation`](Self::code_generation). Per-thread
/// caches remember the counter they were filled at and flush when it moves, so they may skip
/// the lock in between: the [`Tlb`](super::Tlb) follows `generation`, decoded blocks and
/// translations follow `code_generation`. With
/// threads (M5), `munmap` must additionally wait until every running thread has observed
/// the new generation before decommitting host pages.
pub struct GuestMemory {
    base: *mut u8,
    host_page: u64,
    maps: RwLock<Maps>,
    generation: AtomicU64,
    code_generation: AtomicU64,
}

// SAFETY: `base` points to a reservation owned by this object for its whole lifetime, and
// all access to the memory behind it is synchronized as described on the type.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

impl GuestMemory {
    pub fn new() -> std::io::Result<Self> {
        // SAFETY: anonymous PROT_NONE reservation; no existing memory is affected.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                (CONTROL_OFFSET + GUEST_SIZE + GUARD_SIZE) as usize,
                libc::PROT_NONE,
                libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if start == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: making our own reservation's control area read/write.
        if unsafe {
            libc::mprotect(start, CONTROL_SIZE as usize, libc::PROT_READ | libc::PROT_WRITE)
        } != 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let base = start.cast::<u8>().wrapping_add(CONTROL_OFFSET as usize);
        // SAFETY: sysconf has no preconditions.
        let host_page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        assert!(host_page.is_power_of_two() && host_page >= GUEST_PAGE_SIZE);
        Ok(Self {
            base,
            host_page,
            maps: RwLock::default(),
            generation: AtomicU64::new(0),
            code_generation: AtomicU64::new(0),
        })
    }

    /// Host address of guest address 0. Guest address `g` is at `host_base() + g`.
    pub fn host_base(&self) -> *mut u8 {
        self.base
    }

    /// Counter bumped by every mapping change; see the type-level docs.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation.load(Acquire)
    }

    /// Counter bumped by mapping changes that involve executable pages (before or after the
    /// change). Caches of decoded or translated code follow this one, so data-only `mmap`,
    /// `brk`, and `munmap` traffic leaves them intact.
    #[inline]
    /// The interrupt flag of thread slot `slot`: nonzero asks the thread to leave generated
    /// code at the next block entry (for signal delivery).
    pub fn interrupt_flag(&self, slot: usize) -> &AtomicU32 {
        assert!(slot < INTERRUPT_SLOTS);
        // SAFETY: the control area is mapped read/write for the reservation's lifetime and
        // only ever accessed atomically; slots are 4-byte aligned.
        unsafe {
            AtomicU32::from_ptr(self.base.sub(CONTROL_OFFSET as usize).cast::<u32>().add(slot))
        }
    }

    pub fn code_generation(&self) -> u64 {
        self.code_generation.load(Acquire)
    }

    /// Force every thread's caches to flush, e.g. for `riscv_flush_icache`.
    pub fn invalidate_caches(&self) {
        self.generation.fetch_add(1, Release);
        self.code_generation.fetch_add(1, Release);
    }

    fn changed(&self, code: bool) {
        self.generation.fetch_add(1, Release);
        if code {
            self.code_generation.fetch_add(1, Release);
        }
    }

    /// Permissions of the guest page containing `addr`, or `None` if it is unmapped.
    pub fn page_prot(&self, addr: u64) -> Option<Prot> {
        let page = addr & !(GUEST_PAGE_SIZE - 1);
        let maps = self.maps();
        match maps.0.range(..=page).next_back() {
            Some((_, m)) if m.end > page => Some(m.prot),
            _ => None,
        }
    }

    fn maps(&self) -> RwLockReadGuard<'_, Maps> {
        self.maps.read().unwrap_or_else(|e| e.into_inner())
    }

    fn maps_mut(&self) -> RwLockWriteGuard<'_, Maps> {
        self.maps.write().unwrap_or_else(|e| e.into_inner())
    }

    fn check_range(addr: u64, len: u64) -> Result<Range<u64>, MapError> {
        let end = addr.checked_add(len).ok_or(MapError::Invalid)?;
        if len == 0
            || !addr.is_multiple_of(GUEST_PAGE_SIZE)
            || !len.is_multiple_of(GUEST_PAGE_SIZE)
            || end > GUEST_SIZE
        {
            return Err(MapError::Invalid);
        }
        Ok(addr..end)
    }

    /// Map `[addr, addr + len)` with `prot`, zero-filled, replacing anything already mapped
    /// there (like `MAP_FIXED`). `addr` and `len` must be 4 KiB aligned.
    pub fn map(&self, addr: u64, len: u64, prot: Prot) -> Result<(), MapError> {
        let r = Self::check_range(addr, len)?;
        let mut maps = self.maps_mut();
        self.unmap_locked(&mut maps, r.clone());
        self.insert_locked(&mut maps, r, prot);
        Ok(())
    }

    /// Map `[addr, addr + len)` only if no page of it is mapped yet. Returns whether it
    /// did. Check and map happen atomically with respect to other threads.
    pub fn map_if_free(&self, addr: u64, len: u64, prot: Prot) -> Result<bool, MapError> {
        let r = Self::check_range(addr, len)?;
        let mut maps = self.maps_mut();
        if maps.is_any_mapped(r.clone()) {
            return Ok(false);
        }
        self.insert_locked(&mut maps, r, prot);
        Ok(true)
    }

    /// Map `len` bytes in the highest free range ending at or below `limit`, and return its
    /// address. Search and map happen atomically with respect to other threads.
    pub fn map_anywhere(&self, len: u64, limit: u64, prot: Prot) -> Result<u64, MapError> {
        if len == 0 || !len.is_multiple_of(GUEST_PAGE_SIZE) {
            return Err(MapError::Invalid);
        }
        let mut maps = self.maps_mut();
        let addr = maps.find_free(len, limit)?;
        self.insert_locked(&mut maps, addr..addr + len, prot);
        Ok(addr)
    }

    /// Unmap `[addr, addr + len)`. Unmapped pages in the range are ignored.
    pub fn unmap(&self, addr: u64, len: u64) -> Result<(), MapError> {
        let r = Self::check_range(addr, len)?;
        self.unmap_locked(&mut self.maps_mut(), r);
        Ok(())
    }

    /// Change the permissions of `[addr, addr + len)`, which must be fully mapped.
    pub fn protect(&self, addr: u64, len: u64, prot: Prot) -> Result<(), MapError> {
        let r = Self::check_range(addr, len)?;
        let mut maps = self.maps_mut();
        if maps.covering(r.start, len, Prot::NONE).is_err() {
            return Err(MapError::NotMapped);
        }
        let mut code = prot.contains(Prot::EXEC);
        for (start, _) in maps.split_out(r) {
            let m = maps.0.get_mut(&start).unwrap();
            code |= m.prot.contains(Prot::EXEC);
            m.prot = prot;
        }
        self.changed(code);
        Ok(())
    }

    /// The highest free, page-aligned range of `len` bytes that ends at or below `limit`.
    /// Only a snapshot: use [`map_anywhere`](Self::map_anywhere) to claim it.
    pub fn find_free(&self, len: u64, limit: u64) -> Result<u64, MapError> {
        if len == 0 || !len.is_multiple_of(GUEST_PAGE_SIZE) {
            return Err(MapError::Invalid);
        }
        self.maps().find_free(len, limit)
    }

    /// Whether any page of `r` is mapped (a snapshot).
    pub fn is_any_mapped(&self, r: Range<u64>) -> bool {
        self.maps().is_any_mapped(r)
    }

    fn insert_locked(&self, maps: &mut Maps, r: Range<u64>, prot: Prot) {
        self.commit_host(r.clone());
        maps.0.insert(r.start, Mapping { end: r.end, prot });
        self.changed(prot.contains(Prot::EXEC));
    }

    fn unmap_locked(&self, maps: &mut Maps, r: Range<u64>) {
        let mut code = false;
        for (start, m) in maps.split_out(r) {
            code |= m.prot.contains(Prot::EXEC);
            maps.0.remove(&start);
            self.release(maps, start..m.end);
        }
        self.changed(code);
    }

    /// Host-page-aligned range covering guest range `r`.
    fn host_pages(&self, r: Range<u64>) -> Range<u64> {
        let mask = self.host_page - 1;
        (r.start & !mask)..((r.end + mask) & !mask)
    }

    fn commit_host(&self, r: Range<u64>) {
        let hr = self.host_pages(r);
        // SAFETY: the range lies inside our reservation. Making already-committed pages
        // read/write again is a no-op for their contents.
        let rc = unsafe {
            libc::mprotect(
                self.base.add(hr.start as usize).cast(),
                (hr.end - hr.start) as usize,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        assert_eq!(rc, 0, "mprotect failed: {}", std::io::Error::last_os_error());
    }

    /// Restore the zero invariant for a just-unmapped guest range: decommit host pages no
    /// longer used by any mapping, and zero the bytes of pages still shared with one.
    /// Runs under the write lock, so no data access can be in flight.
    fn release(&self, maps: &Maps, r: Range<u64>) {
        let hp = self.host_page;
        let hr = self.host_pages(r.clone());
        let mut page = hr.start;
        while page < hr.end {
            let p = page..page + hp;
            if maps.is_any_mapped(p.clone()) {
                let z = p.start.max(r.start)..p.end.min(r.end);
                // SAFETY: committed (a mapping shares this host page), inside the
                // reservation, and no access is in flight under the write lock.
                unsafe {
                    std::ptr::write_bytes(
                        self.base.add(z.start as usize),
                        0,
                        (z.end - z.start) as usize,
                    )
                };
            } else {
                // SAFETY: replaces the page with a fresh PROT_NONE one inside our own
                // reservation, discarding its contents.
                let rc = unsafe {
                    libc::mmap(
                        self.base.add(p.start as usize).cast(),
                        hp as usize,
                        libc::PROT_NONE,
                        libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_NORESERVE | libc::MAP_FIXED,
                        -1,
                        0,
                    )
                };
                assert_ne!(rc, libc::MAP_FAILED, "{}", std::io::Error::last_os_error());
            }
            page += hp;
        }
    }

    /// Validate `[addr, addr + len)` for `need` and return it as a [`GuestBuf`].
    pub fn host_buf(&self, addr: u64, len: u64, need: Prot) -> Result<GuestBuf, MemFault> {
        if len > 0 {
            self.maps().covering(addr, len, need)?;
        }
        // SAFETY: for len > 0 the range is inside the reservation; for len == 0 the
        // pointer is never dereferenced, and wrapping_add avoids out-of-bounds UB.
        Ok(GuestBuf { ptr: self.base.wrapping_add(addr as usize), len: len as usize })
    }

    /// Copy guest memory into `buf` after checking `need`; atomic per aligned access.
    fn load(&self, addr: u64, buf: &mut [u8], need: Prot) -> Result<(), MemFault> {
        let maps = self.maps();
        maps.covering(addr, buf.len() as u64, need)?;
        // SAFETY: the range is mapped (committed read/write) and stays so while we hold
        // the read lock; accesses are atomic, so racing guest stores are not UB.
        unsafe { atomic_copy_out(self.base.add(addr as usize), buf) };
        Ok(())
    }

    fn store(&self, addr: u64, buf: &[u8], need: Prot) -> Result<(), MemFault> {
        let maps = self.maps();
        maps.covering(addr, buf.len() as u64, need)?;
        // SAFETY: as in `load`.
        unsafe { atomic_copy_in(self.base.add(addr as usize), buf) };
        Ok(())
    }

    /// Copy `len` bytes from guest `src` to guest `dst` regardless of permissions (for
    /// `mremap`). Both ranges must be mapped. Holds the write lock, so no guest access races.
    pub fn copy_privileged(&self, src: u64, dst: u64, len: u64) -> Result<(), MemFault> {
        let maps = self.maps_mut();
        maps.covering(src, len, Prot::NONE)?;
        maps.covering(dst, len, Prot::NONE)?;
        // SAFETY: both ranges are mapped (committed) and the write lock excludes every
        // other access; `copy` allows overlap.
        unsafe {
            std::ptr::copy(self.base.add(src as usize), self.base.add(dst as usize), len as usize)
        };
        Ok(())
    }

    /// Zero `len` mapped bytes at `addr` regardless of permissions (for `MADV_DONTNEED`).
    pub fn zero_privileged(&self, addr: u64, len: u64) -> Result<(), MemFault> {
        let maps = self.maps_mut();
        maps.covering(addr, len, Prot::NONE)?;
        // SAFETY: mapped, and the write lock excludes every other access.
        unsafe { std::ptr::write_bytes(self.base.add(addr as usize), 0, len as usize) };
        Ok(())
    }

    /// Read regardless of guest permissions (for the debugger). The range must be mapped.
    pub fn read_privileged(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        self.load(addr, buf, Prot::NONE)
    }

    /// Write regardless of guest permissions (for the loader and the kernel side of
    /// syscalls). The range must still be mapped.
    pub fn write_privileged(&self, addr: u64, buf: &[u8]) -> Result<(), MemFault> {
        self.store(addr, buf, Prot::NONE)
    }

    /// Read a NUL-terminated string of at most `max` bytes (excluding the NUL).
    pub fn read_cstr(&self, addr: u64, max: usize) -> Result<std::ffi::CString, MemFault> {
        let mut out = Vec::new();
        for i in 0..=max as u64 {
            let mut b = [0u8];
            self.load(addr + i, &mut b, Prot::READ)?;
            if b[0] == 0 {
                return Ok(std::ffi::CString::new(out).unwrap());
            }
            out.push(b[0]);
        }
        Err(MemFault { addr })
    }
}

/// Copy from shared memory using atomic loads: one access for naturally aligned 1/2/4/8-byte
/// values, bytewise otherwise.
///
/// # Safety
/// `src..src + buf.len()` must be valid for reads for the duration of the call.
#[inline]
pub(super) unsafe fn atomic_copy_out(src: *mut u8, buf: &mut [u8]) {
    // SAFETY (all arms): `src` is valid per the caller and aligned where a wider atomic is
    // used.
    unsafe {
        match buf.len() {
            8 if src.cast::<u64>().is_aligned() => {
                buf.copy_from_slice(&AtomicU64::from_ptr(src.cast()).load(Relaxed).to_le_bytes())
            }
            4 if src.cast::<u32>().is_aligned() => {
                buf.copy_from_slice(&AtomicU32::from_ptr(src.cast()).load(Relaxed).to_le_bytes())
            }
            2 if src.cast::<u16>().is_aligned() => {
                buf.copy_from_slice(&AtomicU16::from_ptr(src.cast()).load(Relaxed).to_le_bytes())
            }
            _ => {
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = AtomicU8::from_ptr(src.add(i)).load(Relaxed);
                }
            }
        }
    }
}

/// Counterpart of [`atomic_copy_out`] for stores.
///
/// # Safety
/// `dst..dst + buf.len()` must be valid for writes for the duration of the call.
#[inline]
pub(super) unsafe fn atomic_copy_in(dst: *mut u8, buf: &[u8]) {
    // SAFETY (all arms): as in `atomic_copy_out`.
    unsafe {
        match buf.len() {
            8 if dst.cast::<u64>().is_aligned() => AtomicU64::from_ptr(dst.cast())
                .store(u64::from_le_bytes(buf.try_into().unwrap()), Relaxed),
            4 if dst.cast::<u32>().is_aligned() => AtomicU32::from_ptr(dst.cast())
                .store(u32::from_le_bytes(buf.try_into().unwrap()), Relaxed),
            2 if dst.cast::<u16>().is_aligned() => AtomicU16::from_ptr(dst.cast())
                .store(u16::from_le_bytes(buf.try_into().unwrap()), Relaxed),
            _ => {
                for (i, &b) in buf.iter().enumerate() {
                    AtomicU8::from_ptr(dst.add(i)).store(b, Relaxed);
                }
            }
        }
    }
}

impl Drop for GuestMemory {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly the reservation created in `new`.
        unsafe {
            let start = self.base.wrapping_sub(CONTROL_OFFSET as usize);
            libc::munmap(start.cast(), (CONTROL_OFFSET + GUEST_SIZE + GUARD_SIZE) as usize)
        };
    }
}

impl Memory for GuestMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        self.load(addr, buf, Prot::READ)
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemFault> {
        self.store(addr, buf, Prot::WRITE)
    }

    fn fetch(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        self.load(addr, buf, Prot::EXEC)
    }

    fn atomic_update(
        &self,
        addr: u64,
        size: usize,
        f: &mut dyn FnMut(u64) -> u64,
    ) -> Result<u64, MemFault> {
        let maps = self.maps();
        maps.covering(addr, size as u64, Prot::RW)?;
        let p = self.base.wrapping_add(addr as usize);
        assert!(addr.is_multiple_of(size as u64), "atomics must be naturally aligned");
        // SAFETY: mapped and aligned; the read lock keeps it mapped.
        let old = unsafe {
            match size {
                4 => AtomicU32::from_ptr(p.cast())
                    .try_update(SeqCst, SeqCst, |v| Some(f(v as u64) as u32))
                    .unwrap() as u64,
                8 => AtomicU64::from_ptr(p.cast())
                    .try_update(SeqCst, SeqCst, |v| Some(f(v)))
                    .unwrap(),
                _ => panic!("unsupported atomic size {size}"),
            }
        };
        Ok(old)
    }

    fn compare_exchange(
        &self,
        addr: u64,
        size: usize,
        expected: u64,
        new: u64,
    ) -> Result<bool, MemFault> {
        let maps = self.maps();
        maps.covering(addr, size as u64, Prot::RW)?;
        let p = self.base.wrapping_add(addr as usize);
        assert!(addr.is_multiple_of(size as u64), "atomics must be naturally aligned");
        // SAFETY: mapped and aligned; the read lock keeps it mapped.
        let ok = unsafe {
            match size {
                4 => AtomicU32::from_ptr(p.cast())
                    .compare_exchange(expected as u32, new as u32, SeqCst, SeqCst)
                    .is_ok(),
                8 => AtomicU64::from_ptr(p.cast())
                    .compare_exchange(expected, new, SeqCst, SeqCst)
                    .is_ok(),
                _ => panic!("unsupported atomic size {size}"),
            }
        };
        Ok(ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = GUEST_PAGE_SIZE;

    #[test]
    fn map_read_write_unmap() {
        let m = GuestMemory::new().unwrap();
        m.map(0x10000, 2 * P, Prot::RW).unwrap();
        m.write(0x10ffe, &[1, 2, 3, 4]).unwrap(); // across the page boundary
        let mut b = [0; 4];
        m.read(0x10ffe, &mut b).unwrap();
        assert_eq!(b, [1, 2, 3, 4]);

        assert_eq!(m.read(0x12000, &mut b), Err(MemFault { addr: 0x12000 }));
        assert_eq!(m.read(0x11ffe, &mut b), Err(MemFault { addr: 0x12000 }));

        m.unmap(0x10000, P).unwrap();
        assert_eq!(m.read(0x10ffe, &mut b), Err(MemFault { addr: 0x10ffe }));
        m.read(0x11000, &mut b[..2]).unwrap();
        assert_eq!(b[..2], [3, 4]);
    }

    #[test]
    fn permissions_are_enforced_per_guest_page() {
        let m = GuestMemory::new().unwrap();
        // Two guest pages that share one 16 KiB host page.
        m.map(0x20000, P, Prot::READ | Prot::EXEC).unwrap();
        m.map(0x21000, P, Prot::RW).unwrap();
        assert!(m.write(0x20000, &[1]).is_err());
        m.write(0x21000, &[1]).unwrap();
        let mut b = [0; 2];
        assert!(m.fetch(0x21000, &mut b).is_err());
        m.fetch(0x20000, &mut b).unwrap();

        m.protect(0x20000, P, Prot::RW).unwrap();
        m.write(0x20000, &[1]).unwrap();
        assert_eq!(m.protect(0x20000, 4 * P, Prot::RW), Err(MapError::NotMapped));

        // Privileged writes ignore permissions but not mappings.
        m.protect(0x20000, P, Prot::NONE).unwrap();
        m.write_privileged(0x20000, &[9]).unwrap();
        assert!(m.write_privileged(0x30000, &[9]).is_err());
    }

    #[test]
    fn remapping_reads_as_zero() {
        let m = GuestMemory::new().unwrap();
        m.map(0x40000, 2 * P, Prot::RW).unwrap();
        m.write(0x40000, &[0xaa; 8192]).unwrap();
        // Unmap one guest page whose host page is still shared with the other one.
        m.unmap(0x41000, P).unwrap();
        m.map(0x41000, P, Prot::RW).unwrap();
        let mut b = [0xff; 8];
        m.read(0x41000, &mut b).unwrap();
        assert_eq!(b, [0; 8]);
        // Replacing in place (MAP_FIXED semantics) also zeroes.
        m.map(0x40000, P, Prot::RW).unwrap();
        m.read(0x40000, &mut b).unwrap();
        assert_eq!(b, [0; 8]);
    }

    #[test]
    fn protect_splits_mappings() {
        let m = GuestMemory::new().unwrap();
        m.map(0x50000, 3 * P, Prot::RW).unwrap();
        m.protect(0x51000, P, Prot::READ).unwrap();
        m.write(0x50000, &[1]).unwrap();
        assert!(m.write(0x51000, &[1]).is_err());
        m.write(0x52000, &[1]).unwrap();
    }

    #[test]
    fn find_free_searches_top_down() {
        let m = GuestMemory::new().unwrap();
        let top = 0x100000;
        assert_eq!(m.find_free(P, top), Ok(top - P));
        m.map(top - P, P, Prot::RW).unwrap();
        assert_eq!(m.find_free(P, top), Ok(top - 2 * P));
        m.map(top - 3 * P, P, Prot::RW).unwrap();
        // One-page hole at top - 2P; a two-page request must go below the second mapping.
        assert_eq!(m.find_free(P, top), Ok(top - 2 * P));
        assert_eq!(m.find_free(2 * P, top), Ok(top - 5 * P));
        assert_eq!(m.find_free(P, P), Ok(0));
        assert_eq!(m.find_free(2 * P, P), Err(MapError::NoSpace));
    }

    #[test]
    fn rejects_bad_ranges() {
        let m = GuestMemory::new().unwrap();
        assert_eq!(m.map(0x1001, P, Prot::RW), Err(MapError::Invalid));
        assert_eq!(m.map(0x1000, 100, Prot::RW), Err(MapError::Invalid));
        assert_eq!(m.map(0x1000, 0, Prot::RW), Err(MapError::Invalid));
        assert_eq!(m.map(GUEST_SIZE - P, 2 * P, Prot::RW), Err(MapError::Invalid));
        assert_eq!(m.map(u64::MAX - P + 1, P, Prot::RW), Err(MapError::Invalid));
        let mut b = [0; 1];
        assert!(m.read(u64::MAX, &mut b).is_err());
        m.map(GUEST_SIZE - P, P, Prot::RW).unwrap(); // the very top is usable
    }

    #[test]
    fn read_cstr() {
        let m = GuestMemory::new().unwrap();
        m.map(0x60000, P, Prot::RW).unwrap();
        m.write(0x60ffb, b"abc\0").unwrap();
        assert_eq!(m.read_cstr(0x60ffb, 10).unwrap().as_bytes(), b"abc");
        assert!(m.read_cstr(0x60ffb, 2).is_err()); // too long
        m.write(0x60ffc, b"xyzw").unwrap(); // runs off the mapping
        assert!(m.read_cstr(0x60ffc, 100).is_err());
    }

    #[test]
    fn map_if_free_and_map_anywhere() {
        let m = GuestMemory::new().unwrap();
        assert_eq!(m.map_if_free(0x70000, P, Prot::RW), Ok(true));
        assert_eq!(m.map_if_free(0x6f000, 2 * P, Prot::RW), Ok(false), "overlaps");
        let top = 0x200000;
        let a = m.map_anywhere(2 * P, top, Prot::RW).unwrap();
        assert_eq!(a, top - 2 * P);
        assert_eq!(m.map_anywhere(P, top, Prot::RW), Ok(top - 3 * P));
    }

    #[test]
    fn atomics_are_atomic_across_threads() {
        let m = GuestMemory::new().unwrap();
        m.map(0x80000, P, Prot::RW).unwrap();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..10_000 {
                        m.atomic_update(0x80000, 8, &mut |v| v + 1).unwrap();
                        m.atomic_update(0x80008, 4, &mut |v| v + 1).unwrap();
                    }
                });
            }
        });
        let mut b = [0u8; 12];
        m.read(0x80000, &mut b).unwrap();
        assert_eq!(u64::from_le_bytes(b[..8].try_into().unwrap()), 80_000);
        assert_eq!(u32::from_le_bytes(b[8..].try_into().unwrap()), 80_000);

        assert_eq!(m.compare_exchange(0x80008, 4, 80_000, 1), Ok(true));
        assert_eq!(m.compare_exchange(0x80008, 4, 80_000, 2), Ok(false));
        assert!(m.atomic_update(0x90000, 4, &mut |v| v).is_err(), "unmapped");
    }

    #[test]
    fn host_buf_validates_then_points_into_the_reservation() {
        let m = GuestMemory::new().unwrap();
        m.map(0xa0000, P, Prot::READ).unwrap();
        let b = m.host_buf(0xa0010, 16, Prot::READ).unwrap();
        assert_eq!(b.ptr, m.host_base().wrapping_add(0xa0010));
        assert_eq!(b.len, 16);
        assert!(m.host_buf(0xa0010, 16, Prot::WRITE).is_err());
        assert!(m.host_buf(0xa0ff0, 32, Prot::READ).is_err());
        assert!(m.host_buf(0xb0000, 0, Prot::READ).is_ok(), "empty buffers need no mapping");
    }
}
