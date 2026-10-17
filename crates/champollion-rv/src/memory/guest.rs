//! The guest's address space, backed by one large host reservation.
//!
//! Guest address `g` lives at host address `base + g` for every `g < GUEST_SIZE`. The whole
//! range is reserved `PROT_NONE` up front, which costs no physical memory; pages are
//! committed as the guest maps them. [`GUARD_SIZE`] bytes on either side stay `PROT_NONE`
//! forever, so the JIT may add a small offset to an in-range address without masking the
//! sum: the access either lands in the guest space or faults. A read/write control area
//! holds one interrupt flag per thread slot for the runtime, which generated code checks
//! (see [`GuestMemory::interrupt_flag`]).
//!
//! There are two layouts. The offset layout (macOS, and the fallback) puts the guest space
//! anywhere and the control area just below its lower guard:
//!
//! ```text
//! base - CONTROL_OFFSET   interrupt flags (INTERRUPT_SLOTS x u32)
//! base - GUARD_SIZE       guard
//! base                    guest address 0
//! base + GUEST_SIZE       guard
//! ```
//!
//! The identity layout (Linux, VM mode) puts guest address `g` at host address `g`
//! (`base` is 0), so guest pointers are host pointers and system calls can pass them to the
//! kernel as they are. The guest space then starts at the lowest address the host lets a
//! process map ([`GuestMemory::lowest`]), what lies below acts as the lower guard, and the
//! control area is a separate mapping. macOS reserves the low 4 GiB of every process, so
//! it always uses the offset layout.
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
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

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
    /// The host refused a shared mapping (with this Linux errno).
    Host(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mapping {
    end: u64,
    prot: Prot,
    /// Backed by a host `MAP_SHARED` mapping (a file or memory shared across fork), not
    /// by the reservation's private pages.
    shared: bool,
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

/// See [`GuestMemory::lock_for_fork`].
pub struct ForkGuard<'a> {
    _maps: RwLockWriteGuard<'a, Maps>,
    readers: std::sync::MutexGuard<'a, Vec<Weak<AtomicU64>>>,
    _pending: std::sync::MutexGuard<'a, Vec<Range<u64>>>,
}

impl ForkGuard<'_> {
    /// In the child: the other threads are gone, and their TLBs with them. Without this,
    /// `settle` would wait forever for readers that never move on.
    pub fn forget_other_threads(&mut self) {
        for r in self.readers.iter().filter_map(Weak::upgrade) {
            r.store(QUIESCENT, SeqCst);
        }
    }
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
                self.0.insert(start, Mapping { end: at, ..m });
                self.0.insert(at, m);
            }
        }
        self.0.range(r.start..r.end).map(|(&s, &m)| (s, m)).collect()
    }

    /// The highest free range of `len` bytes that ends at or below `limit`.
    fn find_free(&self, len: u64, limit: u64, floor: u64) -> Result<u64, MapError> {
        self.find_free_aligned(len, limit, GUEST_PAGE_SIZE, floor)
    }

    /// Like `find_free`, for a range starting at a multiple of `align`. Nothing below
    /// `floor` is free.
    fn find_free_aligned(
        &self,
        len: u64,
        limit: u64,
        align: u64,
        floor: u64,
    ) -> Result<u64, MapError> {
        let mut hi = limit.min(GUEST_SIZE) & !(GUEST_PAGE_SIZE - 1);
        // Walk mappings from the top down, looking for a gap of `len` below `hi`.
        for (&start, m) in self.0.range(..hi).rev() {
            let bottom = m.end.max(floor);
            if hi >= bottom && hi - bottom >= len && ((hi - len) & !(align - 1)) >= bottom {
                return Ok((hi - len) & !(align - 1));
            }
            hi = hi.min(start);
        }
        if hi >= floor + len && ((hi - len) & !(align - 1)) >= floor {
            Ok((hi - len) & !(align - 1))
        } else {
            Err(MapError::NoSpace)
        }
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
    /// Host address of guest address 0, with the provenance of the reservation (in the
    /// identity layout, an address of 0 derived from it).
    base: *mut u8,
    /// The lowest guest address that can be mapped.
    lowest: u64,
    /// The interrupt flags (see [`GuestMemory::interrupt_flag`]).
    control: *mut u32,
    /// What `drop` unmaps: the reservation, and in the identity layout the control area.
    owned: [(*mut libc::c_void, usize); 2],
    host_page: u64,
    maps: RwLock<Maps>,
    generation: AtomicU64,
    code_generation: AtomicU64,
    /// The generation each software TLB user is working with (see [`TlbReader`]).
    readers: Mutex<Vec<Weak<AtomicU64>>>,
    /// Unmapped host pages, already emptied, still to be made inaccessible once no TLB can
    /// point at them (see [`GuestMemory::settle`]).
    pending: Mutex<Vec<Range<u64>>>,
    /// Set by [`GuestMemory::leak_at_exit`]: the process is exiting, so the reservation
    /// stays mapped and the kernel reclaims it.
    leaked: std::sync::atomic::AtomicBool,
}

/// A TLB user that is not using its TLB.
const QUIESCENT: u64 = u64::MAX;

/// Registration of a software TLB with its memory (see [`GuestMemory::tlb_reader`]).
///
/// TLB hits skip the mapping lock, so a TLB could still point at a page another thread
/// unmaps. Unmapping therefore replaces the page with fresh, still accessible zero pages
/// at once, and makes it inaccessible (for generated code's precise faults) only after
/// every reader has moved past the generation of the unmap.
pub struct TlbReader(Arc<AtomicU64>);

impl TlbReader {
    /// Start (or keep) using a TLB: publish the current generation and return it. The TLB
    /// must be flushed if it holds entries of an older generation.
    pub fn enter(&self, mem: &GuestMemory) -> u64 {
        loop {
            let generation = mem.generation.load(SeqCst);
            self.0.store(generation, SeqCst);
            if mem.generation.load(SeqCst) == generation {
                return generation;
            }
        }
    }

    /// Stop using the TLB (until the next `enter`).
    pub fn leave(&self) {
        self.0.store(QUIESCENT, SeqCst);
    }
}

impl Drop for TlbReader {
    fn drop(&mut self) {
        self.leave();
    }
}

// SAFETY: `base` points to a reservation owned by this object for its whole lifetime, and
// all access to the memory behind it is synchronized as described on the type.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

impl GuestMemory {
    /// A guest address space in the identity layout (see the module docs), or `None` if the
    /// host has none (macOS) or the guest range is not free: another `GuestMemory` has it,
    /// or the host program was linked into it (see `.cargo/config.toml`).
    #[cfg(not(target_os = "linux"))]
    pub fn new_identity() -> Option<Self> {
        None
    }

    /// See the other `new_identity`.
    #[cfg(target_os = "linux")]
    pub fn new_identity() -> Option<Self> {
        let host_page = Self::host_page_size();
        let min = std::fs::read_to_string("/proc/sys/vm/mmap_min_addr")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(65536);
        let lowest = min.max(GUEST_PAGE_SIZE).next_multiple_of(host_page);
        let len = (GUEST_SIZE + GUARD_SIZE - lowest) as usize;
        // SAFETY: a PROT_NONE reservation that refuses to replace existing mappings.
        let start = unsafe {
            libc::mmap(
                lowest as *mut libc::c_void,
                len,
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS
                    | libc::MAP_PRIVATE
                    | libc::MAP_NORESERVE
                    | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        if start == libc::MAP_FAILED {
            return None;
        }
        if start as u64 != lowest {
            // A kernel older than MAP_FIXED_NOREPLACE (4.17) took it as a hint.
            // SAFETY: unmapping what we just mapped.
            unsafe { libc::munmap(start, len) };
            return None;
        }
        // SAFETY: a fresh anonymous read/write mapping.
        let control = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                CONTROL_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        if control == libc::MAP_FAILED {
            // SAFETY: unmapping our reservation.
            unsafe { libc::munmap(start, len) };
            return None;
        }
        Some(Self::with_layout(
            start.cast::<u8>().wrapping_sub(lowest as usize),
            lowest,
            control.cast(),
            [(start, len), (control, CONTROL_SIZE as usize)],
            host_page,
        ))
    }

    fn host_page_size() -> u64 {
        // SAFETY: sysconf has no preconditions.
        let host_page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        assert!(host_page.is_power_of_two() && host_page >= GUEST_PAGE_SIZE);
        host_page
    }

    fn with_layout(
        base: *mut u8,
        lowest: u64,
        control: *mut u32,
        owned: [(*mut libc::c_void, usize); 2],
        host_page: u64,
    ) -> Self {
        Self {
            readers: Mutex::default(),
            pending: Mutex::default(),
            base,
            lowest,
            control,
            owned,
            host_page,
            maps: RwLock::default(),
            generation: AtomicU64::new(0),
            code_generation: AtomicU64::new(0),
            leaked: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// A guest address space in the offset layout (see the module docs). Any number of them
    /// can exist at once.
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
        let len = (CONTROL_OFFSET + GUEST_SIZE + GUARD_SIZE) as usize;
        let none = (std::ptr::null_mut(), 0);
        Ok(Self::with_layout(base, 0, start.cast(), [(start, len), none], Self::host_page_size()))
    }

    /// Host address of guest address 0. Guest address `g` is at `host_base() + g`. Null in
    /// the identity layout.
    pub fn host_base(&self) -> *mut u8 {
        self.base
    }

    /// Whether guest addresses are host addresses (the identity layout).
    pub fn is_identity(&self) -> bool {
        self.base.is_null()
    }

    /// The lowest guest address that can be mapped (0 in the offset layout).
    pub fn lowest(&self) -> u64 {
        self.lowest
    }

    /// Start of the control area: the interrupt flags, slot by slot.
    pub fn control_area(&self) -> *mut u32 {
        self.control
    }

    /// Host pointer to guest address `g` (which must be below `GUEST_SIZE + GUARD_SIZE`).
    fn host(&self, g: u64) -> *mut u8 {
        // Wrapping: in the identity layout `base` itself lies outside the reservation.
        self.base.wrapping_add(g as usize)
    }

    /// Counter bumped by every mapping change; see the type-level docs.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation.load(Acquire)
    }

    /// The interrupt flag of thread slot `slot`: nonzero asks the thread to leave generated
    /// code at the next block entry (for signal delivery).
    pub fn interrupt_flag(&self, slot: usize) -> &AtomicU32 {
        assert!(slot < INTERRUPT_SLOTS);
        // SAFETY: the control area is mapped read/write for the reservation's lifetime and
        // only ever accessed atomically; slots are 4-byte aligned.
        unsafe { AtomicU32::from_ptr(self.control.add(slot)) }
    }

    /// Counter bumped by mapping changes that involve executable pages (before or after the
    /// change). Caches of decoded or translated code follow this one, so data-only `mmap`,
    /// `brk`, and `munmap` traffic leaves them intact.
    #[inline]
    pub fn code_generation(&self) -> u64 {
        self.code_generation.load(Acquire)
    }

    /// Register a software TLB (see [`TlbReader`]).
    pub fn tlb_reader(&self) -> TlbReader {
        let slot = Arc::new(AtomicU64::new(QUIESCENT));
        let mut readers = self.readers.lock().unwrap_or_else(|e| e.into_inner());
        readers.retain(|r| r.strong_count() > 0);
        readers.push(Arc::downgrade(&slot));
        TlbReader(slot)
    }

    /// Finish unmapping: once no TLB reader works with a generation older than now, make
    /// the emptied host pages inaccessible, unless they were mapped again meanwhile. Runs
    /// without the mapping lock while waiting, since a reader may need it to make progress.
    fn settle(&self) {
        let mut pending =
            std::mem::take(&mut *self.pending.lock().unwrap_or_else(|e| e.into_inner()));
        // Pages mapped again meanwhile (MAP_FIXED replacing a mapping) need nothing.
        let hp = self.host_page;
        pending.retain(|r| !self.maps().is_any_mapped(r.start..r.start + hp));
        if pending.is_empty() {
            return;
        }
        let target = self.generation.load(SeqCst);
        loop {
            let readers = self.readers.lock().unwrap_or_else(|e| e.into_inner());
            let busy = readers.iter().filter_map(Weak::upgrade).any(|r| r.load(SeqCst) < target);
            drop(readers);
            if !busy {
                break;
            }
            std::thread::yield_now();
        }
        let maps = self.maps_mut();
        for r in pending {
            let mut page = r.start;
            while page < r.end {
                if !maps.is_any_mapped(page..page + hp) {
                    // SAFETY: an unmapped host page of our reservation; no TLB points at it.
                    unsafe { libc::mprotect(self.host(page).cast(), hp as usize, libc::PROT_NONE) };
                }
                page += hp;
            }
        }
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

    /// Hold the mapping table still across a host `fork`, so the child (which keeps only
    /// the forking thread) never inherits it locked by a thread that no longer exists.
    pub fn lock_for_fork(&self) -> ForkGuard<'_> {
        ForkGuard {
            _maps: self.maps_mut(),
            readers: self.readers.lock().unwrap_or_else(|e| e.into_inner()),
            _pending: self.pending.lock().unwrap_or_else(|e| e.into_inner()),
        }
    }

    fn check_range(&self, addr: u64, len: u64) -> Result<Range<u64>, MapError> {
        let end = addr.checked_add(len).ok_or(MapError::Invalid)?;
        if len == 0
            || !addr.is_multiple_of(GUEST_PAGE_SIZE)
            || !len.is_multiple_of(GUEST_PAGE_SIZE)
            || addr < self.lowest
            || end > GUEST_SIZE
        {
            return Err(MapError::Invalid);
        }
        Ok(addr..end)
    }

    /// Map `[addr, addr + len)` with `prot`, zero-filled, replacing anything already mapped
    /// there (like `MAP_FIXED`). `addr` and `len` must be 4 KiB aligned.
    pub fn map(&self, addr: u64, len: u64, prot: Prot) -> Result<(), MapError> {
        let r = self.check_range(addr, len)?;
        let mut maps = self.maps_mut();
        self.unmap_locked(&mut maps, r.clone());
        self.insert_locked(&mut maps, r, prot);
        drop(maps);
        self.settle();
        Ok(())
    }

    /// Map `[addr, addr + len)` only if no page of it is mapped yet. Returns whether it
    /// did. Check and map happen atomically with respect to other threads.
    pub fn map_if_free(&self, addr: u64, len: u64, prot: Prot) -> Result<bool, MapError> {
        let r = self.check_range(addr, len)?;
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
        let addr = maps.find_free(len, limit, self.lowest)?;
        self.insert_locked(&mut maps, addr..addr + len, prot);
        Ok(addr)
    }

    /// Unmap `[addr, addr + len)`. Unmapped pages in the range are ignored.
    pub fn unmap(&self, addr: u64, len: u64) -> Result<(), MapError> {
        let r = self.check_range(addr, len)?;
        self.unmap_locked(&mut self.maps_mut(), r);
        self.settle();
        Ok(())
    }

    /// Change the permissions of `[addr, addr + len)`, which must be fully mapped.
    pub fn protect(&self, addr: u64, len: u64, prot: Prot) -> Result<(), MapError> {
        let r = self.check_range(addr, len)?;
        let mut maps = self.maps_mut();
        if maps.covering(r.start, len, Prot::NONE).is_err() {
            return Err(MapError::NotMapped);
        }
        let mut code = prot.contains(Prot::EXEC);
        // Shared mappings start read-only on the host unless writable (see `map_shared`), so
        // making one writable needs the host's consent: a file opened read-only refuses.
        // (Write access is never taken back on the host; software checks enforce that.)
        let splits = maps.split_out(r);
        if prot.contains(Prot::WRITE) {
            for &(start, m) in &splits {
                if m.shared {
                    let hr = self.host_pages(start..m.end);
                    // SAFETY: host pages of a shared mapping, which it has to itself.
                    let rc = unsafe {
                        libc::mprotect(
                            self.host(hr.start).cast(),
                            (hr.end - hr.start) as usize,
                            libc::PROT_READ | libc::PROT_WRITE,
                        )
                    };
                    if rc != 0 {
                        return Err(MapError::Host(13)); // EACCES
                    }
                }
            }
        }
        for (start, _) in splits {
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
        self.maps().find_free(len, limit, self.lowest)
    }

    /// Whether any page of `r` is mapped (a snapshot).
    pub fn is_any_mapped(&self, r: Range<u64>) -> bool {
        self.maps().is_any_mapped(r)
    }

    fn insert_locked(&self, maps: &mut Maps, r: Range<u64>, prot: Prot) {
        self.commit_host(r.clone());
        maps.0.insert(r.start, Mapping { end: r.end, prot, shared: false });
        self.changed(prot.contains(Prot::EXEC));
    }

    fn unmap_locked(&self, maps: &mut Maps, r: Range<u64>) {
        let mut code = false;
        for (start, m) in maps.split_out(r) {
            code |= m.prot.contains(Prot::EXEC);
            maps.0.remove(&start);
            self.release(maps, start..m.end, m.shared);
        }
        self.changed(code);
    }

    /// Map `len` bytes of a file privately (copy-on-write) at the host-page-aligned guest
    /// `addr`, replacing what is there: an executable's segments, whose pages then page
    /// in lazily instead of being copied. `offset` is host-page aligned.
    pub fn map_file(
        &self,
        addr: u64,
        len: u64,
        prot: Prot,
        file: (i32, u64),
    ) -> Result<(), MapError> {
        let hp = self.host_page;
        let len = len.checked_next_multiple_of(hp).filter(|&l| l != 0).ok_or(MapError::Invalid)?;
        let (fd, offset) = file;
        if !addr.is_multiple_of(hp) || !offset.is_multiple_of(hp) {
            return Err(MapError::Invalid);
        }
        let mut maps = self.maps_mut();
        let r = self.check_range(addr, len)?;
        self.unmap_locked(&mut maps, r.clone());
        let host_prot = if prot.contains(Prot::WRITE) {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };
        // SAFETY: replaces whole host pages of our own reservation, which no other
        // mapping uses (they are free, or were just unmapped).
        let rc = unsafe {
            libc::mmap(
                self.host(addr).cast(),
                len as usize,
                host_prot,
                libc::MAP_PRIVATE | libc::MAP_FIXED,
                fd,
                offset as libc::off_t,
            )
        };
        if rc == libc::MAP_FAILED {
            let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            self.release(&maps, r, false);
            return Err(MapError::Host(match e {
                libc::EACCES => 13,
                libc::ENODEV => 19,
                _ => 12,
            }));
        }
        self.insert_locked(&mut maps, r, prot);
        Ok(())
    }

    /// Map host memory shared across processes at a host-page-aligned guest range: a file
    /// (`Some((fd, offset))`, offset host-page aligned) or anonymous memory that a fork
    /// shares. At `addr`, replacing what is there, or anywhere below `limit`. The range is
    /// rounded up to whole host pages, so no other mapping shares them.
    pub fn map_shared(
        &self,
        addr: Option<u64>,
        len: u64,
        limit: u64,
        prot: Prot,
        file: Option<(i32, u64)>,
    ) -> Result<u64, MapError> {
        let hp = self.host_page;
        let len = len.checked_next_multiple_of(hp).filter(|&l| l != 0).ok_or(MapError::Invalid)?;
        if file.is_some_and(|(_, offset)| !offset.is_multiple_of(hp)) {
            return Err(MapError::Invalid);
        }
        let mut maps = self.maps_mut();
        let addr = match addr {
            Some(a) => {
                let r = self.check_range(a, len)?;
                if !a.is_multiple_of(hp) {
                    return Err(MapError::Invalid);
                }
                self.unmap_locked(&mut maps, r);
                a
            }
            None => maps.find_free_aligned(len, limit, hp, self.lowest)?,
        };
        let (flags, fd, offset) = match file {
            Some((fd, offset)) => (libc::MAP_SHARED, fd, offset as libc::off_t),
            None => (libc::MAP_SHARED | libc::MAP_ANON, -1, 0),
        };
        // Read-only unless the guest asks for writes: a file opened read-only cannot be
        // mapped shared and writable (see `protect` for later changes).
        let host_prot = if prot.contains(Prot::WRITE) {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };
        // SAFETY: replaces whole host pages of our own reservation, which no other mapping
        // uses (they are free, or were just unmapped).
        let rc = unsafe {
            libc::mmap(
                self.host(addr).cast(),
                len as usize,
                host_prot,
                flags | libc::MAP_FIXED,
                fd,
                offset,
            )
        };
        if rc == libc::MAP_FAILED {
            let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            // Put the reservation back.
            self.release(&maps, addr..addr + len, true);
            return Err(MapError::Host(if e == libc::EACCES {
                13
            } else if e == libc::ENODEV {
                19
            } else {
                12
            }));
        }
        maps.0.insert(addr, Mapping { end: addr + len, prot, shared: true });
        self.changed(prot.contains(Prot::EXEC));
        drop(maps);
        self.settle();
        Ok(addr)
    }

    /// Whether `addr` lies in a shared mapping (see [`GuestMemory::map_shared`]).
    pub fn is_shared(&self, addr: u64) -> bool {
        let maps = self.maps();
        maps.0.range(..=addr).next_back().is_some_and(|(_, m)| m.end > addr && m.shared)
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
                self.host(hr.start).cast(),
                (hr.end - hr.start) as usize,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        assert_eq!(rc, 0, "mprotect failed: {}", std::io::Error::last_os_error());
    }

    /// Restore the zero invariant for a just-unmapped guest range: decommit host pages no
    /// longer used by any mapping, and zero the bytes of pages still shared with one.
    /// Runs under the write lock, so no data access can be in flight.
    /// Give the host pages of unmapped range `r` back to the reservation. Pages other
    /// mappings still use are kept, with `r`'s part zeroed, unless `r` was `shared`: its
    /// bytes belong to a file or another process, and must not be overwritten.
    fn release(&self, maps: &Maps, r: Range<u64>, shared: bool) {
        let hp = self.host_page;
        let hr = self.host_pages(r.clone());
        let mut page = hr.start;
        while page < hr.end {
            let p = page..page + hp;
            if maps.is_any_mapped(p.clone()) {
                if shared {
                    page += hp;
                    continue;
                }
                let z = p.start.max(r.start)..p.end.min(r.end);
                // SAFETY: committed (a mapping shares this host page), inside the
                // reservation, and no access is in flight under the write lock.
                unsafe { std::ptr::write_bytes(self.host(z.start), 0, (z.end - z.start) as usize) };
            } else {
                // SAFETY: replaces the page with a fresh zero one inside our own reservation,
                // discarding its contents. It stays accessible for a TLB that may still
                // point at it, and becomes PROT_NONE in `settle`.
                let rc = unsafe {
                    libc::mmap(
                        self.host(p.start).cast(),
                        hp as usize,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_NORESERVE | libc::MAP_FIXED,
                        -1,
                        0,
                    )
                };
                assert_ne!(rc, libc::MAP_FAILED, "{}", std::io::Error::last_os_error());
                self.pending.lock().unwrap_or_else(|e| e.into_inner()).push(p);
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
        Ok(GuestBuf { ptr: self.host(addr), len: len as usize })
    }

    /// Copy guest memory into `buf` after checking `need`; atomic per aligned access.
    fn load(&self, addr: u64, buf: &mut [u8], need: Prot) -> Result<(), MemFault> {
        let maps = self.maps();
        maps.covering(addr, buf.len() as u64, need)?;
        // SAFETY: the range is mapped (committed read/write) and stays so while we hold
        // the read lock; accesses are atomic, so racing guest stores are not UB.
        unsafe { atomic_copy_out(self.host(addr), buf) };
        Ok(())
    }

    fn store(&self, addr: u64, buf: &[u8], need: Prot) -> Result<(), MemFault> {
        let maps = self.maps();
        maps.covering(addr, buf.len() as u64, need)?;
        // SAFETY: as in `load`.
        unsafe { atomic_copy_in(self.host(addr), buf) };
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
        unsafe { std::ptr::copy(self.host(src), self.host(dst), len as usize) };
        Ok(())
    }

    /// Zero `len` mapped bytes at `addr` regardless of permissions (for `MADV_DONTNEED`).
    pub fn zero_privileged(&self, addr: u64, len: u64) -> Result<(), MemFault> {
        let maps = self.maps_mut();
        maps.covering(addr, len, Prot::NONE)?;
        // SAFETY: mapped, and the write lock excludes every other access.
        unsafe { std::ptr::write_bytes(self.host(addr), 0, len as usize) };
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
        if self.leaked.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        for (start, len) in self.owned {
            if len > 0 {
                // SAFETY: unmapping exactly what `new` mapped.
                unsafe { libc::munmap(start, len) };
            }
        }
    }
}

impl GuestMemory {
    /// Skip unmapping the reservation when this drops: only for a process that is about
    /// to exit, where the kernel reclaims the address space anyway. Unmapping the 264 GiB
    /// range explicitly costs about a millisecond on macOS, on every rvrun invocation.
    pub fn leak_at_exit(&self) {
        self.leaked.store(true, std::sync::atomic::Ordering::Relaxed);
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
        let p = self.host(addr);
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
        let p = self.host(addr);
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

    #[test]
    fn shared_file_mappings_follow_the_file_access_mode() {
        use std::os::fd::AsRawFd;
        let m = GuestMemory::new().unwrap();
        let hp = m.host_page;
        let path = std::env::temp_dir().join(format!("champollion-ro-{}", std::process::id()));
        std::fs::write(&path, b"read only").unwrap();
        let ro = std::fs::File::open(&path).unwrap();
        let a = m.map_shared(None, hp, GUEST_SIZE, Prot::READ, Some((ro.as_raw_fd(), 0))).unwrap();
        let mut b = [0u8; 9];
        m.read(a, &mut b).unwrap();
        assert_eq!(&b, b"read only");
        assert_eq!(m.protect(a, hp, Prot::RW), Err(MapError::Host(13)), "EACCES, as on Linux");
        let rw = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        let a = m.map_shared(None, hp, GUEST_SIZE, Prot::READ, Some((rw.as_raw_fd(), 0))).unwrap();
        m.protect(a, hp, Prot::RW).unwrap();
        m.write(a, b"R").unwrap();
        m.unmap(a, hp).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"Read only");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn unmap_waits_for_tlb_readers_before_revoking_host_pages() {
        let m = Arc::new(GuestMemory::new().unwrap());
        let hp = m.host_page;
        m.map(0x40000, hp, Prot::RW).unwrap();
        m.write(0x40000, &[7]).unwrap();
        let reader = m.tlb_reader();
        reader.enter(&m);
        let unmapper = {
            let m = m.clone();
            std::thread::spawn(move || m.unmap(0x40000, hp).unwrap())
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!unmapper.is_finished(), "unmap returned under an active reader");
        // A stale TLB entry still reaches the page: zeroed, but no host fault.
        // SAFETY: the page is pending and still readable.
        let byte = unsafe { std::ptr::read_volatile(m.host_base().add(0x40000)) };
        assert_eq!(byte, 0);
        assert!(m.read(0x40000, &mut [0]).is_err(), "unmapped for everyone else");
        reader.leave();
        unmapper.join().unwrap();
        assert!(m.pending.lock().unwrap().is_empty());
        // Entering again sees the new generation; an idle reader holds nothing up.
        reader.enter(&m);
        reader.leave();
        m.map(0x40000, hp, Prot::RW).unwrap();
        m.unmap(0x40000, hp).unwrap();
    }

    /// The sequence an ELF loader uses: map a large R-X region for a library, then cover
    /// its tail with MAP_FIXED read-write parts. Everything before the fixed maps must
    /// keep its permissions.
    #[test]
    fn fixed_maps_do_not_clobber_the_rest_of_a_mapping() {
        let m = GuestMemory::new().unwrap();
        let base = 0x3fbfc81000;
        let span = 0x189000;
        m.map(base, span, Prot::READ | Prot::EXEC).unwrap();
        m.map(base + 0x178000, 0x5000, Prot::READ | Prot::WRITE).unwrap();
        m.map(base + 0x17d000, 0xc000, Prot::READ | Prot::WRITE).unwrap();
        let text = base + 0x8961e;
        assert_eq!(m.page_prot(text), Some(Prot::READ | Prot::EXEC), "library text");
        let mut b = [0u8; 4];
        m.fetch(text, &mut b).unwrap();
    }
}
