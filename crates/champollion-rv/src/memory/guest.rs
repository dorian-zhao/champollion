//! The guest's address space, backed by one large host reservation.
//!
//! Guest address `g` lives at host address `base + g` for every `g < GUEST_SIZE`. The whole
//! range is reserved `PROT_NONE` up front, which costs no physical memory; pages are
//! committed as the guest maps them.
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

use super::{MemFault, Memory};

/// Size of the guest address space: 2^38 bytes (256 GiB), the user half of RISC-V Sv39.
pub const GUEST_SIZE: u64 = 1 << 38;
pub const GUEST_PAGE_SIZE: u64 = 4096;

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

pub struct GuestMemory {
    base: *mut u8,
    host_page: u64,
    /// Non-overlapping guest mappings, keyed by start address. Adjacent mappings are not
    /// merged; nothing depends on that and it keeps splitting simple.
    maps: BTreeMap<u64, Mapping>,
}

impl GuestMemory {
    pub fn new() -> std::io::Result<Self> {
        // SAFETY: anonymous PROT_NONE reservation; no existing memory is affected.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                GUEST_SIZE as usize,
                libc::PROT_NONE,
                libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: sysconf has no preconditions.
        let host_page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        assert!(host_page.is_power_of_two() && host_page >= GUEST_PAGE_SIZE);
        Ok(Self { base: base.cast(), host_page, maps: BTreeMap::new() })
    }

    /// Host address of guest address `addr`. Only meaningful for `addr < GUEST_SIZE`.
    pub fn host_base(&self) -> *mut u8 {
        self.base
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
    pub fn map(&mut self, addr: u64, len: u64, prot: Prot) -> Result<(), MapError> {
        let r = Self::check_range(addr, len)?;
        self.unmap(addr, len)?;
        self.commit_host(r.clone());
        self.maps.insert(r.start, Mapping { end: r.end, prot });
        Ok(())
    }

    /// Unmap `[addr, addr + len)`. Unmapped pages in the range are ignored.
    pub fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MapError> {
        let r = Self::check_range(addr, len)?;
        let touched = self.split_out(r.clone());
        for (start, m) in touched {
            self.maps.remove(&start);
            self.release(start..m.end);
        }
        Ok(())
    }

    /// Change the permissions of `[addr, addr + len)`, which must be fully mapped.
    pub fn protect(&mut self, addr: u64, len: u64, prot: Prot) -> Result<(), MapError> {
        let r = Self::check_range(addr, len)?;
        if !self.is_fully_mapped(r.clone()) {
            return Err(MapError::NotMapped);
        }
        for (start, _) in self.split_out(r) {
            self.maps.get_mut(&start).unwrap().prot = prot;
        }
        Ok(())
    }

    /// The highest free, page-aligned range of `len` bytes that ends at or below `limit`.
    pub fn find_free(&self, len: u64, limit: u64) -> Result<u64, MapError> {
        if len == 0 || !len.is_multiple_of(GUEST_PAGE_SIZE) {
            return Err(MapError::Invalid);
        }
        let mut hi = limit.min(GUEST_SIZE) & !(GUEST_PAGE_SIZE - 1);
        // Walk mappings from the top down, looking for a gap of `len` below `hi`.
        for (&start, m) in self.maps.range(..hi).rev() {
            if m.end <= hi && hi - m.end >= len {
                return Ok(hi - len);
            }
            hi = hi.min(start);
        }
        if hi >= len { Ok(hi - len) } else { Err(MapError::NoSpace) }
    }

    /// Whether any page of `[addr, addr + len)` is mapped.
    pub fn is_any_mapped(&self, r: Range<u64>) -> bool {
        self.maps.range(..r.end).next_back().is_some_and(|(_, m)| m.end > r.start)
    }

    fn is_fully_mapped(&self, r: Range<u64>) -> bool {
        self.covering(r.start, r.end - r.start, Prot::NONE).is_ok()
    }

    /// Split mappings so that `r`'s boundaries fall on mapping boundaries, then return the
    /// mappings inside `r`.
    fn split_out(&mut self, r: Range<u64>) -> Vec<(u64, Mapping)> {
        for at in [r.start, r.end] {
            if let Some((&start, &m)) = self.maps.range(..at).next_back()
                && m.end > at
            {
                self.maps.insert(start, Mapping { end: at, prot: m.prot });
                self.maps.insert(at, m);
            }
        }
        self.maps.range(r.start..r.end).map(|(&s, &m)| (s, m)).collect()
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
    fn release(&mut self, r: Range<u64>) {
        let hp = self.host_page;
        let hr = self.host_pages(r.clone());
        let mut page = hr.start;
        while page < hr.end {
            let p = page..page + hp;
            if self.is_any_mapped(p.clone()) {
                let z = p.start.max(r.start)..p.end.min(r.end);
                // SAFETY: committed (a mapping shares this host page) and inside the
                // reservation.
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

    /// Check that `[addr, addr + len)` is mapped with at least `need` permissions.
    fn covering(&self, addr: u64, len: u64, need: Prot) -> Result<(), MemFault> {
        let end = addr.checked_add(len).filter(|&e| e <= GUEST_SIZE).ok_or(MemFault { addr })?;
        let mut at = addr;
        while at < end {
            match self.maps.range(..=at).next_back() {
                Some((_, m)) if m.end > at && m.prot.contains(need) => at = m.end,
                _ => return Err(MemFault { addr: at }),
            }
        }
        Ok(())
    }

    /// Borrow guest memory as a host slice after checking `need` permissions.
    pub fn slice(&self, addr: u64, len: u64, need: Prot) -> Result<&[u8], MemFault> {
        if len == 0 {
            return Ok(&[]);
        }
        self.covering(addr, len, need)?;
        // SAFETY: the range is mapped, hence committed read/write and inside the
        // reservation; the borrow of `self` prevents unmapping while the slice lives.
        Ok(unsafe { std::slice::from_raw_parts(self.base.add(addr as usize), len as usize) })
    }

    pub fn slice_mut(&mut self, addr: u64, len: u64, need: Prot) -> Result<&mut [u8], MemFault> {
        if len == 0 {
            return Ok(&mut []);
        }
        self.covering(addr, len, need)?;
        // SAFETY: as in `slice`, and `&mut self` makes the borrow exclusive.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.base.add(addr as usize), len as usize) })
    }

    /// Write regardless of guest permissions (for the loader and the kernel side of
    /// syscalls). The range must still be mapped.
    pub fn write_privileged(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemFault> {
        self.slice_mut(addr, buf.len() as u64, Prot::NONE)?.copy_from_slice(buf);
        Ok(())
    }

    /// Read a NUL-terminated string of at most `max` bytes (excluding the NUL).
    pub fn read_cstr(&self, addr: u64, max: usize) -> Result<std::ffi::CString, MemFault> {
        let mut out = Vec::new();
        for i in 0..=max as u64 {
            let b = self.slice(addr + i, 1, Prot::READ)?[0];
            if b == 0 {
                return Ok(std::ffi::CString::new(out).unwrap());
            }
            out.push(b);
        }
        Err(MemFault { addr })
    }
}

impl Drop for GuestMemory {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly the reservation created in `new`.
        unsafe { libc::munmap(self.base.cast(), GUEST_SIZE as usize) };
    }
}

impl Memory for GuestMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        buf.copy_from_slice(self.slice(addr, buf.len() as u64, Prot::READ)?);
        Ok(())
    }

    fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemFault> {
        self.slice_mut(addr, buf.len() as u64, Prot::WRITE)?.copy_from_slice(buf);
        Ok(())
    }

    fn fetch(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        buf.copy_from_slice(self.slice(addr, buf.len() as u64, Prot::EXEC)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = GUEST_PAGE_SIZE;

    #[test]
    fn map_read_write_unmap() {
        let mut m = GuestMemory::new().unwrap();
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
        let mut m = GuestMemory::new().unwrap();
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
        let mut m = GuestMemory::new().unwrap();
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
        let mut m = GuestMemory::new().unwrap();
        m.map(0x50000, 3 * P, Prot::RW).unwrap();
        m.protect(0x51000, P, Prot::READ).unwrap();
        m.write(0x50000, &[1]).unwrap();
        assert!(m.write(0x51000, &[1]).is_err());
        m.write(0x52000, &[1]).unwrap();
    }

    #[test]
    fn find_free_searches_top_down() {
        let mut m = GuestMemory::new().unwrap();
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
        let mut m = GuestMemory::new().unwrap();
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
        let mut m = GuestMemory::new().unwrap();
        m.map(0x60000, P, Prot::RW).unwrap();
        m.write(0x60ffb, b"abc\0").unwrap();
        assert_eq!(m.read_cstr(0x60ffb, 10).unwrap().as_bytes(), b"abc");
        assert!(m.read_cstr(0x60ffb, 2).is_err()); // too long
        m.write(0x60ffc, b"xyzw").unwrap(); // runs off the mapping
        assert!(m.read_cstr(0x60ffc, 100).is_err());
    }
}
