//! Per-thread software TLB: caches guest page permissions so that most interpreter memory
//! accesses skip the mapping lock and the B-tree lookup.
//!
//! A hit costs a tag compare and a permission test, then goes straight to host memory. A
//! miss (or an access that crosses a guest page) falls back to [`GuestMemory`]'s locked
//! path and refills the entry. The owner must [`flush`](Tlb::flush) whenever
//! [`GuestMemory::generation`] changes; see the concurrency notes there.

use std::cell::Cell;

use super::guest::{atomic_copy_in, atomic_copy_out};
use super::{GUEST_PAGE_SIZE, GUEST_SIZE, GuestMemory, MemFault, Memory, Prot};

const ENTRIES: usize = 256;
const PAGE_SHIFT: u32 = GUEST_PAGE_SIZE.trailing_zeros();

#[derive(Clone, Copy, Default)]
struct Entry {
    /// Guest page number + 1; 0 marks an empty entry.
    tag: u64,
    prot: Prot,
}

/// Direct-mapped, indexed by the low bits of the guest page number. Not `Sync`: one per
/// guest thread.
pub struct Tlb {
    entries: Box<[Cell<Entry>]>,
}

impl Default for Tlb {
    fn default() -> Self {
        Self { entries: (0..ENTRIES).map(|_| Cell::default()).collect() }
    }
}

impl Tlb {
    pub fn flush(&self) {
        for e in self.entries.iter() {
            e.set(Entry::default());
        }
    }
}

/// [`Memory`] view of a [`GuestMemory`] through a [`Tlb`].
pub struct TlbMemory<'a> {
    pub mem: &'a GuestMemory,
    pub tlb: &'a Tlb,
}

impl TlbMemory<'_> {
    /// Host pointer for `[addr, addr + len)` if it lies within one guest page that the TLB
    /// says (after a refill if needed) grants `need`. `None` sends the caller down the
    /// slow path, which also produces the precise fault.
    #[inline]
    fn host(&self, addr: u64, len: usize, need: Prot) -> Option<*mut u8> {
        let page = addr >> PAGE_SHIFT;
        let offset = addr & (GUEST_PAGE_SIZE - 1);
        if offset + len as u64 > GUEST_PAGE_SIZE || addr >= GUEST_SIZE {
            return None;
        }
        let slot = &self.tlb.entries[page as usize % ENTRIES];
        let mut e = slot.get();
        if e.tag != page + 1 {
            e = Entry { tag: page + 1, prot: self.mem.page_prot(addr)? };
            slot.set(e);
        }
        e.prot.contains(need).then(|| self.mem.host_base().wrapping_add(addr as usize))
    }
}

impl Memory for TlbMemory<'_> {
    #[inline]
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        match self.host(addr, buf.len(), Prot::READ) {
            // SAFETY: a TLB hit means the page was mapped readable at the current
            // generation, so it is committed host memory inside the reservation.
            Some(p) => unsafe { atomic_copy_out(p, buf) },
            None => return self.mem.read(addr, buf),
        }
        Ok(())
    }

    #[inline]
    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemFault> {
        match self.host(addr, buf.len(), Prot::WRITE) {
            // SAFETY: as in `read`.
            Some(p) => unsafe { atomic_copy_in(p, buf) },
            None => return self.mem.write(addr, buf),
        }
        Ok(())
    }

    #[inline]
    fn fetch(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        match self.host(addr, buf.len(), Prot::EXEC) {
            // SAFETY: as in `read`.
            Some(p) => unsafe { atomic_copy_out(p, buf) },
            None => return self.mem.fetch(addr, buf),
        }
        Ok(())
    }

    // Atomics are rare enough to always take the locked path.
    fn atomic_update(
        &self,
        addr: u64,
        size: usize,
        f: &mut dyn FnMut(u64) -> u64,
    ) -> Result<u64, MemFault> {
        self.mem.atomic_update(addr, size, f)
    }

    fn compare_exchange(
        &self,
        addr: u64,
        size: usize,
        expected: u64,
        new: u64,
    ) -> Result<bool, MemFault> {
        self.mem.compare_exchange(addr, size, expected, new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = GUEST_PAGE_SIZE;

    #[test]
    fn hits_misses_and_faults_match_the_slow_path() {
        let mem = GuestMemory::new().unwrap();
        mem.map(0x10000, P, Prot::READ).unwrap();
        mem.map(0x11000, P, Prot::RW).unwrap();
        mem.write_privileged(0x10ff8, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let tlb = Tlb::default();
        let tm = TlbMemory { mem: &mem, tlb: &tlb };

        let mut b = [0u8; 8];
        tm.read(0x10ff8, &mut b).unwrap(); // miss, then fill
        tm.read(0x10ff8, &mut b).unwrap(); // hit
        assert_eq!(b, [1, 2, 3, 4, 5, 6, 7, 8]);
        // Read-only page: the fault comes from the slow path with the same address.
        assert_eq!(tm.write(0x10000, &[0]), mem.write(0x10000, &[0]));
        // Crossing into the next page takes the slow path and still works.
        tm.write(0x11ffe, &[9, 9]).unwrap();
        assert!(tm.write(0x10ffe, &[9, 9, 9, 9]).is_err(), "first half is read-only");
        // Unmapped pages are not cached as present.
        assert_eq!(tm.read(0x20000, &mut b), Err(MemFault { addr: 0x20000 }));
        assert_eq!(tm.read(GUEST_SIZE, &mut b), mem.read(GUEST_SIZE, &mut b));
    }

    #[test]
    fn stale_entries_survive_until_flushed() {
        let mem = GuestMemory::new().unwrap();
        mem.map(0x10000, P, Prot::RW).unwrap();
        let tlb = Tlb::default();
        let tm = TlbMemory { mem: &mem, tlb: &tlb };
        tm.write(0x10000, &[1]).unwrap();
        let before = mem.generation();
        mem.protect(0x10000, P, Prot::READ).unwrap();
        assert_ne!(mem.generation(), before, "mapping changes bump the generation");
        tlb.flush();
        assert!(tm.write(0x10000, &[1]).is_err());
    }
}
