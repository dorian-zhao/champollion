//! Guest memory access.
//!
//! The interpreter talks to memory only through [`Memory`], so the same interpreter runs
//! against a [`FlatMemory`] test buffer or the real [`GuestMemory`] address space.
//!
//! Every method takes `&self`: guest memory is shared between guest threads, so stores
//! never need exclusive access, and atomics are real host atomics.

use std::cell::Cell;

mod guest;
mod tlb;

pub use guest::{GUEST_PAGE_SIZE, GUEST_SIZE, GuestBuf, GuestMemory, MapError, Prot};
pub use tlb::{Tlb, TlbMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemFault {
    pub addr: u64,
}

pub trait Memory {
    /// Data load. Requires read permission.
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault>;
    /// Data store. Requires write permission.
    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemFault>;

    /// Instruction fetch. Requires execute permission where permissions exist.
    fn fetch(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        self.read(addr, buf)
    }

    /// Atomically replace the naturally aligned `size`-byte value (4 or 8) at `addr` with
    /// `f(old)`, and return `old` zero-extended. Requires read and write permission.
    /// `f` may be called more than once if another thread races with the update.
    fn atomic_update(
        &self,
        addr: u64,
        size: usize,
        f: &mut dyn FnMut(u64) -> u64,
    ) -> Result<u64, MemFault>;

    /// Atomically store `new` at `addr` if the current value equals `expected` (both
    /// zero-extended `size`-byte values). Returns whether the store happened.
    fn compare_exchange(
        &self,
        addr: u64,
        size: usize,
        expected: u64,
        new: u64,
    ) -> Result<bool, MemFault>;

    fn fetch_u16(&self, addr: u64) -> Result<u16, MemFault> {
        let mut b = [0; 2];
        self.fetch(addr, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn fetch_u32(&self, addr: u64) -> Result<u32, MemFault> {
        let mut b = [0; 4];
        self.fetch(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
}

/// A single contiguous region starting at `base`. Good enough for unit tests and
/// `riscv-tests`, which are linked at a fixed address. Single-threaded only (`!Sync`), so
/// its "atomics" are plain read-modify-writes.
pub struct FlatMemory {
    pub base: u64,
    bytes: Box<[Cell<u8>]>,
}

impl FlatMemory {
    pub fn new(base: u64, size: usize) -> Self {
        Self { base, bytes: (0..size).map(|_| Cell::new(0)).collect() }
    }

    fn range(&self, addr: u64, len: usize) -> Result<&[Cell<u8>], MemFault> {
        let fault = MemFault { addr };
        let start = addr.checked_sub(self.base).ok_or(fault)? as usize;
        let end = start.checked_add(len).ok_or(fault)?;
        self.bytes.get(start..end).ok_or(fault)
    }

    /// Copy little-endian instruction words into memory at `addr`.
    pub fn load_words(&self, addr: u64, words: &[u32]) -> Result<(), MemFault> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write(addr, &bytes)
    }

    /// Snapshot of the contents, for tests.
    pub fn bytes(&self) -> Vec<u8> {
        self.bytes.iter().map(Cell::get).collect()
    }

    fn load(&self, addr: u64, size: usize) -> Result<u64, MemFault> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b[..size])?;
        Ok(u64::from_le_bytes(b))
    }
}

impl Memory for FlatMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        let src = self.range(addr, buf.len())?;
        for (dst, src) in buf.iter_mut().zip(src) {
            *dst = src.get();
        }
        Ok(())
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemFault> {
        for (dst, src) in self.range(addr, buf.len())?.iter().zip(buf) {
            dst.set(*src);
        }
        Ok(())
    }

    fn atomic_update(
        &self,
        addr: u64,
        size: usize,
        f: &mut dyn FnMut(u64) -> u64,
    ) -> Result<u64, MemFault> {
        let old = self.load(addr, size)?;
        self.write(addr, &f(old).to_le_bytes()[..size])?;
        Ok(old)
    }

    fn compare_exchange(
        &self,
        addr: u64,
        size: usize,
        expected: u64,
        new: u64,
    ) -> Result<bool, MemFault> {
        if self.load(addr, size)? != expected {
            return Ok(false);
        }
        self.write(addr, &new.to_le_bytes()[..size])?;
        Ok(true)
    }
}
