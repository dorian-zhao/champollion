//! Guest memory access.
//!
//! The interpreter talks to memory only through [`Memory`], so the same interpreter can run
//! against a flat test buffer now and the host-mapped guest address space later (M2).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemFault {
    pub addr: u64,
}

pub trait Memory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault>;
    fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemFault>;

    fn read_u16(&self, addr: u64) -> Result<u16, MemFault> {
        let mut b = [0; 2];
        self.read(addr, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn read_u32(&self, addr: u64) -> Result<u32, MemFault> {
        let mut b = [0; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
}

/// A single contiguous region starting at `base`. Good enough for unit tests and
/// `riscv-tests`, which are linked at a fixed address.
pub struct FlatMemory {
    pub base: u64,
    pub bytes: Vec<u8>,
}

impl FlatMemory {
    pub fn new(base: u64, size: usize) -> Self {
        Self { base, bytes: vec![0; size] }
    }

    fn range(&self, addr: u64, len: usize) -> Result<std::ops::Range<usize>, MemFault> {
        let fault = MemFault { addr };
        let start = addr.checked_sub(self.base).ok_or(fault)? as usize;
        let end = start.checked_add(len).ok_or(fault)?;
        if end > self.bytes.len() {
            return Err(fault);
        }
        Ok(start..end)
    }

    /// Copy little-endian instruction words into memory at `addr`.
    pub fn load_words(&mut self, addr: u64, words: &[u32]) -> Result<(), MemFault> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write(addr, &bytes)
    }
}

impl Memory for FlatMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemFault> {
        let r = self.range(addr, buf.len())?;
        buf.copy_from_slice(&self.bytes[r]);
        Ok(())
    }

    fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemFault> {
        let r = self.range(addr, buf.len())?;
        self.bytes[r].copy_from_slice(buf);
        Ok(())
    }
}
