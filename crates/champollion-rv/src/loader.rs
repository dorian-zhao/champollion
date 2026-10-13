//! Static ELF loader.
//!
//! Parsing is done; M2 adds mapping segments into the real guest address space and
//! building the initial stack (argc, argv, envp, auxv). Until then, [`load_flat`] places
//! an image into a [`FlatMemory`], which is enough for freestanding test binaries.

use crate::memory::{FlatMemory, Memory};

/// `e_machine` value for RISC-V.
pub const EM_RISCV: u16 = 243;

const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const PT_LOAD: u32 = 1;
const PT_INTERP: u32 = 3;
const EHDR_SIZE: usize = 64;
const PHDR_SIZE: usize = 56;

/// Segment permission bits (`p_flags`).
pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    Truncated,
    NotElf,
    Not64BitLittleEndian,
    WrongMachine(u16),
    /// PIE or shared object; static PIE support is a later milestone.
    PositionIndependent,
    /// Has a PT_INTERP; dynamic linking is M5.
    DynamicallyLinked,
    UnsupportedType(u16),
    BadSegment {
        index: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment<'a> {
    pub vaddr: u64,
    /// Size in memory; bytes past `data.len()` are zero-filled (`.bss`).
    pub mem_size: u64,
    /// File-backed contents (`p_filesz` bytes).
    pub data: &'a [u8],
    /// `PF_R` | `PF_W` | `PF_X`.
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elf<'a> {
    pub entry: u64,
    /// `e_flags`: bit 0 is RVC, bits 2:1 the float ABI.
    pub flags: u32,
    pub segments: Vec<Segment<'a>>,
}

impl Elf<'_> {
    /// Lowest and one-past-highest virtual address covered by loadable segments.
    pub fn span(&self) -> Option<(u64, u64)> {
        let lo = self.segments.iter().map(|s| s.vaddr).min()?;
        let hi = self.segments.iter().map(|s| s.vaddr + s.mem_size).max()?;
        Some((lo, hi))
    }
}

fn u16_at(b: &[u8], off: usize) -> Result<u16, LoadError> {
    let bytes = b.get(off..off + 2).ok_or(LoadError::Truncated)?;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn u32_at(b: &[u8], off: usize) -> Result<u32, LoadError> {
    let bytes = b.get(off..off + 4).ok_or(LoadError::Truncated)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn u64_at(b: &[u8], off: usize) -> Result<u64, LoadError> {
    let bytes = b.get(off..off + 8).ok_or(LoadError::Truncated)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

pub fn parse(file: &[u8]) -> Result<Elf<'_>, LoadError> {
    if file.len() < EHDR_SIZE {
        return Err(LoadError::Truncated);
    }
    if &file[..4] != b"\x7fELF" {
        return Err(LoadError::NotElf);
    }
    // EI_CLASS = ELFCLASS64, EI_DATA = ELFDATA2LSB
    if file[4] != 2 || file[5] != 1 {
        return Err(LoadError::Not64BitLittleEndian);
    }
    let machine = u16_at(file, 18)?;
    if machine != EM_RISCV {
        return Err(LoadError::WrongMachine(machine));
    }
    match u16_at(file, 16)? {
        ET_EXEC => {}
        ET_DYN => return Err(LoadError::PositionIndependent),
        t => return Err(LoadError::UnsupportedType(t)),
    }
    let entry = u64_at(file, 24)?;
    let phoff = u64_at(file, 32)? as usize;
    let flags = u32_at(file, 48)?;
    let phentsize = u16_at(file, 54)? as usize;
    let phnum = u16_at(file, 56)? as usize;
    if phentsize != PHDR_SIZE {
        return Err(LoadError::Truncated);
    }

    let mut segments = Vec::new();
    for index in 0..phnum {
        let ph = phoff + index * PHDR_SIZE;
        match u32_at(file, ph)? {
            PT_INTERP => return Err(LoadError::DynamicallyLinked),
            PT_LOAD => {}
            _ => continue,
        }
        let bad = LoadError::BadSegment { index };
        let seg_flags = u32_at(file, ph + 4)?;
        let offset = u64_at(file, ph + 8)?;
        let vaddr = u64_at(file, ph + 16)?;
        let file_size = u64_at(file, ph + 32)?;
        let mem_size = u64_at(file, ph + 40)?;
        if file_size > mem_size || vaddr.checked_add(mem_size).is_none() {
            return Err(bad);
        }
        let end = offset.checked_add(file_size).ok_or(bad)?;
        let data = file.get(offset as usize..end as usize).ok_or(bad)?;
        segments.push(Segment {
            vaddr,
            mem_size,
            data,
            flags: seg_flags,
        });
    }
    Ok(Elf {
        entry,
        flags,
        segments,
    })
}

/// Copy every loadable segment into a single [`FlatMemory`] spanning them all, plus
/// `extra` zeroed bytes past the end (e.g. room for a stack). Permissions are ignored.
pub fn load_flat(elf: &Elf, extra: usize) -> Option<FlatMemory> {
    let (lo, hi) = elf.span()?;
    let mut mem = FlatMemory::new(lo, (hi - lo) as usize + extra);
    for seg in &elf.segments {
        mem.write(seg.vaddr, seg.data).ok()?;
    }
    Some(mem)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO: &[u8] = include_bytes!("../../../tests/fixtures/hello");

    #[test]
    fn parses_static_hello() {
        let elf = parse(HELLO).unwrap();
        assert_eq!(elf.flags & 1, 1, "RVC");
        assert_eq!(elf.flags & 0b110, 0b100, "double-float ABI");
        assert!(!elf.segments.is_empty());
        let text = elf.segments.iter().find(|s| s.flags & PF_X != 0).unwrap();
        assert!((text.vaddr..text.vaddr + text.mem_size).contains(&elf.entry));
        assert!(
            elf.segments
                .iter()
                .all(|s| s.data.len() as u64 <= s.mem_size)
        );
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(parse(&HELLO[..10]), Err(LoadError::Truncated));
        assert_eq!(parse(&[0; 64]), Err(LoadError::NotElf));

        let patched = |off: usize, bytes: &[u8]| {
            let mut v = HELLO.to_vec();
            v[off..off + bytes.len()].copy_from_slice(bytes);
            v
        };
        assert_eq!(
            parse(&patched(4, &[1])),
            Err(LoadError::Not64BitLittleEndian)
        );
        assert_eq!(
            parse(&patched(18, &62u16.to_le_bytes())),
            Err(LoadError::WrongMachine(62))
        );
        assert_eq!(
            parse(&patched(16, &3u16.to_le_bytes())),
            Err(LoadError::PositionIndependent)
        );
        // Headers intact, but a segment's file range runs past the end.
        assert!(matches!(
            parse(&HELLO[..HELLO.len() / 2]),
            Err(LoadError::BadSegment { .. })
        ));
    }
}
