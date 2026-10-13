//! Static ELF loader: parse an executable (fixed-address or static-PIE), map its segments
//! into a [`GuestMemory`], and build the Linux initial stack (argc, argv, envp, auxv).
//!
//! [`load_flat`] instead places an image into a [`FlatMemory`], which is enough for
//! freestanding test binaries such as riscv-tests.

use std::collections::BTreeMap;

use crate::memory::{FlatMemory, GUEST_PAGE_SIZE, GuestMemory, MapError, MemFault, Memory, Prot};

/// `e_machine` value for RISC-V.
pub const EM_RISCV: u16 = 243;

const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const PT_LOAD: u32 = 1;
const PT_INTERP: u32 = 3;
const PT_PHDR: u32 = 6;
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
    /// Guest address of the program headers once loaded (for `AT_PHDR`), if they are
    /// covered by a loadable segment. musl's static startup needs it to find `PT_TLS`.
    pub phdr: Option<u64>,
    pub phnum: u16,
    /// Position-independent (`ET_DYN` without an interpreter: static-PIE, the default for
    /// Rust's musl targets). Must be [`relocate`](Elf::relocate)d to a load base; the
    /// program applies its own relocations at startup.
    pub pie: bool,
}

impl Elf<'_> {
    /// Shift every address by `bias` (the load base of a PIE).
    pub fn relocate(&mut self, bias: u64) {
        self.entry += bias;
        self.phdr = self.phdr.map(|p| p + bias);
        for seg in &mut self.segments {
            seg.vaddr += bias;
        }
    }

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
    let pie = match u16_at(file, 16)? {
        ET_EXEC => false,
        ET_DYN => true,
        t => return Err(LoadError::UnsupportedType(t)),
    };
    let entry = u64_at(file, 24)?;
    let phoff = u64_at(file, 32)? as usize;
    let flags = u32_at(file, 48)?;
    let phentsize = u16_at(file, 54)? as usize;
    let phnum = u16_at(file, 56)? as usize;
    if phentsize != PHDR_SIZE {
        return Err(LoadError::Truncated);
    }

    let mut segments = Vec::new();
    let mut phdr = None;
    for index in 0..phnum {
        let ph = phoff + index * PHDR_SIZE;
        match u32_at(file, ph)? {
            PT_INTERP => return Err(LoadError::DynamicallyLinked),
            PT_PHDR => {
                phdr = Some(u64_at(file, ph + 16)?);
                continue;
            }
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
        // Without PT_PHDR, find the segment whose file range contains the headers.
        let table = phoff as u64..(phoff + phnum * PHDR_SIZE) as u64;
        if phdr.is_none() && offset <= table.start && table.end <= end {
            phdr = Some(vaddr + (table.start - offset));
        }
        segments.push(Segment { vaddr, mem_size, data, flags: seg_flags });
    }
    Ok(Elf { entry, flags, segments, phdr, phnum: phnum as u16, pie })
}

/// Copy every loadable segment into a single [`FlatMemory`] spanning them all, plus
/// `extra` zeroed bytes past the end (e.g. room for a stack). Permissions are ignored.
pub fn load_flat(elf: &Elf, extra: usize) -> Option<FlatMemory> {
    let (lo, hi) = elf.span()?;
    let mem = FlatMemory::new(lo, (hi - lo) as usize + extra);
    for seg in &elf.segments {
        mem.write(seg.vaddr, seg.data).ok()?;
    }
    Some(mem)
}

fn page_down(a: u64) -> u64 {
    a & !(GUEST_PAGE_SIZE - 1)
}

fn page_up(a: u64) -> u64 {
    (a + GUEST_PAGE_SIZE - 1) & !(GUEST_PAGE_SIZE - 1)
}

fn segment_prot(flags: u32) -> Prot {
    let mut p = Prot::NONE;
    if flags & PF_R != 0 {
        p = p | Prot::READ;
    }
    if flags & PF_W != 0 {
        p = p | Prot::WRITE;
    }
    if flags & PF_X != 0 {
        p = p | Prot::EXEC;
    }
    p
}

/// Map every loadable segment into `mem` with its permissions; `.bss` is zero-filled.
/// Returns the initial program break: the page-aligned end of the highest segment.
///
/// If two segments share a page (linkers normally pad to avoid it), the page gets the
/// union of their permissions.
pub fn map_elf(mem: &GuestMemory, elf: &Elf) -> Result<u64, MapError> {
    let mut pages: BTreeMap<u64, Prot> = BTreeMap::new();
    for seg in &elf.segments {
        let prot = segment_prot(seg.flags);
        let mut page = page_down(seg.vaddr);
        while page < seg.vaddr + seg.mem_size {
            *pages.entry(page).or_default() = pages.get(&page).copied().unwrap_or_default() | prot;
            page += GUEST_PAGE_SIZE;
        }
    }
    for &page in pages.keys() {
        mem.map(page, GUEST_PAGE_SIZE, Prot::RW)?;
    }
    for seg in &elf.segments {
        mem.write_privileged(seg.vaddr, seg.data).map_err(|_| MapError::Invalid)?;
    }
    for (&page, &prot) in &pages {
        mem.protect(page, GUEST_PAGE_SIZE, prot)?;
    }
    let end = elf.segments.iter().map(|s| s.vaddr + s.mem_size).max().unwrap_or(0);
    Ok(page_up(end))
}

/// Auxiliary vector entry types (`AT_*`).
pub mod auxv {
    pub const AT_NULL: u64 = 0;
    pub const AT_PHDR: u64 = 3;
    pub const AT_PHENT: u64 = 4;
    pub const AT_PHNUM: u64 = 5;
    pub const AT_PAGESZ: u64 = 6;
    pub const AT_BASE: u64 = 7;
    pub const AT_FLAGS: u64 = 8;
    pub const AT_ENTRY: u64 = 9;
    pub const AT_UID: u64 = 11;
    pub const AT_EUID: u64 = 12;
    pub const AT_GID: u64 = 13;
    pub const AT_EGID: u64 = 14;
    pub const AT_HWCAP: u64 = 16;
    pub const AT_CLKTCK: u64 = 17;
    pub const AT_SECURE: u64 = 23;
    pub const AT_RANDOM: u64 = 25;
    pub const AT_EXECFN: u64 = 31;
}

/// What goes on the initial stack besides the auxv entries the caller computes.
pub struct StackInit<'a> {
    pub argv: &'a [Vec<u8>],
    pub envp: &'a [Vec<u8>],
    /// Path of the executable, for `AT_EXECFN`.
    pub execfn: &'a [u8],
    /// 16 random bytes for `AT_RANDOM` (stack protector and pointer guard seeds).
    pub random: [u8; 16],
    /// Auxv entries, excluding `AT_RANDOM`, `AT_EXECFN`, and the `AT_NULL` terminator.
    pub auxv: &'a [(u64, u64)],
}

/// Build the Linux initial process stack below `top` (which must be mapped below) and
/// return the initial `sp`, which points at `argc` and is 16-byte aligned:
///
/// ```text
/// sp -> argc, argv[0..argc], NULL, envp[..], NULL, auxv pairs.., AT_NULL 0
///       ... padding, 16 random bytes, argv/envp/execfn strings, 8 zero bytes <- top
/// ```
pub fn build_stack(mem: &GuestMemory, top: u64, init: &StackInit) -> Result<u64, MemFault> {
    let mut cur = top - 8; // the topmost word stays zero, as on Linux
    let mut push_bytes = |mem: &GuestMemory, bytes: &[u8]| -> Result<u64, MemFault> {
        cur -= bytes.len() as u64;
        mem.write_privileged(cur, bytes)?;
        Ok(cur)
    };
    let mut push_str = |mem: &GuestMemory, s: &[u8]| -> Result<u64, MemFault> {
        let mut v = s.to_vec();
        v.push(0);
        push_bytes(mem, &v)
    };

    let execfn = push_str(mem, init.execfn)?;
    let envp: Vec<u64> =
        init.envp.iter().rev().map(|s| push_str(mem, s)).collect::<Result<_, _>>()?;
    let argv: Vec<u64> =
        init.argv.iter().rev().map(|s| push_str(mem, s)).collect::<Result<_, _>>()?;
    let random = push_bytes(mem, &init.random)?;

    let mut words = vec![init.argv.len() as u64];
    words.extend(argv.iter().rev());
    words.push(0);
    words.extend(envp.iter().rev());
    words.push(0);
    for &(k, v) in init.auxv {
        words.extend([k, v]);
    }
    words.extend([auxv::AT_RANDOM, random, auxv::AT_EXECFN, execfn, auxv::AT_NULL, 0]);

    let sp = (cur - 8 * words.len() as u64) & !15;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    mem.write_privileged(sp, &bytes)?;
    Ok(sp)
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
        assert!(elf.segments.iter().all(|s| s.data.len() as u64 <= s.mem_size));
        // The program headers are loaded somewhere inside the image.
        let phdr = elf.phdr.unwrap();
        let (lo, hi) = elf.span().unwrap();
        assert!((lo..hi).contains(&phdr));
        assert!(elf.phnum > 0);
    }

    #[test]
    fn maps_hello_with_permissions() {
        let elf = parse(HELLO).unwrap();
        let mem = GuestMemory::new().unwrap();
        let brk = map_elf(&mem, &elf).unwrap();
        assert_eq!(brk % GUEST_PAGE_SIZE, 0);
        assert!(brk >= elf.span().unwrap().1);

        // Entry point is executable and holds the file's bytes.
        let mut insn = [0u8; 2];
        mem.fetch(elf.entry, &mut insn).unwrap();
        let text = elf.segments.iter().find(|s| s.flags & PF_X != 0).unwrap();
        let off = (elf.entry - text.vaddr) as usize;
        assert_eq!(insn, text.data[off..off + 2]);
        assert!(mem.write(elf.entry, &[0]).is_err(), "text must not be writable");

        // AT_PHDR points at a copy of the file's program header table.
        let phoff = u64_at(HELLO, 32).unwrap() as usize;
        let table = elf.phnum as usize * PHDR_SIZE;
        let mut loaded = vec![0u8; table];
        mem.read(elf.phdr.unwrap(), &mut loaded).unwrap();
        assert_eq!(loaded, HELLO[phoff..phoff + table]);

        // .bss (past the file data of the writable segment) reads as zero.
        let data = elf.segments.iter().find(|s| s.flags & PF_W != 0).unwrap();
        if data.mem_size > data.data.len() as u64 {
            let mut b = [0xffu8; 1];
            mem.read(data.vaddr + data.mem_size - 1, &mut b).unwrap();
            assert_eq!(b, [0]);
        }
    }

    #[test]
    fn initial_stack_layout() {
        let mem = GuestMemory::new().unwrap();
        let top = 0x10_0000;
        mem.map(top - 0x4000, 0x4000, Prot::RW).unwrap();
        let argv = [b"prog".to_vec(), b"arg one".to_vec()];
        let envp = [b"A=1".to_vec()];
        let init = StackInit {
            argv: &argv,
            envp: &envp,
            execfn: b"/bin/prog",
            random: [7; 16],
            auxv: &[(auxv::AT_PAGESZ, 4096)],
        };
        let sp = build_stack(&mem, top, &init).unwrap();
        assert_eq!(sp % 16, 0);

        let word = |i: u64| {
            let mut b = [0u8; 8];
            mem.read(sp + 8 * i, &mut b).unwrap();
            u64::from_le_bytes(b)
        };
        let string = |addr: u64| mem.read_cstr(addr, 100).unwrap().into_bytes();
        assert_eq!(word(0), 2);
        assert_eq!(string(word(1)), b"prog");
        assert_eq!(string(word(2)), b"arg one");
        assert_eq!(word(3), 0);
        assert_eq!(string(word(4)), b"A=1");
        assert_eq!(word(5), 0);
        assert_eq!((word(6), word(7)), (auxv::AT_PAGESZ, 4096));
        assert_eq!(word(8), auxv::AT_RANDOM);
        let mut random = [0u8; 16];
        mem.read(word(9), &mut random).unwrap();
        assert_eq!(random, [7; 16]);
        assert_eq!(word(10), auxv::AT_EXECFN);
        assert_eq!(string(word(11)), b"/bin/prog");
        assert_eq!((word(12), word(13)), (auxv::AT_NULL, 0));
        assert!(word(11) < top && sp < word(9));
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
        assert_eq!(parse(&patched(4, &[1])), Err(LoadError::Not64BitLittleEndian));
        assert_eq!(parse(&patched(18, &62u16.to_le_bytes())), Err(LoadError::WrongMachine(62)));
        assert!(parse(&patched(16, &3u16.to_le_bytes())).unwrap().pie, "ET_DYN is a static PIE");
        assert_eq!(parse(&patched(16, &4u16.to_le_bytes())), Err(LoadError::UnsupportedType(4)));
        // Headers intact, but a segment's file range runs past the end.
        assert!(matches!(parse(&HELLO[..HELLO.len() / 2]), Err(LoadError::BadSegment { .. })));
    }
}
