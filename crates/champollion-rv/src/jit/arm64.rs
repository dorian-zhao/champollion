//! A minimal ARM64 (A64) instruction encoder for the JIT.
//!
//! Only what the translator needs, each encoding checked against an assembler by
//! `tests/encode_arm64.rs`. Registers are plain numbers: 0–30 are x0–x30, and 31 is `xzr` or
//! `sp` depending on the instruction (as in the architecture). Methods named `*_w` operate on
//! 32-bit `w` registers.

/// A register number, 0–31.
pub type Reg = u8;

/// Register 31 when it means the zero register.
pub const XZR: Reg = 31;
/// Register 31 when it means the stack pointer.
pub const SP: Reg = 31;
/// The link register.
pub const LR: Reg = 30;

/// Condition codes for `b.cond`, `csel`, and `cset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
    Al = 14,
}

impl Cond {
    pub fn invert(self) -> Cond {
        // Flipping bit 0 inverts every condition except AL (and NV).
        // SAFETY-free: all values 0..=13 are valid variants.
        match self as u8 ^ 1 {
            0 => Cond::Eq,
            1 => Cond::Ne,
            2 => Cond::Hs,
            3 => Cond::Lo,
            4 => Cond::Mi,
            5 => Cond::Pl,
            6 => Cond::Vs,
            7 => Cond::Vc,
            8 => Cond::Hi,
            9 => Cond::Ls,
            10 => Cond::Ge,
            11 => Cond::Lt,
            12 => Cond::Gt,
            13 => Cond::Le,
            _ => Cond::Al,
        }
    }
}

/// LSE atomic operations (`ld<op>al` and `swpal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    /// Bit clear: `[rn] &= !rs`.
    Clr,
    Eor,
    Set,
    Smax,
    Smin,
    Umax,
    Umin,
    Swp,
}

/// Load/store access size and signedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mem {
    /// Zero-extending loads / plain stores of 1, 2, 4, 8 bytes.
    U8,
    U16,
    U32,
    U64,
    /// Sign-extending loads to 64 bits.
    S8,
    S16,
    S32,
}

/// Encode a logical (bitmask) immediate for a 64-bit (`width` = 64) or 32-bit operation as
/// `(N, immr, imms)`, or `None` if `value` is not representable.
pub fn logical_imm(value: u64, width: u32) -> Option<(u32, u32, u32)> {
    let value = if width == 32 { value & 0xffff_ffff } else { value };
    if value == 0 || (width == 64 && value == u64::MAX) || (width == 32 && value == 0xffff_ffff) {
        return None;
    }
    // Find the smallest repeating element size.
    let mut size = width;
    while size > 2 {
        let half = size / 2;
        let mask = (1u64 << half) - 1;
        if (value & mask) != ((value >> half) & mask) {
            break;
        }
        size = half;
    }
    let mask = if size == 64 { u64::MAX } else { (1u64 << size) - 1 };
    let elem = value & mask;
    // The element must be a rotated run of ones.
    let ones = elem.count_ones();
    // Rotate right until the run starts at bit 0 (trailing ones, then zeros).
    let mut rot = 0;
    let mut e = elem;
    while rot < size {
        let run = (1u64 << ones) - 1;
        if e == run {
            break;
        }
        e = ((e >> 1) | ((e & 1) << (size - 1))) & mask;
        rot += 1;
    }
    if rot == size {
        return None;
    }
    // e = elem rotated right by `rot`; the encoding stores the right-rotation that turns
    // the run into elem, which is (size - rot) % size.
    let immr = (size - rot) % size;
    let n = u32::from(size == 64);
    let imms = ((!(size - 1) << 1) & 0x3f) | (ones - 1);
    Some((n, immr, imms))
}

/// An instruction stream under construction.
#[derive(Default)]
pub struct Asm {
    pub code: Vec<u32>,
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index of the next instruction.
    pub fn pos(&self) -> usize {
        self.code.len()
    }

    pub fn emit(&mut self, word: u32) {
        self.code.push(word);
    }

    fn r3(&mut self, base: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }

    // ---- data processing, register ----

    pub fn add(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x8b00_0000, rd, rn, rm);
    }
    pub fn add_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x0b00_0000, rd, rn, rm);
    }
    /// `add rd, rn, rm, lsl #shift`
    pub fn add_lsl(&mut self, rd: Reg, rn: Reg, rm: Reg, shift: u32) {
        self.r3(0x8b00_0000 | shift << 10, rd, rn, rm);
    }
    pub fn sub(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0xcb00_0000, rd, rn, rm);
    }
    pub fn sub_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x4b00_0000, rd, rn, rm);
    }
    pub fn cmp(&mut self, rn: Reg, rm: Reg) {
        self.r3(0xeb00_0000, XZR, rn, rm);
    }
    pub fn and(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x8a00_0000, rd, rn, rm);
    }
    pub fn orr(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0xaa00_0000, rd, rn, rm);
    }
    pub fn eor(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0xca00_0000, rd, rn, rm);
    }
    /// `mov rd, rm` (64-bit, not involving sp).
    pub fn mov(&mut self, rd: Reg, rm: Reg) {
        self.orr(rd, XZR, rm);
    }
    /// `mvn rd, rm`: bitwise not.
    pub fn mvn(&mut self, rd: Reg, rm: Reg) {
        self.r3(0xaa20_0000, rd, XZR, rm);
    }
    /// `mov wd, wm`: copies the low 32 bits and zeroes the rest.
    pub fn mov_w(&mut self, rd: Reg, rm: Reg) {
        self.r3(0x2a00_0000, rd, XZR, rm);
    }
    pub fn lslv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9ac0_2000, rd, rn, rm);
    }
    pub fn lsrv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9ac0_2400, rd, rn, rm);
    }
    pub fn asrv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9ac0_2800, rd, rn, rm);
    }
    pub fn lslv_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x1ac0_2000, rd, rn, rm);
    }
    pub fn lsrv_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x1ac0_2400, rd, rn, rm);
    }
    pub fn asrv_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x1ac0_2800, rd, rn, rm);
    }
    pub fn mul(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9b00_7c00, rd, rn, rm);
    }
    pub fn mul_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x1b00_7c00, rd, rn, rm);
    }
    /// `msub rd, rn, rm, ra`: rd = ra - rn * rm.
    pub fn msub(&mut self, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        self.r3(0x9b00_8000 | (ra as u32) << 10, rd, rn, rm);
    }
    pub fn msub_w(&mut self, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        self.r3(0x1b00_8000 | (ra as u32) << 10, rd, rn, rm);
    }
    pub fn smulh(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9b40_7c00, rd, rn, rm);
    }
    pub fn umulh(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9bc0_7c00, rd, rn, rm);
    }
    pub fn sdiv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9ac0_0c00, rd, rn, rm);
    }
    pub fn udiv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9ac0_0800, rd, rn, rm);
    }
    pub fn sdiv_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x1ac0_0c00, rd, rn, rm);
    }
    pub fn udiv_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x1ac0_0800, rd, rn, rm);
    }
    pub fn csel(&mut self, rd: Reg, rn: Reg, rm: Reg, cond: Cond) {
        self.r3(0x9a80_0000 | (cond as u32) << 12, rd, rn, rm);
    }
    /// `csinv rd, rn, rm, cond`: rd = cond ? rn : !rm.
    pub fn csinv(&mut self, rd: Reg, rn: Reg, rm: Reg, cond: Cond) {
        self.r3(0xda80_0000 | (cond as u32) << 12, rd, rn, rm);
    }
    pub fn csinv_w(&mut self, rd: Reg, rn: Reg, rm: Reg, cond: Cond) {
        self.r3(0x5a80_0000 | (cond as u32) << 12, rd, rn, rm);
    }
    /// `cset rd, cond`: rd = cond ? 1 : 0.
    pub fn cset(&mut self, rd: Reg, cond: Cond) {
        self.r3(0x9a80_0400 | (cond.invert() as u32) << 12, rd, XZR, XZR);
    }

    // ---- data processing, immediate ----

    fn imm12(&mut self, base: u32, rd: Reg, rn: Reg, imm: u32) {
        assert!(imm < 4096, "imm12 out of range: {imm}");
        self.emit(base | imm << 10 | (rn as u32) << 5 | rd as u32);
    }
    /// `add rd, rn, #imm` (0–4095; register 31 is sp).
    pub fn add_imm(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.imm12(0x9100_0000, rd, rn, imm);
    }
    pub fn add_imm_w(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.imm12(0x1100_0000, rd, rn, imm);
    }
    pub fn sub_imm(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.imm12(0xd100_0000, rd, rn, imm);
    }
    pub fn sub_imm_w(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.imm12(0x5100_0000, rd, rn, imm);
    }
    pub fn cmp_imm(&mut self, rn: Reg, imm: u32) {
        self.imm12(0xf100_0000, XZR, rn, imm);
    }
    pub fn cmp_imm_w(&mut self, rn: Reg, imm: u32) {
        self.imm12(0x7100_0000, XZR, rn, imm);
    }

    /// Logical immediate ops; `None` if `imm` is not a bitmask immediate.
    fn logical(&mut self, base: u32, width: u32, rd: Reg, rn: Reg, imm: u64) -> Option<()> {
        let (n, immr, imms) = logical_imm(imm, width)?;
        self.emit(base | n << 22 | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32);
        Some(())
    }
    pub fn and_imm(&mut self, rd: Reg, rn: Reg, imm: u64) -> Option<()> {
        self.logical(0x9200_0000, 64, rd, rn, imm)
    }
    pub fn orr_imm(&mut self, rd: Reg, rn: Reg, imm: u64) -> Option<()> {
        self.logical(0xb200_0000, 64, rd, rn, imm)
    }
    pub fn eor_imm(&mut self, rd: Reg, rn: Reg, imm: u64) -> Option<()> {
        self.logical(0xd200_0000, 64, rd, rn, imm)
    }

    /// `ubfm` / `sbfm` (64-bit).
    pub fn ubfm(&mut self, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.emit(0xd340_0000 | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32);
    }
    pub fn sbfm(&mut self, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.emit(0x9340_0000 | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32);
    }
    pub fn ubfm_w(&mut self, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.emit(0x5300_0000 | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32);
    }
    pub fn sbfm_w(&mut self, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.emit(0x1300_0000 | immr << 16 | imms << 10 | (rn as u32) << 5 | rd as u32);
    }
    pub fn lsl_imm(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.ubfm(rd, rn, (64 - shift) % 64, 63 - shift);
    }
    pub fn lsr_imm(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.ubfm(rd, rn, shift, 63);
    }
    pub fn asr_imm(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.sbfm(rd, rn, shift, 63);
    }
    pub fn lsl_imm_w(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.ubfm_w(rd, rn, (32 - shift) % 32, 31 - shift);
    }
    pub fn lsr_imm_w(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.ubfm_w(rd, rn, shift, 31);
    }
    pub fn asr_imm_w(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.sbfm_w(rd, rn, shift, 31);
    }
    /// `sxtw rd, wn`: sign-extend the low 32 bits.
    pub fn sxtw(&mut self, rd: Reg, rn: Reg) {
        self.sbfm(rd, rn, 0, 31);
    }

    pub fn movz(&mut self, rd: Reg, imm16: u16, shift: u32) {
        self.emit(0xd280_0000 | (shift / 16) << 21 | (imm16 as u32) << 5 | rd as u32);
    }
    pub fn movk(&mut self, rd: Reg, imm16: u16, shift: u32) {
        self.emit(0xf280_0000 | (shift / 16) << 21 | (imm16 as u32) << 5 | rd as u32);
    }
    pub fn movn(&mut self, rd: Reg, imm16: u16, shift: u32) {
        self.emit(0x9280_0000 | (shift / 16) << 21 | (imm16 as u32) << 5 | rd as u32);
    }

    /// Materialize any 64-bit constant in the fewest instructions this simple scheme finds:
    /// a bitmask immediate, else movz/movn plus movk for the remaining halfwords.
    pub fn mov_imm(&mut self, rd: Reg, value: u64) {
        let halves = |v: u64| (0..4).map(move |i| (v >> (16 * i)) as u16);
        let zeros = halves(value).filter(|&h| h == 0).count();
        let ones = halves(value).filter(|&h| h == 0xffff).count();
        if zeros < 3 && ones < 3 && self.orr_imm(rd, XZR, value).is_some() {
            return;
        }
        if ones > zeros {
            // Start from all ones with movn, patch the other halfwords.
            let mut first = true;
            for (i, h) in halves(value).enumerate() {
                if h != 0xffff {
                    if first {
                        self.movn(rd, !h, 16 * i as u32);
                        first = false;
                    } else {
                        self.movk(rd, h, 16 * i as u32);
                    }
                }
            }
            if first {
                self.movn(rd, 0, 0);
            }
        } else {
            let mut first = true;
            for (i, h) in halves(value).enumerate() {
                if h != 0 {
                    if first {
                        self.movz(rd, h, 16 * i as u32);
                        first = false;
                    } else {
                        self.movk(rd, h, 16 * i as u32);
                    }
                }
            }
            if first {
                self.movz(rd, 0, 0);
            }
        }
    }

    // ---- loads and stores ----

    /// Load `rt` from `[rn, rm]` (register offset, no shift).
    pub fn ldr_reg(&mut self, mem: Mem, rt: Reg, rn: Reg, rm: Reg) {
        let base = match mem {
            Mem::U8 => 0x3860_6800,
            Mem::U16 => 0x7860_6800,
            Mem::U32 => 0xb860_6800,
            Mem::U64 => 0xf860_6800,
            Mem::S8 => 0x38a0_6800,
            Mem::S16 => 0x78a0_6800,
            Mem::S32 => 0xb8a0_6800,
        };
        self.r3(base, rt, rn, rm);
    }

    /// Store `rt` to `[rn, rm]`; `mem` must be unsigned (the size).
    pub fn str_reg(&mut self, mem: Mem, rt: Reg, rn: Reg, rm: Reg) {
        let base = match mem {
            Mem::U8 => 0x3820_6800,
            Mem::U16 => 0x7820_6800,
            Mem::U32 => 0xb820_6800,
            Mem::U64 => 0xf820_6800,
            _ => panic!("stores have no signedness"),
        };
        self.r3(base, rt, rn, rm);
    }

    /// `ldr xt, [xn, #offset]` with a scaled unsigned offset (multiple of 8, < 32 KiB).
    pub fn ldr_off(&mut self, rt: Reg, rn: Reg, offset: u32) {
        assert!(offset.is_multiple_of(8) && offset / 8 < 4096, "ldr offset {offset}");
        self.emit(0xf940_0000 | (offset / 8) << 10 | (rn as u32) << 5 | rt as u32);
    }
    pub fn str_off(&mut self, rt: Reg, rn: Reg, offset: u32) {
        assert!(offset.is_multiple_of(8) && offset / 8 < 4096, "str offset {offset}");
        self.emit(0xf900_0000 | (offset / 8) << 10 | (rn as u32) << 5 | rt as u32);
    }
    /// `ldr wt, [xn, #offset]` (multiple of 4).
    pub fn ldr_off_w(&mut self, rt: Reg, rn: Reg, offset: u32) {
        assert!(offset.is_multiple_of(4) && offset / 4 < 4096, "ldr offset {offset}");
        self.emit(0xb940_0000 | (offset / 4) << 10 | (rn as u32) << 5 | rt as u32);
    }
    pub fn str_off_w(&mut self, rt: Reg, rn: Reg, offset: u32) {
        assert!(offset.is_multiple_of(4) && offset / 4 < 4096, "str offset {offset}");
        self.emit(0xb900_0000 | (offset / 4) << 10 | (rn as u32) << 5 | rt as u32);
    }

    /// `stp rt, rt2, [rn, #offset]!` (pre-index; offset a multiple of 8 in -512..=504).
    pub fn stp_pre(&mut self, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        assert!(offset % 8 == 0 && (-512..=504).contains(&offset));
        let imm7 = ((offset / 8) as u32) & 0x7f;
        self.emit(0xa980_0000 | imm7 << 15 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32);
    }
    /// `ldp rt, rt2, [rn], #offset` (post-index).
    pub fn ldp_post(&mut self, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        assert!(offset % 8 == 0 && (-512..=504).contains(&offset));
        let imm7 = ((offset / 8) as u32) & 0x7f;
        self.emit(0xa8c0_0000 | imm7 << 15 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32);
    }

    // ---- branches ----

    /// Offset in instructions from `from` to `to`.
    fn rel(from: usize, to: usize) -> i64 {
        to as i64 - from as i64
    }

    /// `b` by `delta` instructions from here (targets outside this stream).
    pub fn b_rel(&mut self, delta: i64) {
        self.emit(0x1400_0000 | (delta as u32 & 0x03ff_ffff));
    }
    /// `b.cond` by `delta` instructions from here.
    pub fn b_cond_rel(&mut self, cond: Cond, delta: i64) {
        self.emit(0x5400_0000 | (delta as u32 & 0x7ffff) << 5 | cond as u32);
    }

    /// `b` to instruction index `target` (may be patched later with [`Asm::patch`]).
    pub fn b(&mut self, target: usize) {
        let off = Self::rel(self.pos(), target);
        self.emit(0x1400_0000 | (off as u32 & 0x03ff_ffff));
    }
    pub fn bl(&mut self, target: usize) {
        let off = Self::rel(self.pos(), target);
        self.emit(0x9400_0000 | (off as u32 & 0x03ff_ffff));
    }
    pub fn b_cond(&mut self, cond: Cond, target: usize) {
        let off = Self::rel(self.pos(), target);
        self.emit(0x5400_0000 | (off as u32 & 0x7ffff) << 5 | cond as u32);
    }
    pub fn cbz(&mut self, rt: Reg, target: usize) {
        let off = Self::rel(self.pos(), target);
        self.emit(0xb400_0000 | (off as u32 & 0x7ffff) << 5 | rt as u32);
    }
    pub fn cbnz(&mut self, rt: Reg, target: usize) {
        let off = Self::rel(self.pos(), target);
        self.emit(0xb500_0000 | (off as u32 & 0x7ffff) << 5 | rt as u32);
    }
    pub fn br(&mut self, rn: Reg) {
        self.emit(0xd61f_0000 | (rn as u32) << 5);
    }
    pub fn blr(&mut self, rn: Reg) {
        self.emit(0xd63f_0000 | (rn as u32) << 5);
    }
    pub fn ret(&mut self) {
        self.emit(0xd65f_03c0);
    }
    /// `adr rd, target` (instruction index; within ±1 MiB).
    pub fn adr(&mut self, rd: Reg, target: usize) {
        let off = (Self::rel(self.pos(), target) * 4) as u32;
        self.emit(0x1000_0000 | (off & 3) << 29 | ((off >> 2) & 0x7ffff) << 5 | rd as u32);
    }

    /// Re-point the branch at index `at` (b, bl, b.cond, cbz, cbnz) to index `target`.
    pub fn patch(&mut self, at: usize, target: usize) {
        self.code[at] = encode_branch_to(self.code[at], Self::rel(at, target));
    }

    /// `tst rn, #imm` (bitmask immediate; `None` if not encodable).
    pub fn tst_imm(&mut self, rn: Reg, imm: u64) -> Option<()> {
        self.logical(0xf200_0000, 64, XZR, rn, imm)
    }

    /// `cmp wn, wm`.
    pub fn cmp_w(&mut self, rn: Reg, rm: Reg) {
        self.r3(0x6b00_0000, XZR, rn, rm);
    }

    /// LSE atomic read-modify-write with acquire+release semantics:
    /// `rt = [rn]; [rn] = op([rn], rs)`. `wide` selects 64-bit (else 32-bit, zero-extended).
    pub fn atomic(&mut self, op: AtomicOp, wide: bool, rs: Reg, rt: Reg, rn: Reg) {
        let size = if wide { 0xc000_0000 } else { 0x8000_0000 };
        let opc = match op {
            AtomicOp::Add => 0x0000,
            AtomicOp::Clr => 0x1000,
            AtomicOp::Eor => 0x2000,
            AtomicOp::Set => 0x3000,
            AtomicOp::Smax => 0x4000,
            AtomicOp::Smin => 0x5000,
            AtomicOp::Umax => 0x6000,
            AtomicOp::Umin => 0x7000,
            AtomicOp::Swp => 0x8000,
        };
        self.emit(size | 0x38e0_0000 | opc | (rs as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }

    /// `casal rs, rt, [rn]`: if `[rn] == rs` then `[rn] = rt`; `rs` receives the old value.
    pub fn casal(&mut self, wide: bool, rs: Reg, rt: Reg, rn: Reg) {
        let base = if wide { 0xc8e0_fc00 } else { 0x88e0_fc00 };
        self.emit(base | (rs as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }

    pub fn nop(&mut self) {
        self.emit(0xd503_201f);
    }
    pub fn brk(&mut self, imm: u16) {
        self.emit(0xd420_0000 | (imm as u32) << 5);
    }
    pub fn dmb_ish(&mut self) {
        self.emit(0xd503_3bbf);
    }
}

/// Rewrite the offset of an existing branch word to `delta` instructions.
pub fn encode_branch_to(word: u32, delta: i64) -> u32 {
    let off = delta as u32;
    if word & 0x7c00_0000 == 0x1400_0000 {
        // b / bl: imm26
        (word & 0xfc00_0000) | (off & 0x03ff_ffff)
    } else if word & 0xff00_0010 == 0x5400_0000 || word & 0x7e00_0000 == 0x3400_0000 {
        // b.cond / cbz / cbnz: imm19 at bit 5
        (word & !(0x7ffff << 5)) | (off & 0x7ffff) << 5
    } else {
        panic!("not a patchable branch: {word:#010x}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_immediates() {
        // (value, width, expected (N, immr, imms)) from the architecture manual.
        assert_eq!(logical_imm(1, 64), Some((1, 0, 0)));
        assert_eq!(logical_imm(0xff, 64), Some((1, 0, 7)));
        assert_eq!(logical_imm((1 << 38) - 1, 64), Some((1, 0, 37)));
        assert_eq!(logical_imm(0x8000_0000_0000_0000, 64), Some((1, 1, 0)));
        assert_eq!(logical_imm(0x5555_5555_5555_5555, 64), Some((0, 0, 0b111100)));
        assert_eq!(logical_imm(0xffff_ffff_0000_0000, 64), Some((1, 32, 31)));
        assert_eq!(logical_imm(0, 64), None);
        assert_eq!(logical_imm(u64::MAX, 64), None);
        assert_eq!(logical_imm(0x1234, 64), None);
        assert_eq!(logical_imm(0xffff_fffe, 32), Some((0, 31, 30)));
    }

    #[test]
    fn mov_imm_is_short() {
        for (v, max_len) in [
            (0u64, 1),
            (1, 1),
            (u64::MAX, 1),
            (0x1234_0000, 1),
            (0xffff_ffff_ffff_1234, 1),
            (0x1_0000_0001, 2),
            (0x1234_5678_9abc_def0, 4),
        ] {
            let mut a = Asm::new();
            a.mov_imm(0, v);
            assert!(a.code.len() <= max_len, "{v:#x}: {} instructions", a.code.len());
        }
    }

    #[test]
    fn patching_branches() {
        let mut a = Asm::new();
        a.b(0);
        a.b_cond(Cond::Ne, 0);
        a.cbz(3, 0);
        a.nop();
        a.patch(0, 3);
        a.patch(1, 3);
        a.patch(2, 0);
        assert_eq!(a.code[0], 0x1400_0003);
        assert_eq!(a.code[1], 0x5400_0041);
        assert_eq!(a.code[2], 0xb400_0000 | (0x7fffe << 5) | 3);
    }
}
