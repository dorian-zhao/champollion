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

/// Floating-point operand size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp {
    S,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmaKind {
    Fmadd,
    Fmsub,
    Fnmadd,
    Fnmsub,
}

/// Rounding encoded in float-to-integer conversions (`fcvt{n,a,p,m,z}{s,u}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpRound {
    Nearest,
    Away,
    Up,
    Down,
    Zero,
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

impl Mem {
    /// Access size in bytes.
    pub fn size(self) -> u32 {
        match self {
            Mem::U8 | Mem::S8 => 1,
            Mem::U16 | Mem::S16 => 2,
            Mem::U32 | Mem::S32 => 4,
            Mem::U64 => 8,
        }
    }
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
    // The element must be a run of ones rotated by some amount (as in LLVM): either a
    // contiguous run, or one that wraps around the element's top and bottom.
    // A run reaching bit 63 makes `v | (v - 1)` all ones, so the + 1 must wrap: 2^64 is
    // still a power of two for the test below (and a debug build must not panic).
    let shifted_mask = |v: u64| v != 0 && (v | (v - 1)).wrapping_add(1) & (v | (v - 1)) == 0;
    let (rot, ones) = if shifted_mask(elem) {
        let rot = elem.trailing_zeros();
        (rot, (elem >> rot).trailing_ones())
    } else {
        // Wrapping run: its complement (within the element) is a contiguous run of zeros.
        let filled = elem | !mask;
        if !shifted_mask(!filled) {
            return None;
        }
        let leading = filled.leading_ones(); // the run's high part, plus the fill
        let rot = 64 - leading;
        (rot, leading + filled.trailing_ones() - (64 - size))
    };
    // The encoding stores the right-rotation that turns the run (at bit 0) into `elem`.
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
    /// `sub rd, rn, #imm, lsl #12`.
    pub fn sub_imm_lsl12(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.imm12(0xd140_0000, rd, rn, imm);
    }
    pub fn add_imm_lsl12(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.imm12(0x9140_0000, rd, rn, imm);
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

    /// Whether a `size`-byte access at `[rn, #offset]` is encodable by [`Asm::ldr_imm`] and
    /// friends: a scaled unsigned offset (a multiple of `size`, below `4096 * size`) or an
    /// unscaled 9-bit signed one.
    pub fn imm_offset_fits(size: u32, offset: i32) -> bool {
        Self::scaled_offset(size, offset).is_some() || (-256..256).contains(&offset)
    }

    /// The scaled unsigned immediate for `offset`, if it has one.
    fn scaled_offset(size: u32, offset: i32) -> Option<u32> {
        let offset = u32::try_from(offset).ok()?;
        (offset.is_multiple_of(size) && offset / size < 4096).then_some(offset / size)
    }

    /// Emit a load or store with an immediate offset, scaled (`scaled` opcode) when possible,
    /// else unscaled (`unscaled`, i.e. `ldur`/`stur`). False if neither form fits.
    fn mem_imm(
        &mut self,
        scaled: u32,
        unscaled: u32,
        size: u32,
        rt: Reg,
        rn: Reg,
        offset: i32,
    ) -> bool {
        let regs = (rn as u32) << 5 | rt as u32;
        if let Some(imm) = Self::scaled_offset(size, offset) {
            self.emit(scaled | imm << 10 | regs);
        } else if (-256..256).contains(&offset) {
            self.emit(unscaled | (offset as u32 & 0x1ff) << 12 | regs);
        } else {
            return false;
        }
        true
    }

    /// Load `rt` from `[rn, #offset]` (see [`Asm::imm_offset_fits`]); false if not encodable.
    #[must_use]
    pub fn ldr_imm(&mut self, mem: Mem, rt: Reg, rn: Reg, offset: i32) -> bool {
        let (scaled, unscaled, size) = match mem {
            Mem::U8 => (0x3940_0000, 0x3840_0000, 1),
            Mem::U16 => (0x7940_0000, 0x7840_0000, 2),
            Mem::U32 => (0xb940_0000, 0xb840_0000, 4),
            Mem::U64 => (0xf940_0000, 0xf840_0000, 8),
            Mem::S8 => (0x3980_0000, 0x3880_0000, 1),
            Mem::S16 => (0x7980_0000, 0x7880_0000, 2),
            Mem::S32 => (0xb980_0000, 0xb880_0000, 4),
        };
        self.mem_imm(scaled, unscaled, size, rt, rn, offset)
    }

    /// Store `rt` to `[rn, #offset]`; `mem` must be unsigned (the size). False if not
    /// encodable.
    #[must_use]
    pub fn str_imm(&mut self, mem: Mem, rt: Reg, rn: Reg, offset: i32) -> bool {
        let (scaled, unscaled, size) = match mem {
            Mem::U8 => (0x3900_0000, 0x3800_0000, 1),
            Mem::U16 => (0x7900_0000, 0x7800_0000, 2),
            Mem::U32 => (0xb900_0000, 0xb800_0000, 4),
            Mem::U64 => (0xf900_0000, 0xf800_0000, 8),
            _ => panic!("stores have no signedness"),
        };
        self.mem_imm(scaled, unscaled, size, rt, rn, offset)
    }

    /// Load FP register `rt` from `[rn, #offset]`; false if not encodable.
    #[must_use]
    pub fn ldr_fp_imm(&mut self, size: Fp, rt: Reg, rn: Reg, offset: i32) -> bool {
        match size {
            Fp::S => self.mem_imm(0xbd40_0000, 0xbc40_0000, 4, rt, rn, offset),
            Fp::D => self.mem_imm(0xfd40_0000, 0xfc40_0000, 8, rt, rn, offset),
        }
    }

    /// Store FP register `rt` to `[rn, #offset]`; false if not encodable.
    #[must_use]
    pub fn str_fp_imm(&mut self, size: Fp, rt: Reg, rn: Reg, offset: i32) -> bool {
        match size {
            Fp::S => self.mem_imm(0xbd00_0000, 0xbc00_0000, 4, rt, rn, offset),
            Fp::D => self.mem_imm(0xfd00_0000, 0xfc00_0000, 8, rt, rn, offset),
        }
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
    /// `ldp rt, rt2, [rn]`: two adjacent 8-byte loads in one instruction.
    pub fn ldp(&mut self, rt: Reg, rt2: Reg, rn: Reg) {
        self.emit(0xa940_0000 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32);
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
    /// `svc #0`: enter the host kernel.
    pub fn svc(&mut self) {
        self.emit(0xd400_0001);
    }
    /// `adrp rd, #pages`: the address of the 4 KiB page `pages` pages from this one (±4 GiB).
    pub fn adrp(&mut self, rd: Reg, pages: i32) {
        assert!((-(1 << 20)..1 << 20).contains(&pages), "adrp out of range: {pages}");
        let imm = pages as u32;
        self.emit(0x9000_0000 | (imm & 3) << 29 | ((imm >> 2) & 0x7ffff) << 5 | rd as u32);
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

    // ---- floating point (register numbers name v0–v31) ----

    /// `ldr dt|st, [xn, #offset]` (scaled unsigned offset).
    pub fn ldr_fp(&mut self, size: Fp, rt: Reg, rn: Reg, offset: u32) {
        let (base, scale) = if size == Fp::D { (0xfd40_0000, 8) } else { (0xbd40_0000, 4) };
        assert!(offset.is_multiple_of(scale) && offset / scale < 4096);
        self.emit(base | (offset / scale) << 10 | (rn as u32) << 5 | rt as u32);
    }
    pub fn str_fp(&mut self, size: Fp, rt: Reg, rn: Reg, offset: u32) {
        let (base, scale) = if size == Fp::D { (0xfd00_0000, 8) } else { (0xbd00_0000, 4) };
        assert!(offset.is_multiple_of(scale) && offset / scale < 4096);
        self.emit(base | (offset / scale) << 10 | (rn as u32) << 5 | rt as u32);
    }
    /// `ldr dt|st, [xn, xm]` (register offset).
    pub fn ldr_fp_reg(&mut self, size: Fp, rt: Reg, rn: Reg, rm: Reg) {
        self.r3(if size == Fp::D { 0xfc60_6800 } else { 0xbc60_6800 }, rt, rn, rm);
    }
    pub fn str_fp_reg(&mut self, size: Fp, rt: Reg, rn: Reg, rm: Reg) {
        self.r3(if size == Fp::D { 0xfc20_6800 } else { 0xbc20_6800 }, rt, rn, rm);
    }
    /// `fmov dd, dn` (register copy).
    pub fn fmov_d(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x1e60_4000 | (rn as u32) << 5 | rd as u32);
    }
    /// `fneg dd, dn` / `fabs dd, dn`: sign-bit operations, NaNs included, no exceptions.
    pub fn fneg_d(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x1e61_4000 | (rn as u32) << 5 | rd as u32);
    }
    pub fn fabs_d(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x1e60_c000 | (rn as u32) << 5 | rd as u32);
    }
    /// `stp dt, dt2, [xn, #offset]!` / `ldp dt, dt2, [xn], #offset`.
    pub fn stp_pre_d(&mut self, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        assert!(offset % 8 == 0 && (-512..=504).contains(&offset));
        let imm7 = ((offset / 8) as u32) & 0x7f;
        self.emit(0x6d80_0000 | imm7 << 15 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32);
    }
    pub fn ldp_post_d(&mut self, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        assert!(offset % 8 == 0 && (-512..=504).contains(&offset));
        let imm7 = ((offset / 8) as u32) & 0x7f;
        self.emit(0x6cc0_0000 | imm7 << 15 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32);
    }
    /// `fmov sd, wn` / `fmov dd, xn`.
    pub fn fmov_from_gpr(&mut self, size: Fp, rd: Reg, rn: Reg) {
        let base = if size == Fp::D { 0x9e67_0000 } else { 0x1e27_0000 };
        self.emit(base | (rn as u32) << 5 | rd as u32);
    }
    /// `fmov wd, sn` / `fmov xd, dn`.
    pub fn fmov_to_gpr(&mut self, size: Fp, rd: Reg, rn: Reg) {
        let base = if size == Fp::D { 0x9e66_0000 } else { 0x1e26_0000 };
        self.emit(base | (rn as u32) << 5 | rd as u32);
    }
    /// `pmull vd.1q, vn.1d, vm.1d`: the low 128 bits of a carry-less product.
    pub fn pmull_d(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x0ee0_e000 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `umov rd, vn.d[1]`: the high 64-bit lane.
    pub fn umov_d1(&mut self, rd: Reg, vn: Reg) {
        self.emit(0x4e18_3c00 | (vn as u32) << 5 | rd as u32);
    }
    /// `umov wd, vn.s[lane]`.
    pub fn vec_umov_s(&mut self, rd: Reg, vn: Reg, lane: u32) {
        assert!(lane < 4);
        self.emit(0x0e04_3c00 | lane << 19 | (vn as u32) << 5 | rd as u32);
    }
    /// `umov xd, vn.d[lane]`.
    pub fn vec_umov_d(&mut self, rd: Reg, vn: Reg, lane: u32) {
        assert!(lane < 2);
        self.emit(0x4e08_3c00 | lane << 20 | (vn as u32) << 5 | rd as u32);
    }
    /// `extr rd, rn, rm, #lsb`: a 128-bit right shift's low word.
    pub fn extr(&mut self, rd: Reg, rn: Reg, rm: Reg, lsb: u32) {
        assert!(lsb < 64);
        self.emit(0x93c0_0000 | lsb << 10 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    fn fp_type(size: Fp) -> u32 {
        if size == Fp::D { 0x0040_0000 } else { 0 }
    }
    pub fn fp_arith(&mut self, op: FpOp, size: Fp, rd: Reg, rn: Reg, rm: Reg) {
        let opc = match op {
            FpOp::Mul => 0x0800,
            FpOp::Div => 0x1800,
            FpOp::Add => 0x2800,
            FpOp::Sub => 0x3800,
        };
        self.r3(0x1e20_0000 | Self::fp_type(size) | opc, rd, rn, rm);
    }
    /// `fmin`/`fmax` (NaN-propagating) and `fminnm`/`fmaxnm` (NaN-ignoring).
    pub fn fp_minmax(&mut self, max: bool, numeric: bool, size: Fp, rd: Reg, rn: Reg, rm: Reg) {
        let base = if size == Fp::D { 0x1e60_4800 } else { 0x1e20_4800 };
        let op = (if max { 0 } else { 0x1000 }) | if numeric { 0x2000 } else { 0 };
        self.emit(base | op | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    pub fn fsqrt(&mut self, size: Fp, rd: Reg, rn: Reg) {
        self.emit(0x1e21_c000 | Self::fp_type(size) | (rn as u32) << 5 | rd as u32);
    }
    /// Fused multiply-add family: `fmadd` = ra + rn·rm, `fmsub` = ra − rn·rm,
    /// `fnmadd` = −ra − rn·rm, `fnmsub` = −ra + rn·rm.
    pub fn fma(&mut self, op: FmaKind, size: Fp, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        let base = match op {
            FmaKind::Fmadd => 0x1f00_0000,
            FmaKind::Fmsub => 0x1f00_8000,
            FmaKind::Fnmadd => 0x1f20_0000,
            FmaKind::Fnmsub => 0x1f20_8000,
        };
        self.r3(base | Self::fp_type(size) | (ra as u32) << 10, rd, rn, rm);
    }
    /// `fcmp` (quiet: invalid only for signaling NaNs) or `fcmpe` (signaling).
    pub fn fcmp(&mut self, signaling: bool, size: Fp, rn: Reg, rm: Reg) {
        let e = if signaling { 0x10 } else { 0 };
        self.r3(0x1e20_2000 | Self::fp_type(size) | e, 0, rn, rm);
    }
    /// `fcvt` between single and double precision (`to` is the destination size).
    pub fn fcvt(&mut self, to: Fp, rd: Reg, rn: Reg) {
        let base = if to == Fp::D { 0x1e22_c000 } else { 0x1e62_4000 };
        self.emit(base | (rn as u32) << 5 | rd as u32);
    }
    /// Float to integer with the rounding in the opcode; `wide` selects an x destination.
    pub fn fcvt_to_int(
        &mut self,
        round: FpRound,
        signed: bool,
        wide: bool,
        size: Fp,
        rd: Reg,
        rn: Reg,
    ) {
        let opc = match round {
            FpRound::Nearest => 0x0000_0000,
            FpRound::Away => 0x0004_0000,
            FpRound::Up => 0x0008_0000,
            FpRound::Down => 0x0010_0000,
            FpRound::Zero => 0x0018_0000,
        };
        let u = if signed { 0 } else { 0x0001_0000 };
        let sf = if wide { 0x8000_0000 } else { 0 };
        self.emit(0x1e20_0000 | sf | Self::fp_type(size) | opc | u | (rn as u32) << 5 | rd as u32);
    }
    /// Integer to float (`scvtf`/`ucvtf`), rounding per FPCR; `wide` selects an x source.
    pub fn cvt_from_int(&mut self, signed: bool, wide: bool, size: Fp, rd: Reg, rn: Reg) {
        let base = if signed { 0x1e22_0000 } else { 0x1e23_0000 };
        let sf = if wide { 0x8000_0000 } else { 0 };
        self.emit(base | sf | Self::fp_type(size) | (rn as u32) << 5 | rd as u32);
    }
    pub fn mrs_fpcr(&mut self, rt: Reg) {
        self.emit(0xd53b_4400 | rt as u32);
    }
    pub fn msr_fpcr(&mut self, rt: Reg) {
        self.emit(0xd51b_4400 | rt as u32);
    }
    pub fn mrs_fpsr(&mut self, rt: Reg) {
        self.emit(0xd53b_4420 | rt as u32);
    }
    pub fn msr_fpsr(&mut self, rt: Reg) {
        self.emit(0xd51b_4420 | rt as u32);
    }

    /// `mrs rt, CNTVCT_EL0`: the virtual counter.
    pub fn mrs_cntvct(&mut self, rt: Reg) {
        self.emit(0xd53b_e040 | rt as u32);
    }

    /// `mrs rt, CNTFRQ_EL0`: the counter frequency in Hz.
    pub fn mrs_cntfrq(&mut self, rt: Reg) {
        self.emit(0xd53b_e000 | rt as u32);
    }
    /// `rbit wd, wn`: reverse the bits of a 32-bit value.
    pub fn rbit_w(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x5ac0_0000 | (rn as u32) << 5 | rd as u32);
    }
    /// `rbit xd, xn`: reverse the bits of a 64-bit value.
    pub fn rbit(&mut self, rd: Reg, rn: Reg) {
        self.emit(0xdac0_0000 | (rn as u32) << 5 | rd as u32);
    }
    /// `rev xd, xn`: reverse the byte order (RISC-V `rev8`).
    pub fn rev(&mut self, rd: Reg, rn: Reg) {
        self.emit(0xdac0_0c00 | (rn as u32) << 5 | rd as u32);
    }

    // ---- bit manipulation, for the Zba/Zbb/Zbs/Zicond instructions ----

    /// `bic rd, rn, rm`: `rd = rn & !rm` (RISC-V `andn`).
    pub fn bic(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x8a20_0000, rd, rn, rm);
    }
    /// `orn rd, rn, rm`: `rd = rn | !rm`.
    pub fn orn(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0xaa20_0000, rd, rn, rm);
    }
    /// `eon rd, rn, rm`: `rd = !(rn ^ rm)` (RISC-V `xnor`).
    pub fn eon(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0xca20_0000, rd, rn, rm);
    }
    pub fn clz(&mut self, rd: Reg, rn: Reg) {
        self.emit(0xdac0_1000 | (rn as u32) << 5 | rd as u32);
    }
    pub fn clz_w(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x5ac0_1000 | (rn as u32) << 5 | rd as u32);
    }
    /// `rorv rd, rn, rm`: rotate right by the low 6 (`_w`: 5) bits of `rm`.
    pub fn rorv(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x9ac0_2c00, rd, rn, rm);
    }
    pub fn rorv_w(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.r3(0x1ac0_2c00, rd, rn, rm);
    }
    /// `ror rd, rn, #shift` (EXTR with both sources the same).
    pub fn ror_imm(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.emit(0x93c0_0000 | (rn as u32) << 16 | shift << 10 | (rn as u32) << 5 | rd as u32);
    }
    pub fn ror_imm_w(&mut self, rd: Reg, rn: Reg, shift: u32) {
        self.emit(0x1380_0000 | (rn as u32) << 16 | shift << 10 | (rn as u32) << 5 | rd as u32);
    }
    /// `add rd, rn, w_rm, uxtw`: `rd = rn + zero-extended rm` (RISC-V `add.uw`; the
    /// zero-extension is of `rm`'s low word, which is why the 32-bit form adds it).
    pub fn add_uxtw(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.add_uxtw_lsl(rd, rn, rm, 0);
    }
    pub fn add_uxtw_lsl(&mut self, rd: Reg, rn: Reg, rm: Reg, shift: u32) {
        self.emit(0x8b20_4000 | (rm as u32) << 16 | shift << 10 | (rn as u32) << 5 | rd as u32);
    }
    /// NEON scratch calls for `cpop` and `orc.b`: `cnt`/`addv` sum 16 byte counts (the
    /// upper 8 bytes of the vector are zeroed by writing the scalar register), `cmeq`
    /// against zero plus `mvn` plus counts non-zero bytes.
    pub fn cnt16b(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e20_5800 | (rn as u32) << 5 | rd as u32);
    }
    pub fn addv16b(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e31_b800 | (rn as u32) << 5 | rd as u32);
    }
    pub fn cmeq_zero16b(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e20_9800 | (rn as u32) << 5 | rd as u32);
    }
    pub fn mvn16b(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x6e20_5800 | (rn as u32) << 5 | rd as u32);
    }
    /// `rbit v.16b`: reverse the bits within each byte (Zbkb `brev8`).
    pub fn rbit16b(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x6e60_5800 | (rn as u32) << 5 | rd as u32);
    }
    /// `cmeq v.16b, v.16b, v.16b`: per-byte equality, 0xFF where equal.
    pub fn cmeq16b(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x6e20_8c00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `movi v.16b, #1`: one per byte (the constant the mask packing needs).
    pub fn movi16b_one(&mut self, rd: Reg) {
        self.emit(0x4f00_e420 | rd as u32);
    }
    /// `ext v.16b, v.16b, v.16b, #imm`: reach the other half of a vector register.
    pub fn ext16b(&mut self, rd: Reg, rn: Reg, rm: Reg, imm: u32) {
        self.emit(0x6e00_0000 | (rm as u32) << 16 | imm << 11 | (rn as u32) << 5 | rd as u32);
    }

    // ---- 128-bit vector moves and lane-wise ALU, for the RVV JIT subset ----

    /// `ldr q` / `str q`: 16-byte moves, unaligned-capable, in the immediate form
    /// (scaled by 16) and the index form.
    pub fn ldr_q(&mut self, rt: Reg, rn: Reg) {
        self.emit(0x3dc0_0000 | (rn as u32) << 5 | rt as u32);
    }
    pub fn ldr_q_off(&mut self, rt: Reg, rn: Reg, offset: u32) -> bool {
        if !offset.is_multiple_of(16) || offset > 65520 {
            return false;
        }
        self.emit(0x3dc0_0000 | (offset / 16) << 10 | (rn as u32) << 5 | rt as u32);
        true
    }
    pub fn str_q(&mut self, rt: Reg, rn: Reg) {
        self.emit(0x3d80_0000 | (rn as u32) << 5 | rt as u32);
    }
    pub fn str_q_off(&mut self, rt: Reg, rn: Reg, offset: u32) -> bool {
        if !offset.is_multiple_of(16) || offset > 65520 {
            return false;
        }
        self.emit(0x3d80_0000 | (offset / 16) << 10 | (rn as u32) << 5 | rt as u32);
        true
    }
    pub fn ldr_q_reg(&mut self, rt: Reg, rn: Reg, rm: Reg) {
        self.emit(0x3ce0_6800 | (rm as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }
    pub fn str_q_reg(&mut self, rt: Reg, rn: Reg, rm: Reg) {
        self.emit(0x3ca0_6800 | (rm as u32) << 16 | (rn as u32) << 5 | rt as u32);
    }
    /// `ldp q, q, [xn, #offset]`: two adjacent 16-byte vector loads.
    pub fn ldp_q_off(&mut self, rt: Reg, rt2: Reg, rn: Reg, offset: u32) -> bool {
        if !offset.is_multiple_of(16) || offset > 1008 {
            return false;
        }
        self.emit(
            0xad40_0000 | (offset / 16) << 15 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32,
        );
        true
    }
    /// `stp q, q, [xn, #offset]`: the matching pair store.
    pub fn stp_q_off(&mut self, rt: Reg, rt2: Reg, rn: Reg, offset: u32) -> bool {
        if !offset.is_multiple_of(16) || offset > 1008 {
            return false;
        }
        self.emit(
            0xad00_0000 | (offset / 16) << 15 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32,
        );
        true
    }
    /// `ldp q, q, [xn]`.
    pub fn ldp_q(&mut self, rt: Reg, rt2: Reg, rn: Reg) {
        assert!(self.ldp_q_off(rt, rt2, rn, 0));
    }
    /// `stp q, q, [xn]`.
    pub fn stp_q(&mut self, rt: Reg, rt2: Reg, rn: Reg) {
        assert!(self.stp_q_off(rt, rt2, rn, 0));
    }

    /// The size field of the lane-wise integer ops: the element size in bits.
    const fn vec_size(size: u32) -> u32 {
        (size / 8).ilog2() << 22
    }
    /// `add v.16b/8h/4s/2d ...`: lane-wise add of the elements (`size` in bits).
    pub fn vec_add(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x4e20_8400 | Self::vec_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    pub fn vec_sub(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x6e20_8400 | Self::vec_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// Byte-wise `and`/`orr`/`eor` (element-size independent).
    pub fn vec_and(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x4e20_1c00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    pub fn vec_orr(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x4ea0_1c00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    pub fn vec_eor(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x6e20_1c00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `bit vd.16b, vn.16b, vm.16b`: where `vm` is set, take `vn`'s bits; elsewhere keep
    /// `vd`. With the sign mask in `vm` this injects a sign exactly, payload untouched.
    pub fn vec_bit(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x6ea0_1c00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `movi vd.4s, #0x80, lsl #24`: the sign bit set in every 32-bit lane.
    pub fn vec_movi_sign(&mut self, rd: Reg) {
        self.emit(0x4f04_6400 | rd as u32);
    }
    /// `movi vd.4s, #0`.
    pub fn vec_movi_zero(&mut self, rd: Reg) {
        self.emit(0x4f00_0400 | rd as u32);
    }
    /// `movi v.8h, #imm` (an 8-bit replicated immediate).
    pub fn vec_movi_8h(&mut self, rd: Reg, imm: u8) {
        self.emit(0x4f00_8400 | u32::from(imm) << 5 | rd as u32);
    }
    /// `ext vd.16b, vn.16b, vm.16b, #imm`: `vn[imm..16]` followed by `vm[0..imm]`
    /// (0..=15).
    pub fn vec_ext(&mut self, rd: Reg, rn: Reg, rm: Reg, imm: u32) {
        assert!(imm < 16, "ext byte offset {imm}");
        self.emit(0x6e00_0000 | (rm as u32) << 16 | imm << 11 | (rn as u32) << 5 | rd as u32);
    }
    /// `tbl vd.16b, {vn.16b}, vm.16b`: gather bytes from a 16-byte table; indices
    /// outside the table produce zero.
    pub fn vec_tbl(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x4e00_0000 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `ins vd.<T>[last], vn.<T>[0]`: the source's lane 0 into the top element.
    pub fn vec_ins_last(&mut self, rd: Reg, rn: Reg, bytes: u32) {
        assert!(matches!(bytes, 1 | 2 | 4 | 8), "element size {bytes}");
        self.emit(0x6e00_0400 | (32 - bytes) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `dup v, wn`: broadcast the low `bytes` bytes of `rn` (a W register for 1/2/4,
    /// an X register for 8) to every lane.
    pub fn vec_dup(&mut self, rd: Reg, rn: Reg, bytes: u32) {
        self.emit(0x4e00_0c00 | bytes << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `ushr v.16b/8h/4s/2d, vn, #imm`: shift each element right by an immediate,
    /// zero fill. Bits 22:16 hold `2*size - imm`.
    pub fn vec_ushr_imm(&mut self, size: u32, rd: Reg, rn: Reg, imm: u32) {
        self.emit(0x6f00_0400 | (2 * size - imm) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `zip1 v.16b, vn.16b, vm.16b`: interleave the low eight bytes.
    pub fn vec_zip1_16b(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x4e00_3800 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `shl v.16b, vn.16b, #imm`: shift each byte left.
    pub fn vec_shl_16b(&mut self, rd: Reg, rn: Reg, imm: u32) {
        assert!(imm < 8, "byte shift {imm}");
        self.emit(0x4f08_5400 | imm << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `ushr v.16b, vn.16b, #imm`: shift each byte right, zero fill.
    pub fn vec_ushr_16b(&mut self, rd: Reg, rn: Reg, imm: u32) {
        assert!((1..=8).contains(&imm), "byte shift {imm}");
        self.emit(0x6f00_0400 | (16 - imm) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `rev16 v.16b`: reverse each 16-bit halfword's bytes.
    pub fn vec_rev16(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e20_1800 | (rn as u32) << 5 | rd as u32);
    }
    /// `rev32 v.16b`: reverse each 32-bit word's bytes.
    pub fn vec_rev32(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x6e20_0800 | (rn as u32) << 5 | rd as u32);
    }
    /// `rev64 v.16b`: reverse each 64-bit doubleword's bytes.
    pub fn vec_rev64(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e20_0800 | (rn as u32) << 5 | rd as u32);
    }
    /// `clz v.16b/8h/4s`: leading-zero count per element (no 2d form).
    pub fn vec_clz(&mut self, bits: u32, rd: Reg, rn: Reg) {
        let size = match bits {
            8 => 0x0000_0000,
            16 => 0x0040_0000,
            32 => 0x0080_0000,
            _ => panic!("vector clz has no {bits}-bit form"),
        };
        self.emit(0x6e20_4800 | size | (rn as u32) << 5 | rd as u32);
    }
    /// `cnt v.16b`: population count per byte.
    pub fn vec_cnt(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e20_5800 | (rn as u32) << 5 | rd as u32);
    }
    /// `uaddlp`: pairwise add long, widening 16b to 8h, 8h to 4s, or 4s to 2d.
    pub fn vec_uaddlp(&mut self, from_bits: u32, rd: Reg, rn: Reg) {
        let size = match from_bits {
            8 => 0x0000_0000,
            16 => 0x0040_0000,
            32 => 0x0080_0000,
            _ => panic!("uaddlp has no {from_bits}-bit source"),
        };
        self.emit(0x6e20_2800 | size | (rn as u32) << 5 | rd as u32);
    }
    /// `xtn`: narrow elements from 16→8, 32→16, or 64→32 bits.
    pub fn vec_xtn(&mut self, from_bits: u32, rd: Reg, rn: Reg) {
        let size = match from_bits {
            16 => 0x0000_0000,
            32 => 0x0040_0000,
            64 => 0x0080_0000,
            _ => panic!("xtn has no {from_bits}-bit source"),
        };
        self.emit(0x0e21_2800 | size | (rn as u32) << 5 | rd as u32);
    }
    /// `sshr`: the sign-filling counterpart of `vec_ushr_imm`.
    pub fn vec_sshr_imm(&mut self, size: u32, rd: Reg, rn: Reg, imm: u32) {
        self.emit(0x4f00_0400 | (2 * size - imm) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `ushl vd, vn, vm`: per-element shift; the amount is a signed byte, so a
    /// negative one shifts right and zero-fills.
    pub fn vec_ushl(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x6e20_4400 | Self::vec_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `sshl`: as `vec_ushl`, but a right shift sign-fills.
    pub fn vec_sshl(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x4e20_4400 | Self::vec_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `mul vd.16b/8h/4s, ...`: the low half of each product. There is no 64-bit form.
    pub fn vec_mul(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x4e20_9c00 | Self::vec_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `mla vd, vn, vm`: `vd += vn * vm` at 8/16/32-bit lanes.
    pub fn vec_mla(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        assert!(matches!(size, 8 | 16 | 32), "vector mla has no {size}-bit form");
        self.emit(
            0x4e20_9400 | Self::vec_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `sshll`/`ushll vd, vn, #0`: widen each source lane.
    pub fn vec_shift_long(&mut self, signed: bool, from_bits: u32, rd: Reg, rn: Reg) {
        let size = match from_bits {
            8 => 0x0008_0000,
            16 => 0x0010_0000,
            32 => 0x0020_0000,
            _ => panic!("shift-long has no {from_bits}-bit source"),
        };
        let sign = if signed { 0x0f00_0000 } else { 0x2f00_0000 };
        self.emit(sign | 0x0000_a400 | size | (rn as u32) << 5 | rd as u32);
    }
    /// `umin vd.8h/4s, vn, vm`: unsigned lane-wise minimum.
    pub fn vec_umin(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x6e20_6c00 | Self::vec_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `cmhs vd.2d, vn, vm`: unsigned greater-than-or-equal mask, 64-bit lanes.
    pub fn vec_cmhs_2d(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(0x6ee0_3c00 | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `ins vd.s[0], vn.s[0]`: write lane 0 only. Unlike a scalar FP op, an element
    /// insert leaves the other bits alone — the NaN box above the single survives.
    pub fn vec_ins_s0(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x6e04_0400 | (rn as u32) << 5 | rd as u32);
    }
    /// Insert lane 0 of `rn` into lane 0 of `rd` at the given element size in bytes.
    pub fn vec_ins_lane0(&mut self, bytes: u32, rd: Reg, rn: Reg) {
        assert!(matches!(bytes, 1 | 2 | 4 | 8), "element size {bytes}");
        self.emit(0x6e00_0400 | bytes << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// Horizontal add: `addv b/h/s`, reducing 16b/8h/4s to lane 0.
    pub fn vec_addv(&mut self, bits: u32, rd: Reg, rn: Reg) {
        let size = match bits {
            8 => 0x0000_0000,
            16 => 0x0040_0000,
            32 => 0x0080_0000,
            _ => panic!("addv has no {bits}-bit form"),
        };
        self.emit(0x4e31_b800 | size | (rn as u32) << 5 | rd as u32);
    }
    /// `addp d, v.2d`: add the two 64-bit lanes.
    pub fn vec_addp_d(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x5ef1_b800 | (rn as u32) << 5 | rd as u32);
    }
    /// `.4s` vs `.2d` for the float vector ops: bit 22 of the size field.
    const fn fp_size(size: u32) -> u32 {
        if size == 64 { 0x0040_0000 } else { 0 }
    }
    pub fn vec_fadd(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x4e20_d400 | Self::fp_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    pub fn vec_fsub(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x4ea0_d400 | Self::fp_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    pub fn vec_fmul(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x6e20_dc00 | Self::fp_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    pub fn vec_fdiv(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x6e20_fc00 | Self::fp_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `fmla vd, vn, vm`: `vd += vn × vm`, fused (one rounding).
    pub fn vec_fmla(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x4e20_cc00 | Self::fp_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `fmla vd.4s/2d, vn.4s/2d, vm.s/d[0]`: fused multiply-add with lane 0 of `vm`
    /// broadcast, without first materializing a duplicate.
    pub fn vec_fmla_lane0(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        let base = match size {
            32 => 0x4f80_1000,
            64 => 0x4fc0_1000,
            _ => panic!("no by-element fmla for {size}-bit lanes"),
        };
        self.emit(base | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `fmls vd, vn, vm`: `vd -= vn × vm`, fused.
    pub fn vec_fmls(&mut self, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            0x4ea0_cc00 | Self::fp_size(size) | (rm as u32) << 16 | (rn as u32) << 5 | rd as u32,
        );
    }
    /// `fneg vd, vn`: exact sign flip, no exceptions.
    pub fn vec_fneg(&mut self, size: u32, rd: Reg, rn: Reg) {
        self.emit(0x6ea0_f800 | Self::fp_size(size) | (rn as u32) << 5 | rd as u32);
    }
    /// `dup vd.4s/2d, vn.s/d[0]`: broadcast lane 0 of another vector register (the
    /// pinned scalar `f` registers live in this file, so a `.vf` scalar is one op).
    pub fn vec_dup_lane0(&mut self, size: u32, rd: Reg, rn: Reg) {
        let imm5 = if size == 64 { 8 } else { 4 };
        self.emit(0x4e00_0400 | imm5 << 16 | (rn as u32) << 5 | rd as u32);
    }
    /// `fcvtl v.4s, v.4h`: widen the low four f16 lanes.
    pub fn vec_fcvtl(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x0e21_7800 | (rn as u32) << 5 | rd as u32);
    }
    /// `fcvtl2 v.4s, v.8h`: widen the high four f16 lanes.
    pub fn vec_fcvtl2(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e21_7800 | (rn as u32) << 5 | rd as u32);
    }
    /// `fcvtn v.4h, v.4s`: narrow four f32 lanes into the low half.
    pub fn vec_fcvtn(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x0e21_6800 | (rn as u32) << 5 | rd as u32);
    }
    /// `fcvtn2 v.8h, v.4s`: narrow four f32 lanes into the high half.
    pub fn vec_fcvtn2(&mut self, rd: Reg, rn: Reg) {
        self.emit(0x4e21_6800 | (rn as u32) << 5 | rd as u32);
    }
    /// `cmn wn, #imm` (sets flags for wn + imm).
    pub fn cmn_imm_w(&mut self, rn: Reg, imm: u32) {
        self.imm12(0x3100_0000, XZR, rn, imm);
    }
    pub fn and_imm_w(&mut self, rd: Reg, rn: Reg, imm: u64) -> Option<()> {
        self.logical(0x1200_0000, 32, rd, rn, imm)
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
    fn shift_long_encodings() {
        let mut a = Asm::new();
        a.vec_shift_long(true, 8, 20, 21);
        a.vec_shift_long(false, 8, 22, 23);
        a.vec_shift_long(true, 16, 24, 25);
        assert_eq!(a.code, [0x0f08_a6b4, 0x2f08_a6f6, 0x0f10_a738]);
    }

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
    fn half_conversions_match_assembler() {
        let mut a = Asm::new();
        a.vec_fcvtl(16, 17);
        a.vec_fcvtl2(18, 17);
        a.vec_fcvtn(19, 20);
        a.vec_fcvtn2(19, 20);
        assert_eq!(a.code, [0x0e21_7a30, 0x4e21_7a32, 0x0e21_6a93, 0x4e21_6a93]);
    }

    #[test]
    fn fmla_lane0_matches_assembler() {
        let mut a = Asm::new();
        a.vec_fmla_lane0(32, 16, 18, 15);
        a.vec_fmla_lane0(64, 16, 18, 15);
        assert_eq!(a.code, [0x4f8f_1250, 0x4fcf_1250]);
    }

    #[test]
    fn tbl_matches_assembler() {
        let mut a = Asm::new();
        a.vec_tbl(18, 16, 17);
        assert_eq!(a.code, [0x4e11_0212]);
    }

    #[test]
    fn gather_helpers_match_assembler() {
        let mut a = Asm::new();
        a.vec_umin(16, 17, 17, 18);
        a.vec_umin(32, 17, 17, 18);
        a.vec_cmhs_2d(17, 17, 18);
        assert_eq!(a.code, [0x6e72_6e31, 0x6eb2_6e31, 0x6ef2_3e31]);
    }

    /// The architecture's `DecodeBitMasks` for a 64-bit operation.
    fn decode_bit_masks(n: u32, immr: u32, imms: u32) -> Option<u64> {
        let combined = (n << 6) | (!imms & 0x3f);
        if combined < 2 {
            return None; // element size below 2: reserved
        }
        let len = 31 - combined.leading_zeros();
        let size = 1u32 << len;
        let s = imms & (size - 1);
        let r = immr & (size - 1);
        if s == size - 1 {
            return None; // all ones: reserved
        }
        let mask = if size == 64 { u64::MAX } else { (1u64 << size) - 1 };
        let run = (1u64 << (s + 1)) - 1;
        let elem = ((run >> r) | (run << ((size - r) % size))) & mask;
        let mut value = elem;
        let mut width = size;
        while width < 64 {
            value |= value << width;
            width *= 2;
        }
        Some(value)
    }

    #[test]
    fn every_bitmask_immediate_round_trips() {
        let mut count = 0;
        for n in 0..2 {
            for immr in 0..64 {
                for imms in 0..64 {
                    let Some(value) = decode_bit_masks(n, immr, imms) else { continue };
                    let (en, er, es) = logical_imm(value, 64)
                        .unwrap_or_else(|| panic!("{n} {immr} {imms}: {value:#x}"));
                    assert_eq!(decode_bit_masks(en, er, es), Some(value), "{value:#x}");
                    count += 1;
                }
            }
        }
        assert!(count >= 5334, "{count}"); // distinct values, some with several encodings
    }

    #[test]
    fn bitmask_immediates_reject_other_values() {
        // Random values, and runs with a bit flipped: any accepted value must decode back.
        let mut x = 0x2545_f491_4f6c_dd1du64;
        for i in 0..200_000u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = if i % 2 == 0 { x } else { (u64::MAX >> (x % 64)) ^ (1 << (x >> 58)) };
            if let Some((n, immr, imms)) = logical_imm(v, 64) {
                assert_eq!(decode_bit_masks(n, immr, imms), Some(v), "{v:#x}");
            }
        }
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
