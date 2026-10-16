//! RISC-V F/D arithmetic on the host (arm64) FPU.
//!
//! This module is the semantic contract for floating point: the interpreter calls it, and
//! the JIT (M4) will emit the same instruction sequences inline. ARM64 matches RISC-V almost
//! everywhere once `FPCR.DN` (default NaN) is set, because ARM's default NaN *is* the RISC-V
//! canonical NaN and `FPSR`'s cumulative flags map one-to-one onto `fflags`. The remaining
//! differences are fixed up here, each on a path that is rare at run time:
//!
//! - `fmin`/`fmax` with a signaling NaN and a number: RISC-V returns the number, ARM's
//!   `fminnm`/`fmaxnm` return NaN.
//! - float → integer of NaN: RISC-V returns the largest integer, ARM returns 0.
//! - Round-to-nearest, ties-to-max-magnitude (RMM): ARM has it only for conversions to
//!   integer (`fcvtas`), so arithmetic in RMM falls back to SoftFloat.
//! - Underflow: ARM detects tininess before rounding, RISC-V after. They can only disagree
//!   when the result rounds to exactly ±the smallest normal, and ARM raised UF; in that case
//!   we recompute with SoftFloat.
//!
//! Every function is checked against Berkeley SoftFloat (the RISC-V reference) by the fuzz
//! tests at the bottom.

use std::arch::asm;

use softfloat as sf;

pub use crate::decoder::{FArithOp as BinOp, FCmpOp as CmpOp, FWidth as Width, FmaOp, IntType};

#[cfg(not(target_arch = "aarch64"))]
compile_error!("the host FPU backend is arm64-only");

/// Accrued exception flags in RISC-V `fflags` layout: NX=1, UF=2, OF=4, DZ=8, NV=16.
pub type Fflags = u8;
pub const NV: Fflags = 16;
pub const DZ: Fflags = 8;
pub const OF: Fflags = 4;
pub const UF: Fflags = 2;
pub const NX: Fflags = 1;

/// A resolved RISC-V rounding mode (`rm` values 0–4; `dyn` is resolved by the caller).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm {
    Rne,
    Rtz,
    Rdn,
    Rup,
    Rmm,
}

impl Rm {
    pub fn from_bits(bits: u8) -> Option<Rm> {
        Some(match bits {
            0 => Rm::Rne,
            1 => Rm::Rtz,
            2 => Rm::Rdn,
            3 => Rm::Rup,
            4 => Rm::Rmm,
            _ => return None,
        })
    }

    fn softfloat(self) -> sf::Rounding {
        match self {
            Rm::Rne => sf::Rounding::NearEven,
            Rm::Rtz => sf::Rounding::MinMag,
            Rm::Rdn => sf::Rounding::Min,
            Rm::Rup => sf::Rounding::Max,
            Rm::Rmm => sf::Rounding::NearMaxMag,
        }
    }

    /// `FPCR` with default-NaN mode and this rounding mode, or `None` for RMM.
    fn fpcr(self) -> Option<u64> {
        const DN: u64 = 1 << 25;
        let rmode = match self {
            Rm::Rne => 0b00,
            Rm::Rup => 0b01,
            Rm::Rdn => 0b10,
            Rm::Rtz => 0b11,
            Rm::Rmm => return None,
        };
        Some(DN | rmode << 22)
    }
}

/// `FPSR` cumulative flags (IOC, DZC, OFC, UFC, IXC = bits 0..4) to `fflags`.
#[inline]
fn fflags(fpsr: u64) -> Fflags {
    let ioc = fpsr & 1;
    let dzc = fpsr >> 1 & 1;
    let ofc = fpsr >> 2 & 1;
    let ufc = fpsr >> 3 & 1;
    let ixc = fpsr >> 4 & 1;
    (ioc << 4 | dzc << 3 | ofc << 2 | ufc << 1 | ixc) as Fflags
}

/// Run one ARM FP instruction under `fpcr`, returning the result and `fflags`. The template
/// names its operands `{r}`, `{a}`, `{b}`, `{c}` with a `:s` or `:d` modifier.
macro_rules! fp_asm {
    ($fpcr:expr, $insn:literal, $ty:ty, $r:ident; $($name:ident = $val:expr),*) => {{
        let r: $ty;
        let fpsr: u64;
        // SAFETY: only FP registers and the FPCR/FPSR system registers are touched, and the
        // host's FPCR is restored before the block ends.
        unsafe {
            asm!(
                "mrs {saved}, fpcr",
                "msr fpcr, {fpcr}",
                "msr fpsr, xzr",
                $insn,
                "mrs {fpsr}, fpsr",
                "msr fpcr, {saved}",
                fpcr = in(reg) $fpcr,
                saved = out(reg) _,
                fpsr = out(reg) fpsr,
                $r = lateout(vreg) r,
                $($name = in(vreg) $val,)*
                options(nomem, nostack),
            );
        }
        (r, fflags(fpsr))
    }};
}

fn is_nan(w: Width, v: u64) -> bool {
    match w {
        Width::S => f32::from_bits(v as u32).is_nan(),
        Width::D => f64::from_bits(v).is_nan(),
    }
}

fn is_snan(w: Width, v: u64) -> bool {
    match w {
        Width::S => is_nan(w, v) && v & (1 << 22) == 0,
        Width::D => is_nan(w, v) && v & (1 << 51) == 0,
    }
}

fn is_min_normal(w: Width, v: u64) -> bool {
    match w {
        Width::S => v & 0x7fff_ffff == 0x0080_0000,
        Width::D => v & 0x7fff_ffff_ffff_ffff == 0x0010_0000_0000_0000,
    }
}

/// SoftFloat version of a binary op: the RMM path and the tininess fix-up.
fn soft_binop(op: BinOp, w: Width, a: u64, b: u64, rm: Rm) -> (u64, Fflags) {
    // SAFETY (all): foreign but pure functions of their arguments.
    sf::with(rm.softfloat(), || unsafe {
        match w {
            Width::S => {
                let (a, b) = (sf::F32 { v: a as u32 }, sf::F32 { v: b as u32 });
                (match op {
                    BinOp::Add => sf::f32_add(a, b),
                    BinOp::Sub => sf::f32_sub(a, b),
                    BinOp::Mul => sf::f32_mul(a, b),
                    BinOp::Div => sf::f32_div(a, b),
                })
                .v as u64
            }
            Width::D => {
                let (a, b) = (sf::F64 { v: a }, sf::F64 { v: b });
                match op {
                    BinOp::Add => sf::f64_add(a, b),
                    BinOp::Sub => sf::f64_sub(a, b),
                    BinOp::Mul => sf::f64_mul(a, b),
                    BinOp::Div => sf::f64_div(a, b),
                }
                .v
            }
        }
    })
}

/// `fadd`/`fsub`/`fmul`/`fdiv`. Operands and result are raw bits (single precision in the
/// low 32 bits, already unboxed).
pub fn binop(op: BinOp, w: Width, a: u64, b: u64, rm: Rm) -> (u64, Fflags) {
    let Some(fpcr) = rm.fpcr() else { return soft_binop(op, w, a, b, rm) };
    let (r, flags) = match w {
        Width::S => {
            let (a, b) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
            let (r, fl): (f32, _) = match op {
                BinOp::Add => fp_asm!(fpcr, "fadd {r:s}, {a:s}, {b:s}", f32, r; a = a, b = b),
                BinOp::Sub => fp_asm!(fpcr, "fsub {r:s}, {a:s}, {b:s}", f32, r; a = a, b = b),
                BinOp::Mul => fp_asm!(fpcr, "fmul {r:s}, {a:s}, {b:s}", f32, r; a = a, b = b),
                BinOp::Div => fp_asm!(fpcr, "fdiv {r:s}, {a:s}, {b:s}", f32, r; a = a, b = b),
            };
            (r.to_bits() as u64, fl)
        }
        Width::D => {
            let (a, b) = (f64::from_bits(a), f64::from_bits(b));
            let (r, fl): (f64, _) = match op {
                BinOp::Add => fp_asm!(fpcr, "fadd {r:d}, {a:d}, {b:d}", f64, r; a = a, b = b),
                BinOp::Sub => fp_asm!(fpcr, "fsub {r:d}, {a:d}, {b:d}", f64, r; a = a, b = b),
                BinOp::Mul => fp_asm!(fpcr, "fmul {r:d}, {a:d}, {b:d}", f64, r; a = a, b = b),
                BinOp::Div => fp_asm!(fpcr, "fdiv {r:d}, {a:d}, {b:d}", f64, r; a = a, b = b),
            };
            (r.to_bits(), fl)
        }
    };
    if flags & UF != 0 && is_min_normal(w, r) {
        return soft_binop(op, w, a, b, rm);
    }
    (r, flags)
}

pub fn sqrt(w: Width, a: u64, rm: Rm) -> (u64, Fflags) {
    // SAFETY: foreign but pure.
    let soft = || {
        sf::with(rm.softfloat(), || unsafe {
            match w {
                Width::S => sf::f32_sqrt(sf::F32 { v: a as u32 }).v as u64,
                Width::D => sf::f64_sqrt(sf::F64 { v: a }).v,
            }
        })
    };
    // The square root of a positive number never underflows, so only RMM needs SoftFloat.
    let Some(fpcr) = rm.fpcr() else { return soft() };
    match w {
        Width::S => {
            let (r, fl): (f32, _) =
                fp_asm!(fpcr, "fsqrt {r:s}, {a:s}", f32, r; a = f32::from_bits(a as u32));
            (r.to_bits() as u64, fl)
        }
        Width::D => {
            let (r, fl): (f64, _) =
                fp_asm!(fpcr, "fsqrt {r:d}, {a:d}", f64, r; a = f64::from_bits(a));
            (r.to_bits(), fl)
        }
    }
}

/// Fused multiply-add. ARM names the same four operations with the signs swapped around:
/// RISC-V `fmsub` is ARM `fnmsub`, and RISC-V `fnmsub` is ARM `fmsub`.
pub fn fma(op: FmaOp, w: Width, a: u64, b: u64, c: u64, rm: Rm) -> (u64, Fflags) {
    let soft = || {
        let neg = |v: u64| v ^ if w == Width::S { 1 << 31 } else { 1 << 63 };
        // RISC-V's negations apply to the product and the addend; negating an operand of
        // the product is equivalent and exact, and NaN results are canonical either way.
        let (a2, c2) = match op {
            FmaOp::Madd => (a, c),
            FmaOp::Msub => (a, neg(c)),
            FmaOp::Nmsub => (neg(a), c),
            FmaOp::Nmadd => (neg(a), neg(c)),
        };
        // SAFETY: foreign but pure.
        sf::with(rm.softfloat(), || unsafe {
            match w {
                Width::S => {
                    sf::f32_mulAdd(
                        sf::F32 { v: a2 as u32 },
                        sf::F32 { v: b as u32 },
                        sf::F32 { v: c2 as u32 },
                    )
                    .v as u64
                }
                Width::D => {
                    sf::f64_mulAdd(sf::F64 { v: a2 }, sf::F64 { v: b }, sf::F64 { v: c2 }).v
                }
            }
        })
    };
    let Some(fpcr) = rm.fpcr() else { return soft() };
    let (r, flags) = match w {
        Width::S => {
            let (a, b, c) =
                (f32::from_bits(a as u32), f32::from_bits(b as u32), f32::from_bits(c as u32));
            let (r, fl): (f32, _) = match op {
                FmaOp::Madd => {
                    fp_asm!(fpcr, "fmadd {r:s}, {a:s}, {b:s}, {c:s}", f32, r; a = a, b = b, c = c)
                }
                FmaOp::Msub => {
                    fp_asm!(fpcr, "fnmsub {r:s}, {a:s}, {b:s}, {c:s}", f32, r; a = a, b = b, c = c)
                }
                FmaOp::Nmsub => {
                    fp_asm!(fpcr, "fmsub {r:s}, {a:s}, {b:s}, {c:s}", f32, r; a = a, b = b, c = c)
                }
                FmaOp::Nmadd => {
                    fp_asm!(fpcr, "fnmadd {r:s}, {a:s}, {b:s}, {c:s}", f32, r; a = a, b = b, c = c)
                }
            };
            (r.to_bits() as u64, fl)
        }
        Width::D => {
            let (a, b, c) = (f64::from_bits(a), f64::from_bits(b), f64::from_bits(c));
            let (r, fl): (f64, _) = match op {
                FmaOp::Madd => {
                    fp_asm!(fpcr, "fmadd {r:d}, {a:d}, {b:d}, {c:d}", f64, r; a = a, b = b, c = c)
                }
                FmaOp::Msub => {
                    fp_asm!(fpcr, "fnmsub {r:d}, {a:d}, {b:d}, {c:d}", f64, r; a = a, b = b, c = c)
                }
                FmaOp::Nmsub => {
                    fp_asm!(fpcr, "fmsub {r:d}, {a:d}, {b:d}, {c:d}", f64, r; a = a, b = b, c = c)
                }
                FmaOp::Nmadd => {
                    fp_asm!(fpcr, "fnmadd {r:d}, {a:d}, {b:d}, {c:d}", f64, r; a = a, b = b, c = c)
                }
            };
            (r.to_bits(), fl)
        }
    };
    if flags & UF != 0 && is_min_normal(w, r) {
        return soft();
    }
    (r, flags)
}

/// `fminm`/`fmaxm` (Zfa): IEEE 754-2019 minimum/maximum — if either input is NaN the
/// result is the canonical NaN, and a signaling input also raises NV.
pub fn min_max_zfa(max: bool, w: Width, a: u64, b: u64) -> (u64, Fflags) {
    if is_nan(w, a) || is_nan(w, b) {
        let flags = if is_snan(w, a) || is_snan(w, b) { NV } else { 0 };
        let quiet = match w {
            Width::S => f32::NAN.to_bits() as u64,
            Width::D => f64::NAN.to_bits(),
        };
        return (quiet, flags);
    }
    // No NaN input, so the hardware path below is exact.
    min_max(max, w, a, b)
}

/// `fmin`/`fmax` (IEEE 754-2019 minimumNumber/maximumNumber, with -0 < +0).
pub fn min_max(max: bool, w: Width, a: u64, b: u64) -> (u64, Fflags) {
    // RISC-V: a NaN operand is ignored in favor of a number, and a signaling NaN also raises
    // NV. ARM's fminnm/fmaxnm agree except that a signaling NaN poisons the result.
    let (snan_a, snan_b) = (is_snan(w, a), is_snan(w, b));
    if (snan_a || snan_b) && !(is_nan(w, a) && is_nan(w, b)) {
        return (if snan_a { b } else { a }, NV);
    }
    let fpcr = Rm::Rne.fpcr().unwrap();
    match w {
        Width::S => {
            let (a, b) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
            let (r, fl): (f32, _) = if max {
                fp_asm!(fpcr, "fmaxnm {r:s}, {a:s}, {b:s}", f32, r; a = a, b = b)
            } else {
                fp_asm!(fpcr, "fminnm {r:s}, {a:s}, {b:s}", f32, r; a = a, b = b)
            };
            (r.to_bits() as u64, fl)
        }
        Width::D => {
            let (a, b) = (f64::from_bits(a), f64::from_bits(b));
            let (r, fl): (f64, _) = if max {
                fp_asm!(fpcr, "fmaxnm {r:d}, {a:d}, {b:d}", f64, r; a = a, b = b)
            } else {
                fp_asm!(fpcr, "fminnm {r:d}, {a:d}, {b:d}", f64, r; a = a, b = b)
            };
            (r.to_bits(), fl)
        }
    }
}

/// `feq` (quiet: NV only for signaling NaNs), `flt`/`fle` (signaling: NV for any NaN),
/// and Zfa's `fleq`/`fltq` (quiet for quiet NaNs, NV only for signaling ones).
/// ARM's `fcmp` (raises nothing) and `fcmpe` (raises for any NaN) cover the first two;
/// the quiet pair uses `fcmp` plus the signaling-NaN fix-up.
pub fn compare(op: CmpOp, w: Width, a: u64, b: u64) -> (bool, Fflags) {
    let r: u64;
    let fpsr: u64;
    macro_rules! cmp {
        ($insn:literal, $cond:literal, $reg:ident, $ty:ty, $a:expr, $b:expr) => {
            // SAFETY: only FP state and the FPSR are touched.
            unsafe {
                asm!(
                    "msr fpsr, xzr",
                    concat!($insn, " {a:", stringify!($reg), "}, {b:", stringify!($reg), "}"),
                    concat!("cset {r}, ", $cond),
                    "mrs {fpsr}, fpsr",
                    a = in(vreg) $a,
                    b = in(vreg) $b,
                    r = out(reg) r,
                    fpsr = out(reg) fpsr,
                    options(nomem, nostack),
                )
            }
        };
    }
    // Conditions after an FP compare: eq (equal), mi (less than; false when unordered),
    // ls (less or equal; false when unordered).
    match (w, op) {
        (Width::S, CmpOp::Eq) => {
            cmp!("fcmp", "eq", s, f32, f32::from_bits(a as u32), f32::from_bits(b as u32))
        }
        (Width::S, CmpOp::Lt) => {
            cmp!("fcmpe", "mi", s, f32, f32::from_bits(a as u32), f32::from_bits(b as u32))
        }
        (Width::S, CmpOp::Le) => {
            cmp!("fcmpe", "ls", s, f32, f32::from_bits(a as u32), f32::from_bits(b as u32))
        }
        (Width::S, CmpOp::Ltq) => {
            cmp!("fcmp", "mi", s, f32, f32::from_bits(a as u32), f32::from_bits(b as u32))
        }
        (Width::S, CmpOp::Leq) => {
            cmp!("fcmp", "ls", s, f32, f32::from_bits(a as u32), f32::from_bits(b as u32))
        }
        (Width::D, CmpOp::Eq) => cmp!("fcmp", "eq", d, f64, f64::from_bits(a), f64::from_bits(b)),
        (Width::D, CmpOp::Lt) => cmp!("fcmpe", "mi", d, f64, f64::from_bits(a), f64::from_bits(b)),
        (Width::D, CmpOp::Le) => cmp!("fcmpe", "ls", d, f64, f64::from_bits(a), f64::from_bits(b)),
        (Width::D, CmpOp::Ltq) => cmp!("fcmp", "mi", d, f64, f64::from_bits(a), f64::from_bits(b)),
        (Width::D, CmpOp::Leq) => cmp!("fcmp", "ls", d, f64, f64::from_bits(a), f64::from_bits(b)),
    }
    let mut flags = fflags(fpsr);
    if matches!(op, CmpOp::Ltq | CmpOp::Leq) && (is_snan(w, a) || is_snan(w, b)) {
        flags |= NV;
    }
    (r != 0, flags)
}

/// Zfa `fround.s`/`fround.d`: round to an integral value in the same format. Zero and
/// infinite inputs pass through, NaN inputs give the canonical NaN (NV for signaling
/// NaN), and `exact` (`froundnx`) raises NX when the input was not already integral.
/// `Rmm` (round-to-nearest, ties-away) maps onto ARM's `frinta`, which the FPCR
/// cannot express.
pub fn round_to_int(w: Width, a: u64, rm: Rm, exact: bool) -> (u64, Fflags) {
    if rm == Rm::Rmm {
        // frinta rounds ties away but never reports inexact; froundnx compares.
        let fpcr = Rm::Rne.fpcr().unwrap();
        let (r, fl) = match w {
            Width::S => {
                let (r, fl): (f32, _) = fp_asm!(fpcr, "frinta {r:s}, {a:s}", f32, r;
                    a = f32::from_bits(a as u32));
                (r.to_bits() as u64, fl)
            }
            Width::D => {
                let (r, fl): (f64, _) = fp_asm!(fpcr, "frinta {r:d}, {a:d}", f64, r;
                    a = f64::from_bits(a));
                (r.to_bits(), fl)
            }
        };
        let flags = if exact && !is_nan(w, a) && r != a { fl | NX } else { fl };
        return (r, flags);
    }
    let fpcr = rm.fpcr().expect("Rmm was handled above");
    match (w, exact) {
        (Width::S, false) => {
            let (r, fl): (f32, _) = fp_asm!(fpcr, "frinti {r:s}, {a:s}", f32, r;
                a = f32::from_bits(a as u32));
            (r.to_bits() as u64, fl)
        }
        (Width::S, true) => {
            let (r, fl): (f32, _) = fp_asm!(fpcr, "frintx {r:s}, {a:s}", f32, r;
                a = f32::from_bits(a as u32));
            (r.to_bits() as u64, fl)
        }
        (Width::D, false) => {
            let (r, fl): (f64, _) = fp_asm!(fpcr, "frinti {r:d}, {a:d}", f64, r;
                a = f64::from_bits(a));
            (r.to_bits(), fl)
        }
        (Width::D, true) => {
            let (r, fl): (f64, _) = fp_asm!(fpcr, "frintx {r:d}, {a:d}", f64, r;
                a = f64::from_bits(a));
            (r.to_bits(), fl)
        }
    }
}

/// Zfa `fcvtmod.w.d`: `fcvt.w.d` in RTZ with the range check replaced by arithmetic
/// modulo 2^32 — bits 31:0 of the unbounded two's-complement result, sign-extended;
/// NaN and the infinities convert to zero. The flags are what `fcvt.w.d` raises:
/// NV alone when the rounded value leaves the `i32` range or the input is not a
/// number, NX when an in-range input had a fractional part.
pub fn fcvtmod_w_d(a: u64) -> (u64, Fflags) {
    let sign = a >> 63;
    let exp = (a >> 52 & 0x7ff) as i32;
    let frac = a & ((1u64 << 52) - 1);
    if exp == 0x7ff {
        return (0, NV); // NaN and ±inf both convert to zero
    }
    if exp == 0 {
        return (0, if frac == 0 { 0 } else { NX }); // zero or subnormal: |x| < 1
    }
    let e = exp - 1023;
    if e < 0 {
        return (0, NX); // |x| < 1, x != 0
    }
    let m = frac | 1u64 << 52;
    let inexact: bool;
    let mag: u32;
    let out_of_range: bool;
    if e < 52 {
        let drop = (52 - e) as u32;
        inexact = m & ((1u64 << drop) - 1) != 0;
        let trunc = m >> drop;
        mag = trunc as u32;
        // The truncated value decides the range: anything past 2^31 is out, +2^31
        // itself does not fit an `i32`, and -2^31 does.
        out_of_range = trunc > 1u64 << 31 || (trunc == 1u64 << 31 && sign == 0);
    } else {
        // Already integral; the low bits of the magnitude come from a shift.
        inexact = false;
        let up = (e - 52) as u32;
        mag = if up >= 32 { 0 } else { ((m << up) & 0xffff_ffff) as u32 };
        out_of_range = true; // |x| >= 2^63 > i32::MAX
    }
    let flags = if out_of_range {
        NV
    } else if inexact {
        NX
    } else {
        0
    };
    let low = if sign == 1 { mag.wrapping_neg() } else { mag };
    ((low as i32) as i64 as u64, flags)
}

/// `fclass`: a one-hot classification mask. Pure bit inspection, no flags.
pub fn classify(w: Width, a: u64) -> u64 {
    let (sign, exp, frac, exp_max, quiet_bit) = match w {
        Width::S => (a >> 31 & 1, a >> 23 & 0xff, a & 0x7f_ffff, 0xff, 1 << 22),
        Width::D => (a >> 63 & 1, a >> 52 & 0x7ff, a & 0xf_ffff_ffff_ffff, 0x7ff, 1 << 51),
    };
    let neg = sign == 1;
    let bit = match (exp, frac) {
        (e, 0) if e == exp_max => {
            if neg {
                0
            } else {
                7
            }
        } // ±inf
        (e, f) if e == exp_max => {
            if f & quiet_bit != 0 {
                9
            } else {
                8
            }
        } // qNaN / sNaN
        (0, 0) => {
            if neg {
                3
            } else {
                4
            }
        } // ±0
        (0, _) => {
            if neg {
                2
            } else {
                5
            }
        } // ±subnormal
        _ => {
            if neg {
                1
            } else {
                6
            }
        } // ±normal
    };
    1 << bit
}

/// Float → integer. Returns the value as it lands in `x[rd]` (32-bit results sign-extended,
/// including `wu`).
pub fn to_int(w: Width, t: IntType, a: u64, rm: Rm) -> (u64, Fflags) {
    if is_nan(w, a) {
        let max = match t {
            IntType::W => i32::MAX as i64 as u64,
            IntType::Wu => u32::MAX as i32 as i64 as u64,
            IntType::L => i64::MAX as u64,
            IntType::Lu => u64::MAX,
        };
        return (max, NV);
    }
    // One asm block per (source width, destination type, rounding): ARM encodes the
    // rounding in the instruction (fcvt{n,z,m,p,a}{s,u}), which also covers RMM (`a`).
    macro_rules! cvt {
        ($insn:literal, $dst:literal, $src:ident, $val:expr) => {{
            let r: u64;
            let fpsr: u64;
            // SAFETY: only FP state and the FPSR are touched.
            unsafe {
                asm!(
                    "msr fpsr, xzr",
                    concat!($insn, " {r:", $dst, "}, {a:", stringify!($src), "}"),
                    "mrs {fpsr}, fpsr",
                    a = in(vreg) $val,
                    r = out(reg) r,
                    fpsr = out(reg) fpsr,
                    options(nomem, nostack),
                );
            }
            (r, fflags(fpsr))
        }};
    }
    macro_rules! by_rm {
        ($n:literal, $z:literal, $m:literal, $p:literal, $a:literal, $dst:literal, $src:ident, $val:expr) => {
            match rm {
                Rm::Rne => cvt!($n, $dst, $src, $val),
                Rm::Rtz => cvt!($z, $dst, $src, $val),
                Rm::Rdn => cvt!($m, $dst, $src, $val),
                Rm::Rup => cvt!($p, $dst, $src, $val),
                Rm::Rmm => cvt!($a, $dst, $src, $val),
            }
        };
    }
    macro_rules! by_type {
        ($src:ident, $val:expr) => {
            match t {
                IntType::W | IntType::L => {
                    let dst = if t == IntType::W { "w" } else { "x" };
                    if dst == "w" {
                        by_rm!("fcvtns", "fcvtzs", "fcvtms", "fcvtps", "fcvtas", "w", $src, $val)
                    } else {
                        by_rm!("fcvtns", "fcvtzs", "fcvtms", "fcvtps", "fcvtas", "x", $src, $val)
                    }
                }
                IntType::Wu => {
                    by_rm!("fcvtnu", "fcvtzu", "fcvtmu", "fcvtpu", "fcvtau", "w", $src, $val)
                }
                IntType::Lu => {
                    by_rm!("fcvtnu", "fcvtzu", "fcvtmu", "fcvtpu", "fcvtau", "x", $src, $val)
                }
            }
        };
    }
    let (r, flags) = match w {
        Width::S => by_type!(s, f32::from_bits(a as u32)),
        Width::D => by_type!(d, f64::from_bits(a)),
    };
    // A 32-bit `w` register write zero-extends; RISC-V sign-extends both w and wu results.
    let r = match t {
        IntType::W | IntType::Wu => r as u32 as i32 as i64 as u64,
        IntType::L | IntType::Lu => r,
    };
    (r, flags)
}

/// Integer → float. `x` is the full register; `w`/`wu` sources use its low 32 bits.
pub fn from_int(w: Width, t: IntType, x: u64, rm: Rm) -> (u64, Fflags) {
    let Some(fpcr) = rm.fpcr() else {
        // SAFETY: foreign but pure.
        return sf::with(rm.softfloat(), || unsafe {
            match (w, t) {
                (Width::S, IntType::W) => sf::i32_to_f32(x as i32).v as u64,
                (Width::S, IntType::Wu) => sf::ui32_to_f32(x as u32).v as u64,
                (Width::S, IntType::L) => sf::i64_to_f32(x as i64).v as u64,
                (Width::S, IntType::Lu) => sf::ui64_to_f32(x).v as u64,
                (Width::D, IntType::W) => sf::i32_to_f64(x as i32).v,
                (Width::D, IntType::Wu) => sf::ui32_to_f64(x as u32).v,
                (Width::D, IntType::L) => sf::i64_to_f64(x as i64).v,
                (Width::D, IntType::Lu) => sf::ui64_to_f64(x).v,
            }
        });
    };
    macro_rules! cvt {
        ($insn:literal, $ty:ty, $dst:literal, $src:literal, $val:expr) => {{
            let r: $ty;
            let fpsr: u64;
            // SAFETY: only FP registers and the FPCR/FPSR are touched; FPCR is restored.
            unsafe {
                asm!(
                    "mrs {saved}, fpcr",
                    "msr fpcr, {fpcr}",
                    "msr fpsr, xzr",
                    concat!($insn, " {r:", $dst, "}, {x:", $src, "}"),
                    "mrs {fpsr}, fpsr",
                    "msr fpcr, {saved}",
                    fpcr = in(reg) fpcr,
                    saved = out(reg) _,
                    fpsr = out(reg) fpsr,
                    x = in(reg) $val,
                    r = lateout(vreg) r,
                    options(nomem, nostack),
                );
            }
            (r.to_bits() as u64, fflags(fpsr))
        }};
    }
    // Integer → float never underflows, so no tininess fix-up is needed.
    match (w, t) {
        (Width::S, IntType::W) => cvt!("scvtf", f32, "s", "w", x),
        (Width::S, IntType::Wu) => cvt!("ucvtf", f32, "s", "w", x),
        (Width::S, IntType::L) => cvt!("scvtf", f32, "s", "x", x),
        (Width::S, IntType::Lu) => cvt!("ucvtf", f32, "s", "x", x),
        (Width::D, IntType::W) => cvt!("scvtf", f64, "d", "w", x),
        (Width::D, IntType::Wu) => cvt!("ucvtf", f64, "d", "w", x),
        (Width::D, IntType::L) => cvt!("scvtf", f64, "d", "x", x),
        (Width::D, IntType::Lu) => cvt!("ucvtf", f64, "d", "x", x),
    }
}

/// `fcvt.s.d` (narrowing, rounds) and `fcvt.d.s` (widening, exact).
pub fn convert(to: Width, a: u64, rm: Rm) -> (u64, Fflags) {
    let soft = || {
        // SAFETY: foreign but pure.
        sf::with(rm.softfloat(), || unsafe {
            match to {
                Width::S => sf::f64_to_f32(sf::F64 { v: a }).v as u64,
                Width::D => sf::f32_to_f64(sf::F32 { v: a as u32 }).v,
            }
        })
    };
    let Some(fpcr) = rm.fpcr() else { return soft() };
    let (r, flags) = match to {
        Width::S => {
            let (r, fl): (f32, _) =
                fp_asm!(fpcr, "fcvt {r:s}, {a:d}", f32, r; a = f64::from_bits(a));
            (r.to_bits() as u64, fl)
        }
        Width::D => {
            let (r, fl): (f64, _) =
                fp_asm!(fpcr, "fcvt {r:d}, {a:s}", f64, r; a = f32::from_bits(a as u32));
            (r.to_bits(), fl)
        }
    };
    if flags & UF != 0 && is_min_normal(to, r) {
        return soft();
    }
    (r, flags)
}

/// Zfhmin `fcvt.s.h`: widening a half to a single is exact. A signaling NaN raises
/// NV, and every NaN becomes the positive canonical NaN (qemu agrees on the sign).
pub fn f16_to_f32(a: u16) -> (u32, Fflags) {
    let sign = u32::from(a >> 15) << 31;
    let exp = u32::from(a >> 10 & 0x1f);
    let frac = u32::from(a & 0x3ff);
    if exp == 0x1f {
        if frac == 0 {
            return (sign | 0x7f80_0000, 0); // ±inf
        }
        let flags = if frac & 0x200 == 0 { NV } else { 0 };
        return (0x7fc0_0000, flags); // canonical NaN, positive
    }
    if exp == 0 {
        if frac == 0 {
            return (sign, 0); // ±0
        }
        // Subnormal half: normalize frac·2^-24 into IEEE single.
        let n = 31 - frac.leading_zeros();
        return (sign | ((103 + n) << 23) | (frac << (23 - n) & 0x7f_ffff), 0);
    }
    (sign | ((exp + 112) << 23) | (frac << 13), 0) // normal: rebias 15 -> 127
}

/// Zfhmin `fcvt.h.s`: narrow a single to half under `rm`, with the RISC-V flag
/// rules: overflow gives ±inf and OF|NX, a subnormal (or rounded-to-zero) result
/// that lost bits gives NX|UF (tininess after rounding, so rounding up to the
/// smallest normal drops UF), exact subnormals stay silent, and NaNs come out
/// canonical with NV only for signaling inputs.
pub fn f32_to_f16(a: u32, rm: Rm) -> (u16, Fflags) {
    let sign = a >> 31;
    let exp = (a >> 23) & 0xff;
    let frac = a & 0x7f_ffff;
    let s16 = (sign as u16) << 15;
    if exp == 0xff {
        if frac == 0 {
            return (s16 | 0x7c00, 0);
        }
        let flags = if frac & (1 << 22) == 0 { NV } else { 0 };
        return (0x7e00, flags); // canonical NaN, positive
    }
    if exp == 0 && frac == 0 {
        return (s16, 0);
    }
    // value = m · 2^e with a 24-bit significand.
    let (m, e) = if exp == 0 {
        let shift = frac.leading_zeros() as i32 - 8;
        (u64::from(frac) << shift, -149 - shift)
    } else {
        (u64::from(frac | 0x80_0000), exp as i32 - 150)
    };
    let eh = e + 23; // the value's exponent of two
    let negative = sign == 1;
    if eh < -14 {
        // Subnormal: quantum 2^-24; 1024 carries up to the smallest normal.
        let (r, inx) = round_h(m, e, -24, rm, negative);
        if r >= 1024 {
            return (s16 | 1 << 10, if inx { NX } else { 0 });
        }
        return (s16 | r as u16, if inx { NX | UF } else { 0 });
    }
    let (mut r, inx) = round_h(m, e, eh - 10, rm, negative);
    let mut eh = eh;
    if r == 2048 {
        // The significand carried out of the binade: that is 1.0 of the next one.
        r = 1024;
        eh += 1;
    }
    if eh > 15 {
        // Some rounding modes carry the overflow to the largest finite value
        // instead of infinity (IEEE: RTZ, and the modes pointing inward).
        let to_max = matches!((rm, negative), (Rm::Rtz, _) | (Rm::Rdn, false) | (Rm::Rup, true));
        return (s16 | if to_max { 0x7bff } else { 0x7c00 }, OF | NX);
    }
    (s16 | (((eh + 15) as u16) << 10) | (r as u16 - 1024), if inx { NX } else { 0 })
}

/// Round `m · 2^e` to an 11-bit significand with quantum `2^q_exp`, returning the
/// (possibly carrying to 2^11) integer and whether anything was lost.
fn round_h(m: u64, e: i32, q_exp: i32, rm: Rm, negative: bool) -> (u64, bool) {
    let shift = e - q_exp;
    if shift >= 0 {
        return (m << shift, false);
    }
    let drop = -shift;
    if drop >= 64 {
        // Far below half a quantum: only the directed modes tick up.
        let inx = m != 0;
        let inc = inx && ((rm == Rm::Rup && !negative) || (rm == Rm::Rdn && negative));
        return (u64::from(inc), inx);
    }
    let r = m >> drop;
    let rem = m & ((1u64 << drop) - 1);
    let half = 1u64 << (drop - 1);
    let inc = match rm {
        Rm::Rne => rem > half || (rem == half && r & 1 == 1),
        Rm::Rtz => false,
        Rm::Rdn => negative && rem != 0,
        Rm::Rup => !negative && rem != 0,
        Rm::Rmm => rem >= half,
    };
    (r + u64::from(inc), rem != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift, biased toward the values where FP semantics get interesting.
    struct Gen(u64);

    impl Gen {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn float(&mut self, w: Width) -> u64 {
            let r = self.next();
            let (sign, exp_bits, frac_bits) = match w {
                Width::S => (1u64 << 31, 8, 23),
                Width::D => (1u64 << 63, 11, 52),
            };
            let exp_max = (1u64 << exp_bits) - 1;
            let frac_mask = (1u64 << frac_bits) - 1;
            let exp = match r % 8 {
                0 => 0,                             // zero / subnormal
                1 => exp_max,                       // inf / NaN
                2 => 1,                             // smallest normal binade
                3 => (exp_max >> 1) + (r >> 8) % 4, // around 1.0
                _ => (r >> 16) % (exp_max + 1),
            };
            let frac = match (r >> 3) % 4 {
                0 => 0,
                1 => frac_mask,
                2 => 1 << (frac_bits - 1), // quiet bit only
                _ => self.next() & frac_mask,
            };
            let s = if r & 1 << 40 != 0 { sign } else { 0 };
            s | exp << frac_bits | frac
        }

        fn rm(&mut self) -> Rm {
            Rm::from_bits((self.next() % 5) as u8).unwrap()
        }
    }

    const N: usize = 200_000;

    fn soft32(a: u64) -> sf::F32 {
        sf::F32 { v: a as u32 }
    }
    fn soft64(a: u64) -> sf::F64 {
        sf::F64 { v: a }
    }

    #[test]
    fn binops_match_softfloat() {
        let mut g = Gen(0x1234_5678_9abc_def1);
        for _ in 0..N {
            for w in [Width::S, Width::D] {
                let (a, b, rm) = (g.float(w), g.float(w), g.rm());
                for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div] {
                    let want = soft_binop(op, w, a, b, rm);
                    assert_eq!(binop(op, w, a, b, rm), want, "{op:?} {w:?} {a:#x} {b:#x} {rm:?}");
                }
            }
        }
    }

    #[test]
    fn sqrt_fma_and_conversions_match_softfloat() {
        let mut g = Gen(0x0bad_cafe_dead_beef);
        for _ in 0..N {
            for w in [Width::S, Width::D] {
                let (a, b, c, rm) = (g.float(w), g.float(w), g.float(w), g.rm());
                // SAFETY (all): foreign but pure.
                let want = sf::with(rm.softfloat(), || unsafe {
                    match w {
                        Width::S => sf::f32_sqrt(soft32(a)).v as u64,
                        Width::D => sf::f64_sqrt(soft64(a)).v,
                    }
                });
                assert_eq!(sqrt(w, a, rm), want, "sqrt {w:?} {a:#x} {rm:?}");

                for op in [FmaOp::Madd, FmaOp::Msub, FmaOp::Nmsub, FmaOp::Nmadd] {
                    let got = fma(op, w, a, b, c, rm);
                    // Reference: SoftFloat with RISC-V's sign conventions applied to inputs.
                    let neg = |v: u64| v ^ if w == Width::S { 1 << 31 } else { 1 << 63 };
                    let (a2, c2) = match op {
                        FmaOp::Madd => (a, c),
                        FmaOp::Msub => (a, neg(c)),
                        FmaOp::Nmsub => (neg(a), c),
                        FmaOp::Nmadd => (neg(a), neg(c)),
                    };
                    let want = sf::with(rm.softfloat(), || unsafe {
                        match w {
                            Width::S => sf::f32_mulAdd(soft32(a2), soft32(b), soft32(c2)).v as u64,
                            Width::D => sf::f64_mulAdd(soft64(a2), soft64(b), soft64(c2)).v,
                        }
                    });
                    assert_eq!(got, want, "{op:?} {w:?} {a:#x} {b:#x} {c:#x} {rm:?}");
                }

                let to = if w == Width::S { Width::D } else { Width::S };
                let want = sf::with(rm.softfloat(), || unsafe {
                    match w {
                        Width::S => sf::f32_to_f64(soft32(a)).v,
                        Width::D => sf::f64_to_f32(soft64(a)).v as u64,
                    }
                });
                assert_eq!(convert(to, a, rm), want, "fcvt {w:?}->{to:?} {a:#x} {rm:?}");

                for t in [IntType::W, IntType::Wu, IntType::L, IntType::Lu] {
                    let r = rm.softfloat() as u8;
                    let want = sf::with(rm.softfloat(), || unsafe {
                        match (w, t) {
                            (Width::S, IntType::W) => {
                                sf::f32_to_i32(soft32(a), r, true) as i64 as u64
                            }
                            (Width::S, IntType::Wu) => {
                                sf::f32_to_ui32(soft32(a), r, true) as i32 as i64 as u64
                            }
                            (Width::S, IntType::L) => sf::f32_to_i64(soft32(a), r, true) as u64,
                            (Width::S, IntType::Lu) => sf::f32_to_ui64(soft32(a), r, true),
                            (Width::D, IntType::W) => {
                                sf::f64_to_i32(soft64(a), r, true) as i64 as u64
                            }
                            (Width::D, IntType::Wu) => {
                                sf::f64_to_ui32(soft64(a), r, true) as i32 as i64 as u64
                            }
                            (Width::D, IntType::L) => sf::f64_to_i64(soft64(a), r, true) as u64,
                            (Width::D, IntType::Lu) => sf::f64_to_ui64(soft64(a), r, true),
                        }
                    });
                    assert_eq!(to_int(w, t, a, rm), want, "fcvt.{t:?}.{w:?} {a:#x} {rm:?}");

                    let x = match g.next() % 4 {
                        0 => g.next() & 0xffff,
                        1 => g.next() | 1 << 63,
                        _ => g.next(),
                    };
                    let want = sf::with(rm.softfloat(), || unsafe {
                        match (w, t) {
                            (Width::S, IntType::W) => sf::i32_to_f32(x as i32).v as u64,
                            (Width::S, IntType::Wu) => sf::ui32_to_f32(x as u32).v as u64,
                            (Width::S, IntType::L) => sf::i64_to_f32(x as i64).v as u64,
                            (Width::S, IntType::Lu) => sf::ui64_to_f32(x).v as u64,
                            (Width::D, IntType::W) => sf::i32_to_f64(x as i32).v,
                            (Width::D, IntType::Wu) => sf::ui32_to_f64(x as u32).v,
                            (Width::D, IntType::L) => sf::i64_to_f64(x as i64).v,
                            (Width::D, IntType::Lu) => sf::ui64_to_f64(x).v,
                        }
                    });
                    assert_eq!(from_int(w, t, x, rm), want, "fcvt.{w:?}.{t:?} {x:#x} {rm:?}");
                }
            }
        }
    }

    #[test]
    fn compares_and_min_max_match_the_spec() {
        let mut g = Gen(0x5eed_5eed_5eed_5eed);
        for _ in 0..N {
            for w in [Width::S, Width::D] {
                let (a, b) = (g.float(w), g.float(w));
                // SAFETY (all): foreign but pure.
                let soft = |f: &dyn Fn() -> bool| sf::with(sf::Rounding::NearEven, f);
                let (eq, lt, le) = match w {
                    Width::S => (
                        soft(&|| unsafe { sf::f32_eq(soft32(a), soft32(b)) }),
                        soft(&|| unsafe { sf::f32_lt(soft32(a), soft32(b)) }),
                        soft(&|| unsafe { sf::f32_le(soft32(a), soft32(b)) }),
                    ),
                    Width::D => (
                        soft(&|| unsafe { sf::f64_eq(soft64(a), soft64(b)) }),
                        soft(&|| unsafe { sf::f64_lt(soft64(a), soft64(b)) }),
                        soft(&|| unsafe { sf::f64_le(soft64(a), soft64(b)) }),
                    ),
                };
                assert_eq!(compare(CmpOp::Eq, w, a, b), eq, "feq {w:?} {a:#x} {b:#x}");
                assert_eq!(compare(CmpOp::Lt, w, a, b), lt, "flt {w:?} {a:#x} {b:#x}");
                assert_eq!(compare(CmpOp::Le, w, a, b), le, "fle {w:?} {a:#x} {b:#x}");

                // fmin/fmax reference, straight from the RISC-V spec (as Spike implements it).
                let canonical = if w == Width::S { 0x7fc0_0000 } else { 0x7ff8_0000_0000_0000 };
                let nv = if is_snan(w, a) || is_snan(w, b) { NV } else { 0 };
                let lt_quiet = soft(&|| unsafe {
                    match w {
                        Width::S => sf::f32_lt_quiet(soft32(a), soft32(b)),
                        Width::D => sf::f64_lt_quiet(soft64(a), soft64(b)),
                    }
                })
                .0;
                let sign = |v: u64| v >> if w == Width::S { 31 } else { 63 } & 1;
                let both_zero = |v: u64| v << 1 == 0 || (w == Width::S && (v as u32) << 1 == 0);
                let (min, max) = match (is_nan(w, a), is_nan(w, b)) {
                    (true, true) => (canonical, canonical),
                    (true, false) => (b, b),
                    (false, true) => (a, a),
                    _ if both_zero(a) && both_zero(b) => {
                        if sign(a) == 1 {
                            (a, b)
                        } else {
                            (b, a)
                        }
                    }
                    _ => {
                        if lt_quiet {
                            (a, b)
                        } else {
                            (b, a)
                        }
                    }
                };
                assert_eq!(min_max(false, w, a, b), (min, nv), "fmin {w:?} {a:#x} {b:#x}");
                assert_eq!(min_max(true, w, a, b), (max, nv), "fmax {w:?} {a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn tininess_is_detected_after_rounding() {
        // The exact product (2^25 - 1) * 2^-151 lies just below the smallest normal 2^-126
        // and rounds up to it. RISC-V detects tininess after rounding, so this is inexact
        // but not an underflow; ARM detects it before rounding and raises UF.
        // 2^25 - 1 = 18631 * 1801.
        let a = (18631.0f32 * 2f32.powi(-75)).to_bits() as u64;
        let b = (1801.0f32 * 2f32.powi(-76)).to_bits() as u64;
        assert_eq!(binop(BinOp::Mul, Width::S, a, b, Rm::Rne), (0x0080_0000, NX));
        assert_eq!(soft_binop(BinOp::Mul, Width::S, a, b, Rm::Rne), (0x0080_0000, NX));
        // Rounding toward zero keeps it below the smallest normal: a real underflow.
        assert_eq!(binop(BinOp::Mul, Width::S, a, b, Rm::Rtz), (0x007f_ffff, UF | NX));
    }

    #[test]
    fn half_conversions_match_softfloat() {
        let mut g = Gen(0x0f16_0f16_0f16_0f16);
        for _ in 0..N {
            let rm = g.rm();
            let a = g.float(Width::S) as u32;
            let (want, flags) =
                sf::with(rm.softfloat(), || unsafe { sf::f32_to_f16(sf::F32 { v: a }) });
            assert_eq!(f32_to_f16(a, rm), (want.v, flags), "fcvt.h.s {a:#x} {rm:?}");
            // Half -> single is exact, so the mode only shows up in the NaN cases.
            let h = g.next() as u16;
            let (want, flags) =
                sf::with(sf::Rounding::NearEven, || unsafe { sf::f16_to_f32(sf::F16 { v: h }) });
            assert_eq!(f16_to_f32(h), (want.v, flags), "fcvt.s.h {h:#x}");
        }
    }

    #[test]
    fn classify_all_classes() {
        let cases: [(Width, u64, u32); 10] = [
            (Width::D, (-f64::INFINITY).to_bits(), 0),
            (Width::D, (-1.5f64).to_bits(), 1),
            (Width::D, 0x8000_0000_0000_0001, 2),
            (Width::D, (-0.0f64).to_bits(), 3),
            (Width::S, 0, 4),
            (Width::S, 1, 5),
            (Width::S, 1.0f32.to_bits() as u64, 6),
            (Width::S, f32::INFINITY.to_bits() as u64, 7),
            (Width::S, 0x7f80_0001, 8),
            (Width::S, 0x7fc0_0000, 9),
        ];
        for (w, v, bit) in cases {
            assert_eq!(classify(w, v), 1 << bit, "{w:?} {v:#x}");
        }
    }
}
