//! Berkeley SoftFloat 3 with RISC-V specialization (canonical NaNs, tininess detected after
//! rounding): the reference semantics for RISC-V F and D. Champollion runs floating point on
//! the host FPU; this crate is the oracle that implementation is fuzzed against, and the
//! fallback for the one rounding mode ARM lacks (round to nearest, ties to max magnitude).

/// SoftFloat rounding modes, numbered like RISC-V `rm` for the five standard modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Rounding {
    NearEven = 0,
    MinMag = 1,
    Min = 2,
    Max = 3,
    NearMaxMag = 4,
}

/// Exception flags, bit-for-bit the RISC-V `fflags` layout: NX=1, UF=2, OF=4, DZ=8, NV=16.
pub type Flags = u8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct F32 {
    pub v: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct F64 {
    pub v: u64,
}

unsafe extern "C" {
    fn champollion_sf_begin(rounding_mode: u8);
    fn champollion_sf_flags() -> u8;

    pub fn f32_add(a: F32, b: F32) -> F32;
    pub fn f32_sub(a: F32, b: F32) -> F32;
    pub fn f32_mul(a: F32, b: F32) -> F32;
    pub fn f32_div(a: F32, b: F32) -> F32;
    pub fn f32_sqrt(a: F32) -> F32;
    pub fn f32_mulAdd(a: F32, b: F32, c: F32) -> F32;
    pub fn f32_eq(a: F32, b: F32) -> bool;
    pub fn f32_lt(a: F32, b: F32) -> bool;
    pub fn f32_le(a: F32, b: F32) -> bool;
    pub fn f32_lt_quiet(a: F32, b: F32) -> bool;
    pub fn f32_isSignalingNaN(a: F32) -> bool;
    pub fn f32_to_f64(a: F32) -> F64;
    pub fn f32_to_i32(a: F32, rm: u8, exact: bool) -> i32;
    pub fn f32_to_ui32(a: F32, rm: u8, exact: bool) -> u32;
    pub fn f32_to_i64(a: F32, rm: u8, exact: bool) -> i64;
    pub fn f32_to_ui64(a: F32, rm: u8, exact: bool) -> u64;
    pub fn i32_to_f32(a: i32) -> F32;
    pub fn ui32_to_f32(a: u32) -> F32;
    pub fn i64_to_f32(a: i64) -> F32;
    pub fn ui64_to_f32(a: u64) -> F32;

    pub fn f64_add(a: F64, b: F64) -> F64;
    pub fn f64_sub(a: F64, b: F64) -> F64;
    pub fn f64_mul(a: F64, b: F64) -> F64;
    pub fn f64_div(a: F64, b: F64) -> F64;
    pub fn f64_sqrt(a: F64) -> F64;
    pub fn f64_mulAdd(a: F64, b: F64, c: F64) -> F64;
    pub fn f64_eq(a: F64, b: F64) -> bool;
    pub fn f64_lt(a: F64, b: F64) -> bool;
    pub fn f64_le(a: F64, b: F64) -> bool;
    pub fn f64_lt_quiet(a: F64, b: F64) -> bool;
    pub fn f64_isSignalingNaN(a: F64) -> bool;
    pub fn f64_to_f32(a: F64) -> F32;
    pub fn f64_to_i32(a: F64, rm: u8, exact: bool) -> i32;
    pub fn f64_to_ui32(a: F64, rm: u8, exact: bool) -> u32;
    pub fn f64_to_i64(a: F64, rm: u8, exact: bool) -> i64;
    pub fn f64_to_ui64(a: F64, rm: u8, exact: bool) -> u64;
    pub fn i32_to_f64(a: i32) -> F64;
    pub fn ui32_to_f64(a: u32) -> F64;
    pub fn i64_to_f64(a: i64) -> F64;
    pub fn ui64_to_f64(a: u64) -> F64;
}

/// Run `f` with SoftFloat's (thread-local) rounding mode set to `rm` and flags cleared, and
/// return its result with the exception flags it raised.
///
/// The extern functions are safe to call with any bit patterns; they are `unsafe` only
/// because they are foreign.
pub fn with<R>(rm: Rounding, f: impl FnOnce() -> R) -> (R, Flags) {
    // SAFETY: plain C functions that only touch this thread's SoftFloat state.
    unsafe { champollion_sf_begin(rm as u8) };
    let r = f();
    // SAFETY: as above.
    (r, unsafe { champollion_sf_flags() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riscv_semantics() {
        // 2^53 + 1 is a tie in binary64: nearest-even rounds down, ties-away rounds up.
        let a = F64 { v: (9007199254740992.0f64).to_bits() };
        let b = F64 { v: 1.0f64.to_bits() };
        // SAFETY: foreign but pure.
        let (r, nx) = with(Rounding::NearMaxMag, || unsafe { f64_add(a, b) });
        assert_eq!(f64::from_bits(r.v), 9007199254740994.0);
        assert_eq!(nx, 1, "inexact");
        let (r, _) = with(Rounding::NearEven, || unsafe { f64_add(a, b) });
        assert_eq!(f64::from_bits(r.v), 9007199254740992.0);

        // NaN results are canonical, and a signaling NaN input raises invalid.
        let (r, flags) =
            with(Rounding::NearEven, || unsafe { f32_add(F32 { v: 0x7f80_0001 }, F32 { v: 0 }) });
        assert_eq!((r.v, flags), (0x7fc0_0000, 16));
        // NaN converts to the largest integer.
        let (i, flags) =
            with(Rounding::MinMag, || unsafe { f64_to_i32(F64 { v: u64::MAX }, 1, true) });
        assert_eq!((i, flags), (i32::MAX, 16));
    }
}
