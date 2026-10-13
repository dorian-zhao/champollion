//! Architectural state of one RISC-V hart.

/// Register file and control state for a single RV64GC hart.
///
/// `#[repr(C)]` so the JIT and the arm64 trampolines can address fields by fixed offsets.
#[repr(C)]
#[derive(Clone, PartialEq, Eq)]
pub struct CpuState {
    /// Integer registers `x0..x31`. `x[0]` must always read as zero; writers go through
    /// [`CpuState::set_x`] or reset it after the fact.
    pub x: [u64; 32],
    /// Floating-point registers `f0..f31` as raw bits. Single-precision values are
    /// NaN-boxed (upper 32 bits all ones), as required by the D extension.
    pub f: [u64; 32],
    pub pc: u64,
    /// `fcsr`: `frm` in bits 7:5, `fflags` in bits 4:0.
    pub fcsr: u32,
    /// `(address, value)` loaded by the last `lr`, cleared by any `sc`. `sc` succeeds only
    /// if memory still holds `value` (checked with a host compare-and-swap), so stores by
    /// other threads in between make it fail, as on real hardware (except for ABA).
    pub reservation: Option<(u64, u64)>,
}

impl CpuState {
    pub fn new(pc: u64) -> Self {
        Self { x: [0; 32], f: [0; 32], pc, fcsr: 0, reservation: None }
    }

    #[inline]
    pub fn x(&self, r: u8) -> u64 {
        self.x[r as usize]
    }

    #[inline]
    pub fn set_x(&mut self, r: u8, v: u64) {
        if r != 0 {
            self.x[r as usize] = v;
        }
    }
}

impl Default for CpuState {
    fn default() -> Self {
        Self::new(0)
    }
}

/// ABI names, indexed by register number.
pub const ABI_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

impl std::fmt::Debug for CpuState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "pc   = {:#018x}  fcsr = {:#x}", self.pc, self.fcsr)?;
        for (i, chunk) in self.x.chunks(4).enumerate() {
            for (j, v) in chunk.iter().enumerate() {
                write!(f, "{:>4} = {:#018x}  ", ABI_NAMES[i * 4 + j], v)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
