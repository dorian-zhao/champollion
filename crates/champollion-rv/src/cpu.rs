//! Architectural state of one RISC-V hart.

/// `reserved_addr` when no reservation is held. Never a valid atomic address: atomics are
/// naturally aligned.
pub const NO_RESERVATION: u64 = u64::MAX;

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
    /// Address loaded by the last `lr` ([`NO_RESERVATION`] if none), cleared by any `sc`.
    /// `sc` succeeds only if memory still holds `reserved_value` (checked with a host
    /// compare-and-swap), so stores by other threads in between make it fail, as on real
    /// hardware (except for ABA). Plain fields so generated code can reach them.
    pub reserved_addr: u64,
    pub reserved_value: u64,
    /// Instructions retired (the `instret` CSR; also what `cycle` reports).
    pub instret: u64,
    /// Vector registers `v0..v31`. VLEN is 128 bits; only the low `vl` elements (of the
    /// byte width `vtype` selects) are ever read or written. 16-byte aligned so the JIT
    /// can reach a whole register with one load/store immediate offset.
    pub v: [[u8; 16]; 32],
    /// `vl`, from the last `vset*`.
    pub vl: u64,
    /// `vtype` as written: `vma` bit 7, `vta` bit 6, `vsew` bits 5:3, `vlmul` bits 2:0.
    /// `vstart` is not modeled and always reads zero.
    pub vtype: u64,
    /// `vtype | vl << 8`: one word for the JIT's entry guard, which otherwise compares
    /// both fields separately every time it enters a vector block.
    pub vkey: u64,
    /// `fcsr`: `frm` in bits 7:5, `fflags` in bits 4:0. Kept last so `v`'s offset stays
    /// a multiple of 16 before it.
    pub fcsr: u32,
}

/// The JIT reaches whole vector registers with `LDR/STR Q` immediate offsets, which must
/// be multiples of 16: that only works while `v` itself stays 16-byte aligned, so keep
/// any new trailing field behind these too.
const _: () = assert!(std::mem::offset_of!(CpuState, v) % 16 == 0);

impl CpuState {
    pub fn new(pc: u64) -> Self {
        Self {
            x: [0; 32],
            f: [0; 32],
            pc,
            reserved_addr: NO_RESERVATION,
            reserved_value: 0,
            instret: 0,
            v: [[0; 16]; 32],
            vl: 0,
            vtype: 0,
            vkey: 0,
            fcsr: 0,
        }
    }

    /// Set the vector configuration and its combined guard key together.
    #[inline]
    pub fn set_vstate(&mut self, vtype: u64, vl: u64) {
        self.vtype = vtype;
        self.vl = vl;
        self.vkey = crate::vector::vkey(vtype, vl);
    }

    /// Change `vl` alone (the fault-only-first forms do this), keeping the key exact.
    #[inline]
    pub fn set_vl(&mut self, vl: u64) {
        self.vl = vl;
        self.vkey = crate::vector::vkey(self.vtype, vl);
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
