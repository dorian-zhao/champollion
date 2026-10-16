//! RISC-V instruction decoder.
//!
//! Covers RV64I, M, A, C (see [`compressed`]), Zicsr (for the CSRs user programs can
//! reach), Zifencei, and the data-movement subset of F and D (loads, stores, `fmv`,
//! `fsgnj`), and all of F and D (arithmetic runs in [`crate::fpu`]).
//!
//! Immediates are stored already sign-extended (`lui`/`auipc` already shifted left by 12),
//! and branch/jump offsets are byte offsets relative to the instruction's own pc. Every
//! immediate fits in an `i32`, which keeps [`Inst`] at 8 bytes.

/// A decoded instruction.
///
/// - `OpImm`: for shifts, `imm` is the shift amount.
/// - `OpImm32` / `Op32` (`*w` ops): operate on the low 32 bits, sign-extend the result.
/// - `Fence`: `pred`/`succ` are IORW bit sets; `fm` is 0b1000 for `fence.tso`.
/// - `FenceI` (Zifencei): make this hart's earlier stores visible to its instruction fetches.
/// - `Csr`: `src` is `rs1`, or the 5-bit immediate if `imm`. Only CSRs in [`Csr`] decode, and
///   writes to read-only ones are rejected here, so execution never sees an illegal access.
/// - `Lr` / `Sc` / `Amo`: `aq`/`rl` are the acquire/release ordering bits. `Sc` writes 0 to
///   `rd` on success, nonzero on failure; `Amo` writes the old memory value to `rd`.
/// - F/D instructions: register numbers name `f` registers, except the base address of
///   loads/stores, the `x` destination of `FMvToInt`/`FCmp`/`FClass`/`FCvtToInt`, and the
///   `x` source of `FMvFromInt`/`FCvtFromInt`. `rm` is the raw rounding-mode field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
    Lui {
        rd: u8,
        imm: i32,
    },
    Auipc {
        rd: u8,
        imm: i32,
    },
    Jal {
        rd: u8,
        offset: i32,
    },
    Jalr {
        rd: u8,
        rs1: u8,
        offset: i32,
    },
    Branch {
        cond: BranchCond,
        rs1: u8,
        rs2: u8,
        offset: i32,
    },
    Load {
        width: LoadWidth,
        rd: u8,
        rs1: u8,
        offset: i32,
    },
    Store {
        width: StoreWidth,
        rs1: u8,
        rs2: u8,
        offset: i32,
    },
    OpImm {
        op: AluOp,
        rd: u8,
        rs1: u8,
        imm: i32,
    },
    Op {
        op: AluOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    OpImm32 {
        op: AluOp32,
        rd: u8,
        rs1: u8,
        imm: i32,
    },
    Op32 {
        op: AluOp32,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fence {
        fm: u8,
        pred: u8,
        succ: u8,
    },
    FenceI,
    Csr {
        op: CsrOp,
        csr: Csr,
        rd: u8,
        src: u8,
        imm: bool,
    },
    Ecall,
    Ebreak,
    Lr {
        width: AmoWidth,
        rd: u8,
        rs1: u8,
        aq: bool,
        rl: bool,
    },
    Sc {
        width: AmoWidth,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    },
    Amo {
        op: AmoOp,
        width: AmoWidth,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    },
    FLoad {
        width: FWidth,
        rd: u8,
        rs1: u8,
        offset: i32,
    },
    FStore {
        width: FWidth,
        rs1: u8,
        rs2: u8,
        offset: i32,
    },
    FMvToInt {
        width: FWidth,
        rd: u8,
        rs1: u8,
    },
    /// Zfa `fli.s`/`fli.d`: load a floating-point constant by index (0–31).
    Fli {
        width: FWidth,
        rd: u8,
        imm: u8,
    },
    FMvFromInt {
        width: FWidth,
        rd: u8,
        rs1: u8,
    },
    FSgnj {
        kind: SgnjKind,
        width: FWidth,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    FArith {
        op: FArithOp,
        width: FWidth,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rm: u8,
    },
    FSqrt {
        width: FWidth,
        rd: u8,
        rs1: u8,
        rm: u8,
    },
    FFma {
        op: FmaOp,
        width: FWidth,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: u8,
        rm: u8,
    },
    FMinMax {
        max: bool,
        /// Zfa `fminm`/`fmaxm`: a NaN input makes the result the canonical NaN.
        canonical_nan: bool,
        width: FWidth,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    FCmp {
        op: FCmpOp,
        width: FWidth,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    FClass {
        width: FWidth,
        rd: u8,
        rs1: u8,
    },
    FCvtToInt {
        from: FWidth,
        to: IntType,
        rd: u8,
        rs1: u8,
        rm: u8,
    },
    FCvtFromInt {
        from: IntType,
        to: FWidth,
        rd: u8,
        rs1: u8,
        rm: u8,
    },
    FCvtFloat {
        to: FWidth,
        rd: u8,
        rs1: u8,
        rm: u8,
    },
    /// Zfa `fround.s`/`fround.d` (and `froundnx...` when `exact`): fcvt.s.d/fcvt.d.s
    /// with rs2 = 4/5, rounding to an integral value in the same format.
    FRound {
        width: FWidth,
        rd: u8,
        rs1: u8,
        rm: u8,
        exact: bool,
    },
    /// Zfa `fcvtmod.w.d`: the low 32 bits of the RTZ-rounded value, sign-extended.
    FCvtMod {
        rd: u8,
        rs1: u8,
    },
    /// `vsetvli`/`vsetivli`/`vsetvl`: sets `vl` and `vtype`. `rs2` carries the 10-bit
    /// vtype value for the immediate forms and the vtype register number for `vsetvl`;
    /// the avl is `x[rs1]` except for `vsetivli`, where `rs1` is the 5-bit immediate.
    VSetvl {
        rd: u8,
        rs1: u8,
        rs2: u16,
        kind: VSetKind,
    },
    VLoad {
        width: VWidth,
        kind: VMemKind,
        vd: u8,
        rs1: u8,
        rs2: u8,
        mask: bool,
        /// Number of registers for whole-register ops (`vl1r` = 0, `vl2r` = 1 extra);
        /// zero for every other kind.
        nf: u8,
    },
    VStore {
        width: VWidth,
        kind: VStoreKind,
        vs3: u8,
        rs1: u8,
        rs2: u8,
        mask: bool,
        nf: u8,
    },
    VArith {
        op: VArithOp,
        vd: u8,
        vs2: u8,
        src: VSel,
        mask: bool,
    },
    VCmp {
        op: VCmpOp,
        vd: u8,
        vs2: u8,
        src: VSel,
        mask: bool,
    },
    VMask {
        op: VMaskOp,
        vd: u8,
        vs1: u8,
        vs2: u8,
    },
    Vmv {
        vd: u8,
        src: VSel,
    },
    VmvNr {
        vd: u8,
        vs2: u8,
        /// Registers moved: `nf` extra (vmv1r = 0 .. vmv8r = 7).
        nf: u8,
    },
    Vmerge {
        vd: u8,
        vs1: u8,
        vs2: u8,
    },
    VRed {
        op: VRedOp,
        vd: u8,
        vs1: u8,
        vs2: u8,
    },
    VmvXs {
        rd: u8,
        vs2: u8,
    },
    VmvSx {
        vd: u8,
        rs1: u8,
    },
    Vcpop {
        rd: u8,
        vs2: u8,
    },
    Vfirst {
        rd: u8,
        vs2: u8,
    },
    Vid {
        vd: u8,
    },
    /// Vector sign/zero-extend (`vsext.vfN`/`vzext.vfN`, `narrow` = the source is SEW/2,
    /// /4, or /8 wide).
    VExt {
        sext: bool,
        narrow: u8,
        vd: u8,
        vs2: u8,
    },
    /// Vector multiply-accumulate (`vmacc` = vd += src × vs2) and multiply-add
    /// (`vmadd` = vd = src × vd + vs2).
    VMacc {
        add: bool,
        vd: u8,
        vs2: u8,
        src: VSel,
    },
    Vnsrl {
        vd: u8,
        vs2: u8,
        imm: u8,
    },
}

/// Vector element width (`vsew`) for the loads, stores, and arithmetic ops in the subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VWidth {
    E8,
    E16,
    E32,
    E64,
}

impl VWidth {
    /// Element size in bytes.
    #[inline]
    pub fn bytes(self) -> u64 {
        match self {
            VWidth::E8 => 1,
            VWidth::E16 => 2,
            VWidth::E32 => 4,
            VWidth::E64 => 8,
        }
    }
}

/// The three `vset*` forms, which differ in where the avl and vtype come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VSetKind {
    /// `vsetvli rd, rs1, vtype`: avl in `x[rs1]` (`x0` means "the largest available").
    RegisterAvl,
    /// `vsetivli rd, avl, vtype`: the avl is the 5-bit immediate.
    ImmediateAvl,
    /// `vsetvl rd, rs1, rs2`: avl in `x[rs1]`, vtype in `x[rs2]`.
    RegisterVtype,
}

/// Vector memory access kind (`mop`/`lumop`/`sumop` fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VMemKind {
    /// Unit-stride `vle<n>.v`.
    Unit,
    /// `vle<n>ff.v`: stops at the first fault with a shortened `vl`.
    FaultOnlyFirst,
    /// Strided `vlse<n>.v`, element i at `rs1 + i * rs2`.
    Stride,
    /// Whole-register `vl1re<n>.v`: VLEN bits, no `vl` involved.
    WholeRegister,
    /// `vlm.v`: one mask register, no `vl` involved.
    Mask,
}

/// Vector store kind (unit-stride, strided, or whole-register).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VStoreKind {
    Unit,
    Stride,
    WholeRegister,
}

/// The vector integer arithmetic in the supported subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VArithOp {
    Add,
    /// `vrsub`: vs2 is subtracted from the operand's source.
    Rsub,
    And,
    Or,
    Xor,
    Andn,
    Sll,
    Mul,
    SlideUp,
    SlideDown,
    Slide1Down,
    Gather,
}

/// Second source of a vector op: another vector, a scalar register, or an immediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VSel {
    V(u8),
    X(u8),
    I(u8),
}

/// Mask-producing comparisons (`vmseq`/`vmsne`/`vmsleu`/`vmsgtu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VCmpOp {
    Eq,
    Ne,
    LeU,
    GtU,
}

/// Mask register logic (`vmor.mm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VMaskOp {
    Or,
}

/// Mask-form reductions (`vredand.vs`, `vredor.vs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VRedOp {
    And,
    Or,
    /// `vredsum.vs`: the elements summed modulo the element width.
    Sum,
}

/// `rm` value meaning "use `frm`". Instructions store `rm` raw (0–4 or 7); reserved values
/// are rejected at decode, and `dyn` is resolved at execution.
pub const RM_DYN: u8 = 0b111;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Fused multiply-add: `Madd` = a·b + c, `Msub` = a·b − c, `Nmsub` = −(a·b) + c,
/// `Nmadd` = −(a·b) − c.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmaOp {
    Madd,
    Msub,
    Nmsub,
    Nmadd,
}

/// `feq` is a quiet comparison; `flt` and `fle` are signaling. Zfa's `fleq`/`fltq`
/// compare like `fle`/`flt` but stay quiet for quiet NaN inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FCmpOp {
    Eq,
    Lt,
    Le,
    Ltq,
    Leq,
}

/// Integer side of `fcvt`: 32-bit (`w`/`wu`) or 64-bit (`l`/`lu`), signed or unsigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntType {
    W,
    Wu,
    L,
    Lu,
}

/// Floating-point operand width: single (F) or double (D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FWidth {
    S,
    D,
}

/// Sign injection: take the sign of `rs2` (`Sgnj`), its inverse (`Sgnjn`), or the XOR of
/// both signs (`Sgnjx`). `fmv`, `fneg`, and `fabs` are aliases of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgnjKind {
    Sgnj,
    Sgnjn,
    Sgnjx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchCond {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadWidth {
    B,
    H,
    W,
    D,
    Bu,
    Hu,
    Wu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreWidth {
    B,
    H,
    W,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
    // M extension; register-register only.
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
    // Zba (address generation), Zbb (basic bit manipulation), Zbs (single-bit), and
    // Zicond (integer conditional operations, V.1.0.2). The unary operations (`Clz`,
    // `Ctz`, `Cpop`, `SextB`, `SextH`, `ZextH`, `Rev8`, `Brev8`, `OrcB`) ignore their
    // second operand, which their encodings fix to a constant. Zbkb adds `Pack`,
    // `Packh`, and `Packw`, Zbc the carry-less multiplies `Clmul`, `Clmulh`, and
    // `Clmulr`, and Zbkx the lookups `Xperm4` and `Xperm8`.
    Sh1Add,
    Sh2Add,
    Sh3Add,
    AddUw,
    Sh1AddUw,
    Sh2AddUw,
    Sh3AddUw,
    SllUw,
    Andn,
    Orn,
    Xnor,
    Max,
    Maxu,
    Min,
    Minu,
    Rol,
    Ror,
    Bclr,
    Bext,
    Binv,
    Bset,
    CzeroEqz,
    CzeroNez,
    Clz,
    Ctz,
    Cpop,
    SextB,
    SextH,
    ZextH,
    Rev8,
    Brev8,
    OrcB,
    Pack,
    Packh,
    Packw,
    Clmul,
    Clmulh,
    Clmulr,
    Xperm4,
    Xperm8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp32 {
    Add,
    Sub,
    Sll,
    Srl,
    Sra,
    // M extension; register-register only.
    Mul,
    Div,
    Divu,
    Rem,
    Remu,
    // Zbb word rotations.
    Rol,
    Ror,
    // Zbb word counts (`clzw`/`ctzw`/`cpopw`).
    Clz,
    Ctz,
    Cpop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmoWidth {
    W,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmoOp {
    Swap,
    Add,
    Xor,
    And,
    Or,
    Min,
    Max,
    Minu,
    Maxu,
}

/// `csrrw` (swap), `csrrs` (set bits), `csrrc` (clear bits), and their immediate forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsrOp {
    Rw,
    Rs,
    Rc,
}

/// The CSRs reachable from user mode that we implement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Csr {
    Fflags,
    Frm,
    Fcsr,
    Cycle,
    Time,
    Instret,
    /// `vl`: the active element count (read-only to the guest for our subset).
    Vl,
    /// `vtype`: the last `vset*`'s raw configuration (read-only here).
    Vtype,
    /// `vlenb`: VLEN in bytes, a constant 16.
    Vlenb,
}

impl Csr {
    fn from_number(n: u32) -> Option<Csr> {
        Some(match n {
            0x001 => Csr::Fflags,
            0x002 => Csr::Frm,
            0x003 => Csr::Fcsr,
            0xc00 => Csr::Cycle,
            0xc01 => Csr::Time,
            0xc02 => Csr::Instret,
            0xc20 => Csr::Vl,
            0xc21 => Csr::Vtype,
            0xc22 => Csr::Vlenb,
            _ => return None,
        })
    }

    /// CSR numbers with bits 11:10 = 0b11 are read-only.
    fn read_only(self) -> bool {
        matches!(self, Csr::Cycle | Csr::Time | Csr::Instret | Csr::Vl | Csr::Vtype | Csr::Vlenb)
    }
}

mod compressed;
mod vector;

pub use compressed::decode_compressed;

/// The raw bits of an instruction that is illegal or not supported yet (16 bits for a
/// compressed instruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Illegal(u32),
}

const LOAD: u32 = 0b000_0011;
const LOAD_FP: u32 = 0b000_0111;
const STORE_FP: u32 = 0b010_0111;
const OP_V: u32 = 0b101_0111;
const OP_FP: u32 = 0b101_0011;
const MADD: u32 = 0b100_0011;
const MSUB: u32 = 0b100_0111;
const NMSUB: u32 = 0b100_1011;
const NMADD: u32 = 0b100_1111;
const MISC_MEM: u32 = 0b000_1111;
const OP_IMM: u32 = 0b001_0011;
const AUIPC: u32 = 0b001_0111;
const OP_IMM_32: u32 = 0b001_1011;
const STORE: u32 = 0b010_0011;
const AMO: u32 = 0b010_1111;
const OP: u32 = 0b011_0011;
const LUI: u32 = 0b011_0111;
const OP_32: u32 = 0b011_1011;
const BRANCH: u32 = 0b110_0011;
const JALR: u32 = 0b110_0111;
const JAL: u32 = 0b110_1111;
const SYSTEM: u32 = 0b111_0011;

const ECALL: u32 = 0x0000_0073;
const EBREAK: u32 = 0x0010_0073;

#[inline]
fn rd(w: u32) -> u8 {
    ((w >> 7) & 0x1f) as u8
}
#[inline]
fn rs1(w: u32) -> u8 {
    ((w >> 15) & 0x1f) as u8
}
#[inline]
fn rs2(w: u32) -> u8 {
    ((w >> 20) & 0x1f) as u8
}
#[inline]
fn funct3(w: u32) -> u32 {
    (w >> 12) & 0x7
}
#[inline]
fn funct7(w: u32) -> u32 {
    w >> 25
}

/// Sign-extend the low `bits` bits of `v`.
#[inline]
fn sext(v: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((v << shift) as i32) >> shift
}

fn imm_i(w: u32) -> i32 {
    (w as i32) >> 20
}

fn imm_s(w: u32) -> i32 {
    ((w as i32) >> 25) << 5 | ((w >> 7) & 0x1f) as i32
}

fn imm_b(w: u32) -> i32 {
    let v = ((w >> 31) & 1) << 12
        | ((w >> 7) & 1) << 11
        | ((w >> 25) & 0x3f) << 5
        | ((w >> 8) & 0xf) << 1;
    sext(v, 13)
}

fn imm_u(w: u32) -> i32 {
    (w & 0xffff_f000) as i32
}

fn imm_j(w: u32) -> i32 {
    let v = ((w >> 31) & 1) << 20
        | ((w >> 12) & 0xff) << 12
        | ((w >> 20) & 1) << 11
        | ((w >> 21) & 0x3ff) << 1;
    sext(v, 21)
}

/// Length in bytes of the instruction whose low 16 bits are `lo`.
pub fn inst_len(lo: u16) -> u64 {
    if lo & 0b11 == 0b11 { 4 } else { 2 }
}

/// Decode one instruction. If the low two bits are not `0b11`, only the low 16 bits are
/// used and decoded as a compressed instruction.
pub fn decode(w: u32) -> Result<Inst, DecodeError> {
    if w & 0b11 != 0b11 {
        return decode_compressed(w as u16);
    }
    let illegal = Err(DecodeError::Illegal(w));
    let (rd, rs1, rs2, f3, f7) = (rd(w), rs1(w), rs2(w), funct3(w), funct7(w));
    let inst = match w & 0x7f {
        LUI => Inst::Lui { rd, imm: imm_u(w) },
        AUIPC => Inst::Auipc { rd, imm: imm_u(w) },
        JAL => Inst::Jal { rd, offset: imm_j(w) },
        JALR if f3 == 0 => Inst::Jalr { rd, rs1, offset: imm_i(w) },
        BRANCH => {
            let cond = match f3 {
                0b000 => BranchCond::Eq,
                0b001 => BranchCond::Ne,
                0b100 => BranchCond::Lt,
                0b101 => BranchCond::Ge,
                0b110 => BranchCond::Ltu,
                0b111 => BranchCond::Geu,
                _ => return illegal,
            };
            Inst::Branch { cond, rs1, rs2, offset: imm_b(w) }
        }
        LOAD => {
            let width = match f3 {
                0b000 => LoadWidth::B,
                0b001 => LoadWidth::H,
                0b010 => LoadWidth::W,
                0b011 => LoadWidth::D,
                0b100 => LoadWidth::Bu,
                0b101 => LoadWidth::Hu,
                0b110 => LoadWidth::Wu,
                _ => return illegal,
            };
            Inst::Load { width, rd, rs1, offset: imm_i(w) }
        }
        STORE => {
            let width = match f3 {
                0b000 => StoreWidth::B,
                0b001 => StoreWidth::H,
                0b010 => StoreWidth::W,
                0b011 => StoreWidth::D,
                _ => return illegal,
            };
            Inst::Store { width, rs1, rs2, offset: imm_s(w) }
        }
        OP_IMM => {
            // RV64 shifts take a 6-bit shamt, leaving funct6 (bits 31:26) as the selector.
            let shamt = ((w >> 20) & 0x3f) as i32;
            let (op, imm) = match (f3, w >> 26) {
                (0b000, _) => (AluOp::Add, imm_i(w)),
                (0b010, _) => (AluOp::Slt, imm_i(w)),
                (0b011, _) => (AluOp::Sltu, imm_i(w)),
                (0b100, _) => (AluOp::Xor, imm_i(w)),
                (0b110, _) => (AluOp::Or, imm_i(w)),
                (0b111, _) => (AluOp::And, imm_i(w)),
                (0b001, 0b00_0000) => (AluOp::Sll, shamt),
                (0b101, 0b00_0000) => (AluOp::Srl, shamt),
                (0b101, 0b01_0000) => (AluOp::Sra, shamt),
                // Zbb: clz/ctz/cpop/sext.b/sext.h share funct6 011000, the rs2 field
                // selects one; rori is the unguarded funct6 011000 with funct3 101.
                (0b001, 0b011_000) if rs2 == 0b00000 => (AluOp::Clz, 0),
                (0b001, 0b011_000) if rs2 == 0b00001 => (AluOp::Ctz, 0),
                (0b001, 0b011_000) if rs2 == 0b00010 => (AluOp::Cpop, 0),
                (0b001, 0b011_000) if rs2 == 0b00100 => (AluOp::SextB, 0),
                (0b001, 0b011_000) if rs2 == 0b00101 => (AluOp::SextH, 0),
                (0b101, 0b011_000) => (AluOp::Ror, shamt),
                (0b101, 0b001_010) => (AluOp::OrcB, 0),
                // Zbkb brev8 shares this funct6 with Zbb rev8; rs2 picks (00111 vs
                // 11000), the way the two extension groups were aligned.
                (0b101, 0b011_010) if rs2 == 0b00111 => (AluOp::Brev8, 0),
                (0b101, 0b011_010) if rs2 == 0b11000 => (AluOp::Rev8, 0),
                // Zbs immediate forms; the 6-bit shamt is the bit index.
                (0b001, 0b010_010) => (AluOp::Bclr, shamt),
                (0b101, 0b010_010) => (AluOp::Bext, shamt),
                (0b001, 0b011_010) => (AluOp::Binv, shamt),
                (0b001, 0b001_010) => (AluOp::Bset, shamt),
                _ => return illegal,
            };
            Inst::OpImm { op, rd, rs1, imm }
        }
        OP => {
            let op = match (f3, f7) {
                (0b000, 0b000_0000) => AluOp::Add,
                (0b000, 0b010_0000) => AluOp::Sub,
                (0b001, 0b000_0000) => AluOp::Sll,
                (0b010, 0b000_0000) => AluOp::Slt,
                (0b011, 0b000_0000) => AluOp::Sltu,
                (0b100, 0b000_0000) => AluOp::Xor,
                (0b101, 0b000_0000) => AluOp::Srl,
                (0b101, 0b010_0000) => AluOp::Sra,
                (0b110, 0b000_0000) => AluOp::Or,
                (0b111, 0b000_0000) => AluOp::And,
                (0b000, 0b000_0001) => AluOp::Mul,
                (0b001, 0b000_0001) => AluOp::Mulh,
                (0b010, 0b000_0001) => AluOp::Mulhsu,
                (0b011, 0b000_0001) => AluOp::Mulhu,
                (0b100, 0b000_0001) => AluOp::Div,
                (0b101, 0b000_0001) => AluOp::Divu,
                (0b110, 0b000_0001) => AluOp::Rem,
                (0b111, 0b000_0001) => AluOp::Remu,
                // Zba shifted adds (64-bit result).
                (0b010, 0b001_0000) => AluOp::Sh1Add,
                (0b100, 0b001_0000) => AluOp::Sh2Add,
                (0b110, 0b001_0000) => AluOp::Sh3Add,
                // Zbb register-register.
                (0b111, 0b010_0000) => AluOp::Andn,
                (0b110, 0b010_0000) => AluOp::Orn,
                (0b100, 0b010_0000) => AluOp::Xnor,
                (0b100, 0b000_0101) => AluOp::Min,
                (0b101, 0b000_0101) => AluOp::Minu,
                (0b110, 0b000_0101) => AluOp::Max,
                (0b111, 0b000_0101) => AluOp::Maxu,
                (0b001, 0b011_0000) => AluOp::Rol,
                (0b101, 0b011_0000) => AluOp::Ror,
                // Zbs register-register.
                (0b001, 0b001_0100) => AluOp::Bset,
                (0b001, 0b010_0100) => AluOp::Bclr,
                (0b101, 0b010_0100) => AluOp::Bext,
                (0b001, 0b011_0100) => AluOp::Binv,
                // Zicond.
                (0b101, 0b000_0111) => AluOp::CzeroEqz,
                (0b111, 0b000_0111) => AluOp::CzeroNez,
                // Zbkb pack/packh, Zbc's carry-less multiplies, and Zbkx's element
                // permutations.
                (0b100, 0b000_0100) => AluOp::Pack,
                (0b111, 0b000_0100) => AluOp::Packh,
                (0b001, 0b000_0101) => AluOp::Clmul,
                (0b011, 0b000_0101) => AluOp::Clmulh,
                (0b010, 0b000_0101) => AluOp::Clmulr,
                (0b010, 0b001_0100) => AluOp::Xperm4,
                (0b100, 0b001_0100) => AluOp::Xperm8,
                _ => return illegal,
            };
            Inst::Op { op, rd, rs1, rs2 }
        }
        OP_IMM_32 => {
            // 32-bit shifts take a 5-bit shamt (the sixth bit is part of funct7).
            match (f3, f7) {
                (0b000, _) => Inst::OpImm32 { op: AluOp32::Add, rd, rs1, imm: imm_i(w) },
                (0b001, 0b000_0000) => Inst::OpImm32 { op: AluOp32::Sll, rd, rs1, imm: rs2 as i32 },
                (0b101, 0b000_0000) => Inst::OpImm32 { op: AluOp32::Srl, rd, rs1, imm: rs2 as i32 },
                (0b101, 0b010_0000) => Inst::OpImm32 { op: AluOp32::Sra, rd, rs1, imm: rs2 as i32 },
                // Zba slli.uw: a 6-bit shift of the zero-extended low word; the result
                // is a full 64-bit value, so the shamt's top bit is part of funct7.
                (0b001, 0b000_0100) | (0b001, 0b000_0101) => {
                    let shamt6 = ((w >> 20) & 0x3f) as i32;
                    Inst::OpImm { op: AluOp::SllUw, rd, rs1, imm: shamt6 }
                }
                // Zbb roriw.
                (0b101, 0b011_0000) => Inst::OpImm32 { op: AluOp32::Ror, rd, rs1, imm: rs2 as i32 },
                // Zbb clzw/ctzw/cpopw share funct6 011000; the rs2 field selects one.
                (0b001, 0b011_0000) if rs2 == 0b00000 => {
                    Inst::OpImm32 { op: AluOp32::Clz, rd, rs1, imm: 0 }
                }
                (0b001, 0b011_0000) if rs2 == 0b00001 => {
                    Inst::OpImm32 { op: AluOp32::Ctz, rd, rs1, imm: 0 }
                }
                (0b001, 0b011_0000) if rs2 == 0b00010 => {
                    Inst::OpImm32 { op: AluOp32::Cpop, rd, rs1, imm: 0 }
                }
                _ => return illegal,
            }
        }
        OP_32 => {
            match (f3, f7) {
                (0b000, 0b000_0000) => Inst::Op32 { op: AluOp32::Add, rd, rs1, rs2 },
                (0b000, 0b010_0000) => Inst::Op32 { op: AluOp32::Sub, rd, rs1, rs2 },
                (0b001, 0b000_0000) => Inst::Op32 { op: AluOp32::Sll, rd, rs1, rs2 },
                (0b101, 0b000_0000) => Inst::Op32 { op: AluOp32::Srl, rd, rs1, rs2 },
                (0b101, 0b010_0000) => Inst::Op32 { op: AluOp32::Sra, rd, rs1, rs2 },
                (0b000, 0b000_0001) => Inst::Op32 { op: AluOp32::Mul, rd, rs1, rs2 },
                (0b100, 0b000_0001) => Inst::Op32 { op: AluOp32::Div, rd, rs1, rs2 },
                (0b101, 0b000_0001) => Inst::Op32 { op: AluOp32::Divu, rd, rs1, rs2 },
                (0b110, 0b000_0001) => Inst::Op32 { op: AluOp32::Rem, rd, rs1, rs2 },
                (0b111, 0b000_0001) => Inst::Op32 { op: AluOp32::Remu, rd, rs1, rs2 },
                // Zba: word-sourced shifted adds and add.uw; the source is zero-extended
                // but the sum is a full 64-bit value.
                (0b010, 0b001_0000) => Inst::Op { op: AluOp::Sh1AddUw, rd, rs1, rs2 },
                (0b100, 0b001_0000) => Inst::Op { op: AluOp::Sh2AddUw, rd, rs1, rs2 },
                (0b110, 0b001_0000) => Inst::Op { op: AluOp::Sh3AddUw, rd, rs1, rs2 },
                (0b000, 0b000_0100) => Inst::Op { op: AluOp::AddUw, rd, rs1, rs2 },
                // Zbb zext.h has a 64-bit result (it is `packw` on the host side).
                (0b100, 0b000_0100) if rs2 == 0 => Inst::Op { op: AluOp::ZextH, rd, rs1, rs2: 0 },
                // Zbkb packw: the same halfword packing, sign-extended from the word.
                (0b100, 0b000_0100) => Inst::Op { op: AluOp::Packw, rd, rs1, rs2 },
                // Zbb word rotations.
                (0b001, 0b011_0000) => Inst::Op32 { op: AluOp32::Rol, rd, rs1, rs2 },
                (0b101, 0b011_0000) => Inst::Op32 { op: AluOp32::Ror, rd, rs1, rs2 },
                _ => return illegal,
            }
        }
        // rd, rs1, and unknown fm values are reserved for future fence variants and must be
        // ignored, which makes every such encoding a (possibly stronger-than-needed) fence.
        MISC_MEM if f3 == 0 => Inst::Fence {
            fm: (w >> 28) as u8,
            pred: ((w >> 24) & 0xf) as u8,
            succ: ((w >> 20) & 0xf) as u8,
        },
        // Vector loads and stores share the FP load/store opcodes; funct3 picks between
        // them (FP uses 010/011, vector widths use 000 and 101..111).
        LOAD_FP | STORE_FP if matches!(f3, 0b000 | 0b101 | 0b110 | 0b111) => vector::decode_mem(w)?,
        LOAD_FP | STORE_FP => {
            let width = match f3 {
                0b010 => FWidth::S,
                0b011 => FWidth::D,
                _ => return illegal, // flh/flq: Zfh, Q
            };
            if w & 0x7f == LOAD_FP {
                Inst::FLoad { width, rd, rs1, offset: imm_i(w) }
            } else {
                Inst::FStore { width, rs1, rs2, offset: imm_s(w) }
            }
        }
        OP_V if f3 == 0b111 => vector::decode_vset(w)?,
        OP_V => vector::decode_op(w)?,
        MADD | MSUB | NMSUB | NMADD => {
            // R4 format: rs3 in bits 31:27, fmt in bits 26:25, rm in funct3.
            let width = match f7 & 0b11 {
                0b00 => FWidth::S,
                0b01 => FWidth::D,
                _ => return illegal,
            };
            if f3 == 0b101 || f3 == 0b110 {
                return illegal;
            }
            let op = match w & 0x7f {
                MADD => FmaOp::Madd,
                MSUB => FmaOp::Msub,
                NMSUB => FmaOp::Nmsub,
                _ => FmaOp::Nmadd,
            };
            Inst::FFma { op, width, rd, rs1, rs2, rs3: (w >> 27) as u8, rm: f3 as u8 }
        }
        OP_FP => {
            // funct7 = funct5 << 2 | fmt, with fmt 0 = S, 1 = D.
            let width = match f7 & 0b11 {
                0b00 => FWidth::S,
                0b01 => FWidth::D,
                _ => return illegal,
            };
            // Rounding-mode field: 101 and 110 are reserved.
            let rm = f3 as u8;
            let rm_ok = !(rm == 0b101 || rm == 0b110);
            let int = |n: u8| match n {
                0 => Some(IntType::W),
                1 => Some(IntType::Wu),
                2 => Some(IntType::L),
                3 => Some(IntType::Lu),
                _ => None,
            };
            let arith = |op| Inst::FArith { op, width, rd, rs1, rs2, rm };
            match (f7 >> 2, f3, rs2) {
                (0b00000, _, _) if rm_ok => arith(FArithOp::Add),
                (0b00001, _, _) if rm_ok => arith(FArithOp::Sub),
                (0b00010, _, _) if rm_ok => arith(FArithOp::Mul),
                (0b00011, _, _) if rm_ok => arith(FArithOp::Div),
                (0b01011, _, 0) if rm_ok => Inst::FSqrt { width, rd, rs1, rm },
                (0b00100, 0b000, _) => Inst::FSgnj { kind: SgnjKind::Sgnj, width, rd, rs1, rs2 },
                (0b00100, 0b001, _) => Inst::FSgnj { kind: SgnjKind::Sgnjn, width, rd, rs1, rs2 },
                (0b00100, 0b010, _) => Inst::FSgnj { kind: SgnjKind::Sgnjx, width, rd, rs1, rs2 },
                (0b00101, 0b000, _) => {
                    Inst::FMinMax { max: false, canonical_nan: false, width, rd, rs1, rs2 }
                }
                (0b00101, 0b001, _) => {
                    Inst::FMinMax { max: true, canonical_nan: false, width, rd, rs1, rs2 }
                }
                (0b00101, 0b010, _) => {
                    Inst::FMinMax { max: false, canonical_nan: true, width, rd, rs1, rs2 }
                }
                (0b00101, 0b011, _) => {
                    Inst::FMinMax { max: true, canonical_nan: true, width, rd, rs1, rs2 }
                }
                // fcvt.s.d (fmt S, source D = 1) and fcvt.d.s (fmt D, source S = 0).
                (0b01000, _, 1) if rm_ok && width == FWidth::S => {
                    Inst::FCvtFloat { to: FWidth::S, rd, rs1, rm }
                }
                (0b01000, _, 0) if rm_ok && width == FWidth::D => {
                    Inst::FCvtFloat { to: FWidth::D, rd, rs1, rm }
                }
                // Zfa: fround/froundnx are fcvt.s.d/fcvt.d.s with rs2 = 4 and 5.
                (0b01000, _, 4) if rm_ok => Inst::FRound { width, rd, rs1, rm, exact: false },
                (0b01000, _, 5) if rm_ok => Inst::FRound { width, rd, rs1, rm, exact: true },
                (0b10100, 0b010, _) => Inst::FCmp { op: FCmpOp::Eq, width, rd, rs1, rs2 },
                (0b10100, 0b001, _) => Inst::FCmp { op: FCmpOp::Lt, width, rd, rs1, rs2 },
                (0b10100, 0b000, _) => Inst::FCmp { op: FCmpOp::Le, width, rd, rs1, rs2 },
                (0b10100, 0b100, _) => Inst::FCmp { op: FCmpOp::Leq, width, rd, rs1, rs2 },
                (0b10100, 0b101, _) => Inst::FCmp { op: FCmpOp::Ltq, width, rd, rs1, rs2 },
                // Zfa: fcvtmod.w.d is fcvt.w.d with rs2 = 8, fixed to RTZ.
                (0b11000, 0b001, 8) if width == FWidth::D => Inst::FCvtMod { rd, rs1 },
                (0b11000, _, n) if rm_ok => match int(n) {
                    Some(to) => Inst::FCvtToInt { from: width, to, rd, rs1, rm },
                    None => return illegal,
                },
                (0b11010, _, n) if rm_ok => match int(n) {
                    Some(from) => Inst::FCvtFromInt { from, to: width, rd, rs1, rm },
                    None => return illegal,
                },
                (0b11100, 0b000, 0) => Inst::FMvToInt { width, rd, rs1 },
                (0b11100, 0b001, 0) => Inst::FClass { width, rd, rs1 },
                (0b11110, 0b000, 0) => Inst::FMvFromInt { width, rd, rs1 },
                // Zfa fli.s/fli.d: fmv.w.x/fmv.d.x with rs2 = 00001; the constant
                // index rides in the rs1 field.
                (0b11110, 0b000, 1) => Inst::Fli { width, rd, imm: rs1 },
                _ => return illegal,
            }
        }
        AMO => {
            let width = match f3 {
                0b010 => AmoWidth::W,
                0b011 => AmoWidth::D,
                _ => return illegal, // Zabha byte/halfword AMOs are not supported
            };
            let (aq, rl) = (w >> 26 & 1 != 0, w >> 25 & 1 != 0);
            let op = match w >> 27 {
                0b00010 if rs2 == 0 => return Ok(Inst::Lr { width, rd, rs1, aq, rl }),
                0b00011 => return Ok(Inst::Sc { width, rd, rs1, rs2, aq, rl }),
                0b00001 => AmoOp::Swap,
                0b00000 => AmoOp::Add,
                0b00100 => AmoOp::Xor,
                0b01100 => AmoOp::And,
                0b01000 => AmoOp::Or,
                0b10000 => AmoOp::Min,
                0b10100 => AmoOp::Max,
                0b11000 => AmoOp::Minu,
                0b11100 => AmoOp::Maxu,
                _ => return illegal,
            };
            Inst::Amo { op, width, rd, rs1, rs2, aq, rl }
        }
        // fence.i: imm, rs1, and rd are reserved and ignored, like fence's.
        MISC_MEM if f3 == 0b001 => Inst::FenceI,
        SYSTEM if f3 != 0b000 && f3 != 0b100 => {
            let csr = Csr::from_number(w >> 20).ok_or(DecodeError::Illegal(w))?;
            let op = match f3 & 0b11 {
                0b01 => CsrOp::Rw,
                0b10 => CsrOp::Rs,
                _ => CsrOp::Rc,
            };
            // csrrw always writes; csrrs/csrrc write unless the source is x0 / 0.
            let writes = op == CsrOp::Rw || rs1 != 0;
            if writes && csr.read_only() {
                return illegal;
            }
            Inst::Csr { op, csr, rd, src: rs1, imm: f3 & 0b100 != 0 }
        }
        SYSTEM if w == ECALL => Inst::Ecall,
        SYSTEM if w == EBREAK => Inst::Ebreak,
        _ => return illegal,
    };
    Ok(inst)
}

/// Instruction encoders for tests, so programs read like assembly. Immediates and offsets
/// are truncated to their field width; they are checked against real assembler output
/// indirectly, since `decode` is (see `tests/decode_rv64i.rs`) and these must round-trip.
#[cfg(test)]
pub(crate) mod enc {
    use super::*;

    pub fn r(op: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
    }

    pub fn i(op: u32, f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32) & 0xfff) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
    }

    pub fn s(op: u32, f3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
        let o = imm as u32;
        ((o >> 5) & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (o & 0x1f) << 7 | op
    }

    pub fn b(f3: u32, rs1: u32, rs2: u32, off: i32) -> u32 {
        let o = off as u32;
        ((o >> 12) & 1) << 31
            | ((o >> 5) & 0x3f) << 25
            | rs2 << 20
            | rs1 << 15
            | f3 << 12
            | ((o >> 1) & 0xf) << 8
            | ((o >> 11) & 1) << 7
            | BRANCH
    }

    pub fn u(op: u32, rd: u32, imm20: u32) -> u32 {
        (imm20 & 0xfffff) << 12 | rd << 7 | op
    }

    pub fn jal(rd: u32, off: i32) -> u32 {
        let o = off as u32;
        ((o >> 20) & 1) << 31
            | ((o >> 1) & 0x3ff) << 21
            | ((o >> 11) & 1) << 20
            | ((o >> 12) & 0xff) << 12
            | rd << 7
            | JAL
    }

    pub fn lui(rd: u32, imm20: u32) -> u32 {
        u(LUI, rd, imm20)
    }
    pub fn auipc(rd: u32, imm20: u32) -> u32 {
        u(AUIPC, rd, imm20)
    }
    pub fn jalr(rd: u32, rs1: u32, off: i32) -> u32 {
        i(JALR, 0, rd, rs1, off)
    }

    pub fn beq(rs1: u32, rs2: u32, off: i32) -> u32 {
        b(0b000, rs1, rs2, off)
    }

    /// Load with funct3 `f3` (0=lb 1=lh 2=lw 3=ld 4=lbu 5=lhu 6=lwu).
    pub fn load(f3: u32, rd: u32, rs1: u32, off: i32) -> u32 {
        i(LOAD, f3, rd, rs1, off)
    }
    /// Store with funct3 `f3` (0=sb 1=sh 2=sw 3=sd).
    pub fn store(f3: u32, rs1: u32, rs2: u32, off: i32) -> u32 {
        s(STORE, f3, rs1, rs2, off)
    }

    pub fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
        i(OP_IMM, 0b000, rd, rs1, imm)
    }
    pub fn op_imm(f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        i(OP_IMM, f3, rd, rs1, imm)
    }
    /// OP-IMM whose immediate field doubles as a selector: the Zbb/Zbs unary and
    /// immediate forms pack a funct6 above a 5-bit rs2/shift field, so the caller
    /// passes the combined 12-bit value.
    pub fn op_imm_selector(f3: u32, selector: u32, rd: u32, rs1: u32) -> u32 {
        i(OP_IMM, f3, rd, rs1, selector as i32)
    }
    pub fn op(f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        r(OP, f3, f7, rd, rs1, rs2)
    }
    pub fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
        op(0b000, 0, rd, rs1, rs2)
    }
    pub fn op_imm_32(f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        i(OP_IMM_32, f3, rd, rs1, imm)
    }
    pub fn op_32(f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        r(OP_32, f3, f7, rd, rs1, rs2)
    }

    /// AMO with funct5 `f5` (2=lr 3=sc 1=swap 0=add ...) and funct3 `f3` (2=w 3=d).
    /// FP load/store with funct3 `f3` (2=w 3=d).
    pub fn fload(f3: u32, rd: u32, rs1: u32, off: i32) -> u32 {
        i(LOAD_FP, f3, rd, rs1, off)
    }
    pub fn fstore(f3: u32, rs1: u32, rs2: u32, off: i32) -> u32 {
        s(STORE_FP, f3, rs1, rs2, off)
    }
    /// OP-FP with funct5 `f5`, fmt (0=S 1=D), and funct3 `f3`.
    pub fn op_fp(f5: u32, fmt: u32, f3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        r(OP_FP, f3, f5 << 2 | fmt, rd, rs1, rs2)
    }

    pub fn amo(f5: u32, f3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        r(AMO, f3, f5 << 2, rd, rs1, rs2)
    }

    pub const FENCE: u32 = 0x0ff0_000f;
    pub const ECALL: u32 = super::ECALL;
    pub const EBREAK: u32 = super::EBREAK;
}

#[cfg(test)]
mod tests {
    use super::enc::*;
    use super::*;

    #[test]
    fn inst_stays_small() {
        assert!(std::mem::size_of::<Inst>() <= 8);
    }

    #[test]
    fn immediate_extremes_roundtrip() {
        for imm in [-2048, -1, 0, 1, 2047] {
            assert_eq!(
                decode(addi(1, 2, imm)),
                Ok(Inst::OpImm { op: AluOp::Add, rd: 1, rs1: 2, imm })
            );
            assert_eq!(
                decode(store(3, 2, 1, imm)),
                Ok(Inst::Store { width: StoreWidth::D, rs1: 2, rs2: 1, offset: imm })
            );
        }
        for offset in [-4096, -2, 2, 4094] {
            assert_eq!(
                decode(beq(3, 4, offset)),
                Ok(Inst::Branch { cond: BranchCond::Eq, rs1: 3, rs2: 4, offset })
            );
        }
        for offset in [-(1 << 20), -2, 2, (1 << 20) - 2] {
            assert_eq!(decode(jal(1, offset)), Ok(Inst::Jal { rd: 1, offset }));
        }
        assert_eq!(decode(lui(1, 0x80000)), Ok(Inst::Lui { rd: 1, imm: i32::MIN }));
    }

    #[test]
    fn rejects_reserved_encodings() {
        let illegal = [
            0xffff_ffff,
            0x0000_607f,                          // unknown major opcode
            jalr(1, 2, 0) | 0b001 << 12,          // jalr with funct3 != 0
            b(0b010, 1, 2, 8),                    // branch funct3 010
            load(0b111, 1, 2, 0),                 // ldu does not exist
            store(0b100, 1, 2, 0),                // sq is RV128
            op_imm(0b001, 1, 2, 0x400 | 3),       // slli with funct6 != 0
            op_imm(0b101, 1, 2, 0x200 | 3),       // srli/srai with bad funct6
            op(0b001, 0b010_0000, 1, 2, 3),       // "subl"
            op(0b000, 0b000_0011, 1, 2, 3),       // unassigned funct7
            op_32(0b001, 0b000_0001, 1, 2, 3),    // no mulhw
            op_imm_32(0b001, 1, 2, 32),           // slliw with shamt[5] set
            op_imm_32(0b101, 1, 2, 0x400 | 32),   // sraiw with shamt[5] set
            op_32(0b010, 0, 1, 2, 3),             // no sltw
            0x0020_0073,                          // uret-shaped SYSTEM encoding
            0x0000_1073,           // csrrw on ustatus: not a user-mode CSR we implement
            fload(0b001, 1, 2, 0), // flh: Zfh
            op_fp(0b00000, 0b01, 0b101, 1, 2, 3), // fadd.d with reserved rm 101
            op_fp(0b01011, 0b00, 0b000, 1, 2, 3), // fsqrt.s with rs2 != 0
            op_fp(0b01000, 0b00, 0b000, 1, 2, 0), // fcvt.s.s
            op_fp(0b11000, 0b01, 0b000, 1, 2, 4), // fcvt to an unknown integer type
            op_fp(0b10100, 0b00, 0b011, 1, 2, 3), // compare funct3 011
            0x0600_0043 | 0b110 << 12, // fmadd.q (fmt 11) / reserved rm
            op_fp(0b11100, 0b01, 0b000, 1, 2, 3), // fmv.x.d with rs2 != 0
            op_fp(0b00100, 0b10, 0b000, 1, 2, 3), // fsgnj.h: Zfh
            0xc000_1073,           // csrrw cycle: read-only CSR
            0xc011_6073,           // csrrsi time, 2: writes a read-only CSR
            0x0010_4073,           // SYSTEM funct3 100: reserved
            amo(0b00010, 0b010, 1, 2, 3), // lr.w with rs2 != 0
            amo(0b00000, 0b000, 1, 2, 3), // amoadd.b: Zabha
            amo(0b00101, 0b011, 1, 2, 3), // amocas.d: Zacas
            amo(0b00110, 0b010, 1, 2, 3), // unassigned funct5
        ];
        for w in illegal {
            assert_eq!(decode(w), Err(DecodeError::Illegal(w)), "{w:#010x}");
        }
        let illegal_c = [
            0x0000, // all zeros: defined illegal
            0x0010, // c.addi4spn with nzuimm = 0
            0x2001, // c.addiw with rd = 0
            0x6101, // c.addi16sp with nzimm = 0
            0x6001, // c.lui with nzimm = 0 and an even register (c.mop.N are the odds)
            0x9c79, // Zcb unary slot, bits 4:2 = 110 (unassigned)
            0x9c7d, // Zcb unary slot, bits 4:2 = 111 (unassigned)
            0x4002, // c.lwsp with rd = 0
            0x8002, // c.jr with rs1 = 0
        ];
        for h in illegal_c {
            assert_eq!(decode(h), Err(DecodeError::Illegal(h)), "{h:#06x}");
        }
        // Upper bits are ignored when the low 16 bits are a compressed instruction.
        assert_eq!(decode(0xffff_0001), decode(0x0001));
    }

    #[test]
    fn fence_ignores_reserved_fields() {
        // rd and rs1 set, which software must not do but hardware must tolerate.
        assert_eq!(
            decode(FENCE | 5 << 7 | 6 << 15),
            Ok(Inst::Fence { fm: 0, pred: 0xf, succ: 0xf })
        );
    }
}
