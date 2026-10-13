//! Champollion: a user-mode RISC-V (RV64GC) Linux binary translator for macOS arm64.
//!
//! Layering, bottom-up:
//! - [`decoder`]: raw instruction bits -> [`decoder::Inst`]
//! - [`interpreter`]: executes `Inst`s against [`cpu::CpuState`] and [`memory::Memory`];
//!   always kept as the correctness oracle for the JIT
//! - [`loader`]: static ELF loading, initial stack / auxv (M2)
//! - [`syscall`]: Linux syscall ABI -> [`syscall::HostOs`]
//! - [`jit`]: basic-block translation to arm64 (M4)

pub mod cpu;
pub mod decoder;
pub mod interpreter;
pub mod jit;
pub mod loader;
pub mod memory;
pub mod syscall;

pub use cpu::CpuState;
pub use decoder::{Inst, decode};
