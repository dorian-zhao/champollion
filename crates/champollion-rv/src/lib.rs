//! Champollion: a user-mode RISC-V (RV64GC) Linux binary translator for macOS arm64.
//!
//! Layering, bottom-up:
//! - [`decoder`]: raw instruction bits -> [`decoder::Inst`]
//! - [`interpreter`]: executes `Inst`s against [`cpu::CpuState`] and [`memory::Memory`];
//!   always kept as the correctness oracle for the JIT
//! - [`engine`]: the fast interpreter (decoded-block cache + software TLB), checked against
//!   the oracle
//! - [`memory`]: the guest address space ([`memory::GuestMemory`])
//! - [`loader`]: static ELF loading, initial stack / auxv
//! - [`syscall`]: Linux syscall ABI -> [`syscall::HostOs`]
//! - [`process`]: ties it together: spawn a guest, run it, serve memory syscalls
//! - [`jit`]: basic-block translation to arm64 (M4)

pub mod cpu;
pub mod decoder;
pub mod engine;
pub mod fpu;
pub mod host;
pub mod interpreter;
pub mod jit;
pub mod loader;
pub mod memory;
pub mod process;
pub mod syscall;

pub use cpu::CpuState;
pub use decoder::{Inst, decode};
