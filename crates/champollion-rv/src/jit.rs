//! Basic-block JIT to arm64 (M4).
//!
//! - [`arm64`]: instruction encoder, checked against an assembler.
//! - [`code`]: executable memory (MAP_JIT and W^X on macOS).

pub mod arm64;
pub mod code;
