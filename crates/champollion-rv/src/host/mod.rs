//! [`HostOs`](crate::syscall::HostOs) backends.

#[cfg(target_os = "macos")]
pub mod darwin;
#[cfg(target_os = "macos")]
mod dirs;
#[cfg(target_os = "macos")]
mod kqueue;
mod signals;
#[cfg(target_os = "macos")]
mod socket;

pub use signals::forward_signals;

/// The backend for the host we run on.
#[cfg(target_os = "macos")]
pub type Native = darwin::DarwinHost;

#[cfg(target_os = "macos")]
use darwin::linux_signal;
