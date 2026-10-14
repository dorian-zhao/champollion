//! [`HostOs`](crate::syscall::HostOs) backends: Darwin for Direct mode, Linux for VM mode.

#[cfg(target_os = "macos")]
pub mod darwin;
#[cfg(target_os = "macos")]
mod dirs;
#[cfg(target_os = "macos")]
mod kqueue;
#[cfg(target_os = "linux")]
pub mod linux;
mod signals;
#[cfg(target_os = "macos")]
mod socket;

pub use signals::forward_signals;

/// The backend for the host we run on.
#[cfg(target_os = "macos")]
pub type Native = darwin::DarwinHost;
#[cfg(target_os = "linux")]
pub type Native = linux::LinuxHost;

#[cfg(target_os = "macos")]
use darwin::linux_signal;

/// Host signal number -> Linux: the same numbers.
#[cfg(target_os = "linux")]
fn linux_signal(sig: i32) -> i32 {
    sig
}
