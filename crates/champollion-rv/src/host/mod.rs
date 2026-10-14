//! [`HostOs`](crate::syscall::HostOs) backends.

#[cfg(target_os = "macos")]
pub mod darwin;
#[cfg(target_os = "macos")]
mod dirs;
#[cfg(target_os = "macos")]
mod kqueue;
#[cfg(target_os = "macos")]
mod socket;
