//! [`HostOs`](crate::syscall::HostOs) backends.

#[cfg(target_os = "macos")]
pub mod darwin;
#[cfg(target_os = "macos")]
mod kqueue;
