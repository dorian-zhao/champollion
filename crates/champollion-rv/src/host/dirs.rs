//! Linux `getdents64` on Darwin, which has no public equivalent: each directory fd the
//! guest reads gets a `DIR` stream on a duplicate of it. The duplicate shares the file
//! offset, but the stream buffers entries, so a guest `lseek` drops the stream and the
//! next read starts a fresh one at the new offset. `d_off` values are entry indices.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use crate::syscall::errno;

struct Entry {
    ino: u64,
    kind: u8,
    name: Vec<u8>,
}

struct Stream {
    dir: *mut libc::DIR,
    /// Entries returned so far (the next `d_off`).
    index: i64,
    /// An entry read but not returned: it did not fit the guest's buffer.
    held: Option<Entry>,
}

// SAFETY: a DIR is only used under the table's lock.
unsafe impl Send for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: we own the stream (and the duplicate fd it closes).
        unsafe { libc::closedir(self.dir) };
    }
}

#[derive(Default)]
pub struct Dirs {
    streams: Mutex<HashMap<i32, Stream>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Linux `DT_*` from Darwin's (the values agree).
fn linux_type(t: u8) -> u8 {
    t
}

impl Dirs {
    /// `getdents64`: `linux_dirent64` records for the next entries of directory `fd`, at
    /// most `max` bytes. `EINVAL` if not even one entry fits.
    pub fn getdents64(&self, fd: i32, max: usize) -> Result<Vec<u8>, i32> {
        let mut streams = lock(&self.streams);
        let s = match streams.entry(fd) {
            std::collections::hash_map::Entry::Occupied(s) => s.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                // SAFETY: dup and fdopendir on the guest's fd; the stream owns the duplicate.
                let dup = unsafe { libc::dup(fd) };
                if dup < 0 {
                    return Err(errno::EBADF);
                }
                let dir = unsafe { libc::fdopendir(dup) };
                if dir.is_null() {
                    let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    unsafe { libc::close(dup) };
                    return Err(if e == libc::ENOTDIR { 20 } else { errno::EBADF });
                }
                slot.insert(Stream { dir, index: 0, held: None })
            }
        };
        let mut out = Vec::new();
        loop {
            let entry = match s.held.take() {
                Some(e) => e,
                None => {
                    // SAFETY: a live stream; the entry is copied out before the next call.
                    let d = unsafe { libc::readdir(s.dir) };
                    if d.is_null() {
                        break;
                    }
                    let d = unsafe { &*d };
                    let name =
                        unsafe { std::ffi::CStr::from_ptr(d.d_name.as_ptr()) }.to_bytes().to_vec();
                    Entry { ino: d.d_ino, kind: linux_type(d.d_type), name }
                }
            };
            // ino, off, reclen, type, name, NUL, padded to 8.
            let reclen = (19 + entry.name.len() + 1).next_multiple_of(8);
            if out.len() + reclen > max {
                s.held = Some(entry);
                if out.is_empty() {
                    return Err(errno::EINVAL);
                }
                break;
            }
            s.index += 1;
            let start = out.len();
            out.extend(entry.ino.to_le_bytes());
            out.extend(s.index.to_le_bytes());
            out.extend((reclen as u16).to_le_bytes());
            out.push(entry.kind);
            out.extend(&entry.name);
            out.resize(start + reclen, 0);
        }
        Ok(out)
    }

    /// The guest moved or closed `fd`: forget its stream.
    pub fn forget(&self, fd: i32) {
        lock(&self.streams).remove(&fd);
    }
}
