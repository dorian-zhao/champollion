//! Linux futexes for the threads of one guest process.
//!
//! A futex is a guest address that threads wait on until another thread wakes them. Only
//! process-private futexes exist here (guest processes never share memory), so instead of
//! a host kernel facility this is a parking lot: waiters queue per address in a bucket
//! guarded by a mutex, each with its own condition variable. The value check in `wait`
//! and the dequeue in `wake` happen under the bucket lock, so a wake issued after the guest
//! changed the futex word cannot slip in between a waiter's check and its sleep.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use crate::memory::Memory;
use crate::syscall::errno;

const BUCKETS: usize = 64;

/// Wake every waiter regardless of its bitset (`FUTEX_BITSET_MATCH_ANY`).
pub const MATCH_ANY: u32 = u32::MAX;

pub struct Futexes {
    buckets: [Mutex<HashMap<u64, Vec<Arc<Waiter>>>>; BUCKETS],
}

impl Default for Futexes {
    fn default() -> Self {
        Self { buckets: std::array::from_fn(|_| Mutex::default()) }
    }
}

struct Waiter {
    bitset: u32,
    woken: Mutex<bool>,
    cv: Condvar,
}

fn bucket_of(addr: u64) -> usize {
    ((addr >> 2).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 58) as usize % BUCKETS
}

/// The futex word at `addr` (one atomic load: `Memory` reads are atomic per aligned access).
fn read_u32(mem: &impl Memory, addr: u64) -> Result<u32, i32> {
    let mut b = [0u8; 4];
    mem.read(addr, &mut b).map_err(|_| errno::EFAULT)?;
    Ok(u32::from_le_bytes(b))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Futexes {
    fn bucket(&self, addr: u64) -> MutexGuard<'_, HashMap<u64, Vec<Arc<Waiter>>>> {
        lock(&self.buckets[bucket_of(addr)])
    }

    /// `FUTEX_WAIT(_BITSET)`: if the 32-bit word at `addr` still holds `expected`, sleep
    /// until woken (0) or until `deadline` (`ETIMEDOUT`); otherwise `EAGAIN`.
    pub fn wait(
        &self,
        mem: &impl Memory,
        addr: u64,
        expected: u32,
        deadline: Option<Instant>,
        bitset: u32,
    ) -> Result<u64, i32> {
        if !addr.is_multiple_of(4) || bitset == 0 {
            return Err(errno::EINVAL);
        }
        let waiter = Arc::new(Waiter { bitset, woken: Mutex::new(false), cv: Condvar::new() });
        {
            let mut bucket = self.bucket(addr);
            if read_u32(mem, addr)? != expected {
                return Err(errno::EAGAIN);
            }
            bucket.entry(addr).or_default().push(waiter.clone());
        }
        let mut woken = lock(&waiter.woken);
        while !*woken {
            match deadline {
                None => woken = waiter.cv.wait(woken).unwrap_or_else(|e| e.into_inner()),
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    woken = waiter
                        .cv
                        .wait_timeout(woken, deadline - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            }
        }
        if *woken {
            return Ok(0);
        }
        drop(woken);
        // Timed out: leave the queue, unless a wake got to us first.
        let mut bucket = self.bucket(addr);
        if *lock(&waiter.woken) {
            return Ok(0);
        }
        if let Some(queue) = bucket.get_mut(&addr) {
            queue.retain(|w| !Arc::ptr_eq(w, &waiter));
            if queue.is_empty() {
                bucket.remove(&addr);
            }
        }
        Err(errno::ETIMEDOUT)
    }

    /// `FUTEX_WAKE(_BITSET)`: wake up to `count` waiters on `addr` whose bitset intersects
    /// `bitset`, oldest first. Returns how many were woken.
    pub fn wake(&self, addr: u64, count: u32, bitset: u32) -> Result<u64, i32> {
        if !addr.is_multiple_of(4) || bitset == 0 {
            return Err(errno::EINVAL);
        }
        let mut bucket = self.bucket(addr);
        Ok(Self::wake_locked(&mut bucket, addr, count, bitset))
    }

    fn wake_locked(
        bucket: &mut HashMap<u64, Vec<Arc<Waiter>>>,
        addr: u64,
        count: u32,
        bitset: u32,
    ) -> u64 {
        let Some(queue) = bucket.get_mut(&addr) else { return 0 };
        let mut woken = 0;
        queue.retain(|w| {
            if woken >= count as u64 || w.bitset & bitset == 0 {
                return true;
            }
            *lock(&w.woken) = true;
            w.cv.notify_one();
            woken += 1;
            false
        });
        if queue.is_empty() {
            bucket.remove(&addr);
        }
        woken
    }

    /// `FUTEX_REQUEUE` / `FUTEX_CMP_REQUEUE`: wake up to `wake` waiters on `addr`, then move
    /// up to `requeue` of the rest to `addr2`. With `expected`, first check that the word at
    /// `addr` still holds it (`EAGAIN` otherwise). Returns the number woken, plus the number
    /// moved for `CMP_REQUEUE`.
    pub fn requeue(
        &self,
        mem: &impl Memory,
        addr: u64,
        wake: u32,
        addr2: u64,
        requeue: u32,
        expected: Option<u32>,
    ) -> Result<u64, i32> {
        if !addr.is_multiple_of(4) || !addr2.is_multiple_of(4) {
            return Err(errno::EINVAL);
        }
        // Lock both buckets in index order (once if they are the same).
        let (b1, b2) = (bucket_of(addr), bucket_of(addr2));
        let mut first = lock(&self.buckets[b1.min(b2)]);
        let mut second = (b1 != b2).then(|| lock(&self.buckets[b1.max(b2)]));
        if let Some(expected) = expected
            && read_u32(mem, addr)? != expected
        {
            return Err(errno::EAGAIN);
        }
        let (from, to) = match &mut second {
            None => (&mut *first, None),
            Some(s) if b1 < b2 => (&mut *first, Some(&mut **s)),
            Some(s) => (&mut **s, Some(&mut *first)),
        };
        let woken = Self::wake_locked(from, addr, wake, MATCH_ANY);
        let mut moved = Vec::new();
        if let Some(queue) = from.get_mut(&addr) {
            let n = queue.len().min(requeue as usize);
            moved = queue.drain(..n).collect();
            if queue.is_empty() {
                from.remove(&addr);
            }
        }
        let count = moved.len() as u64;
        if !moved.is_empty() {
            to.unwrap_or(from).entry(addr2).or_default().extend(moved);
        }
        Ok(woken + if expected.is_some() { count } else { 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{FlatMemory, GuestMemory, Prot};
    use std::time::Duration;

    #[test]
    fn wait_checks_the_value_and_times_out() {
        let mem = FlatMemory::new(0, 0x1000);
        let f = Futexes::default();
        mem.write(0x10, &5u32.to_le_bytes()).unwrap();
        assert_eq!(f.wait(&mem, 0x10, 4, None, MATCH_ANY), Err(errno::EAGAIN));
        let soon = Some(Instant::now() + Duration::from_millis(10));
        assert_eq!(f.wait(&mem, 0x10, 5, soon, MATCH_ANY), Err(errno::ETIMEDOUT));
        assert_eq!(f.wake(0x10, 1, MATCH_ANY), Ok(0), "the timed-out waiter left the queue");
        assert_eq!(f.wait(&mem, 0x11, 5, None, MATCH_ANY), Err(errno::EINVAL));
        assert_eq!(f.wait(&mem, 0x2000, 5, None, MATCH_ANY), Err(errno::EFAULT));
    }

    #[test]
    fn wake_and_requeue_release_waiters() {
        let mem = Arc::new(GuestMemory::new().unwrap());
        mem.map(0, 0x1000, Prot::RW).unwrap();
        let f = Arc::new(Futexes::default());
        let spawn_waiter = |addr| {
            let (mem, f) = (mem.clone(), f.clone());
            std::thread::spawn(move || f.wait(&*mem, addr, 0, None, MATCH_ANY))
        };
        let waiters: Vec<_> = (0..3).map(|_| spawn_waiter(0x20)).collect();
        let queued = |addr| f.bucket(addr).get(&addr).map_or(0, Vec::len);
        while queued(0x20) < 3 {
            std::thread::yield_now();
        }
        // Wake one, move one to 0x40 (CMP_REQUEUE counts both), leave one.
        assert_eq!(f.requeue(&*mem, 0x20, 1, 0x40, 1, Some(0)), Ok(2));
        assert_eq!((queued(0x20), queued(0x40)), (1, 1));
        assert_eq!(f.requeue(&*mem, 0x20, 1, 0x40, 1, Some(9)), Err(errno::EAGAIN));
        assert_eq!(f.wake(0x40, 10, MATCH_ANY), Ok(1));
        assert_eq!(f.wake(0x20, 10, MATCH_ANY), Ok(1));
        for w in waiters {
            assert_eq!(w.join().unwrap(), Ok(0));
        }
    }
}
