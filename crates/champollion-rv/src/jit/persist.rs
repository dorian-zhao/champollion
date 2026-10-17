//! Persistent code cache: save a run's translations and reuse them in a later run.
//!
//! Generated code holds no host addresses (the jump cache is addressed PC-relatively, the
//! exit stub by relative branches), so the used part of the code buffer can be written out
//! and loaded at the same offsets in another process, together with the tables that
//! describe it: block offsets, exits not chained yet, and fault sites. The jump cache holds
//! host addresses, so it is not saved but refilled from the block table.
//!
//! A snapshot is only valid for the same guest code at the same addresses, which the caller
//! guarantees by keying it on the executable (our loader places it deterministically) and
//! on the build of rvrun. The JIT itself only saves translations of that startup image:
//! once translations are dropped (executable mappings changed, `fence.i`, a full buffer),
//! nothing is saved for the run.
//!
//! A snapshot is trusted like the executable it was made from; loading only checks that it
//! is well formed and fits.

use super::fault::Site;
use super::{Jit, translate};
use crate::memory::GuestMemory;

/// Snapshot format and translation-semantics version. Bump it whenever generated code
/// changes in a way an older snapshot would not match, even if the trampoline layout did
/// not change.
const MAGIC: &[u8; 8] = b"CHMPJIT9";

impl Jit {
    /// Whether the translations so far can be saved (see the module docs).
    pub fn cacheable(&self) -> bool {
        self.pristine && !self.blocks.is_empty()
    }

    /// Serialize the translations for [`Jit::restore`], tagged with `key`. `None` if they
    /// are not [`cacheable`](Jit::cacheable).
    pub fn save(&self, key: u64) -> Option<Vec<u8>> {
        if !self.cacheable() {
            return None;
        }
        let mut w = Writer(Vec::with_capacity(self.code.used() + 64 * self.blocks.len()));
        w.0.extend_from_slice(MAGIC);
        w.u64(key);
        w.u64(self.stub_word as u64);
        w.u64(u64::from(self.identity));
        w.u64(self.code.used() as u64);
        w.0.extend_from_slice(self.code.code());
        w.u64(self.blocks.len() as u64);
        for (&pc, &at) in &self.blocks {
            w.u64(pc);
            w.u64(at as u64);
        }
        w.u64(self.unchained.len() as u64);
        for (&target, branches) in &self.unchained {
            w.u64(target);
            w.u64(branches.len() as u64);
            for &b in branches {
                w.u64(b as u64);
            }
        }
        w.u64(self.faults.sites.len() as u64);
        for (&at, site) in &self.faults.sites {
            w.u64(at as u64);
            w.u64(site.guest_pc);
            w.u64(u64::from(site.instret_undo) << 1 | u64::from(site.store));
            w.u64(match site.ff {
                None => 0,
                Some((base, eew, vd)) => {
                    1 | u64::from(base) << 8 | u64::from(eew) << 16 | u64::from(vd) << 24
                }
            });
        }
        Some(w.0)
    }

    /// Replace all translations with a snapshot from [`Jit::save`] made with the same `key`,
    /// for guest code as it is mapped in `mem` now. Returns the number of blocks restored;
    /// on error the JIT is left empty.
    pub fn restore(&mut self, bytes: &[u8], key: u64, mem: &GuestMemory) -> Result<usize, String> {
        self.reset();
        self.identity = mem.is_identity();
        let result = self.load(bytes, key);
        if result.is_err() {
            self.reset();
        }
        self.generation = mem.code_generation();
        self.pristine = true;
        result
    }

    fn load(&mut self, bytes: &[u8], key: u64) -> Result<usize, String> {
        let mut r = Reader(bytes);
        if r.take(8)? != MAGIC {
            return Err("not a code cache".into());
        }
        if r.u64()? != key {
            return Err("made for another program".into());
        }
        if r.u64()? != self.stub_word as u64 {
            return Err("made by another build".into());
        }
        if r.u64()? != u64::from(self.identity) {
            return Err("made for another memory layout".into());
        }
        let used = r.u64()? as usize;
        if used > self.code.capacity() || !used.is_multiple_of(4) {
            return Err("code does not fit".into());
        }
        let code = r.take(used)?;
        let words: Vec<u32> =
            code.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect();
        // Our own trampoline is already at the start; the snapshot's must be identical.
        let prefix = self.code.used();
        if used < prefix || code[..prefix] != self.code.code()[..] {
            return Err("made by another build".into());
        }
        self.code.append(&words[prefix / 4..]).ok_or("code does not fit")?;
        let offset = |v: u64| -> Result<usize, String> {
            let v = v as usize;
            if v.is_multiple_of(4) && v < used { Ok(v) } else { Err("offset out of range".into()) }
        };
        for _ in 0..r.u64()? {
            let pc = r.u64()?;
            let at = offset(r.u64()?)?;
            self.blocks.insert(pc, at);
        }
        for _ in 0..r.u64()? {
            let target = r.u64()?;
            let n = r.u64()?;
            let mut branches = Vec::new();
            for _ in 0..n {
                branches.push(offset(r.u64()?)?);
            }
            self.unchained.insert(target, branches);
        }
        for _ in 0..r.u64()? {
            let at = offset(r.u64()?)?;
            let guest_pc = r.u64()?;
            let packed = r.u64()?;
            let instret_undo = u32::try_from(packed >> 1).map_err(|_| "bad fault site")?;
            let ff = match r.u64()? {
                0 => None,
                f => Some(((f >> 8) as u8, (f >> 16) as u8, (f >> 24) as u8)),
            };
            self.faults
                .sites
                .insert(at, Site { guest_pc, instret_undo, store: packed & 1 != 0, ff });
        }
        if !r.0.is_empty() {
            return Err("trailing data".into());
        }
        // Prefill the jump cache, or every indirect jump would first miss into the
        // dispatcher.
        let entries: Vec<(u64, usize)> = self.blocks.iter().map(|(&pc, &at)| (pc, at)).collect();
        for (pc, at) in entries {
            let host = self.code.ptr(at) as u64;
            self.jump_cache()[translate::jump_cache_index(pc)] = [pc, host];
        }
        self.stats.restored = self.blocks.len() as u64;
        Ok(self.blocks.len())
    }
}

struct Writer(Vec<u8>);

impl Writer {
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.0.len() {
            return Err("truncated".into());
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}
