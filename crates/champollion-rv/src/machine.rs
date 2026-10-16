//! The machine we describe to guests.
//!
//! One table feeds everything software can ask about the CPU: `AT_HWCAP`,
//! `riscv_hwprobe` (which glibc's `ifunc` targets are chosen from), the ISA string in
//! `/proc/cpuinfo`, and the kernel the guest is told it runs on (`uname` and
//! `/proc/version`). It must stay one table: software believes the widest claim it
//! finds, so a feature named in any one of these and implemented in none of the
//! decoder, engine, and JIT is a trap in the middle of a real program.
//!
//! `RVRUN_V=1` adds the vector extension for experiments. V stays off by default
//! until the vector paths of real libraries (glibc's `ifunc` selections first) run.

/// The kernel the guest is told it runs on. `uname` and `/proc/version` both say this;
/// the version matches the Kata kernel VM mode boots with, see `docs/design.md`.
pub const KERNEL_RELEASE: &str = "6.12.0-champollion";
pub const KERNEL_VERSION: &str = "#1 SMP champollion";

/// How many CPUs the guest may use: the host's, capped at the 64 that
/// `sched_getaffinity` can express. `/proc/cpuinfo` and the affinity mask must agree.
pub fn harts() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).min(64)
}

/// Multi-letter extensions, in canonical order: `(name, riscv_hwprobe bit)`. The bit
/// is `None` for extensions the kernel reports only through the ISA string.
const MULTI: &[(&str, Option<u64>)] = &[
    ("zicsr", None),
    ("zifencei", None),
    ("zba", Some(1 << 3)),
    ("zbb", Some(1 << 4)),
    ("zbs", Some(1 << 5)),
    ("zicond", Some(1 << 35)),
];

/// `riscv_hwprobe` bits for key `RISCV_HWPROBE_KEY_IMA_EXT_0` that are not derived from
/// [`MULTI`], in the kernel's numbering.
const IMA_FD: u64 = 1 << 0;
const IMA_C: u64 = 1 << 1;
const IMA_V: u64 = 1 << 2;

/// The machine description with the switches resolved.
#[derive(Clone, Copy, Debug)]
pub struct Machine {
    vector: bool,
}

impl Machine {
    /// Read the switches from the environment.
    pub fn current() -> Machine {
        Machine { vector: std::env::var_os("RVRUN_V").is_some_and(|v| v == "1") }
    }

    pub fn with_vector(vector: bool) -> Machine {
        Machine { vector }
    }

    /// Single-letter extensions, in canonical order (bit n of `AT_HWCAP` is the letter
    /// `'a' + n`).
    fn letters(&self) -> &'static str {
        if self.vector { "imafdcv" } else { "imafdc" }
    }

    /// `AT_HWCAP` for the initial stack.
    pub fn hwcap(&self) -> u64 {
        self.letters().bytes().map(|c| 1u64 << (c - b'a')).fold(0, |a, b| a | b)
    }

    /// `riscv_hwprobe`, key `RISCV_HWPROBE_KEY_IMA_EXT_0`.
    pub fn hwprobe_extensions(&self) -> u64 {
        let mut bits = IMA_FD | IMA_C;
        if self.vector {
            bits |= IMA_V;
        }
        MULTI.iter().filter_map(|(_, bit)| *bit).fold(bits, |a, b| a | b)
    }

    /// The `isa` field of `/proc/cpuinfo` (and of any future description): single
    /// letters, then the multi-letter extensions the table lists.
    pub fn isa_string(&self) -> String {
        let mut isa = String::from("rv64");
        isa.push_str(self.letters());
        for (name, _) in MULTI {
            isa.push('_');
            isa.push_str(name);
        }
        isa
    }

    /// `/proc/cpuinfo`, in the kernel's format for a hart without a vendor.
    pub fn cpuinfo(&self) -> String {
        let isa = self.isa_string();
        let mut text = String::new();
        for hart in 0..harts() {
            text.push_str(&format!(
                "processor\t: {hart}\nhart\t\t: {hart}\nisa\t\t: {isa}\nmmu\t\t: sv39\n\
                 mvendorid\t: 0x0\nmarchid\t\t: 0x0\nmimpid\t\t: 0x0\n\n"
            ));
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hwcap_has_one_bit_per_letter() {
        // i m a f d c
        assert_eq!(Machine::with_vector(false).hwcap(), 0x112d);
        // ... plus 'v' at bit 21.
        assert_eq!(Machine::with_vector(true).hwcap(), 0x20_112d);
    }

    #[test]
    fn hwprobe_lists_exactly_what_we_implement() {
        // IMA_FD | IMA_C | Zba | Zbb | Zbs | Zicond.
        assert_eq!(Machine::with_vector(false).hwprobe_extensions(), 0x8_0000_003b);
        assert_eq!(Machine::with_vector(true).hwprobe_extensions(), 0x8_0000_003f);
    }

    #[test]
    fn the_isa_string_names_every_extension() {
        let plain = Machine::with_vector(false).isa_string();
        assert_eq!(plain, "rv64imafdc_zicsr_zifencei_zba_zbb_zbs_zicond");
        assert!(Machine::with_vector(true).isa_string().starts_with("rv64imafdcv_"));
    }

    #[test]
    fn cpuinfo_has_one_block_per_hart_and_no_host_fields() {
        let text = Machine::with_vector(false).cpuinfo();
        assert_eq!(text.matches("processor\t: ").count(), harts());
        assert_eq!(
            text.matches(&format!("isa\t\t: {}", Machine::with_vector(false).isa_string())).count(),
            harts()
        );
        assert!(text.contains("mmu\t\t: sv39"));
        // None of the host's (arm64) vocabulary may appear.
        assert!(!text.contains("implementer") && !text.contains("Features"));
    }
}
