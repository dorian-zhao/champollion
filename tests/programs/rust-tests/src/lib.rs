//! Exercises Rust std on riscv64 Linux under rvrun: allocation, collections, formatting,
//! floats, panics caught by the test harness, files, and the environment.

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::io::{Read, Write};

    #[test]
    fn collections() {
        let v: Vec<u64> = (0..100_000).collect();
        assert_eq!(v.iter().sum::<u64>(), 4_999_950_000);
        let mut m = HashMap::new();
        for i in 0..1000 {
            *m.entry(i % 7).or_insert(0) += i;
        }
        assert_eq!(m[&3], 71_500);
        let b: BTreeMap<_, _> = (0..10).map(|i| (i, i * i)).collect();
        assert_eq!(b.range(3..6).map(|(_, v)| v).sum::<i32>(), 9 + 16 + 25);
    }

    #[test]
    fn strings_and_floats() {
        assert_eq!(format!("{:.3} {:e} {:x}", std::f64::consts::PI, 1e-5, 255), "3.142 1e-5 ff");
        assert_eq!("héllo".chars().rev().collect::<String>(), "olléh");
        assert_eq!(2f64.sqrt().mul_add(2f64.sqrt(), -2.0).abs() < 1e-15, true);
        assert_eq!("3.25".parse::<f32>(), Ok(3.25));
    }

    #[test]
    #[should_panic(expected = "boom")]
    fn panics_are_caught() {
        panic!("boom");
    }

    #[test]
    fn files_and_env() {
        let path = std::env::temp_dir().join(format!("rust-tests-{}", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(b"champollion").unwrap();
        let mut s = String::new();
        std::fs::File::open(&path).unwrap().read_to_string(&mut s).unwrap();
        assert_eq!(s, "champollion");
        std::fs::remove_file(&path).unwrap();
        assert!(std::env::current_dir().unwrap().is_absolute());
        assert!(std::env::args().count() >= 1);
    }
}
