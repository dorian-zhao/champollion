//! Exercises Rust std on riscv64 Linux under rvrun: allocation, collections, formatting,
//! floats, panics caught by the test harness, files, the environment, and threads.

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::io::{Read, Write};
    use std::sync::{Arc, Barrier, Condvar, Mutex, mpsc};
    use std::thread;

    #[test]
    fn threads() {
        // Spawn and join, with names and results.
        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::Builder::new()
                    .name(format!("worker-{i}"))
                    .spawn(move || (thread::current().name().unwrap().to_string(), i * i))
                    .unwrap()
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results[3], ("worker-3".to_string(), 9));

        // Shared state under a mutex, a barrier, scoped threads borrowing the stack.
        let total = Mutex::new(0u64);
        let barrier = Barrier::new(4);
        thread::scope(|s| {
            for i in 0..4u64 {
                let (total, barrier) = (&total, &barrier);
                s.spawn(move || {
                    barrier.wait();
                    for _ in 0..10_000 {
                        *total.lock().unwrap() += i;
                    }
                });
            }
        });
        assert_eq!(*total.lock().unwrap(), 60_000);

        // Channels, a condition variable, park/unpark, and a panicking thread.
        let (tx, rx) = mpsc::channel();
        for i in 0..8 {
            let tx = tx.clone();
            thread::spawn(move || tx.send(i).unwrap());
        }
        drop(tx);
        assert_eq!(rx.iter().sum::<i32>(), 28);
        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair2 = pair.clone();
        let main = thread::current();
        let t = thread::spawn(move || {
            *pair2.0.lock().unwrap() = true;
            pair2.1.notify_one();
            main.unpark();
        });
        let mut ready = pair.0.lock().unwrap();
        while !*ready {
            ready = pair.1.wait(ready).unwrap();
        }
        drop(ready);
        thread::park_timeout(std::time::Duration::from_secs(5));
        t.join().unwrap();
        assert!(thread::spawn(|| panic!("in a thread")).join().is_err());
        assert!(thread::available_parallelism().unwrap().get() >= 1);
    }

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
