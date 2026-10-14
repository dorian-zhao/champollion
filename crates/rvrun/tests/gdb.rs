//! `rvrun -g PORT` speaks the GDB remote serial protocol. These tests drive it with a
//! minimal client, so they need no GDB installed.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

struct Session {
    child: Child,
    conn: TcpStream,
    stderr: Option<std::io::BufReader<std::process::ChildStderr>>,
}

impl Session {
    /// Start `rvrun -g` on a free port and connect to it.
    fn start(args: &[&std::ffi::OsStr]) -> Session {
        // Port 0: rvrun picks a free port and reports it on stderr.
        let mut child = Command::new(env!("CARGO_BIN_EXE_rvrun"))
            .args(["-g", "0"])
            .env("RVRUN_CACHE", "0")
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stderr = std::io::BufReader::new(child.stderr.take().unwrap());
        let mut line = String::new();
        std::io::BufRead::read_line(&mut stderr, &mut line).unwrap();
        let port: u16 = line
            .split("127.0.0.1:")
            .nth(1)
            .and_then(|rest| rest.split(' ').next()?.parse().ok())
            .unwrap_or_else(|| panic!("no port in {line:?}"));
        let conn = TcpStream::connect(("127.0.0.1", port)).unwrap();
        conn.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        Session { child, conn, stderr: Some(stderr) }
    }

    fn send_raw(&mut self, bytes: &[u8]) {
        self.conn.write_all(bytes).unwrap();
    }

    fn send(&mut self, payload: &str) {
        let sum = payload.bytes().fold(0u8, |a, b| a.wrapping_add(b));
        self.send_raw(format!("${payload}#{sum:02x}").as_bytes());
    }

    /// Send a packet and return the reply payload (acks are consumed).
    fn request(&mut self, payload: &str) -> String {
        self.send(payload);
        self.reply()
    }

    /// `k` has no reply: the stub ends the session and rvrun exits.
    fn kill(mut self) -> std::process::ExitStatus {
        self.send("k");
        self.child.wait().unwrap()
    }

    fn read_byte(&mut self) -> u8 {
        let mut byte = [0u8];
        if let Err(e) = self.conn.read_exact(&mut byte) {
            let status = self.child.try_wait();
            let mut err = String::new();
            if let Some(mut s) = self.stderr.take() {
                let _ = s.read_to_string(&mut err);
            }
            panic!("read: {e}; rvrun status {status:?}; stderr: {err}");
        }
        byte[0]
    }

    fn reply(&mut self) -> String {
        let mut byte = [0u8];
        // Skip acks until the start of a packet.
        loop {
            byte[0] = self.read_byte();
            if byte[0] == b'$' {
                break;
            }
        }
        let mut payload = Vec::new();
        loop {
            byte[0] = self.read_byte();
            if byte[0] == b'#' {
                break;
            }
            payload.push(byte[0]);
        }
        let mut sum = [0u8; 2];
        self.conn.read_exact(&mut sum).unwrap();
        self.send_raw(b"+");
        rle_decode(&payload)
    }

    /// The pc from a `g` (read all registers) reply: x0..x31, then pc, little-endian hex.
    fn pc(&mut self) -> u64 {
        let regs = self.request("g");
        let pc_hex = &regs[32 * 16..33 * 16];
        u64::from_le_bytes(hex(pc_hex).try_into().unwrap())
    }
}

/// Undo the protocol's run-length encoding: `X*N` means X repeated (N - 29) more times.
fn rle_decode(payload: &[u8]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < payload.len() {
        if payload[i] == b'*' {
            let last = *out.last().unwrap();
            out.extend(std::iter::repeat_n(last, (payload[i + 1] - 29) as usize));
            i += 2;
        } else {
            out.push(payload[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

/// Whether `reply` is a stop reply (`S` or `T` packet) for signal `sig`.
fn is_stop(reply: &str, sig: u8) -> bool {
    (reply.starts_with('S') || reply.starts_with('T')) && reply[1..3] == format!("{sig:02x}")
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn elf_entry(path: &PathBuf) -> u64 {
    let f = std::fs::read(path).unwrap();
    u64::from_le_bytes(f[24..32].try_into().unwrap())
}

#[test]
fn registers_memory_steps_breakpoints_and_exit() {
    let hello = fixture("hello");
    let entry = elf_entry(&hello);

    // Session 1: stopped at the entry point; read memory; single-step ten times.
    let mut s = Session::start(&[hello.as_os_str()]);
    assert!(is_stop(&s.request("?"), 5), "stopped with SIGTRAP at the entry point");
    assert_eq!(s.pc(), entry);
    let file = std::fs::read(&hello).unwrap();
    let text = hex(&s.request(&format!("m{entry:x},4")));
    assert_eq!(text.len(), 4);
    assert!(file.windows(4).any(|w| w == text), "memory at entry matches the file");
    let mut pcs = Vec::new();
    for _ in 0..10 {
        assert!(is_stop(&s.request("s"), 5));
        pcs.push(s.pc());
    }
    assert!(pcs.windows(2).any(|w| w[0] != w[1]), "stepping moves the pc");
    assert_eq!(s.kill().code(), Some(128 + 9));

    // Session 2: a breakpoint at the tenth instruction stops there; continuing exits.
    let mut s = Session::start(&[hello.as_os_str()]);
    let target = pcs[9];
    assert_eq!(s.request(&format!("Z0,{target:x},4")), "OK");
    assert!(is_stop(&s.request("c"), 5));
    assert_eq!(s.pc(), target);
    assert_eq!(s.request(&format!("z0,{target:x},4")), "OK");
    assert_eq!(s.request("c"), "W00", "the program exits with status 0");
    let out = s.child.wait_with_output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello from riscv64\n");
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn ctrl_c_interrupts_a_running_guest() {
    let lua = fixture("programs/lua");
    let mut s = Session::start(&[lua.as_os_str(), "-e".as_ref(), "while true do end".as_ref()]);
    s.send_raw(b"$c#63");
    std::thread::sleep(Duration::from_millis(300));
    s.send_raw(&[0x03]);
    let stop = s.reply();
    assert!(is_stop(&stop, 2), "SIGINT stop, got {stop}");
    assert_eq!(s.kill().code(), Some(128 + 9));
}
