use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: rvrun <riscv64-linux-elf> [args...]");
        return ExitCode::from(2);
    };
    eprintln!(
        "rvrun: ELF loading lands in M2; cannot run {} yet",
        path.to_string_lossy()
    );
    ExitCode::FAILURE
}
