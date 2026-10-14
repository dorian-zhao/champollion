//! Host signals for the guest: a thread takes them with `sigwait` and forwards them.

/// Host signals that reach the guest: sent from outside (a terminal, `kill`) or by the
/// host (SIGCHLD when a child rvrun exits, SIGWINCH).
const FORWARDED: [i32; 13] = [
    libc::SIGHUP,
    libc::SIGINT,
    libc::SIGQUIT,
    libc::SIGTERM,
    libc::SIGUSR1,
    libc::SIGUSR2,
    libc::SIGCHLD,
    libc::SIGWINCH,
    libc::SIGALRM,
    libc::SIGTSTP,
    libc::SIGCONT,
    libc::SIGTTIN,
    libc::SIGTTOU,
];

extern "C" fn noop_handler(_: libc::c_int) {}

fn forwarded_set() -> libc::sigset_t {
    // SAFETY: building a signal set in a local.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        for sig in FORWARDED {
            libc::sigaddset(&mut set, sig);
        }
        set
    }
}

/// Forward host signals to the guest of `process`. Blocks them in the calling thread, so
/// call it before starting other threads (they inherit the mask), and starts a thread that
/// takes them with `sigwait` and acts as Linux would for the guest's disposition: queue the
/// signal for a handler, end or stop the process for a default action, or drop it.
pub fn forward_signals(process: std::sync::Arc<crate::process::Process>) {
    for sig in FORWARDED {
        // SAFETY: reading and replacing this process's dispositions.
        unsafe {
            let mut old: libc::sigaction = std::mem::zeroed();
            libc::sigaction(sig, std::ptr::null(), &mut old);
            if old.sa_sigaction == libc::SIG_IGN {
                // Ignored when we started (a shell's background job): ignored in the
                // guest too, as a Linux exec would leave it.
                let ignore =
                    crate::signal::Action { handler: crate::signal::SIG_IGN, ..Default::default() };
                process.set_action(super::linux_signal(sig), ignore);
            } else {
                // A no-op handler: with SIG_DFL, the kernel may discard a blocked signal
                // whose default is to be ignored (SIGCHLD, SIGWINCH) instead of keeping it
                // pending for sigwait.
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = noop_handler as *const () as usize;
                sa.sa_flags = libc::SA_RESTART;
                libc::sigemptyset(&mut sa.sa_mask);
                libc::sigaction(sig, &sa, std::ptr::null_mut());
            }
        }
    }
    let set = forwarded_set();
    // SAFETY: changing this thread's signal mask.
    unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) };
    // A forked child has only the forking thread: it needs its own pump.
    process.set_on_fork(Box::new(start_pump));
    start_pump(process);
}

fn start_pump(process: std::sync::Arc<crate::process::Process>) {
    let _ = std::thread::Builder::new().name("rvrun signals".into()).spawn(move || {
        let set = forwarded_set();
        loop {
            let mut host = 0;
            // SAFETY: waiting for one of the (blocked) forwarded signals.
            if unsafe { libc::sigwait(&set, &mut host) } != 0 {
                continue;
            }
            forward(&process, super::linux_signal(host));
        }
    });
}

fn forward(process: &crate::process::Process, sig: i32) {
    use crate::signal::{self, DefaultAction, Info};
    match process.action(sig).handler {
        signal::SIG_IGN => {}
        signal::SIG_DFL => match signal::default_action(sig) {
            DefaultAction::Terminate => {
                crate::process::Exit::Signal(sig).terminate(&crate::cpu::CpuState::new(0))
            }
            // There is no job control in the guest, but the whole process can stop (and a
            // SIGCONT from the shell resumes it).
            DefaultAction::Stop => {
                // SAFETY: plain raise.
                unsafe { libc::raise(libc::SIGSTOP) };
            }
            DefaultAction::Ignore => {}
        },
        _ => process.send_to_process(sig, Info::Sent { code: signal::SI_USER, pid: 0, uid: 0 }),
    }
}
