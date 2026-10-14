// Signals: handlers, siginfo, masks and pending signals, faults caught with
// siglongjmp, the alternate stack, editing the interrupted context, and signals sent to
// another thread.
//   signals        -> prints a line per check (one thread spins without syscalls)
//   signals term   -> raise(SIGTERM) with the default action: killed (status 143)
//   signals abort  -> abort(): killed by SIGABRT (status 134)
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

static volatile sig_atomic_t usr1_count;
static void on_usr1(int sig) { usr1_count += sig == SIGUSR1; }

static volatile int info_ok;
static void on_usr2(int sig, siginfo_t *si, void *uc) {
    (void)uc;
    info_ok = sig == SIGUSR2 && si->si_signo == SIGUSR2 && si->si_code == SI_USER &&
              si->si_pid == getpid();
}

static sigjmp_buf env;
static volatile uintptr_t fault_addr;
static void on_segv(int sig, siginfo_t *si, void *uc) {
    (void)sig, (void)uc;
    fault_addr = (uintptr_t)si->si_addr;
    siglongjmp(env, 1);
}

static char altstack[65536];
static volatile int on_alt;
static void on_alt_usr1(int sig) {
    (void)sig;
    char local;
    on_alt = &local >= altstack && &local < altstack + sizeof altstack;
}

static void on_ill(int sig, siginfo_t *si, void *ctx) {
    (void)sig, (void)si;
    ucontext_t *uc = ctx;
    uc->uc_mcontext.__gregs[0] += 4; // skip the illegal instruction
    uc->uc_mcontext.__gregs[10] = 42; // and return 42 in a0
}

static pthread_t target;
static atomic_int thread_hit;
static void on_thread_usr1(int sig) {
    (void)sig;
    if (pthread_equal(pthread_self(), target)) atomic_store(&thread_hit, 1);
}
static void *spin(void *arg) {
    (void)arg;
    while (!atomic_load(&thread_hit)) sched_yield();
    return NULL;
}

// No syscalls once spinning: only an interrupt of translated code can get the signal in.
static atomic_int spinning;
static void *spin_hard(void *arg) {
    (void)arg;
    atomic_store(&spinning, 1);
    while (!atomic_load_explicit(&thread_hit, memory_order_relaxed)) {
    }
    return NULL;
}

static void handle(int sig, void (*f)(int), int flags) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = f;
    sa.sa_flags = flags;
    sigaction(sig, &sa, NULL);
}

static void handle_info(int sig, void (*f)(int, siginfo_t *, void *)) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = f;
    sa.sa_flags = SA_SIGINFO;
    sigaction(sig, &sa, NULL);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1 && strcmp(argv[1], "term") == 0) {
        raise(SIGTERM);
        return 1;
    }
    if (argc > 1 && strcmp(argv[1], "abort") == 0) abort();

    handle(SIGUSR1, on_usr1, 0);
    raise(SIGUSR1);
    raise(SIGUSR1);
    printf("handler ran %d times\n", (int)usr1_count);

    handle_info(SIGUSR2, on_usr2);
    kill(getpid(), SIGUSR2);
    printf("siginfo %s\n", info_ok ? "ok" : "wrong");

    sigset_t set, pending;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    sigprocmask(SIG_BLOCK, &set, NULL);
    raise(SIGUSR1);
    sigpending(&pending);
    printf("blocked: pending %d, ran %d\n", sigismember(&pending, SIGUSR1), (int)usr1_count);
    sigprocmask(SIG_UNBLOCK, &set, NULL);
    printf("unblocked: ran %d\n", (int)usr1_count);

    handle(SIGUSR1, SIG_IGN, 0);
    raise(SIGUSR1);
    printf("ignored: ran %d\n", (int)usr1_count);

    handle_info(SIGSEGV, on_segv);
    for (int i = 1; i <= 2; i++) {
        if (sigsetjmp(env, 1) == 0) *(volatile int *)(uintptr_t)(0x10 * i) = 1;
        printf("caught fault %d at %#lx\n", i, (unsigned long)fault_addr);
    }

    stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack, .ss_flags = 0};
    sigaltstack(&ss, NULL);
    handle(SIGUSR1, on_alt_usr1, SA_ONSTACK);
    raise(SIGUSR1);
    printf("alternate stack %s\n", on_alt ? "used" : "not used");

    handle_info(SIGILL, on_ill);
    long r;
    __asm__ volatile("li a0, 7\n\t.4byte 0xffffffff\n\tmv %0, a0" : "=r"(r) : : "a0");
    printf("resumed after illegal instruction with %ld\n", r);

    handle(SIGUSR1, on_thread_usr1, 0);
    pthread_create(&target, NULL, spin, NULL);
    pthread_kill(target, SIGUSR1);
    pthread_join(target, NULL);
    printf("signal to thread %s\n", atomic_load(&thread_hit) ? "delivered" : "lost");

    atomic_store(&thread_hit, 0);
    pthread_create(&target, NULL, spin_hard, NULL);
    while (!atomic_load(&spinning)) sched_yield();
    struct timespec ts = {0, 20 * 1000000};
    nanosleep(&ts, NULL);
    pthread_kill(target, SIGUSR1);
    pthread_join(target, NULL);
    printf("signal to spinning thread %s\n", atomic_load(&thread_hit) ? "delivered" : "lost");
    return 0;
}
