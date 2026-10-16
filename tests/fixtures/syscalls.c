// Self-checking syscall exercise for rvrun. Prints one line per check; the test
// harness compares stdout exactly. argv[1] is a scratch directory.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define CHECK(name, cond) printf("%s %s\n", (cond) ? "ok" : "FAIL", name)

int main(int argc, char **argv) {
    CHECK("argc", argc == 3);
    CHECK("argv", argc == 3 && strcmp(argv[2], "second arg") == 0);
    const char *env = getenv("CHAMPOLLION_TEST");
    CHECK("environ", env && strcmp(env, "yes") == 0);

    // Files: create, write, seek, read back, stat.
    char path[4096];
    snprintf(path, sizeof path, "%s/file.txt", argc > 1 ? argv[1] : ".");
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0644);
    CHECK("open", fd >= 0);
    CHECK("write", write(fd, "champollion", 11) == 11);
    CHECK("lseek", lseek(fd, 5, SEEK_SET) == 5);
    char buf[16] = {0};
    CHECK("read", read(fd, buf, sizeof buf) == 6 && memcmp(buf, "ollion", 6) == 0);
    struct stat st;
    CHECK("fstat", fstat(fd, &st) == 0 && st.st_size == 11 && S_ISREG(st.st_mode));
    CHECK("close", close(fd) == 0);
    CHECK("stat", stat(path, &st) == 0 && st.st_size == 11);
    CHECK("enoent", open("/nonexistent/champollion", O_RDONLY) < 0 && errno == ENOENT);
    CHECK("ebadf", close(fd) < 0 && errno == EBADF);

    // stdio through a FILE (exercises readv/writev paths in musl).
    FILE *f = fopen(path, "r");
    CHECK("fopen", f != NULL);
    CHECK("fgets", f && fgets(buf, sizeof buf, f) && strcmp(buf, "champollion") == 0);
    if (f) fclose(f);

    // Heap: small allocations come from brk, large ones from mmap.
    char *small = malloc(100);
    CHECK("malloc small", small != NULL);
    memset(small, 'x', 100);
    char *big = malloc(8 << 20);
    CHECK("malloc big", big != NULL);
    memset(big, 'y', 8 << 20);
    CHECK("heap contents", small[99] == 'x' && big[(8 << 20) - 1] == 'y');
    free(big);
    free(small);

    // Anonymous mmap, mprotect, munmap.
    char *m = mmap(NULL, 3 * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK("mmap", m != MAP_FAILED && m[0] == 0 && m[3 * 4096 - 1] == 0);
    m[4096] = 42;
    CHECK("mprotect", mprotect(m, 4096, PROT_READ) == 0);
    CHECK("munmap", munmap(m, 3 * 4096) == 0);

    // File-backed private mmap.
    fd = open(path, O_RDONLY);
    char *fm = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK("mmap file", fm != MAP_FAILED && memcmp(fm, "champollion", 11) == 0 && fm[11] == 0);
    close(fd);

    struct timespec t1, t2;
    CHECK("clock_gettime", clock_gettime(CLOCK_MONOTONIC, &t1) == 0 &&
                               clock_gettime(CLOCK_MONOTONIC, &t2) == 0 &&
                               (t2.tv_sec > t1.tv_sec ||
                                (t2.tv_sec == t1.tv_sec && t2.tv_nsec >= t1.tv_nsec)));
    CHECK("realtime", time(NULL) > 1700000000);

    unsigned char r1[32] = {0}, r2[32] = {0};
    CHECK("getrandom", getrandom(r1, 32, 0) == 32 && getrandom(r2, 32, 0) == 32 &&
                           memcmp(r1, r2, 32) != 0);

    struct utsname u;
    CHECK("uname", uname(&u) == 0 && strcmp(u.sysname, "Linux") == 0 &&
                       strcmp(u.machine, "riscv64") == 0);

    // The machine identity: uname, /proc/version, and /proc/cpuinfo must all describe
    // the machine we emulate (never the arm64 host under it), and V stays unclaimed
    // until the vector paths of real libraries run.
    CHECK("uname release", strcmp(u.release, "6.12.0-champollion") == 0);
    CHECK("hwcap no v", (getauxval(AT_HWCAP) & (1UL << 21)) == 0);
    char line[256];
    FILE *ver = fopen("/proc/version", "r");
    CHECK("proc version", ver != NULL);
    CHECK("proc version text", ver && fgets(line, sizeof line, ver) &&
                                   strncmp(line, "Linux version 6.12.0-champollion", 32) == 0);
    if (ver) fclose(ver);
    static char cpuinfo[16384];
    FILE *ci = fopen("/proc/cpuinfo", "r");
    size_t ci_len = ci ? fread(cpuinfo, 1, sizeof cpuinfo - 1, ci) : 0;
    cpuinfo[ci_len] = '\0';
    if (ci) fclose(ci);
    int blocks = 0;
    for (char *p = cpuinfo; (p = strstr(p, "processor\t: ")) != NULL; p++) blocks++;
    cpu_set_t set;
    CHECK("cpuinfo harts", sched_getaffinity(0, sizeof set, &set) == 0 &&
                               CPU_COUNT(&set) == blocks && blocks > 0);
    CHECK("cpuinfo isa", strstr(cpuinfo, "isa\t\t: rv64imafdc_zicsr_") != NULL);
    CHECK("cpuinfo mmu", strstr(cpuinfo, "mmu\t\t: sv39") != NULL);
    CHECK("cpuinfo ours", strncmp(cpuinfo, "processor\t: 0", 13) == 0 &&
                              strstr(cpuinfo, "CPU implementer") == NULL &&
                              strstr(cpuinfo, "asimd") == NULL);
    CHECK("getpid", getpid() > 0);
    CHECK("process group", getpgid(0) == getpgrp() && getpgrp() > 0 && getsid(0) > 0);

    struct statx stx;
    CHECK("statx", statx(AT_FDCWD, path, 0, STATX_BASIC_STATS, &stx) == 0 &&
                       stx.stx_size == 11 && S_ISREG(stx.stx_mode) &&
                       (stx.stx_mask & STATX_SIZE) && stat(path, &st) == 0 &&
                       makedev(stx.stx_dev_major, stx.stx_dev_minor) == st.st_dev);
    fd = open(path, O_RDONLY);
    // Zig's libc returns -1 for errors here, where POSIX says the error number.
    CHECK("fadvise", posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL) == 0 &&
                         posix_fadvise(-1, 0, 0, POSIX_FADV_NORMAL) != 0);
    close(fd);
    struct sysinfo si;
    CHECK("sysinfo", sysinfo(&si) == 0 && si.totalram > 0 && si.mem_unit > 0 && si.uptime > 0);

    // stdin: the harness pipes in a line.
    CHECK("stdin", fgets(buf, sizeof buf, stdin) && strcmp(buf, "from stdin\n") == 0);

    fflush(stdout);
    return 7;
}
