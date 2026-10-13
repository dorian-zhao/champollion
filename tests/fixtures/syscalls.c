// Self-checking syscall exercise for rvrun. Prints one line per check; the test
// harness compares stdout exactly. argv[1] is a scratch directory.
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
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
    CHECK("getpid", getpid() > 0);

    // stdin: the harness pipes in a line.
    CHECK("stdin", fgets(buf, sizeof buf, stdin) && strcmp(buf, "from stdin\n") == 0);

    fflush(stdout);
    return 7;
}
