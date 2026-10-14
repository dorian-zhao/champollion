// System calls rvrun does not emulate, which on Linux (route A) go to the host kernel as
// they are. Prints one line per check, like syscalls.c. argv[1] is a scratch directory.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/sendfile.h>
#include <sys/timerfd.h>
#include <unistd.h>

#define CHECK(name, cond) printf("%s %s\n", (cond) ? "ok" : "FAIL", name)

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    char src[4096], dst[4096], buf[64] = {0};
    snprintf(src, sizeof src, "%s/src.txt", dir);
    snprintf(dst, sizeof dst, "%s/dst.txt", dir);

    int ino = inotify_init1(IN_CLOEXEC);
    CHECK("inotify", ino >= 0 && inotify_add_watch(ino, dir, IN_CREATE) >= 0);

    int in = open(src, O_CREAT | O_RDWR | O_TRUNC, 0644);
    write(in, "passthrough", 11);
    lseek(in, 0, SEEK_SET);
    int out = open(dst, O_CREAT | O_RDWR | O_TRUNC, 0644);
    CHECK("sendfile", sendfile(out, in, NULL, 11) == 11 && pread(out, buf, 11, 0) == 11 &&
                          memcmp(buf, "passthrough", 11) == 0);

    char events[4096];
    ssize_t n = read(ino, events, sizeof events);
    struct inotify_event *ev = (struct inotify_event *)events;
    CHECK("inotify event", n > 0 && (ev->mask & IN_CREATE) && strcmp(ev->name, "src.txt") == 0);

    int tfd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    struct itimerspec its = {.it_value = {.tv_nsec = 1000000}};
    uint64_t ticks = 0;
    CHECK("timerfd", tfd >= 0 && timerfd_settime(tfd, 0, &its, NULL) == 0 &&
                         read(tfd, &ticks, 8) == 8 && ticks >= 1);

    int pipes[2];
    CHECK("pipe size", pipe(pipes) == 0 && fcntl(pipes[0], F_GETPIPE_SZ) >= 4096);

    void *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK("mlock", mlock(page, 4096) == 0 && munlock(page, 4096) == 0);

    uid_t r, e, s;
    CHECK("getresuid", getresuid(&r, &e, &s) == 0 && r == getuid() && e == geteuid());

    errno = 0;
    int prio = getpriority(PRIO_PROCESS, 0);
    CHECK("getpriority", errno == 0 && prio >= -20 && prio <= 19);

    fflush(stdout);
    return 0;
}
