// vfork (CLONE_VM | CLONE_VFORK): the parent does not resume until the child has execed
// or exited. A failed exec is reported synchronously, the way glibc's posix_spawn needs
// its spawner to see it (issue #8): the raw clone here gets the -errno directly. A later
// successful exec after a failed candidate must win, as glibc's posix_spawnp searches PATH.
//   vfork        -> prints a line per check
//   vfork child  -> exits 7 (the target of the successful exec)
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static long vfork_child(void) {
    return syscall(SYS_clone, 0x100 | 0x4000 | 17 /* SIGCHLD */, 0, 0, 0, 0);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1 && strcmp(argv[1], "child") == 0) return 7;

    int bad = 0;

    long ret = vfork_child();
    if (ret == 0) {
        char *args[] = {"missing", NULL};
        syscall(SYS_execve, "/nonexistent/vfork-program", args, environ);
        _exit(127);
    }
    if (ret == -1 && errno == ENOENT) {
        printf("vfork exec missing: ENOENT\n");
    } else {
        printf("vfork exec missing: no (ret=%ld errno=%d)\n", ret, errno);
        if (ret > 0) waitpid((pid_t)ret, NULL, 0);
        bad = 1;
    }

    int status = 0;
    ret = vfork_child();
    if (ret == 0) {
        char *args[] = {argv[0], "child", NULL};
        syscall(SYS_execve, argv[0], args, environ);
        _exit(127);
    }
    if (ret > 0 && waitpid((pid_t)ret, &status, 0) == ret) {
        printf("vfork exec: exited %d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 7) bad = 1;
    } else {
        printf("vfork exec: pid missing (ret=%ld errno=%d)\n", ret, errno);
        bad = 1;
    }

    status = 0;
    ret = vfork_child();
    if (ret == 0) {
        char *missing[] = {"missing", NULL};
        syscall(SYS_execve, "/nonexistent/vfork-first", missing, environ);
        char *args[] = {argv[0], "child", NULL};
        syscall(SYS_execve, argv[0], args, environ);
        _exit(127);
    }
    if (ret > 0 && waitpid((pid_t)ret, &status, 0) == ret) {
        printf("vfork exec retry: exited %d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 7) bad = 1;
    } else {
        printf("vfork exec retry: pid missing (ret=%ld errno=%d)\n", ret, errno);
        bad = 1;
    }

    status = 0;
    ret = vfork_child();
    if (ret == 0) _exit(5);
    if (ret > 0 && waitpid((pid_t)ret, &status, 0) == ret) {
        printf("vfork exit: exited %d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 5) bad = 1;
    } else {
        printf("vfork exit: pid missing (ret=%ld errno=%d)\n", ret, errno);
        bad = 1;
    }
    return bad;
}
