// Processes: fork, wait statuses, a pipe between parent and child, execve, posix_spawn.
//   processes        -> prints a line per check
//   processes child  -> prints its arguments (the target of execve and posix_spawn)
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1 && strcmp(argv[1], "child") == 0) {
        printf("child argv0=%s args=%d last=%s env=%s\n", argv[0], argc, argv[argc - 1],
               getenv("PROCESSES_TEST") ? getenv("PROCESSES_TEST") : "-");
        return 0;
    }
    int status;
    pid_t pid = fork();
    if (pid == 0) _exit(42);
    waitpid(pid, &status, 0);
    printf("fork: exited %d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);

    pid = fork();
    if (pid == 0) {
        raise(SIGTERM);
        _exit(1);
    }
    waitpid(pid, &status, 0);
    printf("fork: killed by %d\n", WIFSIGNALED(status) ? WTERMSIG(status) : -1);

    int fds[2];
    pipe(fds);
    pid = fork();
    if (pid == 0) {
        close(fds[0]);
        const char msg[] = "through a pipe";
        write(fds[1], msg, sizeof msg);
        _exit(0);
    }
    close(fds[1]);
    char buf[32] = {0};
    read(fds[0], buf, sizeof buf - 1);
    waitpid(pid, &status, 0);
    printf("pipe: %s, getppid ok %d\n", buf, getppid() > 0);

    setenv("PROCESSES_TEST", "inherited", 1);
    pid = fork();
    if (pid == 0) {
        char *args[] = {"renamed", "child", "x", "y", NULL};
        execve(argv[0], args, environ);
        _exit(127);
    }
    waitpid(pid, &status, 0);
    printf("execve: exited %d\n", WEXITSTATUS(status));

    char *args[] = {"spawned", "child", "z", NULL};
    if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ) != 0) return 1;
    waitpid(pid, &status, 0);
    printf("posix_spawn: exited %d\n", WEXITSTATUS(status));

    char *bad[] = {"nope", NULL};
    printf("execve missing: %d\n", execve("/nonexistent/program", bad, environ));
    return 0;
}
