// Threads: pthread_create/join, atomics, mutexes, condition variables, TLS, and how
// exit and pthread_exit end a multithreaded process.
//   threads             -> counters and ping-pong
//   threads exit-group  -> a thread calls exit(7) while main waits: status 7
//   threads main-exits  -> main calls pthread_exit; the process ends with the last thread
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define THREADS 8
#define ITERS 100000

static atomic_long atomic_count;
static long locked_count;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static int turn;
static __thread long tls_value;

static void *worker(void *arg) {
    tls_value = (long)arg;
    for (int i = 0; i < ITERS; i++) {
        atomic_fetch_add(&atomic_count, 1);
        pthread_mutex_lock(&lock);
        locked_count++;
        pthread_mutex_unlock(&lock);
    }
    return (void *)(tls_value * 10);
}

static void *pong(void *arg) {
    (void)arg;
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&lock);
        while (turn != 1) pthread_cond_wait(&cond, &lock);
        turn = 0;
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

static void nap(long ms) {
    struct timespec ts = {0, ms * 1000000};
    nanosleep(&ts, NULL);
}

static void *exit_seven(void *arg) {
    (void)arg;
    nap(20);
    exit(7);
}

static void *outlive_main(void *arg) {
    (void)arg;
    nap(50);
    printf("child outlived main\n");
    fflush(stdout);
    return NULL;
}

int main(int argc, char **argv) {
    pthread_t t[THREADS];
    if (argc > 1 && strcmp(argv[1], "exit-group") == 0) {
        pthread_create(&t[0], NULL, exit_seven, NULL);
        pthread_join(t[0], NULL); // never returns
        return 1;
    }
    if (argc > 1 && strcmp(argv[1], "main-exits") == 0) {
        pthread_create(&t[0], NULL, outlive_main, NULL);
        pthread_exit(NULL);
    }

    long sum = 0;
    for (long i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < THREADS; i++) {
        void *r;
        pthread_join(t[i], &r);
        sum += (long)r;
    }
    printf("atomic %ld locked %ld results %ld\n", (long)atomic_count, locked_count, sum);

    pthread_t p;
    pthread_create(&p, NULL, pong, NULL);
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&lock);
        turn = 1;
        pthread_cond_signal(&cond);
        while (turn != 0) pthread_cond_wait(&cond, &lock);
        pthread_mutex_unlock(&lock);
    }
    pthread_join(p, NULL);
    printf("ping-pong done\n");
    return 0;
}
