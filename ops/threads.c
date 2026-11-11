/* ops/threads.c — the worker pool that large kernels split their work across.
 *
 * Off by default: until fe_set_num_threads(n > 1) is called, fe_parallel_for runs
 * every task on the calling thread, which is exactly Ferrite's single-threaded
 * behaviour. Kernels split work into disjoint pieces that are each computed the
 * same way whatever the thread count, so results do not depend on it.
 *
 * Built without threads (FERRITE_NO_PARALLEL, e.g. the ESP32 component) it is a
 * plain loop.
 */
#include "ops.h"

#ifdef FERRITE_NO_PARALLEL

void fe_set_num_threads(int n) { (void)n; }
int fe_get_num_threads(void) { return 1; }
void fe_parallel_for(int n, void (*fn)(void *ctx, int index), void *ctx) {
    for (int i = 0; i < n; i++) fn(ctx, i);
}

#else

#include <pthread.h>

#define FE_MAX_WORKERS 31

static pthread_mutex_t s_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t s_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t s_done = PTHREAD_COND_INITIALIZER;
/* Held for the whole of one parallel_for, so two callers on different threads
 * don't share the pool; a second caller that finds it busy runs serially. */
static pthread_mutex_t s_call = PTHREAD_MUTEX_INITIALIZER;

static pthread_t s_workers[FE_MAX_WORKERS];
static int s_n_workers;
static int s_quit;
static unsigned long s_generation;
static int s_active;                 /* workers still on the current job */

static void (*s_fn)(void *, int);
static void *s_ctx;
static int s_n_tasks;
static int s_next;                   /* next unclaimed task, taken atomically */

static void run_tasks(void) {
    for (;;) {
        int i = __atomic_fetch_add(&s_next, 1, __ATOMIC_RELAXED);
        if (i >= s_n_tasks) return;
        s_fn(s_ctx, i);
    }
}

static void *worker(void *arg) {
    (void)arg;
    unsigned long seen = 0;
    pthread_mutex_lock(&s_mu);
    for (;;) {
        while (s_generation == seen && !s_quit) pthread_cond_wait(&s_work, &s_mu);
        if (s_quit) break;
        seen = s_generation;
        pthread_mutex_unlock(&s_mu);
        run_tasks();
        pthread_mutex_lock(&s_mu);
        if (--s_active == 0) pthread_cond_signal(&s_done);
    }
    pthread_mutex_unlock(&s_mu);
    return NULL;
}

static void stop_workers(void) {
    pthread_mutex_lock(&s_mu);
    s_quit = 1;
    pthread_cond_broadcast(&s_work);
    pthread_mutex_unlock(&s_mu);
    for (int i = 0; i < s_n_workers; i++) pthread_join(s_workers[i], NULL);
    s_n_workers = 0;
    s_quit = 0;
}

void fe_set_num_threads(int n) {
    if (n < 1) n = 1;
    if (n > FE_MAX_WORKERS + 1) n = FE_MAX_WORKERS + 1;
    pthread_mutex_lock(&s_call);
    if (n - 1 != s_n_workers) {
        stop_workers();
        for (int i = 0; i < n - 1; i++) {
            if (pthread_create(&s_workers[i], NULL, worker, NULL) != 0) break;
            s_n_workers++;
        }
    }
    pthread_mutex_unlock(&s_call);
}

int fe_get_num_threads(void) {
    return s_n_workers + 1;
}

void fe_parallel_for(int n, void (*fn)(void *ctx, int index), void *ctx) {
    if (n <= 0) return;
    if (n == 1 || s_n_workers == 0 || pthread_mutex_trylock(&s_call) != 0) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    pthread_mutex_lock(&s_mu);
    s_fn = fn;
    s_ctx = ctx;
    s_n_tasks = n;
    s_next = 0;
    s_active = s_n_workers;
    s_generation++;
    pthread_cond_broadcast(&s_work);
    pthread_mutex_unlock(&s_mu);

    run_tasks();                      /* the caller works too */

    pthread_mutex_lock(&s_mu);
    while (s_active > 0) pthread_cond_wait(&s_done, &s_mu);
    pthread_mutex_unlock(&s_mu);
    pthread_mutex_unlock(&s_call);
}

#endif /* FERRITE_NO_PARALLEL */
