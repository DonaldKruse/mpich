/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef MPIR_THREAD_H_INCLUDED
#define MPIR_THREAD_H_INCLUDED

#include "mpichconfconst.h"
#include "mpichconf.h"
#include "utlist.h"

typedef struct {
    int thread_provided;        /* Provided level of thread support */

    /* This is a special case for is_thread_main, which must be
     * implemented even if MPICH itself is single threaded.  */
#if MPICH_THREAD_LEVEL >= MPI_THREAD_SERIALIZED
    MPID_Thread_id_t main_thread;       /* Thread that started MPI */
#endif

#if defined MPICH_IS_THREADED
    int isThreaded;             /* Set to true if user requested
                                 * THREAD_MULTIPLE */
#endif                          /* MPICH_IS_THREADED */
} MPIR_Thread_info_t;
extern MPIR_Thread_info_t MPIR_ThreadInfo;

/* During Init time, `isThreaded` is not set until the very end of init -- preventing
 * usage of mutexes during init-time; `thread_provided` is set by MPID_Init_thread_level
 * early in the stage so it can be used instead.
 */
#if defined(MPICH_IS_THREADED)
#define MPIR_THREAD_CHECK_BEGIN if (MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
#define MPIR_THREAD_CHECK_END   }
#else
#define MPIR_THREAD_CHECK_BEGIN
#define MPIR_THREAD_CHECK_END
#endif /* MPICH_IS_THREADED */

/* During run time, `isThreaded` should be used, but it still need to be guarded */
#if defined(MPICH_IS_THREADED)
#define MPIR_IS_THREADED    MPIR_ThreadInfo.isThreaded
#else
#define MPIR_IS_THREADED    0
#endif

/* ------------------------------------------------------------ */
/* Global thread model, used for non-performance-critical paths */
/* CONSIDER:
 * - should we restrict to MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX only?
 * - once we isolate the mutexes, we should replace MPID with MPL
 */

#if defined(MPICH_IS_THREADED)
extern MPID_Thread_mutex_t MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX;

/* CS macros with runtime bypass */
#define MPIR_THREAD_CS_ENTER(mutex) \
    if (MPIR_ThreadInfo.isThreaded) { \
        int err_ = 0; \
        MPID_Thread_mutex_lock(&mutex, &err_); \
        MPIR_Assert(err_ == 0); \
    }

#define MPIR_THREAD_CS_EXIT(mutex) \
    if (MPIR_ThreadInfo.isThreaded) { \
        int err_ = 0; \
        MPID_Thread_mutex_unlock(&mutex, &err_); \
        MPIR_Assert(err_ == 0); \
    }

#else
#define MPIR_THREAD_CS_ENTER(mutex)
#define MPIR_THREAD_CS_EXIT(mutex)
#endif

/* ------------------------------------------------------------ */
/* Other thread models, for performance-critical paths          */

#if defined(MPICH_IS_THREADED)
#if MPICH_THREAD_GRANULARITY == MPICH_THREAD_GRANULARITY__VCI
extern MPID_Thread_mutex_t MPIR_THREAD_VCI_HANDLE_MUTEX;
extern MPID_Thread_mutex_t MPIR_THREAD_VCI_CTX_MUTEX;
extern MPID_Thread_mutex_t MPIR_THREAD_VCI_PMI_MUTEX;
extern MPID_Thread_mutex_t MPIR_THREAD_VCI_BSEND_MUTEX;

#define MPIDIU_THREAD_GLOBAL_OFFSET           (-1000)
#define MPIDIU_THREAD_PROGRESS_MUTEX_ID       (MPIDIU_THREAD_GLOBAL_OFFSET + 0)
#define MPIDIU_THREAD_UTIL_MUTEX_ID           (MPIDIU_THREAD_GLOBAL_OFFSET + 1)
#define MPIDIU_THREAD_MPIDIG_GLOBAL_MUTEX_ID  (MPIDIU_THREAD_GLOBAL_OFFSET + 2)
#define MPIDIU_THREAD_SCHED_LIST_MUTEX_ID     (MPIDIU_THREAD_GLOBAL_OFFSET + 3)
#define MPIDIU_THREAD_TSP_QUEUE_MUTEX_ID      (MPIDIU_THREAD_GLOBAL_OFFSET + 4)
#define MPIDIU_THREAD_HCOLL_MUTEX_ID          (MPIDIU_THREAD_GLOBAL_OFFSET + 5)
#define MPIDIU_THREAD_DYNPROC_MUTEX_ID        (MPIDIU_THREAD_GLOBAL_OFFSET + 6)
#define MPIDIU_THREAD_ALLOC_MEM_MUTEX_ID      (MPIDIU_THREAD_GLOBAL_OFFSET + 7)

#define MPID_MUTEX_DBG_LOCK_ID (-2000)

#define MPID_THREAD_REQUEST_MEM_LOCK_OFFSET 0

#define MPIR_THREAD_VCI_HANDLE_MUTEX_ID (-4000)
#define MPIR_THREAD_VCI_CTX_MUTEX_ID    (-3999)
#define MPIR_THREAD_VCI_PMI_MUTEX_ID    (-3998)
#define MPIR_THREAD_VCI_BSEND_MUTEX_ID  (-3997)


#if defined(VCIEXP_LOCK_PTHREADS) || defined(VCIEXP_LOCK_ARGOBOTS)

typedef struct {
    char dummy1[64];
    int debug_enabled;
    int print_rank;
    int print_enabled; /* 0: disabled, 1:lightly, 2: verbose, 3: very verbose */
#if defined(VCIEXP_LOCK_PTHREADS)
    int no_lock;
    #endif
    char dummy2[64];
} MPIU_exp_data_t;
extern MPIU_exp_data_t g_MPIU_exp_data;

typedef struct {
    char dummy1[64];
    int vci_mask;
#if defined(VCIEXP_LOCK_PTHREADS)
    int local_tid;
    #endif
    char dummy2[64];
} MPIU_exp_data_tls_t;
extern __thread MPIU_exp_data_tls_t l_MPIU_exp_data;

#undef VCIEXP_LOCK_PTHREADS_COND_OR_FALSE
#if defined(VCIEXP_LOCK_PTHREADS)
#define VCIEXP_LOCK_PTHREADS_COND_OR_FALSE(cond) (cond)
#else
#define VCIEXP_LOCK_PTHREADS_COND_OR_FALSE(...) 0
#endif

#undef MPIDUI_THREAD_CHECK_ERROR
#if defined(HAVE_ERROR_CHECKING)
#if HAVE_ERROR_CHECKING == MPID_ERROR_LEVEL_ALL
#define MPIDUI_THREAD_CHECK_ERROR 1
#else
#define MPIDUI_THREAD_CHECK_ERROR 0
#endif
#else
#define MPIDUI_THREAD_CHECK_ERROR 0
#endif


void MPIDUI_Thread_cs_vci_check(MPIDU_Thread_mutex_t *p_mutex, int mutex_id, const char *mutex_str,
                                const char *function, const char *file, int line);

void MPIDUI_Thread_cs_vci_print(MPIDU_Thread_mutex_t *p_mutex, int mutex_id, const char *msg,
                                const char *mutex_str, const char *function, const char *file,
                                int line);


static inline
void MPIDUI_Thread_cs_enter_vci_impl(MPIDU_Thread_mutex_t *p_mutex, int mutex_id, int print_level,
                                     bool recursive, const char *mutex_str, const char *function,
                                     const char *file, int line)
{
    if (mutex_id <= 0 || VCIEXP_LOCK_PTHREADS_COND_OR_FALSE(!g_MPIU_exp_data.no_lock)) {
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, recursive ? "racquire" : "acquire",
                                           mutex_str, function, file, line);
        }
        if (recursive) {
            MPIDUI_THREAD_CS_ENTER_REC((*p_mutex));
        } else {
            MPIDUI_THREAD_CS_ENTER((*p_mutex));
        }
    } else {
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "empty-acquire", mutex_str, function,
                                           file, line);
            MPIDUI_Thread_cs_vci_check(p_mutex, mutex_id, mutex_str, function, file, line);
        }
    }
        
}


static inline
void MPIDUI_Thread_cs_enter_or_skip_vci_impl(MPIDU_Thread_mutex_t *p_mutex, int mutex_id,
                                             int *p_skip, int print_level, const char *mutex_str,
                                             const char *function, const char *file, int line)
{
    if (mutex_id <= 0 || VCIEXP_LOCK_PTHREADS_COND_OR_FALSE(!g_MPIU_exp_data.no_lock)) {
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "acquire", mutex_str, function, file,
                                               line);
        }
        MPIDUI_THREAD_CS_ENTER((*p_mutex));
        *p_skip = 0;
    } else if (VCIEXP_LOCK_PTHREADS_COND_OR_FALSE((1 << mutex_id) & l_MPIU_exp_data.vci_mask)) {
            /* This VCI should be checked without lock. */
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "empty-acquire", mutex_str,
                                           function, file, line);
            MPIDUI_Thread_cs_vci_check(p_mutex, mutex_id, mutex_str, function, file, line);
        }
        *p_skip = 0;
    } else {
            /* This VCI is not associated with it. */
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "skip-acquire", mutex_str, function,
                                           file, line);
        }
        *p_skip = 1;
    }
}



static inline
void MPIDUI_Thread_cs_exit_vci_impl(MPIDU_Thread_mutex_t *p_mutex, int mutex_id, int print_level,
                                    const char *mutex_str, const char *function, const char *file,
                                    int line)
{
    if (mutex_id <= 0 || VCIEXP_LOCK_PTHREADS_COND_OR_FALSE(!g_MPIU_exp_data.no_lock)) {
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "release", mutex_str, function, file,
                                           line);
        }
        MPIDUI_THREAD_CS_EXIT((*p_mutex));
    } else {
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "empty-release", mutex_str, function,
                                           file, line);
            MPIDUI_Thread_cs_vci_check(p_mutex, mutex_id, mutex_str, function, file, line);
        }
    }
}


static inline
void MPIDUI_Thread_cs_yield_vci_impl(MPIDU_Thread_mutex_t *p_mutex, int mutex_id, int print_level,
                                     const char *mutex_str, const char *function, const char *file,
                                     int line)
{
    if (mutex_id <= 0 || VCIEXP_LOCK_PTHREADS_COND_OR_FALSE(!g_MPIU_exp_data.no_lock)) {
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "yield", mutex_str, function, file,
                                           line);
        }
        MPIDUI_THREAD_CS_YIELD((*p_mutex));
    } else {
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
            if (g_MPIU_exp_data.print_enabled >= print_level)
                MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "empty-yield", mutex_str, function,
                                           file, line);
            MPIDUI_Thread_cs_vci_check(p_mutex, mutex_id, mutex_str, function, file, line);
        }
    }
}

#define MPIDUI_THREAD_CS_ENTER_VCI(_mutex, _mutex_id) \
    MPIDUI_Thread_cs_enter_vci_impl(&(_mutex), _mutex_id, false, 1, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_ENTER_REC_VCI(_mutex, _mutex_id)               \
    MPIDUI_Thread_cs_enter_vci_impl(&(_mutex), _mutex_id, true, 1, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_ENTER_OR_SKIP_VCI(_mutex, _mutex_id, _p_skip) \
    MPIDUI_Thread_cs_enter_or_skip_vci_impl(&(_mutex), _mutex_id, _p_skip, 1, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_EXIT_VCI(_mutex, _mutex_id)                    \
    MPIDUI_Thread_cs_exit_vci_impl(&(_mutex), _mutex_id, 1, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_ENTER_VCI_NOPRINT(_mutex, _mutex_id) \
    MPIDUI_Thread_cs_enter_vci_impl(&(_mutex), _mutex_id, false, 2, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_ENTER_REC_VCI_NOPRINT(_mutex, _mutex_id)       \
    MPIDUI_Thread_cs_enter_vci_impl(&(_mutex), _mutex_id, true, 2, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_ENTER_OR_SKIP_VCI_NOPRINT(_mutex, _mutex_id, _p_skip) \
    MPIDUI_Thread_cs_enter_or_skip_vci_impl(&(_mutex), _mutex_id, _p_skip, 2, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_EXIT_VCI_NOPRINT(_mutex, _mutex_id)            \
    MPIDUI_Thread_cs_exit_vci_impl(&(_mutex), _mutex_id, 2, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_CS_YIELD_VCI(_mutex, _mutex_id)                   \
    MPIDUI_Thread_cs_yield_vci_impl(&(_mutex), _mutex_id, 1, #_mutex, __FUNCTION__, __FILE__, __LINE__)
#define MPIDUI_THREAD_ASSERT_IN_CS_VCI(_mutex, _mutex_id) do {} while (0) /* no-op */

#else /* !(defined(VCIEXP_LOCK_PTHREADS) || defined(VCIEXP_LOCK_ARGOBOTS)) */

#define MPIDUI_THREAD_CS_ENTER_VCI(mutex, mutex_id) MPIDUI_THREAD_CS_ENTER(mutex)
#define MPIDUI_THREAD_CS_ENTER_REC_VCI(mutex, mutex_id) MPIDUI_THREAD_CS_ENTER_REC(mutex)
#define MPIDUI_THREAD_CS_ENTER_OR_SKIP_VCI(mutex, mutex_id, p_skip) \
    do {                                                            \
        MPIDUI_THREAD_CS_ENTER_VCI(mutex, mutex_id);                \
        *(p_skip) = 0;                                              \
    } while (0)
#define MPIDUI_THREAD_CS_EXIT_VCI(mutex, mutex_id) MPIDUI_THREAD_CS_EXIT(mutex)

#define MPIDUI_THREAD_CS_ENTER_VCI_NOPRINT(mutex, mutex_id) MPIDUI_THREAD_CS_ENTER(mutex)
#define MPIDUI_THREAD_CS_ENTER_REC_VCI_NOPRINT(mutex, mutex_id) MPIDUI_THREAD_CS_ENTER_REC(mutex)
#define MPIDUI_THREAD_CS_ENTER_OR_SKIP_VCI_NOPRINT(mutex, mutex_id, p_skip) \
    do {                                                                \
    MPIDUI_THREAD_CS_ENTER_VCI(mutex, mutex_id);                        \
    *(p_skip) = 0;                                                      \
    } while (0)                                                         
#define MPIDUI_THREAD_CS_EXIT_VCI_NOPRINT(mutex, mutex_id) MPIDUI_THREAD_CS_EXIT(mutex)
#define MPIDUI_THREAD_CS_YIELD_VCI(mutex, mutex_id) MPIDUI_THREAD_CS_YIELD(mutex)
#define MPIDUI_THREAD_ASSERT_IN_CS_VCI(mutex, mutex_id) MPIDUI_THREAD_ASSERT_IN_CS(mutex)

#endif /* !(defined(VCIEXP_LOCK_PTHREADS) || defined(VCIEXP_LOCK_ARGOBOTS)) */

#endif /* MPICH_THREAD_GRANULARITY */
#endif /* MPICH_IS_THREADED */

#endif /* MPIR_THREAD_H_INCLUDED */
