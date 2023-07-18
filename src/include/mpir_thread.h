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
        //int vci_mask;
    uint64_t vci_mask;
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


#if defined(VCIEXP_LOCK_ARGOBOTS)
/* Argobots-only data structures and functions. */
typedef struct MPIDUI_Thread_abt_tls_t {
    uint64_t vci_history;
    ABT_pool original_pool;
} MPIDUI_Thread_abt_tls_t;
ABT_pool MPIDUI_Thread_cs_get_target_pool(int mutex_id);

static inline
void MPIDUI_Thread_cs_update_history(MPIDUI_Thread_abt_tls_t *p_tls, int mutex_id)
{
        /* Update the history. */
    p_tls->vci_history = (((uint64_t)(mutex_id)) | (p_tls->vci_history << (uint64_t)8));
}

/* Return true if this ULT should migrate. */
static inline
bool MPIDUI_Thread_cs_update_history_and_decide(MPIDUI_Thread_abt_tls_t *p_tls, int mutex_id)
{
    uint64_t vci_history = p_tls->vci_history;
        /* Update the history. */
    p_tls->vci_history = (((uint64_t)(mutex_id)) | (vci_history << (uint64_t)8));
#define MPIDUI_THREAD_CS_CHECK_EQ_TMP(i)                                \
    (mutex_id == ((vci_history >> (uint64_t)(i * 8)) & (uint64_t)0xFF) ? 1 : 0) \
        int count = MPIDUI_THREAD_CS_CHECK_EQ_TMP(0) + MPIDUI_THREAD_CS_CHECK_EQ_TMP(1) \
        + MPIDUI_THREAD_CS_CHECK_EQ_TMP(2) + MPIDUI_THREAD_CS_CHECK_EQ_TMP(3) \
        + MPIDUI_THREAD_CS_CHECK_EQ_TMP(4) + MPIDUI_THREAD_CS_CHECK_EQ_TMP(5) \
        + MPIDUI_THREAD_CS_CHECK_EQ_TMP(6) + MPIDUI_THREAD_CS_CHECK_EQ_TMP(7);
#undef MPIDUI_THREAD_CS_CHECK_EQ_TMP                                
        /* Currently, if this thread is offloaded to the same thread N out of 8 times (including this), migration happens. */
    const int N = 5;
    return count >= N;
}
#endif


                                                
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
#if defined(VCIEXP_LOCK_ARGOBOTS)
            /* Schedule this ULT on a certain execution stream. */
        MPL_thread_id_t owner_id;
        MPIDUI_Thread_abt_tls_t *p_tls =
            (MPIDUI_Thread_abt_tls_t *)MPL_thread_get_tls_ptr_and_self_fast(&owner_id);
        while (1) {
            if (!((((uint64_t)1) << (uint64_t)(mutex_id - 1)) & l_MPIU_exp_data.vci_mask)) {
                    /* Check if we should "migrate" this thread, not "offload". */
                if (MPIDUI_Thread_cs_update_history_and_decide(p_tls, mutex_id)) {
                        /* Migration should happen. */
                    if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
                        if (g_MPIU_exp_data.print_enabled >= print_level)
                            MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "resched-acquire-nomig",
                                                       mutex_str, function, file, line);
                    }
                } else {
                        /* Migration should not happen. */
                    if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
                        if (g_MPIU_exp_data.print_enabled >= print_level)
                            MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "resched-acquire-mig",
                                                       mutex_str, function, file, line);
                    }
                    p_tls->original_pool = ABTX_FAST_SELF_GET_ASSOCIATED_POOL();
                }
                ABT_pool target_pool = MPIDUI_Thread_cs_get_target_pool(mutex_id);
                int ret = ABTX_FAST_SET_ASSOCIATED_POOL_AND_YIELD(target_pool);
                MPIR_Assert(ret == ABT_SUCCESS);
            } else {
                    /* This VCI operation can be done on this execution stream. */
                MPIDUI_Thread_cs_update_history(p_tls, mutex_id);
            }
            if (likely(p_mutex->count == 0)) {
                    /* This thread becomes an owner. */
                p_mutex->owner = owner_id;
                p_mutex->count = 1;
                break;
            } else if (recursive) {
                    /* If this thread is the owner, it's fine. */
                if (owner_id == p_mutex->owner) {
                    p_mutex->count++;
                    break;
                }
            }
                /* It seems that another ULT is taking this lock. */
            if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
                if (g_MPIU_exp_data.print_enabled >= print_level)
                    MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "retry-acquire", mutex_str,
                                               function, file, line);
            }
                /* Try again after yield. */
            int ret = ABT_self_yield();
            MPIR_Assert(ret == ABT_SUCCESS);
        }
            /* fallthrough */
#endif /* defined(VCIEXP_LOCK_ARGOBOTS) */
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
//    } else if (VCIEXP_LOCK_PTHREADS_COND_OR_FALSE((1 << mutex_id) & l_MPIU_exp_data.vci_mask)) {
    } else if ((1 << mutex_id) & l_MPIU_exp_data.vci_mask) {
        
#if defined(VCIEXP_LOCK_ARGOBOTS)
            /* Since multiple ULTs can be associated with a single execution stream, we need to check
             *          * the owner. */
        if (unlikely(p_mutex->count != 0)) {
                /* Someone has taken this lock (NOTE: this function does not take a recursive lock). */
            if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
                if (g_MPIU_exp_data.print_enabled >= print_level)
                    MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "skip-empty-acquire", mutex_str,
                                               function, file, line);
            }
            *p_skip = 1;
            return;
        } else {
                /* To take a lock, set owner and count. */
            p_mutex->count = 1;
            p_mutex->owner = MPL_thread_get_self_fast();
        }
            /* fallthrough */
#endif
        
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
#if defined(VCIEXP_LOCK_ARGOBOTS)
        if (likely(p_mutex->count == 1)) {
            p_mutex->count = 0;
            MPIDUI_Thread_abt_tls_t *p_tls =
                (MPIDUI_Thread_abt_tls_t *)ABTX_FAST_SELF_GET_TLS_PTR();
            ABT_pool original_pool = p_tls->original_pool;
            if (original_pool) {
                    /* This ULT was offloaded. Go back to the original pool. */
                p_tls->original_pool = NULL;
                int ret = ABTX_FAST_SET_ASSOCIATED_POOL_AND_YIELD(original_pool);
                MPIR_Assert(ret == ABT_SUCCESS);
            }
        } else {
            p_mutex->count -= 1;
        }
#endif
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
#if defined(VCIEXP_LOCK_ARGOBOTS)
        if (p_mutex->count == 1) {
            MPL_thread_id_t self = p_mutex->owner;
            p_mutex->owner = 0;
            p_mutex->count = 0;
                /* This yield is not very desirable since there's no guarantee of progress while it is
                 *              * yielding.  This routine should not be called very often. */
            ABT_pool target_pool = MPIDUI_Thread_cs_get_target_pool(mutex_id);
            while (1) {
                int ret = ABTX_FAST_SET_ASSOCIATED_POOL_AND_YIELD(target_pool);
                MPIR_Assert(ret == ABT_SUCCESS);
                if (p_mutex->owner == 0) {
                    p_mutex->owner = self;
                    p_mutex->count = 1;
                    break;
                } else {
                    if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
                        if (g_MPIU_exp_data.print_enabled >= print_level)
                            MPIDUI_Thread_cs_vci_print(p_mutex, mutex_id, "retry-yacquire",
                                                       mutex_str, function, file, line);
                    }
                }
            }
        } else {
                /* Yielding recursive lock does not make sense. */
            MPIR_Assert(0);
        }
        if (MPIDUI_THREAD_CHECK_ERROR && unlikely(g_MPIU_exp_data.debug_enabled)) {
                /* Check the VCI association again. */
            MPIDUI_Thread_cs_vci_check(p_mutex, mutex_id, mutex_str, function, file, line);
        }
#endif
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
