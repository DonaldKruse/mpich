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

#endif /* MPICH_THREAD_GRANULARITY */
#endif /* MPICH_IS_THREADED */

#endif /* MPIR_THREAD_H_INCLUDED */
