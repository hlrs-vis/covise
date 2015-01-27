/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PosixMutexAndCondition
//
//  Posix realisation of MutexAndCondition
//  for multithreaded computations for
//  master-worker computations
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _POSIX_MUTEX_AND_CONDITION_H_
#define _POSIX_MUTEX_AND_CONDITION_H_

#include <pthread.h>
#include "MutexAndCondition.h"

class PosixMutexAndCondition : public MutexAndCondition
{
public:
    /// constructor
    PosixMutexAndCondition();
    /// destructor
    ~PosixMutexAndCondition();
    /// Signal wake threads that are waiting on this condition
    virtual int Signal();
    /// Wait wait a signal on this condition
    virtual int Wait();
    /// Lock locks the mutex
    virtual int Lock();
    /// Unlock unlocks the mutex
    virtual int Unlock();

protected:
private:
    PosixMutexAndCondition(const PosixMutexAndCondition &);
    friend class PosixThread; // this is related to the problem
    // posed by the fact that pthread_cleanup_push
    // and pthread_cleanup_pop are macros
    static void cleanup_handler(void *arg);
    pthread_mutex_t _mutex;
    pthread_cond_t _cond;
    // _mutex_state and _cond_state are flags that keep
    // information of a successful initialissation
    enum MutexState
    {
        MUTEX_OK = 0,
        MUTEX_NOT_OK = 1
    } _mutex_state;
    enum CondState
    {
        COND_OK = 0,
        COND_NOT_OK = 1
    } _cond_state;
};
#endif
