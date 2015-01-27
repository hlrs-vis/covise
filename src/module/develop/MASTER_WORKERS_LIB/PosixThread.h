/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PosixThread
//
//  Implementation of Thread with posix threads
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _POSIX_THREAD_H_
#define _POSIX_THREAD_H_

#include "Thread.h"
#include <pthread.h>

class PosixThread : public Thread
{
public:
    /// Constructor
    PosixThread();
    /// Destructor
    ~PosixThread();
    /// Start assigns portAddress to _portAddress and starts the worker function
    virtual int Start(ThreadTask **portAddress);
    /// Cancel cancels the thread
    virtual int Cancel();
    /// Join is called to join the thread
    virtual int Join();

protected:
private:
    int WorkerFunction();
    static void *worker_function(void *arg);
    pthread_t _posix_thread;
};
#endif
