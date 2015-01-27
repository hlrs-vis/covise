/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PosixThread.h"

void *
PosixThread::worker_function(void *arg)
{
    PosixThread *thread = static_cast<PosixThread *>(arg);
    thread->WorkerFunction();
    return NULL;
}

PosixThread::PosixThread()
    : Thread()
{
}

PosixThread::~PosixThread()
{
}

int
PosixThread::Start(ThreadTask **portAddress)
{
    _portAddress = portAddress;

    /*
      pthread_attr_t thread_attr;
      int status = pthread_attr_init(&thread_attr);
      if(status != 0){
         cerr << "Error when creating thread attribute"<<endl;
         abort();
      }
      status = pthread_attr_setdetachstate(&thread_attr,
                                PTHREAD_CREATE_DETACHED);
      if(status != 0){
         cerr << "Error when creating detached thread attribute"<<endl;
   abort();
   }
   return pthread_create(&_posix_thread,&thread_attr,worker_function,this);
   */

    return pthread_create(&_posix_thread, NULL, worker_function, this);
}

int
PosixThread::Cancel()
{
    return pthread_cancel(_posix_thread);
}

int
PosixThread::Join()
{
    void *result;
    return pthread_join(_posix_thread, &result);
}

/*
void
PosixThread::CleanUpPop(bool execute)
{
   pthread_cleanup_pop(execute);
}
*/

#include "ThreadInitialise.h"
#include "PosixMutexAndCondition.h"

int
PosixThread::WorkerFunction()
{
    if (*_portAddress == NULL)
    {
        cerr << "error: _portAddress points to NULL" << endl;
        return -1;
    }

    int diagnostics;
    // localMutex->CleanUpPush();
    // this is not precisely object oriented, but unavoidable (!?)
    // if pthread_cleanup_push and pthread_cleanup_pop are macros
    PosixMutexAndCondition *p_localMutex = dynamic_cast<PosixMutexAndCondition *>(_localMutex);
    PosixMutexAndCondition *p_globalMutex = dynamic_cast<PosixMutexAndCondition *>(_globalMutex);

    pthread_cleanup_push(PosixMutexAndCondition::cleanup_handler,
                         (void *)&(p_localMutex->_mutex));
    diagnostics = _localMutex->Lock();
    if (diagnostics != 0)
    {
        cerr << "Thread::WorkerFunction: locking local mutex failed" << endl;
        return diagnostics;
    }

    delete (*_portAddress);
    *_portAddress = NULL;

    diagnostics = p_localMutex->Signal();

    while (1)
    {
        while ((*_portAddress) == NULL
               || (*_portAddress)->IsWaitingForGathering())
        {
            _localMutex->Wait();
        }
        (*_portAddress)->Solve();
        (*_portAddress)->WaitForGathering(true);
        // globalMutex->CleanUpPush();
        pthread_cleanup_push(PosixMutexAndCondition::cleanup_handler,
                             (void *)&(p_globalMutex->_mutex));
        diagnostics = _globalMutex->Lock();
        if (diagnostics != 0)
        {
            cerr << "Thread::WorkerFunction: locking global mutex failed" << endl;
            return diagnostics;
        }
        _doneStack->push(_label);
        _globalMutex->Signal();
        // CleanUpPop(true); // true -> execute unlocking on globalMutex
        pthread_cleanup_pop(1);
        // diagnostics = _globalMutex->Unlock();
    }
    // this is in fact never executed...
    // CleanUpPop(true); // true -> execute unlocking on localMutex
    pthread_cleanup_pop(1);
    // _localMutex->Unlock();
    return 0;
}
