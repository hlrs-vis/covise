/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "MasterAndWorkers.h"
#include "Thread.h"
#include "MutexAndCondition.h"
#include "ThreadTask.h"
#include "TaskGroup.h"

#ifdef _STANDARD_C_PLUS_PLUS
#include <memory>
using namespace std;
#endif

int
MasterAndWorkers::compute()
{
    // threads are started
    int diagnostics = StartThreads();
    if (diagnostics != 0)
    {
        TerminateThreads();
        return FailureValue();
    }
// *taskGroup manages the problem:
// it creates ThreadTasks and assigns them to the threads
#ifdef _STANDARD_C_PLUS_PLUS
    auto_ptr<TaskGroup> taskGroup(CreateTaskGroup());
    if (taskGroup.get() == NULL)
    {
#else
    TaskGroup *taskGroup = CreateTaskGroup();
    if (taskGroup == NULL)
    {
#endif
        TerminateThreads();
        return FailureValue();
    }
    while (!taskGroup->End()) // when the whole problem is over
    {
        taskGroup->CreateThreadTasks(); // ThreadTasks are created for this iteration
        // all ThreadTasks of this iteration are done
        while (!taskGroup->AllThreadTaskFinished())
        {
            if (_crewSize > 0) // if multithreading is necessary at all
            {
                while (taskGroup->SomeThreadTasksUnserviced()
                       && _lazyThreads.size() > 0)
                {
                    int label = _lazyThreads.top(); // get identification of a lazy thread
                    _lazyThreads.pop();
                    int diagnostics = _workerMutexes[label]->Lock();
                    if (diagnostics != 0)
                    {
                        TerminateThreads();
                        return FailureValue();
                    }
                    //assign a task to a thread
                    taskGroup->AssignTask(&_threadPorts[label]);
                    diagnostics = _workerMutexes[label]->Signal();
                    if (diagnostics != 0)
                    {
                        TerminateThreads();
                        return FailureValue();
                    }
                    diagnostics = _workerMutexes[label]->Unlock();
                    if (diagnostics != 0)
                    {
                        TerminateThreads();
                        return FailureValue();
                    }
                }
                while (_doneThreads.size() == 0) // wait until some thread(s) are done
                {
                    (*_globalMutex)->Wait();
                }
                do // "read" and process work done by threads
                {
                    int label = _doneThreads.top(); // get identification of one of these threads
                    _doneThreads.pop();
                    _lazyThreads.push(label);
                    int diagnostics = _workerMutexes[label]->Lock();
                    if (diagnostics != 0)
                    {
                        TerminateThreads();
                        return FailureValue();
                    }
                    taskGroup->GatherThreadTask(_threadPorts[label]);
                    _threadPorts[label] = NULL;

                    diagnostics = _workerMutexes[label]->Unlock();
                    if (diagnostics != 0)
                    {
                        TerminateThreads();
                        return FailureValue();
                    }
                } while (_doneThreads.size() > 0);
            }
            else // if _crewSize == 0 -> do not use multithreading
            {
                taskGroup->Solve();
            }
        }
        taskGroup->GatherStep(); // process all ThreadTasks of this iteration
    }
    taskGroup->GatherAll(); // processing when all iterations are done
#ifndef _STANDARD_C_PLUS_PLUS
    delete taskGroup;
#endif
    if (_crewSize > 0)
    {
        (*_globalMutex)->Unlock();
    }
    diagnostics = TerminateThreads();
    if (diagnostics != 0)
    {
        return FailureValue();
    }
    return SuccessValue();
}

int
MasterAndWorkers::TerminateThreads(int noThreads)
{
    int thread;

    if (_threads != NULL)
    {
        if (noThreads < 0 || noThreads > _crewSize)
            noThreads = _crewSize;
        for (thread = 0; thread < noThreads; ++thread)
        {
            int diagnostics = _threads[thread]->Cancel();
            if (diagnostics != 0)
            {
                cerr << "MasterAndWorkers::TerminateThreads: cancelling thread failed" << endl;
                return diagnostics;
            }
            diagnostics = _threads[thread]->Join();
            if (diagnostics != 0)
            {
                cerr << "MasterAndWorkers::TerminateThreads: joining thread failed" << endl;
                return diagnostics;
            }
        }
    }

    if (_threads != NULL)
    {
        for (thread = 0; thread < _crewSize; ++thread)
        {
            delete _threads[thread];
            delete _workerMutexes[thread];
        }
    }
    delete[] _threads;
    _threads = NULL;
    delete[] _workerMutexes;
    _workerMutexes = NULL;

    if (_globalMutex != NULL)
    {
        delete (*_globalMutex);
    }
    delete[] _globalMutex;
    _globalMutex = NULL;

    return 0;
}

MasterAndWorkers::MasterAndWorkers()
    : _threads(NULL)
    , _workerMutexes(NULL)
    , _globalMutex(NULL)
{
}

MasterAndWorkers::~MasterAndWorkers()
{
    TerminateThreads();
}

void
MasterAndWorkers::SetCrewSizeExt(int crewSize)
{
    _crewSize = crewSize;
}

int
MasterAndWorkers::SetConcurrency() const
{
    return 0;
}

int
MasterAndWorkers::FailureValue() const
{
    return -1;
}

int
MasterAndWorkers::SuccessValue() const
{
    return 0;
}
#include "ThreadInitialise.h"
#include "PosixMutexAndCondition.h"

// start threads and wait until they have got
// their ports for communication
int
MasterAndWorkers::StartThreads()
{
    SetCrewSizeInt();

    if (_crewSize > 0)
    {
        _threads = CreateThreads(_crewSize); // threads are really not started here,
        // but later in this function
        _workerMutexes = CreateMutexAndConditions(_crewSize);

        // _globalMutex is
        _globalMutex = CreateMutexAndConditions(1);
        // used for synchonising access to _doneThreads

        int diagnostics = (*_globalMutex)->Lock();
        if (diagnostics != 0)
        {
            int thread;
            for (thread = 0; thread < _crewSize; ++thread)
            {
                delete _threads[thread];
            }
            delete[] _threads;
            _threads = NULL;
            TerminateThreads(); // this also destroys mutexes,
            // on the other hand it would not be
            // a good idea cancelling and joining threads,
            // which have not yet been started!?
            cerr << "MasterAndWorkers::StartThreads: locking global mutex failed" << endl;
            return diagnostics;
        }

        // init stacks
        while (!_lazyThreads.empty())
        {
            _lazyThreads.pop();
        }
        while (!_doneThreads.empty())
        {
            _doneThreads.pop();
        }

        // all threads are initially lazy
        int thread;
        for (thread = 0; thread < _crewSize; ++thread)
        {
            _lazyThreads.push(thread);
        }

        // create thread ports
        _threadPorts.clear();
        _threadPorts.resize(_crewSize, NULL);

        SetConcurrency();

        // inform threads of the mutex they work with and of global mutex
        for (thread = 0; thread < _crewSize; ++thread)
        {
            _threadPorts[thread] = new ThreadInitialise();
            int diagnostics = _workerMutexes[thread]->Lock();
            if (diagnostics != 0)
            {
                TerminateThreads(thread);
                cerr << "MasterAndWorkers::StartThreads: locking worker mutex failed" << endl;
                return diagnostics;
            }
            // here are the threads really started
            // here they get the information they need for synchnosisation:
            // global and private mutex and the address of their communication
            // port
            _threads[thread]->SetInfo(thread, _workerMutexes[thread],
                                      *_globalMutex,
                                      &_doneThreads);

            diagnostics = _threads[thread]->Start(&_threadPorts[thread]);
            if (diagnostics != 0)
            {
                TerminateThreads(thread);
                cerr << "MasterAndWorkers::StartThreads: starting worker thread failed" << endl;
                return diagnostics;
            }
            // wait until the thread has performed its initialisation
            while (_threadPorts[thread] != NULL)
            {
                _workerMutexes[thread]->Wait();
            }
            diagnostics = _workerMutexes[thread]->Unlock();
            if (diagnostics != 0)
            {
                TerminateThreads(thread);
                cerr << "MasterAndWorkers::StartThreads: unlocking worker mutex failed" << endl;
                return diagnostics;
            }
        }
    }
    return 0;
}
