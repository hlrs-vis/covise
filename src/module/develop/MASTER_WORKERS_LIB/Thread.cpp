/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Thread.h"
#include "ThreadInitialise.h"

Thread::Thread()
    : _portAddress(NULL)
{
}

Thread::~Thread()
{
}

#include "MutexAndCondition.h"

void
Thread::SetInfo(int label, MutexAndCondition *local, MutexAndCondition *global,
                stack<int> *done)
{
    _label = label;
    _localMutex = local;
    _globalMutex = global;
    _doneStack = done;
}

int
Thread::WorkerFunction()
{
    if (*_portAddress == NULL)
    {
        cerr << "error: _portAddress points to NULL" << endl;
        return -1;
    }

    int diagnostics;
    CleanUpPush(_localMutex);
    diagnostics = _localMutex->Lock();
    if (diagnostics != 0)
    {
        cerr << "Thread::WorkerFunction: locking local mutex failed" << endl;
        return diagnostics;
    }

    delete (*_portAddress);
    *_portAddress = NULL;

    diagnostics = _localMutex->Signal();

    while (1)
    {
        while ((*_portAddress) == NULL
               || (*_portAddress)->IsWaitingForGathering())
        {
            _localMutex->Wait();
        }
        (*_portAddress)->Solve();
        (*_portAddress)->WaitForGathering(true);
        CleanUpPush(_globalMutex);
        diagnostics = _globalMutex->Lock();
        if (diagnostics != 0)
        {
            cerr << "Thread::WorkerFunction: locking global mutex failed" << endl;
            return diagnostics;
        }
        _doneStack->push(_label);
        _globalMutex->Signal();
        CleanUpPop(true); // true -> execute unlocking on globalMutex
        // diagnostics = globalMutex->Unlock();
    }
    // this is in fact never executed...
    CleanUpPop(true); // true -> execute unlocking on localMutex
    return 0;
}

void
Thread::CleanUpPop(bool)
{
}

void
Thread::CleanUpPush(MutexAndCondition *)
{
}
