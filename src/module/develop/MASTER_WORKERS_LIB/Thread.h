/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Thread
//
//  Abstract class defining the interface of a thread
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _COVISE_THREAD_H_
#define _COVISE_THREAD_H_

class ThreadTask;
class MutexAndCondition;

#ifdef _STANDARD_C_PLUS_PLUS
#include <stack>
using namespace std;
#else
#include <stack.h>
#endif

class Thread
{
public:
    /// constructor
    Thread();
    /// destructor
    virtual ~Thread();
    /// Start gives the thread the address through which
    /// it may find the tasks that are assigned to it and
    /// also runs the worker function: WorkerFunction
    virtual int Start(ThreadTask **portAddress) = 0;
    /// Cancel cancels a thread
    virtual int Cancel() = 0;
    /// Joins a thread
    virtual int Join() = 0;
    /// SetInfo sets label identification and pointers to
    /// objects used by the thread
    void SetInfo(int label, MutexAndCondition *local, MutexAndCondition *global,
                 stack<int> *done);

protected:
    /// CleanUpPop removes (and may execute) an unlock action from the stack
    virtual void CleanUpPush(MutexAndCondition *mutCond);
    /// CleanUpPop removes (and may execute) an unlock action from the stack
    virtual void CleanUpPop(bool execute);

    ThreadTask **_portAddress;

    int _label;
    MutexAndCondition *_localMutex;
    MutexAndCondition *_globalMutex;
    stack<int> *_doneStack;

private:
    /// WorkerFunction defines how the MutexAndCondition objects
    /// are used in order to carry out the computation at issue.
    /// The default realisation has to be redone in the case of
    /// posix threads because the pthread_cleanup are in fact macros
    int WorkerFunction();
};
#endif
