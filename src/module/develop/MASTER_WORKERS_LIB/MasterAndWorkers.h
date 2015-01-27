/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS MasterAndWorkers
//
//  Basis abstract class for multithreaded computations for
//  master-worker computations
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _MASTER_AND_WORKERS_H_
#define _MASTER_AND_WORKERS_H_

#ifdef _STANDARD_C_PLUS_PLUS
#include <stack>
#include <vector>
using namespace std;
#else
#include <stack.h>
#include <vector.h>
#endif

class Thread;
class ThreadTask;
class MutexAndCondition;
class TaskGroup;

class MasterAndWorkers
{
public:
    /// constructor
    MasterAndWorkers();
    /// destructor
    virtual ~MasterAndWorkers();
    /// compute is called by the client code in order to have the problem solved
    virtual int compute();
    /// SetCrewSizeExt version used by clients
    virtual void SetCrewSizeExt(int crewSize);

protected:
    /// SetCrewSizeInt used in compute
    virtual void SetCrewSizeInt() = 0;
    /// fabricates a TaskGroup object defining the problem at issue
    virtual TaskGroup *CreateTaskGroup() = 0;
    /// fabricates an array of $number pointers to Threads
    virtual Thread **CreateThreads(int number) = 0;
    /// fabricates an array of $number pointers to MutexAndConditions
    // fabric method
    virtual MutexAndCondition **CreateMutexAndConditions(int number) = 0;
    /// SetConcurrency's default version does nothing, but you may
    /// have to redefine it on some platforms in order to use several CPUs
    virtual int SetConcurrency() const;
    /// FailureValue returns a default value of -1, you may change
    /// this behaviour at your convenience
    virtual int FailureValue() const;
    /// SuccessValue returns a default value of -1, you may change
    /// this behaviour at your convenience
    virtual int SuccessValue() const;
    /// TerminateThreads finishes all threads (up to noThreads, if
    /// this number is positive) and destroys MutexAndCondition objects
    int TerminateThreads(int noThreads = -1);
    /// CrewSize returns the crew size
    int CrewSize() const
    {
        return _crewSize;
    }

private:
    int StartThreads();

    Thread **_threads;
    MutexAndCondition **_workerMutexes;
    vector<ThreadTask *> _threadPorts; // as many as workers

    MutexAndCondition **_globalMutex;

    stack<int> _lazyThreads;
    stack<int> _doneThreads;
    int _crewSize;
};
#endif
