/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS MyMasterAndWorkers
//
//  This class illustrates how you manage a pool of threads
//  by derivation from MasterAndWorkers
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _MY_MASTER_AND_WORKERS_H_
#define _MY_MASTER_AND_WORKERS_H_

#include "MasterAndWorkers.h"

class Thread;
class ThreadTask;
class MutexAndCondition;
class TaskGroup;

class MyMasterAndWorkers : public MasterAndWorkers
{
public:
    /// CreateTaskGroup creates a MyTaskGroup object, which
    /// defines the problem to be solved and how it is decomposed
    /// in a group of ThreadTasks.
    virtual TaskGroup *CreateTaskGroup();
    /// CreateThreads creates an array to pointers to PosixThread objects.
    /// This is the pool of threads managed by MyMasterAndWorkers
    virtual Thread **CreateThreads(int number);
    /// CreateMutexAndConditions creates an array to pointers to PosixMutexAndCondition
    /// variables.
    virtual MutexAndCondition **CreateMutexAndConditions(int number);

protected:
    /// SetCrewSizeInt defines the thread pool size (calls public inherited
    /// function SetCreSizeExt(int)
    virtual void SetCrewSizeInt();
    /// SetConcurrency may be important on some plattforms in order to
    /// use several CPUs (pthread_setconcurrency)
    virtual int SetConcurrency() const;

private:
};
#endif
