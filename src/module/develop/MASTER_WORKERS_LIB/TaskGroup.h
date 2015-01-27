/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS TaskGroup
//
//  TaskGroup defines the way a multihreaded computation is carried out.
//  The computation consists of a loop over iterations. In each iteration
//  a TaskGroup object generates a list of micro-trasks (ThreadTask objects),
//  which are assigned to threads with the master-worker-pool model.
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _TASK_GROUP_H_
#define _TASK_GROUP_H_

#ifdef _STANDARD_C_PLUS_PLUS
#include <vector>
using namespace std;
#else
#include <vector.h>
#endif

class ThreadTask;

class TaskGroup
{
public:
    /// CreateThreadTasks is fabric method that creates a group of
    /// microtasks and manges a vector of pointers to them (_tasks)
    virtual void CreateThreadTasks() = 0; // internal fabric method
    /// AssignTask assigns one of the values of _tasks to portAdress
    /// so that a thread has something to do: *portAdress = _tasks[onrOfThem]
    virtual void AssignTask(ThreadTask **portAdress) = 0;
    /// AllThreadTaskFinished returns true when all ThreadTasks are ready
    /// in one iteration
    virtual bool AllThreadTaskFinished() = 0;
    /// SomeThreadTasksUnserviced returns true if there are still
    /// ThreadTasks which have not yet been assigned
    virtual bool SomeThreadTasksUnserviced() = 0;
    /// GatherThreadTask defined what the main thread does
    /// with a ThreadTask when the thread has signaled, that it
    /// has finished it
    virtual void GatherThreadTask(ThreadTask *portAdress) = 0;
    /// GatherStep defines what the main thread does when all
    /// ThreadTask for one iteration are done
    virtual void GatherStep() = 0;
    /// GatherAll defines what the main thread does when all iterations
    /// have been completed
    virtual void GatherAll() = 0;
    /// End returns true when all iterations are finished and we are ready
    /// to execute GatherAll
    virtual bool End() = 0;
    /// destructor
    virtual ~TaskGroup();
    /// Solve calls the Solve function for all ThreadTasks of an iteration.
    /// it is used when no multithreading is to be used
    virtual void Solve();

protected:
    vector<ThreadTask *> _tasks;
};
#endif
