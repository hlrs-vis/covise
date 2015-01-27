/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS ThreadTask
//
//  Basis abstract class for "microtasks" that are assigned to threads
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _THREAD_TASK_H_
#define _THREAD_TASK_H_

class ThreadTask
{
public:
    /// constructor
    ThreadTask();
    /// destructor
    virtual ~ThreadTask();
    /// Solve carries out a "micro-task"
    virtual void Solve() = 0;
    /// IsWaitingForGathering returns true when a thread is done with this
    /// micro-task and it has to be read in
    bool IsWaitingForGathering() const;
    /// WaitForGathering is used by a Thread in order to mark this
    /// micro-task as finished. The main thread may read it in.
    void WaitForGathering(bool state);

private:
    bool _isWaitingForGathering;
};
#endif
