/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS MyTaskGroup
//
//  This class illustrates how you may define a multithreaded computation
//  by class derivation from TaskGroup. This class defines how the
//  coomputation is distributed in smaller independent parts which are
//  solved by and assigned to a group of threads.
//  See MyTaskGroup.cpp for comments.
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _MY_TASK_GROUP_H_
#define _MY_TASK_GROUP_H_

#include "TaskGroup.h"

class MyThreadTask;

class MyTaskGroup : public TaskGroup
{
public:
    MyTaskGroup();
    virtual void CreateThreadTasks();
    virtual void AssignTask(ThreadTask **portAdress);
    virtual bool AllThreadTaskFinished();
    virtual bool SomeThreadTasksUnserviced();
    virtual void GatherThreadTask(ThreadTask *portAdress);
    virtual void GatherStep();
    virtual void GatherAll();
    virtual bool End();

private:
    const int _numtasks;
    bool _end;
    int _no_finished;
    MyThreadTask *_my_tasks;
};
#endif
