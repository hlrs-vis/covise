/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TaskGroup.h"
#include "ThreadTask.h"

void
TaskGroup::Solve()
{
    int task;
    for (task = 0; task < _tasks.size(); ++task)
    {
        _tasks[task]->Solve();
    }
}

TaskGroup::~TaskGroup()
{
}
