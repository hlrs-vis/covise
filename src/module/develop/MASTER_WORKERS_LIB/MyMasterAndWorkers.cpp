/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MyMasterAndWorkers.h"
#include "MyThreadTask.h"
#include "MyTaskGroup.h"

#include "PosixMutexAndCondition.h"
#include "PosixThread.h"

void
MyMasterAndWorkers::SetCrewSizeInt()
{
    SetCrewSizeExt(5); // CREW SIZE
}

TaskGroup *
MyMasterAndWorkers::CreateTaskGroup()
{
    return new MyTaskGroup();
}

Thread **
MyMasterAndWorkers::CreateThreads(int number)
{
    Thread **ptr = new Thread *[number];
    int i;
    for (i = 0; i < number; ++i)
    {
        ptr[i] = new PosixThread;
    }
    return ptr;
}

MutexAndCondition **
MyMasterAndWorkers::CreateMutexAndConditions(int number)
{
    MutexAndCondition **ptr = new MutexAndCondition *[number];
    int i;
    for (i = 0; i < number; ++i)
    {
        ptr[i] = new PosixMutexAndCondition;
    }
    return ptr;
}

int
MyMasterAndWorkers::SetConcurrency() const
{
#if defined(__sgi)
    pthread_setconcurrency(CrewSize());
#endif
    return 0;
}
