/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ThreadTask.h"

bool
ThreadTask::IsWaitingForGathering() const
{
    return _isWaitingForGathering;
}

void
ThreadTask::WaitForGathering(bool state)
{
    _isWaitingForGathering = state;
}

ThreadTask::ThreadTask()
    : _isWaitingForGathering(false)
{
}

ThreadTask::~ThreadTask()
{
}
