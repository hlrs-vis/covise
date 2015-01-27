/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MyTaskGroup.h"
#include "MyThreadTask.h"

// _numtasks defines the number of ThreadTask objects
// that are generated for the thread team

// _no_finished is a counter of finished ThreadTasks.
// It is used by AllThreadTaskFinished to determine
// when all ThreadTasks of the iteration at issue are done.
// In our problem a single iteration is performed.

// _end is a flag used by End in order to decide whether
// a further iteration is requiered.
MyTaskGroup::MyTaskGroup()
    : _numtasks(20)
    , _no_finished(0)
    , _end(false) // false because otherwise
// we would have 0 iterations
{
}

// create the ThreadTasks which are assigned to the threads
void
MyTaskGroup::CreateThreadTasks()
{
    _my_tasks = new MyThreadTask[_numtasks];
    int i;
    for (i = 0; i < _numtasks; ++i)
    {
        _tasks.push_back(_my_tasks + i); // DO NOT FORGET: load the container _tasks
        // with pointers to our tasks
    }
}

// assign a ThreadTask to a thread.
// portAdress is the address of a pointer to ThreadTask. This
// address is used by one of the threads in order to get the information
// what it has to do (calling ThreadTask::Solve).
void
MyTaskGroup::AssignTask(ThreadTask **portAdress)
{
    int i;
    for (i = 0; i < _numtasks; ++i)
    {
        if (!_my_tasks[i].Assigned()) // We look for one ThreadTask which
        {
            // has not yet been assigned
            _my_tasks[i].Assign(true); // We have found one. We mark it as assigned...
            *portAdress = _my_tasks + i; // and assign it
            return;
        }
    }
}

// returns true when all ThreadTasks of this iteration have been
// completed
bool
MyTaskGroup::AllThreadTaskFinished()
{
    if (_no_finished == _numtasks)
    {
        return true;
    }
    return false;
}

// returns true when in this iteration still
// some ThreadTasks are waiting to be assigned to a thread
bool
MyTaskGroup::SomeThreadTasksUnserviced()
{
    int i;
    for (i = 0; i < _numtasks; ++i)
    {
        if (!_my_tasks[i].Assigned())
        {
            return true;
        }
    }
    return false;
}

// when a thread is done with its assigned ThreadTask, the
// main thread is eventually signalled to do something
// with it. It may have to check its state or use in
// some way the information that has been added by the worker
// thread. In our case we let the main thread check the
// computation done by the worker. This way we are testing
// that the synchonisation machnisms are working properly.
void
MyTaskGroup::GatherThreadTask(ThreadTask *portAdress)
{
    MyThreadTask *myTask = (MyThreadTask *)portAdress;
    if (myTask->CheckCalculation())
    {
        cerr << "The calculation was correct\n" << endl;
    }
    else
    {
        cerr << "The calculation was incorrect: this is a bug\n" << endl;
    }
    ++_no_finished;
}

// do here the required clean-up when all the ThreadTasks
// for an iteration have been done. You might also want
// to create more ThreadTask objects or manipulate those
// you have alredy created in order to start a new iteration.
void
MyTaskGroup::GatherStep()
{
    delete[] _my_tasks;
    _my_tasks = NULL;
    _end = true; // this is used in End in order to decide
    // if another iteration should be started,
    // in our case, we set _end to true as we do not want
    // to start another iteration
}

// cleanup and actions when all iterations have been finished
void
MyTaskGroup::GatherAll()
{
}

// if true, the main threads gets out of the loop of iterations.
// In this example a single iteration is executed
bool
MyTaskGroup::End()
{
    return _end;
}
