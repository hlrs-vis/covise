/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS MyThreadTask
//
//  This class defines a task assigned to a thread.
//  The task is carried out by calling the virtual fucntion Solve.
//  You implement by derivation from ThreadTask.
//
//  One object of this class describes the computation of
//  an array of _numbers random integers and the calculation of
//  their sum. All this is carried out in Solve by a worker thread.
//  Afterwards, the main thread calls CheckCalculation in order
//  to test that everything works as expected.
//
//  See MyThreadTask.cpp for more information.
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _MY_THREAD_TASK_H_
#define _MY_THREAD_TASK_H_

#include "ThreadTask.h"
#ifdef _STANDARD_C_PLUS_PLUS
#include <vector>
using namespace std;
#else
#include <vector.h>
#endif

class MyThreadTask : public ThreadTask
{
public:
    MyThreadTask();
    virtual void Solve();
    bool Assigned() const;
    void Assign(bool assign);
    bool CheckCalculation();

private:
    vector<int> _numbers;
    int _sum;
    bool _assigned;
    unsigned int _seed;
    static unsigned int _seed_start;
};
#endif
