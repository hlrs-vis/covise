/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MyThreadTask.h"

#ifdef _STANDARD_C_PLUS_PLUS
#include <iostream>
using namespace std;
#else
#include <iostream.h>
#endif

// this variable is used so as to generate different
// arrays of integer numbers in Solve when different objects
// of this class are instantiated.
unsigned int
    MyThreadTask::_seed_start = 0;

MyThreadTask::MyThreadTask()
    : ThreadTask()
    , _numbers(10)
    , _sum(0)
    , _assigned(false)
{
    _seed = _seed_start;
    ++_seed_start;
}

#include <stdlib.h>

// an array of random numbers between 1 and 10 is generated
// and the sum is computed.
void
MyThreadTask::Solve()
{
    int i;
    _sum = 0;
    for (i = 0; i < _numbers.size(); ++i)
    {
        _numbers[i] = (1 + (int)(10.0 * rand_r(&_seed) / (RAND_MAX + 1.0)));
        _sum += _numbers[i];
    }
}

// Assigned is used by MyTaskGroup in order to test whether
// this task has already been assigned to a thread.
bool
MyThreadTask::Assigned() const
{
    return _assigned;
}

// Assign is used by MyTaskGroup in order to tag
// a MyThreadTask object as having been assigned
// to a thread.
void
MyThreadTask::Assign(bool assign)
{
    _assigned = assign;
}

// CheckCalculation is used by the main thread in order
// to test that the synchronisation mechnisms are working
// properly. It is used by MyTaskGroup::GatherThreadTask.
bool
MyThreadTask::CheckCalculation()
{
    int i;
    int sum = 0;
    for (i = 0; i < _numbers.size(); ++i)
    {
        cerr << _numbers[i];
        sum += _numbers[i];
        if (i < _numbers.size() - 1)
        {
            cerr << '+';
        }
        else
        {
            cerr << '=';
        }
    }
    cerr << _sum << '=' << sum << endl;
    return (sum == _sum);
}
