/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS ThreadInitialise
//
//  This describes a subtype of ThreadTask instrumental
//  in the correct creation of the team of worker threads.
//  You will never use this class.
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _THREAD_INITIALISER_H_
#define _THREAD_INITIALISER_H_

#ifdef _STANDARD_C_PLUS_PLUS
#include <stack>
using namespace std;
#else
#include <stack.h>
#endif

#include "ThreadTask.h"

// class MutexAndCondition;

class ThreadInitialise : public ThreadTask
{
public:
    /// constructor
    ThreadInitialise();
    /// destructor
    virtual ~ThreadInitialise();

protected:
    /// Solve does nothing
    virtual void Solve();

private:
};
#endif
