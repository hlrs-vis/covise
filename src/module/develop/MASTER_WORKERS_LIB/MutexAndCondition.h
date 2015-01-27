/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS MutexAndCondition
//
//  Basis abstract class representing a couple (mutex-condition).
//  A concrete realisation is PosixMutexAndCondition
//
//  Initial version: 03.02.2003 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _MUTEX_AND_CONDITION_H_
#define _MUTEX_AND_CONDITION_H_

class MutexAndCondition
{
public:
    /// destructor
    virtual ~MutexAndCondition()
    {
    }
    /// signal wakes waiting threads
    virtual int Signal() = 0;
    /// Wait starts waiting on a condition, and returns with the mutex locked
    virtual int Wait() = 0;
    /// Lock locks the mutex
    virtual int Lock() = 0;
    /// Unlock unlocks the mutex
    virtual int Unlock() = 0;

private:
};
#endif
