/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _STANDARD_C_PLUS_PLUS
#include <iostream>
using namespace std;
#else
#include <iostream.h>
#endif

#include "PosixMutexAndCondition.h"

// initialises the mutex and variable condition
PosixMutexAndCondition::PosixMutexAndCondition()
{
    int status = pthread_mutex_init(&_mutex, NULL);
    if (status == 0)
    {
        _mutex_state = MUTEX_OK;
    }
    else
    {
        _mutex_state = MUTEX_NOT_OK;
    }
    status = pthread_cond_init(&_cond, NULL);
    if (status == 0)
    {
        _cond_state = COND_OK;
    }
    else
    {
        _cond_state = COND_NOT_OK;
    }
}

// destroys the mutex and variable condition
// this code is safe when worker threads have been cancelled
PosixMutexAndCondition::~PosixMutexAndCondition()
{
    if (_mutex_state == MUTEX_OK)
    {
        pthread_mutex_unlock(&_mutex);
        pthread_mutex_destroy(&_mutex);
    }
    if (_cond_state == COND_OK)
        pthread_cond_destroy(&_cond);
}

int
PosixMutexAndCondition::Signal()
{
    if (_cond_state == COND_NOT_OK)
    {
        cerr << "Condition variable is not correctly initialised" << endl;
        return -1;
    }
    return pthread_cond_signal(&_cond);
}

int
PosixMutexAndCondition::Wait()
{
    if (_cond_state == COND_NOT_OK)
    {
        cerr << "Condition variable is not correctly initialised" << endl;
        return -1;
    }
    return pthread_cond_wait(&_cond, &_mutex);
}

int
PosixMutexAndCondition::Lock()
{
    if (_mutex_state == MUTEX_NOT_OK)
    {
        cerr << "Mutex is not correctly initialised" << endl;
        return -1;
    }
    int val = pthread_mutex_lock(&_mutex);
    return val;
}

int
PosixMutexAndCondition::Unlock()
{
    if (_mutex_state == MUTEX_NOT_OK)
    {
        cerr << "Mutex is not correctly initialised" << endl;
        return -1;
    }
    int val = pthread_mutex_unlock(&_mutex);
    return val;
}

void
PosixMutexAndCondition::cleanup_handler(void *arg)
{
    pthread_mutex_t *ptr = static_cast<pthread_mutex_t *>(arg);
    int status = pthread_mutex_unlock(ptr);
    if (status != 0)
    {
        cerr << "PosixMutexAndCondition::cleanup_handler failed" << endl;
    }
}
