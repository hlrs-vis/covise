/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Mutex.h"

#include <stdio.h>
#include <iostream>
#include <string>

Mutex::Mutex()
{
#ifndef _WIN32
    _mutex = new pthread_mutex_t;
#else
    _mutex = new HANDLE;
#endif
}

Mutex::~Mutex()
{
    delete _mutex;
}

int Mutex::init()
{
#ifndef _WIN32
    return pthread_mutex_init(_mutex, NULL);
    ;
#else
    *_mutex = CreateMutex(0, FALSE, 0);
    return (*_mutex == 0);
#endif
    return -1;
}

int Mutex::lock()
{
#ifndef _WIN32
    return pthread_mutex_lock(_mutex);
#else
    return (WaitForSingleObject(*_mutex, INFINITE) == WAIT_FAILED ? 1 : 0);
#endif
    return -1;
}

int Mutex::trylock()
{
#ifndef _WIN32
    return pthread_mutex_trylock(_mutex);
#else
    return (WaitForSingleObject(*_mutex, INFINITE) == WAIT_FAILED ? 1 : 0);
#endif
    return -1;
}

int Mutex::unlock()
{
#ifndef _WIN32
    return pthread_mutex_unlock(_mutex);
#else
    return (ReleaseMutex(*_mutex) == 0);
#endif
    return -1;
}
