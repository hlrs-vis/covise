/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MUTEX_H
#define _MUTEX_H

#ifndef _WIN32
#include <pthread.h>
#else
#include <windows.h>
#include <process.h>
#endif

class Mutex
{
public:
    Mutex();
    virtual ~Mutex();

    int init();
    int lock();
    int unlock();
    int trylock();

private:
#ifndef _WIN32
    pthread_mutex_t *_mutex;
#else
    HANDLE *_mutex;
#endif
};

#endif
