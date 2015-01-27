/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef __OCU_UTIL_TIMER_H__
#define __OCU_UTIL_TIMER_H__

#include <cuda.h>

#ifdef _WIN32

#include <windows.h>

namespace ocu
{

struct CPUTimer
{
    LARGE_INTEGER _start_time, _end_time;

    CPUTimer()
    {
    }
    ~CPUTimer()
    {
    }

    void start()
    {
        QueryPerformanceCounter(&_start_time);
    }
    void stop()
    {
        QueryPerformanceCounter(&_end_time);
    }

    float elapsed_sec()
    {
        LARGE_INTEGER diff;
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        diff.QuadPart = _end_time.QuadPart - _start_time.QuadPart;
        return (float)diff.QuadPart / (float)freq.QuadPart;
    }

    float elapsed_ms()
    {
        return elapsed_sec() * 1000.0f;
    }
};

} // end namespace

#else // non-windows environment, use gettimeofday()

#include <sys/time.h>

namespace ocu
{

struct CPUTimer
{
    double _start_time, _end_time;

    CPUTimer()
    {
    }
    ~CPUTimer()
    {
    }

    void start()
    {
        struct timeval t;
        gettimeofday(&t, 0);
        _start_time = t.tv_sec + (1e-6 * t.tv_usec);
    }
    void stop()
    {
        struct timeval t;
        gettimeofday(&t, 0);
        _end_time = t.tv_sec + (1e-6 * t.tv_usec);
    }

    float elapsed_sec()
    {
        return (float)(_end_time - _start_time);
    }

    float elapsed_ms()
    {
        return (float)1000 * (_end_time - _start_time);
    }
};

} // end namespace

#endif

namespace ocu
{

struct GPUTimer
{
    void *e_start;
    void *e_stop;

    GPUTimer();
    ~GPUTimer();

    void start();
    void stop();

    float elapsed_ms();
    float elapsed_sec()
    {
        return elapsed_ms() / 1000.0f;
    }
};

void disable_timing();
void enable_timing();

} // end namespace

#endif
