/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// FPSCounter.cpp

#include "FPSCounter.h"

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

FPSCounter::FPSCounter()
    : m_numframes(0)
    , m_start(0.0)
    , m_last(0.0)
    , m_fps(0.0)
{
#ifdef _WIN32
    timeBeginPeriod(1);
#endif
}

FPSCounter::~FPSCounter()
{
#ifdef _WIN32
    timeEndPeriod(1);
#endif
}

double FPSCounter::getTime()
{
#ifdef _WIN32
    return (1.0 / 1000.0) * (double)timeGetTime();
#else
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (double)tv.tv_sec + 1.e-6 * tv.tv_usec;
#endif
}

double FPSCounter::registerFrame()
{
    // increase frame counter
    m_numframes++;

    double now = getTime();

    // returns the time in seconds since the last call to registerFrame()
    double elapsed = now - m_last;

    m_last = now;

    return elapsed;
}

double FPSCounter::update()
{
    double now = getTime();

    double elapsed = now - m_start;

    // compute current fps
    m_fps = (double)m_numframes / elapsed;

    // reset state
    m_numframes = 0;
    m_start = now;

    return m_fps;
}
