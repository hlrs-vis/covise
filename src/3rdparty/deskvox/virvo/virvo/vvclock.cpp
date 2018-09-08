// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include "vvplatform.h"

#include <iostream>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvclock.h"

#ifdef _WIN32
clock_t vvClock::baseTime = 0;
LARGE_INTEGER vvClock::freq;
bool vvClock::useQueryPerformance = false;
#else
timeval vvClock::baseTime = { 0, 0 };
#endif
bool vvClock::initialized = false;

void vvClock::init()
{
#ifdef _WIN32
  baseTime = clock();
  useQueryPerformance = (QueryPerformanceFrequency(&freq)) ? true : false;
#else
  gettimeofday(&baseTime, NULL);
#endif
  initialized = true;
}

double vvClock::getTime()
{
  if(!initialized) init();

#ifdef _WIN32
  if (useQueryPerformance)
  {
	LARGE_INTEGER currTimeQP;
    QueryPerformanceCounter((LARGE_INTEGER*)&currTimeQP);
	return double(currTimeQP.QuadPart) / double(freq.QuadPart);
  }
  else
  {
    clock_t currTime;
    currTime = clock();
    return double(currTime - baseTime / CLOCKS_PER_SEC);
  }
#else
  timeval currTime = {0,0};
  gettimeofday(&currTime, NULL);
  return double(currTime.tv_sec-baseTime.tv_sec) + double(currTime.tv_usec-baseTime.tv_usec) / 1000000.0;
#endif
}

//----------------------------------------------------------------------------
/// Constructor. Initializes time variable with zero.
vvStopwatch::vvStopwatch()
{
  lastTime = 0.0f;
  baseTime = 0.0f;
}

//----------------------------------------------------------------------------
/// Start or restart measurement but don't reset counter.
void vvStopwatch::start()
{
  baseTime = float(vvClock::getTime());
  lastTime = 0.0f;
}

//----------------------------------------------------------------------------
/// Return the time passed since the last start command [seconds].
float vvStopwatch::getTime()
{
  float dt = float(vvClock::getTime()) - baseTime;
  lastTime = dt;
  return dt;
}

//----------------------------------------------------------------------------
/// Return the time passed since the last getTime or getDiff command [seconds].
float vvStopwatch::getDiff()
{
  float last = lastTime;
  return getTime() - last;
}

vvTimer::vvTimer(const char *label)
: label(label)
{
  watch.start();
}

vvTimer::~vvTimer()
{
  if (label)
  {
    std::cerr << label << ": " << watch.getTime() * 1000.0f << " ms" << std::endl;
  }
  else
  {
    std::cerr << watch.getTime() * 1000.0f << " ms" << std::endl;
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
