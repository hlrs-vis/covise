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

#ifndef VV_CLOCK_H
#define VV_CLOCK_H

#include "vvplatform.h"

#include "vvexport.h"

/** static clock function returning current time in seconds
  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
  */
class VIRVO_FILEIOEXPORT vvClock
{
public:
  static double getTime();

private:
  static void init();

  static bool initialized;
#ifdef WIN32
  static clock_t baseTime;                             ///< system time
  static bool useQueryPerformance;                     ///< true=use QueryPerformance API
  static LARGE_INTEGER freq;                     ///< system time when using QueryPerformance API
#else
  static timeval baseTime;                             ///< system time when stop watch was triggered last
#endif
};

/** System independent implementation of a stop watch.
  The stop watch can be started, stopped, reset, and its time
  can be read. Stopping does not reset the watch. Initially, the
  stop watch is reset. Resetting stops the watch. Reading the time
  does not stop the watch. <P>

  Example usage:<PRE>

  vvStopwatch sw = new vvStopwatch(); // create new stop watch instance
  sw->start();                        // reset counter
  // *** do something ***
float time1 = sw->getTime();        // get an intermediate time but don't stop counting
// *** do something ***
float time2 = sw->getTime();        // get another intermediate time
// *** do something ***
float time3 = sw->getTime();        // get the final time
cout << "The total time measured is: " << time3 << " seconds." << endl;
delete sw;                          // remove stop watch from memory
</PRE>

@author Jurgen Schulze (jschulze@ucsd.edu)
*/
class VIRVO_FILEIOEXPORT vvStopwatch
{
  private:

    float baseTime;
    float lastTime;

  public:
    vvStopwatch();
    void  start();
    float getTime();
    float getDiff();
};

class VIRVO_FILEIOEXPORT vvTimer
{
  public:
    vvTimer(const char *label = 0);
    ~vvTimer();
  private:
    const char *label;
    vvStopwatch watch;
};

#define TIME_BLOCK vvTimer automatic_timer(__PRETTY_FUNCTION__)
#define TIME_BLOCK_BEGIN vvTimer* itIsVeryUnlikelyThatSomeoneWillChooseThisNameForHisVariable = new vvTimer();
#define TIME_BLOCK_END delete itIsVeryUnlikelyThatSomeoneWillChooseThisNameForHisVariable;

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
