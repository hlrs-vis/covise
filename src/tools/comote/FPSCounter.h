/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// fpscounter.h

#ifndef FPSCOUNTER_H
#define FPSCOUNTER_H

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
class FPSCounter
{
public:
    FPSCounter();
    ~FPSCounter();

    // register a frame
    // returns the frame-delta, ie. the time in seconds since the last call to registerFrame()
    double registerFrame();

    // update and return current fps
    // has to be called every now and then, eg.  using a timer
    double update();

    // returns current fps
    double getFPS() const
    {
        return m_fps;
    }

private:
    // returns the current system time in seconds
    static double getTime();

private:
    // number of frames rendered since the last call to update()
    int m_numframes;
    // last update() time
    double m_start;
    // last registerFrame() time
    double m_last;
    // current frames per second
    double m_fps;
};
#endif
