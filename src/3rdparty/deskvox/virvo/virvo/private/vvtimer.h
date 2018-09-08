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

#ifndef VV_PRIVATE_TIMER_H
#define VV_PRIVATE_TIMER_H

#include <boost/chrono/chrono.hpp>

namespace virvo
{

//------------------------------------------------------------------------------
// Timer
//------------------------------------------------------------------------------

class Timer
{
public:
    typedef boost::chrono::high_resolution_clock clock;
    typedef clock::time_point time_point;
    typedef clock::duration duration;

public:
    Timer()
        : start_(clock::now())
        , lap_(start_)
    {
    }

    void reset()
    {
        start_ = clock::now();
        lap_ = start_;
    }

    duration elapsed() const {
        return clock::now() - start_;
    }

    double elapsedSeconds() const {
        return toSeconds(elapsed());
    }

    duration lap()
    {
        time_point now = clock::now();
        duration d = now - lap_;
        lap_ = now;
        return d;
    }

    double lapSeconds() {
        return toSeconds(lap());
    }

private:
    static double toSeconds(duration d) {
        return boost::chrono::duration<double>(d).count();
    }

private:
    time_point start_;
    time_point lap_;
};

//------------------------------------------------------------------------------
// FrameCounter
//------------------------------------------------------------------------------

class FrameCounter
{
public:
    typedef boost::chrono::high_resolution_clock clock;
    typedef clock::time_point time_point;
    typedef clock::duration duration;

public:
    FrameCounter()
        : start_(clock::now())
        , counter_(0)
        , fps_(0.0)
    {
    }

    double getFPS() const {
        return fps_;
    }

    double registerFrame()
    {
        ++counter_;

        time_point now = clock::now();

        double elapsed = toSeconds(now - start_);
        if (elapsed > 0.5/*sec*/)
        {
            fps_ = counter_ / elapsed;
            start_ = now;
            counter_ = 0;
        }

        return fps_;
    }

private:
    static double toSeconds(duration d) {
        return boost::chrono::duration<double>(d).count();
    }

private:
    // Time-point of last call to updateFPS
    time_point start_;
    // Number of frames since the last call to updateFPS
    unsigned counter_;
    // Current frames-per-second
    double fps_;
};

} // namespace virvo

#endif // !VV_PRIVATE_TIMER_H
