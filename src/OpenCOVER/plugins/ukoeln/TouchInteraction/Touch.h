/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Touch.h

#pragma once

#include <iostream>
#include <iomanip>
#include <map>

//
// This class represents a single touch point
//
struct Touch
{
    friend class TouchInteraction;

    enum State
    {
        Undefined,
        Pressed,
        Moved,
        Released
    };

    Touch()
        : id(-1)
        , time(0.0)
        , x()
        , y()
        , state(Undefined)
    {
    }

    Touch(int id, double time, float x, float y, State state)
        : id(id)
        , time(time)
        , x(x)
        , y(y)
        , state(state)
    {
    }

    // Unique touch point identifier
    // Note: a valid id must be non-negative
    int id;
    // The last time this touch point was updated
    double time;
    // Current touch point position
    float x;
    float y;

private:
    State state; // Used internally by TouchInteraction
};

typedef std::map<int /*id*/, Touch> Touches;

inline std::ostream &operator<<(std::ostream &stream, Touch const &touch)
{
    stream << "[" << touch.id << " (" << touch.x << ", " << touch.y << ")]";
    return stream;
}

inline std::ostream &operator<<(std::ostream &stream, Touches const &touches)
{
    for (Touches::const_iterator it = touches.begin(); it != touches.end(); ++it)
    {
        stream << it->second;
    }
    return stream;
}
