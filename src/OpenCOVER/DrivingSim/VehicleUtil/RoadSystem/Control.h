/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Control_h
#define Control_h

//#include "Element.h"
#include "RoadSignal.h"

class Control
{
public:
    Control(TrafficLightSignal *, const std::string &);

    TrafficLightSignal *getSignal()
    {
        return signal;
    }
    const std::string &getControlType()
    {
        return type;
    }

    void switchGreenLight(bool);
    void switchYellowLight(bool);
    void switchRedLight(bool);

    bool getGreenLight()
    {
        return greenState;
    }
    bool getYellowLight()
    {
        return yellowState;
    }
    bool getRedLight()
    {
        return redState;
    }

protected:
    TrafficLightSignal *signal;
    std::string type;
    bool greenState;
    bool yellowState;
    bool redState;
};

#endif
