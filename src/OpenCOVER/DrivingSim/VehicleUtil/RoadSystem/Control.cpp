/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Control.h"

Control::Control(TrafficLightSignal *signal_, const std::string &type_)
    : signal(signal_)
    , type(type_)
    , greenState(false)
    , yellowState(false)
    , redState(false)
{
}

void Control::switchGreenLight(bool state)
{
    greenState = state;
    switch (state)
    {
    case true:
        signal->switchGreenSignal(TrafficLightSignal::ON);
        break;
    case false:
        signal->switchGreenSignal(TrafficLightSignal::OFF);
        break;
    }
}

void Control::switchYellowLight(bool state)
{
    yellowState = state;
    switch (state)
    {
    case true:
        signal->switchYellowSignal(TrafficLightSignal::ON);
        break;
    case false:
        signal->switchYellowSignal(TrafficLightSignal::OFF);
        break;
    }
}

void Control::switchRedLight(bool state)
{
    redState = state;
    switch (state)
    {
    case true:
        signal->switchRedSignal(TrafficLightSignal::ON);
        break;
    case false:
        signal->switchRedSignal(TrafficLightSignal::OFF);
        break;
    }
}
