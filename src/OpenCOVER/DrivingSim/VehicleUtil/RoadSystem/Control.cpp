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
    if (state)
        signal->switchGreenSignal(TrafficLightSignal::ON);
    else
        signal->switchGreenSignal(TrafficLightSignal::OFF);
}

void Control::switchYellowLight(bool state)
{
    yellowState = state;
    if (state)
        signal->switchYellowSignal(TrafficLightSignal::ON);
    else
        signal->switchYellowSignal(TrafficLightSignal::ON);
}

void Control::switchRedLight(bool state)
{
    redState = state;
    if (state)
        signal->switchRedSignal(TrafficLightSignal::ON);
    else
        signal->switchRedSignal(TrafficLightSignal::OFF);
}
