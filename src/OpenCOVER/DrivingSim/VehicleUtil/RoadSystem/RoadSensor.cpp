/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RoadSensor.h"

RoadSensor::RoadSensor(const std::string &id_, const double &s_)
    : Element(id_)
    , s(s_)
    , triggerAction(NULL)
{
}

void RoadSensor::setTriggerAction(RoadSensorTriggerAction *triggerAction_)
{
    triggerAction = triggerAction_;
}
