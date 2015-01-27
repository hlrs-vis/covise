/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VehicleDynamics.h"

using covise::coCoviseConfig;

VehicleDynamics::VehicleDynamics()
{
    testLengthUp = fabs(coCoviseConfig::getFloat("testLengthUp", "COVER.Plugin.SteeringWheel.Dynamics.IntersectionTest", 1.0));
    testLengthDown = fabs(coCoviseConfig::getFloat("testLengthDown", "COVER.Plugin.SteeringWheel.Dynamics.IntersectionTest", 1.0));
    roadType = 0;
}
