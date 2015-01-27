/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "../include/time.hxx"
#include "timestandard.h"

TimeSteps::TimeSteps(void)
{
}

TimeSteps::~TimeSteps()
{
}

TimeSteps *TimeSteps::getInstance()
{
    return new TimeStepsStandard();
}
