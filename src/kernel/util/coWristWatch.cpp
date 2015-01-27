/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coWristWatch.h"

using namespace covise;

coWristWatch::coWristWatch()
{
    reset();
}

void coWristWatch::reset()
{
    gettimeofday(&myClock, NULL);
}

float coWristWatch::elapsed()
{
    timeval now;
    gettimeofday(&now, NULL);
    return ((float)(now.tv_sec - myClock.tv_sec)) + ((float)(now.tv_usec - myClock.tv_usec)) / 1000000.0f;
}
