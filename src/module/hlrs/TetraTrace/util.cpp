/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <api/coModule.h>
#include "util.h"

#include <util/coviseCompat.h>

#ifndef _WIN32
#include <sys/time.h>
#endif

WristWatch::WristWatch()
{
}

WristWatch::~WristWatch()
{
}

void WristWatch::start()
{
#ifndef _WIN32
    gettimeofday(&myClock, NULL);
#endif
    return;
}

void WristWatch::stop(const char *s)
{
#ifndef _WIN32
    timeval now;
    float dur;

    gettimeofday(&now, NULL);
    dur = ((float)(now.tv_sec - myClock.tv_sec)) + ((float)(now.tv_usec - myClock.tv_usec)) / 1000000.0;
    // sk: 11.04.2001
    // fprintf( stderr, "%s took %6.3f seconds\n", s, dur );
    covise::Covise::sendInfo("%s took %6.3f seconds.", s, dur);
#endif
    return;
}
