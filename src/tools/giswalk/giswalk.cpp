/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>

#include "giswalk.h"
#include "gwApp.h"
#include "gwTier.h"
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_TIFF
#include <tiffio.h>
#endif
#ifndef _WIN32
#include <inttypes.h>
#include <sys/time.h>
#else
#include <sys/types.h>
#include <sys/timeb.h>

#endif

#ifdef _WIN32
#ifndef _WINSOCKAPI_
struct timeval
{
    int tv_sec;
    int tv_usec;
};
#endif

int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    struct __timeb64 currentTime;
    (void)tz;
#if _MSC_VER < 1400
    _ftime64(&currentTime);
#else
    _ftime64_s(&currentTime);
#endif
    tv->tv_sec = (int)currentTime.time;
    tv->tv_usec = currentTime.millitm * 1000;

    return 0;
}
#endif

int main(int argc, char **argv)
{
    if (argc == 2)
    {
        timeval currentTime;
        gettimeofday(&currentTime, NULL);
        double startTime = currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0;
        srand((int)startTime);
        gwApp *app = new gwApp(argv[1]);
        gettimeofday(&currentTime, NULL);
        double initTime = currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0;
        app->run();
        gettimeofday(&currentTime, NULL);
        double computeTime = currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0;
        app->writeSVG();
        app->writeShape();
        gettimeofday(&currentTime, NULL);
        double writeTime = currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0;
        fprintf(stderr, "init %lf\n compute %lf\n write %lf\n complete %lf\n", initTime - startTime, computeTime - initTime, writeTime - computeTime, writeTime - startTime);
        delete app;
    }
    else
    {

        fprintf(stderr, "GisWalk (c)2010 Uwe Woessner V1.3.6\n");
        fprintf(stderr, "usage: giswalk mapFile[.tif,.hdr,.txt]\n");
    }
}
