/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covise.h"
#include "covise_time.h"
#include "covise_global.h"
#include <util/coLog.h>

/*
 $Log: covise_time.C,v $
Revision 1.1  1993/11/15  14:01:02  zrfg0125
Initial revision

*/

using namespace covise;

CoviseTime *covise_time = new CoviseTime(10000);

CoviseTime::CoviseTime(int sz)
{
    size = sz;
    count = 0;

    line = new int[size];
    text = new const char *[size];
    mark(__LINE__, "init");
}

#ifndef _WIN32
void CoviseTime::print()
{
    double fl_time = 0.0;
#if !defined(__linux__) && !defined(__APPLE__)
    double fluni_time = 0.0;
    static double fl_firsttime = 0.0;
#endif
    int i;
    char tmp_str[255];
    static int first_line = 0;

    if (first_line == 0)
    {
#if !defined(__linux__) && !defined(__APPLE__)
        fl_firsttime = fl_time = val.tv_sec + ((double)val.tv_usec) / 1000000.0;
#endif
        sprintf(tmp_str, "System base time: %12.9f", fl_time);
        print_time(tmp_str);
    }
    //    print_time("-----------------------------------");
    for (i = 1; i < count; i++)
    {
#if !defined(__linux__) && !defined(__APPLE__)
        fl_time = (double)(list[i] - list[0]) / (double)CLK_TCK;
        fluni_time = ((int)fl_firsttime) % 1000 + fl_firsttime - (int)fl_firsttime + fl_time;
        sprintf(tmp_str, "%12.9f => %12.9f: (%3d) %s",
                fluni_time, fl_time, line[i], text[i]);
#endif
        print_time(tmp_str);
    }
    if (COVISE_time_hdl != NULL)
    {
        fflush(COVISE_time_hdl);
    }
    first_line = 1;
    //    list[0].tv_sec = list[count-1].tv_sec;
    //    list[0].tv_usec = list[count-1].tv_usec;
    count = 1;
}
#endif
