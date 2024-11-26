/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_TIME_H
#define EC_TIME_H

#include "covise.h"
#include <sys/types.h>
#ifndef _WIN32
#include <sys/time.h>
#include <sys/times.h>
#endif

/*
 $Log: covise_time.h,v $
 * Revision 1.1  1993/11/15  14:00:38  zrfg0125
 * Initial revision
 *
*/

namespace covise
{

extern COVISEEXPORT FILE *COVISE_time_hdl;

class COVISEEXPORT CoviseTime
{
private:
    int size;
    int count;
#ifndef _WIN32
    struct tms *user_sys;
    clock_t *list;
    struct timeval val;
#endif
    int *line;
    const char **text;

public:
    CoviseTime(int sz);
    ~CoviseTime()
    {
#ifndef _WIN32
        delete[] list;
#endif
        delete[] line;
        delete[] text;
    };
    void mark(int l, const char *t)
    {
        if (COVISE_time_hdl)
        {
#ifndef _WIN32
            list[count] = times(&user_sys[count]);
            line[count] = l;
            text[count++] = t;
#endif
        }
    };
    void init_mark(int l, const char *t);
    void print();
};
}
#endif
