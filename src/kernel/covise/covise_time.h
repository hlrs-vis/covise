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
#endif
#ifndef __sgi
#ifndef _WIN32
#include <sys/times.h>
#endif
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
#ifdef __sgi
    volatile unsigned *iotimer_addr;
    struct timeval *list;
    unsigned *counter_val;
#else
#ifndef _WIN32
    struct tms *user_sys;
    clock_t *list;
    struct timeval val;
#endif
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
#ifdef __sgi
    void mark(int l, const char *t)
    {
        if (COVISE_time_hdl)
        {
            gettimeofday(&list[count], 0);
            counter_val[count] = *iotimer_addr;
            line[count] = l;
            text[count++] = t;
        }
    };
    void mark_fine(int l, const char *t)
    {
        if (COVISE_time_hdl)
        {
            list[count].tv_sec = 0;
            counter_val[count] = *iotimer_addr;
            line[count] = l;
            text[count++] = t;
        }
    };
#else
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
#endif
    void print();
};
}
#endif
