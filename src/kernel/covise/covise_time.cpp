/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covise.h"
#include "covise_time.h"
#include "covise_global.h"
#include <util/coLog.h>

#if defined(__sgi)
#include <sys/mman.h>
#include <sys/syssgi.h>
#include <sys/immu.h>
#endif
#ifdef CRAY
#include <sys/times.h>
#endif

/*
 $Log: covise_time.C,v $
Revision 1.1  1993/11/15  14:01:02  zrfg0125
Initial revision

*/

using namespace covise;

CoviseTime *covise_time = new CoviseTime(10000);

#ifdef CRAY
void CoviseTime::init_mark(int l, char *t)
{
    char tmp_str[255];
    if (COVISE_time_hdl)
    {
        if (gettimeofday(&val, (struct timezone *)0L) == -1)
        {
            switch (errno)
            {
            case EFAULT:
                print_comment(__LINE__, __FILE__, "EFAULT");
                break;
            case EINVAL:
                print_comment(__LINE__, __FILE__, "EINVAL");
                break;
            case EPERM:
                print_comment(__LINE__, __FILE__, "EPERM");
                break;
            }
        }
        sprintf(tmp_str, "gettimeofday: %d.%d", val.tv_sec, val.tv_usec);
        print_comment(__LINE__, __FILE__, tmp_str);
        list[count] = times(&user_sys[count]);
        line[count] = l;
        text[count++] = t;
    }
};
#endif

CoviseTime::CoviseTime(int sz)
{

    size = sz;
    count = 0;
#ifdef __sgi
    list = new struct timeval[size];
    counter_val = new unsigned[size];

#if CYCLE_COUNTER_IS_64BIT
    typedef unsigned long long iotimer_t;
#else
    typedef unsigned int iotimer_t;
#endif
    __psunsigned_t phys_addr, raddr;
    unsigned int cycleval;
    volatile iotimer_t counter_value, *iotimer_addr;
    int fd, poffmask;

    poffmask = getpagesize() - 1;
    phys_addr = syssgi(SGI_QUERY_CYCLECNTR, &cycleval);
    char tmp_str[255];
    sprintf(tmp_str, "picoseconds per count: %d", (int)cycleval);
    print_comment(__LINE__, __FILE__, tmp_str);
    raddr = phys_addr & ~poffmask;
    fd = open("/dev/mmem", O_RDONLY);
#ifndef INSURE
    iotimer_addr = (volatile iotimer_t *)mmap(0, poffmask, PROT_READ,
                                              MAP_PRIVATE, fd, (off_t)raddr);
    iotimer_addr = (iotimer_t *)((__psunsigned_t)iotimer_addr + (phys_addr & poffmask));
    close(fd);
#endif
#endif
#ifdef CRAY
    user_sys = new struct tms[size];
    list = new clock_t[size];
#endif
    line = new int[size];
    text = new const char *[size];
#ifndef CRAY
    mark(__LINE__, "init");
#else
    init_mark(__LINE__, "init");
#endif
}

#ifndef _WIN32
void CoviseTime::print()
{
    double fl_time = 0.0;
#if defined __sgi || (!defined(__linux__) && !defined(__APPLE__))
    double fluni_time = 0.0;
#endif

#if defined(__sgi)
    static double fl_o_time = 0.0;
    int last_proc_count;
#endif
#if !defined(__linux__) && !defined(__APPLE__)
    static double fl_firsttime = 0.0;
#endif
    int i;
    char tmp_str[255];
    static int first_line = 0;

    if (first_line == 0)
    {
#ifdef __sgi
        fl_time = list[0].tv_sec + ((double)list[0].tv_usec) / 1000000.0;
#endif
#if !defined(__linux__) && !defined(__APPLE__)
        fl_firsttime = fl_time = val.tv_sec + ((double)val.tv_usec) / 1000000.0;
#endif
        sprintf(tmp_str, "System base time: %12.9f", fl_time);
        print_time(tmp_str);
    }
    //    print_time("-----------------------------------");
    for (i = 1; i < count; i++)
    {
#ifndef __sgi
// ???
#if !defined(__linux__) && !defined(__APPLE__)
        fl_time = (double)(list[i] - list[0]) / (double)CLK_TCK;
        fluni_time = ((int)fl_firsttime) % 1000 + fl_firsttime - (int)fl_firsttime + fl_time;
        sprintf(tmp_str, "%12.9f => %12.9f: (%3d) %s",
                fluni_time, fl_time, line[i], text[i]);
#endif
#else
        if (list[i].tv_sec != 0)
        {
            fl_time = (list[i].tv_sec - list[0].tv_sec) + ((double)(list[i].tv_usec - list[0].tv_usec)) / 1000000.0;
            fluni_time = (list[i].tv_sec % 1000) + ((double)(list[i].tv_usec)) / 1000000.0;
            last_proc_count = counter_val[i];
            double diff = 0.0;
            if (i > 0)
                diff = fl_time - fl_o_time;
            sprintf(tmp_str, "%12.6f => %12.6f | %8.6f : (%3d) %s",
                    fluni_time, fl_time, diff, line[i], text[i]);
            fl_o_time = fl_time;
        }
        else
        {
            sprintf(tmp_str, "%12u: (%3d) %s",
                    counter_val[i] - last_proc_count, line[i], text[i]);
        }
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
