/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TIMER_H_
#define _TIMER_H_

#ifdef _MSC_VER
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif
#include "coSignal.h"

#include <iostream>
#include <cstdlib>
#include <cstring>

// define different time measurement methods
#if defined(__sgi)
#define POSIX_TIME
//#define CLOCK_ID      CLOCK_SGI_CYCLE
#define CLOCK_ID CLOCK_REALTIME
#endif

namespace covise
{

// +++++++ POSIX time-calls
#if defined(POSIX_TIME)
typedef struct timespec TimeType;

// this is a macro to save time (inlined by Preprocesor)
#define TIME_CALL(x) clock_gettime(CLOCK_ID, x)

inline void TIME_SET(TimeType &x, double ti)
{
    x.tv_sec = (long)(ti);
    x.tv_nsec = (long)((ti - x.tv_sec) * 1.0e9);
}

inline double coTimerGetFloat(const TimeType &x)
{
    return (double)x.tv_sec + 1.0e-9 * x.tv_nsec;
}

// +++++++ SysV time-calls
#else
#ifdef _MSC_VER

typedef struct __timeb64 TimeType;
#define TIME_CALL(x) \
    {                \
        _ftime64(x); \
    }

inline void TIME_SET(TimeType &x, double ti)
{
    x.time = (long)(ti);
    x.millitm = (unsigned short)((ti - x.time) * 1.0e6f);
}

inline double coTimerGetFloat(const TimeType &x)
{
    return x.time + 1e-6 * x.millitm;
}

#else
typedef struct timeval TimeType;
#define TIME_CALL(x) gettimeofday(x, NULL)

inline void TIME_SET(TimeType &x, double ti)
{
    x.tv_sec = (long)(ti);
    x.tv_usec = (long)((ti - x.tv_sec) * 1.0e6);
}

inline double coTimerGetFloat(const TimeType &x)
{
    return x.tv_sec + 1e-6 * x.tv_usec;
}
#endif
#endif

/// none of the following is reqired for the getTimeBase program
#ifndef CREATE_TIMEBASE_PROG

#ifdef NO_TIMER
#else
#define MARK0(text)          \
    {                        \
        coTimer::mark(text); \
    }
#define MARK1(mask, d1)                       \
    {                                         \
        sprintf(coTimer::mark(), mask, (d1)); \
    }
#define MARK2(mask, d1, d2)                         \
    {                                               \
        sprintf(coTimer::mark(), mask, (d1), (d2)); \
    }
#define MARK3(mask, d1, d2, d3)                           \
    {                                                     \
        sprintf(coTimer::mark(), mask, (d1), (d2), (d3)); \
    }

#define MARK4(mask, d1, d2, d3, d4)                             \
    {                                                           \
        sprintf(coTimer::mark(), mask, (d1), (d2), (d3), (d4)); \
    }

#define MARK5(mask, d1, d2, d3, d4, d5)                               \
    {                                                                 \
        sprintf(coTimer::mark(), mask, (d1), (d2), (d3), (d4), (d5)); \
    }

#define MARK6(mask, d1, d2, d3, d4, d5, d6)                                 \
    {                                                                       \
        sprintf(coTimer::mark(), mask, (d1), (d2), (d3), (d4), (d5), (d6)); \
    }
#endif

class UTILEXPORT coTimer : public coSignalHandler
{
private:
    enum
    {
        MAX_LEN = 128
    };

    int maxElem, actElem;
    char *descField;
    TimeType *timeField, *actTval, startTime;
    std::ofstream *stream;

    static coTimer *s_timerObj;
    static char s_buffer[MAX_LEN];

    virtual void sigHandler(int sigNo);
    virtual const char *sigHandlerName()
    {
        return "coTimer";
    }

protected:
    coTimer(const char *fileNameBase, int length, bool handleSignals);

    char *int_mark()
    {
#ifndef NOTIMER
        if (actElem >= maxElem)
            flush();
        actTval++;
        actElem++;
        TIME_CALL(actTval);
        return descField + MAX_LEN * actElem;
#else
        static char dummy[MAX_LEN] return dummy;
#endif
    }

#ifndef NOTIMER
    void int_mark(const char *string)
    {
        if (actElem >= maxElem)
            flush();
        actTval++;
        actElem++;
        TIME_CALL(actTval);
        strcpy(descField + MAX_LEN * actElem, string);
    }
#else
    void mark(const char *)
    {
    }
#endif

    void flush();

public:
    virtual ~coTimer();

    static void init(const char *fileNameBase, int length, bool handleSignals = true);
    static void quit();

    static char *mark()
    {
        char *res;
        if (s_timerObj)
            res = s_timerObj->int_mark();
        else
            res = s_buffer;
        return res;
    }

    static void mark(const char *string)
    {
        if (s_timerObj)
            s_timerObj->int_mark(string);
    }
};
#endif // getTimeBase
}
#endif
