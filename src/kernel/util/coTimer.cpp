/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTimer.h"
#include <fstream>
#ifdef WIN32
#include <unixcompat.h>
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace covise;

/// we create the getTimeBase program in here, too
#ifndef CREATE_TIMEBASE_PROG

coTimer *coTimer::s_timerObj = NULL;
#if defined(__sun)
char coTimer::s_buffer[128];
#else
char coTimer::s_buffer[MAX_LEN];
#endif

coTimer::coTimer(const char *filename, int length, bool handleSignals)
{
    descField = new char[length * MAX_LEN];
    timeField = actTval = new TimeType[length];
    maxElem = length - 1; // sure to have one more for flush() times
    actElem = -1;
    actTval--;

    // if thie variable "COVISE_TIMEBASE" is set
    const char *timebase = getenv("COVISE_TIMEBASE");
    if (timebase)
    {
        double fltime = atof(timebase);
        TIME_SET(startTime, fltime);
    }
    else
    {
        TIME_CALL(&startTime);
    }

#ifdef WIN32
    sprintf(descField, "%s.%d.tim", filename, _getpid());
#else
    sprintf(descField, "%s.%d.tim", filename, getpid());
#endif
    stream = new std::ofstream(descField);
    *descField = '\0';
#ifndef _WIN32
    if (handleSignals)
    {
        coSignal::addSignal(SIGHUP, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGINT, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGQUIT, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGILL, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGABRT, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGFPE, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGBUS, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGSEGV, *this, coSignal::KEEP_DFL);
#ifndef __linux__
        coSignal::addSignal(SIGSYS, *this, coSignal::KEEP_DFL);
#endif
        coSignal::addSignal(SIGPIPE, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGTERM, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGUSR1, *this, coSignal::REPLACE_DFL);
        coSignal::addSignal(SIGTSTP, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGTTIN, *this, coSignal::KEEP_DFL);
        coSignal::addSignal(SIGTTOU, *this, coSignal::KEEP_DFL);
    }
#endif
    atexit(coTimer::quit);
}

void coTimer::sigHandler(int sigNo)
{
    sprintf(int_mark(), "coTimer::sigHandler for signal %d", sigNo);
    flush();
    stream->flush();
}

void coTimer::flush()
{
    if (stream)
    {
        int i;
        actTval++;
        actElem++;
        TIME_CALL(actTval);
        strcpy(descField + MAX_LEN * actElem, " --- FLUSH --- FLUSH --- FLUSH --- ");

        actTval = timeField;
        for (i = 0; i <= actElem; i++)
        {
            stream->flags(std::ios::right | std::ios::fixed);
            stream->width(14);
            stream->precision(6);
#ifdef POSIX_TIME
            *stream << (actTval->tv_sec - startTime.tv_sec) + (actTval->tv_nsec - startTime.tv_nsec) * 0.000000001
                    << " sec, "
                    << (descField + MAX_LEN * i) << endl;

#else
#ifdef _MSC_VER
            *stream << (actTval->time - startTime.time) + (actTval->millitm - startTime.millitm) * 0.000001
                    << " sec, "
                    << (descField + MAX_LEN * i) << std::endl;
#else
            *stream << (actTval->tv_sec - startTime.tv_sec) + (actTval->tv_usec - startTime.tv_usec) * 0.000001
                    << " sec, "
                    << (descField + MAX_LEN * i) << std::endl;
#endif
#endif

            actTval++;
        }
        stream->flags();
        actElem = -1;
        actTval = timeField - 1;
        mark("coTimer::flush() finished: Re-Start");
    }
}

coTimer::~coTimer()
{
    if (stream)
    {
        flush();
        *stream << "\n\n ------- FINISHED -------" << std::endl;
        stream->close();
        stream = NULL;
    }
}

void coTimer::init(const char *fileNameBase, int length, bool handleSignals)
{
    if (s_timerObj)
        delete s_timerObj;
    s_timerObj = new coTimer(fileNameBase, length, handleSignals);
}

void coTimer::quit()
{
    delete s_timerObj;
    s_timerObj = NULL;
}

//////////// code for creating getTimeBase program
#else

int main(int, char *[])
{
    TimeType startTime;
    TIME_CALL(&startTime);
    printf("%20.8f\n", coTimerGetFloat(startTime));
}
#endif
