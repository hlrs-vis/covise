/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_SERVER_H
#define TILED_DISPLAY_SERVER_H

//#define TILED_DISPLAY_SYNC

#include <util/unixcompat.h>

#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>
#include <OpenThreads/Thread>

#include <osg/Image>

#include <string>

#ifndef _WIN32
#include <sys/time.h>
#endif

#include "TiledDisplayDefines.h"
#include "TiledDisplayDimension.h"

class TiledDisplayCompositor;

class TiledDisplayServer : public OpenThreads::Thread
{

public:
    TiledDisplayServer(int number);
    virtual ~TiledDisplayServer();

    virtual bool accept() = 0;

    virtual void copySubImage(TiledDisplayCompositor *compositor);

    virtual bool isImageAvailable();

    virtual void run() = 0;
    virtual void exit();

    double getFrameTime()
    {
        return frameTime;
    }

protected:
#ifdef TILED_DISPLAY_SYNC
    OpenThreads::Barrier sendBarrier;
#else
    OpenThreads::Mutex sendLock;
#endif

    unsigned char *pixels;

    int number;

    bool dataAvailable;
    bool bufferAvailable;
    bool keepRunning;

    bool isRunning;

    double frameTime;

    TiledDisplayDimension dimension;

    struct Timer
    {

        Timer()
        {
            for (int ctr = 0; ctr < 4; ++ctr)
                started[ctr] = false;
        }

        void start(int n)
        {
            ::gettimeofday(&startTime[n], 0);
            restartTime[n] = startTime[n];
            restarts[n] = 0;
            started[n] = true;
        }

        void restart(int n)
        {
            if (!started[n])
            {
                start(n);
            }
            else
            {
                gettimeofday(&restartTime[n], 0);
                ++restarts[n];
            }
        }

        int elapsed(int n)
        {
            gettimeofday(&currentTime[n], 0);
            return (currentTime[n].tv_sec - restartTime[n].tv_sec) * 1000000 + (currentTime[n].tv_usec - restartTime[n].tv_usec);
        }

        float cps(int n)
        {
            gettimeofday(&currentTime[n], 0);
            int running = currentTime[n].tv_sec - startTime[n].tv_sec;
            return ((float)restarts[n]) / ((float)running);
        }

        timeval startTime[4];
        timeval restartTime[4];
        timeval currentTime[4];

        int restarts[4];
        bool started[4];
    };
};

#endif
