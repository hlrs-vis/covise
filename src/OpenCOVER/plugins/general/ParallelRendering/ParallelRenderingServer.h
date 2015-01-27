/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_SERVER_H
#define PARALLELRENDERING_SERVER_H

#include <util/unixcompat.h>
#include <osg/Image>
#include <string>

#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>
#include <OpenThreads/Thread>

#ifndef _WIN32
#include <sys/time.h>
#endif

#include "ParallelRenderingDefines.h"
#include "ParallelRenderingDimension.h"

class ParallelRenderingCompositor;

class ParallelRenderingServer : public OpenThreads::Thread
{

public:
    ParallelRenderingServer(int number, bool compositorRenders);
    virtual ~ParallelRenderingServer();

    double getFrameTime();
    virtual bool isConnected();

    virtual void run() = 0;
    virtual void render() = 0;
    virtual void acceptConnection() = 0;

    virtual void readBackImage();
    virtual void exit();

    bool addCompositor(int channel, ParallelRenderingCompositor *compositor);

protected:
    unsigned char **pixels;

    unsigned char *image;
    int numClients;
    double frameTime;

    ParallelRenderingDimension *dimension;
    bool keepRunning;
    bool connected;
    ParallelRenderingCompositor **compositors;

    int startClient;
    bool compositorRenders;
    int width;
    int height;
    GLenum externalPixelFormat;
};

#endif
