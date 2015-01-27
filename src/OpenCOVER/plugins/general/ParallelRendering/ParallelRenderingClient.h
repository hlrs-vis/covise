/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_CLIENT_H
#define PARALLELRENDERING_CLIENT_H

#include "ParallelRenderingDefines.h"
#include <string>
#include <osg/Camera>

#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>
#include <OpenThreads/Thread>

class ParallelRenderingClient : public OpenThreads::Thread
{

public:
    ParallelRenderingClient(int number, const std::string &compositor);
    virtual ~ParallelRenderingClient();

    virtual void run() = 0;
    virtual void send() = 0;
    virtual bool isConnected();
    virtual void connectToServer() = 0;

    virtual void readBackImage();
    virtual void exit();

protected:
    unsigned char *image;

    std::string compositor;
    int number;

    int width;
    int height;
    bool keepRunning;

    GLenum externalPixelFormat;

    bool connected;

#ifdef TILE_ENCODE_JPEG
    SGJpegEncoder encoder;
    SGJpegImage jpegImage;
#endif
};

#endif
