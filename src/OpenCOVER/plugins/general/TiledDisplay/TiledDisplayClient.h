/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_CLIENT_H
#define TILED_DISPLAY_CLIENT_H

#include "TiledDisplayDefines.h"

#include <OpenThreads/Mutex>
#include <OpenThreads/Thread>

#include <osg/Camera>

#ifdef TILE_ENCODE_JPEG
#include <util/JPEG/SGJpegEncoder.h>
#endif

#include <string>

class TiledDisplayClient : public OpenThreads::Thread
{

public:
    TiledDisplayClient(int number, const std::string &compositor);
    virtual ~TiledDisplayClient();

    virtual bool connect() = 0;
    virtual void readBackImage(const osg::Camera &cam);

    virtual bool isImageAvailable();

    virtual void run() = 0;
    virtual void exit();

protected:
    OpenThreads::Mutex sendLock;
    OpenThreads::Mutex fillLock;

    unsigned char *image;

    std::string compositor;
    int number;

    int width;
    int height;

    bool dataAvailable;
    bool bufferAvailable;
    bool keepRunning;

    GLenum externalPixelFormat;

#ifdef TILE_ENCODE_JPEG
    SGJpegEncoder encoder;
    SGJpegImage jpegImage;
#endif
};

#endif
