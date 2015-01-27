/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coTypes.h>
#include "TiledDisplayServer.h"

#include "TiledDisplayOGLTexQuadCompositor.h"
#include "TiledDisplayOSGTexQuadCompositor.h"

#include <iostream>
using std::cerr;
using std::endl;

TiledDisplayServer::TiledDisplayServer(int number)
{
    this->number = number;
    this->keepRunning = true;
    this->bufferAvailable = true;
    this->dataAvailable = false;
    this->pixels = 0;
    this->isRunning = false;
    frameTime = 0.0;
}

TiledDisplayServer::~TiledDisplayServer()
{
    delete[] pixels;
}

//#define TILED_DISPLAY_SERVER_TIME_IMAGE_AVAILABLE

bool TiledDisplayServer::isImageAvailable()
{
#ifdef TILED_DISPLAY_SERVER_TIME_IMAGE_AVAILABLE
    static Timer timer;
    timer.restart(number);
#endif

    bool rv;

#ifdef TILED_DISPLAY_SYNC
    rv = dataAvailable & isRunning;
#else
    int locked = sendLock.trylock();
    if (locked == 0 && dataAvailable)
    {
        sendLock.unlock();
        rv = true;
    }
    else
    {
        if (locked == 0)
            sendLock.unlock();
        rv = false;
    }
#endif

#ifdef TILED_DISPLAY_SERVER_TIME_IMAGE_AVAILABLE
    cerr << "TiledDisplayServer::isImageAvailable info: server " << number << " check in " << timer.elapsed(number)
         << " usec, avg. cps = " << timer.cps(number) << endl;
#endif

    return rv;
}

//#define TILED_DISPLAY_SERVER_TIME_COPY_SUB_IMAGE

void TiledDisplayServer::copySubImage(TiledDisplayCompositor *compositor)
{

    if (!isRunning)
        return;

#ifdef TILED_DISPLAY_SERVER_TIME_COPY_SUB_IMAGE
    static Timer timer;
    timer.restart(number);
#endif

//cerr << "TiledDisplayServer::copySubImage info: server " << number << " before lock" << endl;
#ifdef TILED_DISPLAY_SYNC
    sendBarrier.block(2);
#else
    sendLock.lock();
#endif

    //cerr << "TiledDisplayServer::copySubImage info: server " << number << " copy image ["
    //     << dimension.width << "|" << dimension.height << "]" << endl;
    if (pixels)
        compositor->setSubTexture(dimension.width, dimension.height, pixels);

    bufferAvailable = true;
    dataAvailable = false;

#ifndef TILED_DISPLAY_SYNC
    sendLock.unlock();
#endif

#ifdef TILED_DISPLAY_SERVER_TIME_COPY_SUB_IMAGE
    cerr << "TiledDisplayServer::copySubImage info: server " << number << " tex subload in " << timer.elapsed(number)
         << " usec, avg. cps = " << timer.cps(number) << endl;
#endif
}

void TiledDisplayServer::exit()
{
    keepRunning = false;
}
