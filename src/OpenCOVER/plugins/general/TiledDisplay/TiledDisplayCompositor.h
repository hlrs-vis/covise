/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_COMPOSITOR_H
#define TILED_DISPLAY_COMPOSITOR_H

#include "TiledDisplayDefines.h"
#include "TiledDisplayDimension.h"

class TiledDisplayServer;

class TiledDisplayCompositor
{

public:
    TiledDisplayCompositor(int channel, TiledDisplayServer *server);
    virtual ~TiledDisplayCompositor();

    virtual void initSlaveChannel() = 0;
    bool updateTextures();

    // Called by the server
    virtual void setSubTexture(int width, int height, const unsigned char *pixels) = 0;

    virtual TiledDisplayServer *getServer()
    {
        return server;
    }
    virtual int getChannel()
    {
        return channel;
    }

protected:
    // Called by updateTextures
    virtual void updateTexturesImplementation() = 0;

    double lastUpdate;

    int channel;
    TiledDisplayServer *server;
    TiledDisplayDimension dimension;
};

#endif
