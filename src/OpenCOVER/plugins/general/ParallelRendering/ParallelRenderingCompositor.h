/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_COMPOSITOR_H
#define PARALLELRENDERING_COMPOSITOR_H

#include "ParallelRenderingDefines.h"
#include "ParallelRenderingDimension.h"

#ifndef __APPLE__
#include <GL/gl.h>
#else
#include <OpenGL/gl.h>
#endif

class ParallelRenderingServer;

class ParallelRenderingCompositor
{

public:
    ParallelRenderingCompositor(int channel);
    virtual ~ParallelRenderingCompositor();

    virtual void initSlaveChannel(bool replaceOSG) = 0;
    bool updateTextures();

    virtual void setTexture(int width, int height, const unsigned char *pixels) = 0;

    virtual int getChannel()
    {
        return channel;
    }
    virtual void render() = 0;

protected:
    int channel;
    mutable GLuint *texture;

    int texSize;
};

#endif
