/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_OGL_TEX_QUAD_COMPOSITOR_H
#define PARALLELRENDERING_OGL_TEX_QUAD_COMPOSITOR_H

#include "ParallelRenderingCompositor.h"

#include <osg/Geode>

class ParallelRenderingOGLTexQuadCompositor : public ParallelRenderingCompositor
{
public:
    ParallelRenderingOGLTexQuadCompositor(int channel);
    virtual ~ParallelRenderingOGLTexQuadCompositor();

    virtual void initSlaveChannel(bool replaceOSG);
    virtual void setTexture(int width, int height, const unsigned char *pixels);
    virtual void render();

private:
    int width;
    int height;

    int frame;
};

#endif
