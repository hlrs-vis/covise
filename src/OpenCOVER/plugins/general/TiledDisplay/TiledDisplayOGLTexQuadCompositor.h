/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_OGL_TEX_QUAD_COMPOSITOR_H
#define TILED_DISPLAY_OGL_TEX_QUAD_COMPOSITOR_H

#include "TiledDisplayCompositor.h"

#include <osg/Geode>

class TiledDisplayOGLTexQuadCompositor : public TiledDisplayCompositor
{
public:
    TiledDisplayOGLTexQuadCompositor(int channel, TiledDisplayServer *server);
    virtual ~TiledDisplayOGLTexQuadCompositor();

    virtual void initSlaveChannel();
    virtual void setSubTexture(int width, int height, const unsigned char *pixels);

private:
    osg::ref_ptr<osg::Geode> compositorNode;
};

#endif
