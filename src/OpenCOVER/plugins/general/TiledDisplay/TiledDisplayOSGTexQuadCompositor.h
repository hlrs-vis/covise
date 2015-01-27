/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_OSG_TEX_QUAD_COMPOSITOR_H
#define TILED_DISPLAY_OSG_TEX_QUAD_COMPOSITOR_H

#include "TiledDisplayCompositor.h"

#include <osg/Texture2D>

#include <osg/Geode>

class TiledDisplayOSGTexQuadCompositor : public TiledDisplayCompositor, public osg::Texture2D::SubloadCallback
{
public:
    TiledDisplayOSGTexQuadCompositor(int channel, TiledDisplayServer *server);
    virtual ~TiledDisplayOSGTexQuadCompositor();

    virtual void initSlaveChannel();
    virtual void setSubTexture(int width, int height, const unsigned char *pixels);

    virtual void load(const osg::Texture2D &texture, osg::State &state) const;
    virtual void subload(const osg::Texture2D &texture, osg::State &state) const;

private:
    virtual void updateTexturesImplementation();

    osg::ref_ptr<osg::Group> compositorNode;
    osg::ref_ptr<osg::Texture2D> compositorTexture;
};

#endif
