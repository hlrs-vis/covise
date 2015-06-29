/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_OGL_TEX_QUAD_DRAWABLE_H
#define TILED_DISPLAY_OGL_TEX_QUAD_DRAWABLE_H

#include <osg/Drawable>
#include <osg/Texture2D>
#include <osg/Version>

#include "TiledDisplayDefines.h"
#include "TiledDisplayDimension.h"

class TiledDisplayServer;
class TiledDisplayCompositor;

class TiledDisplayOGLTexQuadDrawable : public osg::Drawable
{

public:
    TiledDisplayOGLTexQuadDrawable(TiledDisplayCompositor *compositor);
    ~TiledDisplayOGLTexQuadDrawable();

    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    virtual osg::Object *cloneType() const;
    virtual osg::Object *clone(const osg::CopyOp &) const;

private:
    TiledDisplayServer *server;
    TiledDisplayCompositor *compositor;
    int channel;

    TiledDisplayDimension dimension;

    GLenum pixelFormat;
    GLenum pixelType;

#ifdef SGCOMPOSE_ENCODE_JPEG
    SGJpegImage jpegImage;
    SGJpegDecoder jpegDecoder;
#endif

    GLubyte *pixels;
    GLubyte *texImage;

    osg::StateSet *currentState;

    mutable GLuint *texture;
    mutable bool initPending;
};

#endif
