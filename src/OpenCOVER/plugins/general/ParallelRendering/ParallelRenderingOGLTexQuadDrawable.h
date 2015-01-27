/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_OGL_TEX_QUAD_DRAWABLE_H
#define PARALLELRENDERING_OGL_TEX_QUAD_DRAWABLE_H

#include <osg/Drawable>
#include <osg/Texture2D>

#include "ParallelRenderingDefines.h"
#include "ParallelRenderingDimension.h"

class ParallelRenderingServer;
class ParallelRenderingCompositor;

class ParallelRenderingOGLTexQuadDrawable : public osg::Drawable
{

public:
    ParallelRenderingOGLTexQuadDrawable(ParallelRenderingCompositor *compositor);
    ~ParallelRenderingOGLTexQuadDrawable();

    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

    virtual osg::BoundingBox computeBound() const;

    virtual osg::Object *cloneType() const;
    virtual osg::Object *clone(const osg::CopyOp &) const;

private:
    ParallelRenderingServer *server;
    ParallelRenderingCompositor *compositor;
    int channel;

    ParallelRenderingDimension dimension;

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
