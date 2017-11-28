/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#ifndef CUDATEXTURERECTANGLE_H
#define CUDATEXTURERECTANGLE_H

#include <osg/State>
#include <osg/TextureRectangle>

#include "CudaGraphicsResource.h"


namespace opencover
{

class CudaTextureRectangle : public osg::TextureRectangle
{
public:

    CudaTextureRectangle();
    ~CudaTextureRectangle();

    virtual void    apply(osg::State& state) const;

    void    resize(osg::State* state, int w, int h, int dataTypeSize);
    void*   resourceData();
    void    clear();

protected:

    osg::ref_ptr<osg::PixelDataBufferObject> pbo_;
    CudaGraphicsResource resource_;
    int resourceDataSize_;
};

}

#endif

#endif
