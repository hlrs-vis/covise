/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#ifndef CUDATEXTURE2D_H
#define CUDATEXTURE2D_H

#include <osg/State>
#include <osg/Texture2D>

#include "CudaGraphicsResource.h"


namespace opencover
{

class CudaTexture2D : public osg::Texture2D
{
public:

    CudaTexture2D();
    ~CudaTexture2D();

    virtual void    apply(osg::State& state) const;

    void    resize(osg::State* state, int w, int h, int dataTypeSize);
    void*   resourceData() const;
    size_t  getTotalSizeInBytes() const;
    void    clear();

protected:

    GLuint pbo_{0};
    CudaGraphicsResource resource_;
    size_t resourceDataSize_;
};

}

#endif

#endif
