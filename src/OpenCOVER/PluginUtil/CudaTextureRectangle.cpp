/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#include <GL/glew.h>

#include "CudaTextureRectangle.h"


namespace opencover
{

CudaTextureRectangle::CudaTextureRectangle() :
    pbo_(new osg::PixelDataBufferObject),
    resourceDataSize_(0)
{
    pbo_->setTarget(GL_PIXEL_UNPACK_BUFFER);

    resource_.map();
}

CudaTextureRectangle::~CudaTextureRectangle()
{
    resource_.unmap();
}

void CudaTextureRectangle::apply(osg::State& state) const
{
    osg::GLBufferObject* glBufferObject = pbo_->getGLBufferObject(state.getContextID());
    if (glBufferObject == nullptr) {
        osg::TextureRectangle::apply(state);

        return;
    }

    const_cast<CudaGraphicsResource*>(&resource_)->unmap();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBufferObject->getGLObjectID());

    osg::TextureRectangle::apply(state);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    const_cast<CudaGraphicsResource*>(&resource_)->map();
}

void CudaTextureRectangle::resize(osg::State* state, int w, int h, int dataTypeSize)
{
    resource_.unmap();

    resourceDataSize_ = w * h * dataTypeSize;

    pbo_->setDataSize(resourceDataSize_);
    pbo_->compileBuffer(*state);

    resource_.register_buffer(pbo_->getGLBufferObject(state->getContextID())->getGLObjectID(), cudaGraphicsRegisterFlagsWriteDiscard);

    resource_.map();
}

void* CudaTextureRectangle::resourceData()
{
    return resource_.dev_ptr();
}

void CudaTextureRectangle::clear()
{
    if (resourceData() == nullptr)
        return;

    cudaMemset(resourceData(), 0, resourceDataSize_);
}

}

#endif
