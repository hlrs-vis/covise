/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#include <GL/glew.h>
#include <osg/Version>

#include "CudaSafeCall.h"
#include "CudaTextureRectangle.h"


namespace opencover
{

CudaTextureRectangle::CudaTextureRectangle() : resourceDataSize_(0)
{
    resource_.map();
}

CudaTextureRectangle::~CudaTextureRectangle()
{
    resource_.unmap();
    glDeleteBuffers(1, &pbo_);
}

void CudaTextureRectangle::apply(osg::State& state) const
{
    const_cast<CudaGraphicsResource*>(&resource_)->unmap();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);

    //osg::TextureRectangle::apply(state);

    // On TextureRectangle::apply(), OSG decides which code to be called for
    // texture allocation based on the source and internal formats. The
    // following branch is what needs to be executed; and it _is_ if the
    // internal format is not "sized". There's a bug (or inconsistency) in OSG
    // where GL_DEPTH_COMPONENT(_32F) is _not_ considered sized (but sized
    // _and_ stenciled), whereas GL_RGBA(8) is considered sized. So the two PBO
    // textures take different code paths internally. We do *not* want to rely
    // on this behavior, but instead just replicate the desired behavior of OSG
    // directly, instead of calling TextureRectangle::apply()

    GLenum internalFormat = getSourceFormat() ? getSourceFormat() : getInternalFormat();
    TextureObject* textureObject = generateAndAssignTextureObject(
        state.getContextID(),
        GL_TEXTURE_RECTANGLE,
        0,
        internalFormat,
        getTextureWidth(),
        getTextureHeight(),
        1,
        0);
#if OSG_VERSION_GREATER_OR_EQUAL(3, 7, 0)
    textureObject->bind(state);
#else
    textureObject->bind();
#endif
    applyTexParameters(GL_TEXTURE_RECTANGLE, state);
    glTexImage2D( GL_TEXTURE_RECTANGLE, 0, getInternalFormat(),
             getTextureWidth(), getTextureHeight(), getBorderWidth(),
             internalFormat,
             getSourceType() ? getSourceType() : GL_UNSIGNED_BYTE,
             0);
    textureObject->setAllocated(0, getInternalFormat(), getTextureWidth(), getTextureHeight(), 1, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    const_cast<CudaGraphicsResource*>(&resource_)->map();
}

void CudaTextureRectangle::resize(osg::State* state, int w, int h, int dataTypeSize)
{
    if (!pbo_) {
        glGenBuffers(1, &pbo_);
    }

    resource_.unmap();

    resourceDataSize_ = w * h * dataTypeSize;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, resourceDataSize_, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_SAFE_CALL(resource_.register_buffer(pbo_, cudaGraphicsRegisterFlagsWriteDiscard));

    resource_.map();
}

void* CudaTextureRectangle::resourceData() const
{
    return resource_.dev_ptr();
}

void CudaTextureRectangle::clear()
{
    if (resourceData() == nullptr)
        return;

    CUDA_SAFE_CALL(cudaMemset(resourceData(), 0, resourceDataSize_));
}

}

#endif
