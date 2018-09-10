// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2012 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#include "vvrendertarget.h"
#include "gl/util.h"

#include <GL/glew.h>

#include <assert.h>
#include <string.h>


namespace gl = virvo::gl;

using virvo::RenderTarget;
using virvo::NullRT;
using virvo::FramebufferObjectRT;
using virvo::HostBufferRT;
using virvo::PixelFormat;


//--------------------------------------------------------------------------------------------------
// RenderTarget
//--------------------------------------------------------------------------------------------------


RenderTarget::RenderTarget()
    : Width(0)
    , Height(0)
    , Bound(false)
{
}


RenderTarget::~RenderTarget()
{
    assert( !Bound );
}


bool RenderTarget::beginFrame(unsigned clearMask)
{
    assert( !Bound && "already bound" );

    Bound = BeginFrameImpl(clearMask);
    return Bound;
}


bool RenderTarget::endFrame()
{
    assert( Bound && "not bound" );

    bool Success = EndFrameImpl();

    Bound = false;

    return Success;
}


bool RenderTarget::resize(int w, int h)
{
    assert( !Bound && "resize while bound" );

    if (Width == w && Height == h)
        return true;

    bool Success = ResizeImpl(w, h);

    if (Success)
    {
        Width = w;
        Height = h;
    }

    return Success;
}


bool RenderTarget::displayColorBuffer() const
{
    assert( !Bound && "display while bound" );

    return DisplayColorBufferImpl();
}


bool RenderTarget::downloadColorBuffer(unsigned char* buffer, size_t size) const
{
    assert( !Bound && "download color buffer while bound" );

    return DownloadColorBufferImpl(buffer, size);
}


bool RenderTarget::downloadDepthBuffer(unsigned char* buffer, size_t size) const
{
    assert( !Bound && "download color buffer while bound" );

    return DownloadDepthBufferImpl(buffer, size);
}


bool RenderTarget::downloadColorBuffer(std::vector<unsigned char>& buffer) const
{
    assert( !Bound && "download color buffer while bound" );

    // TODO: Check for overflow...
    buffer.resize(width() * height() * getPixelSize(colorFormat()));

    return DownloadColorBufferImpl(&buffer[0], buffer.size());
}


bool RenderTarget::downloadDepthBuffer(std::vector<unsigned char>& buffer) const
{
    assert( !Bound && "download depth buffer while bound" );

    // TODO: Check for overflow...
    buffer.resize(width() * height() * getPixelSize(depthFormat()));

    return DownloadDepthBufferImpl(&buffer[0], buffer.size());
}


//--------------------------------------------------------------------------------------------------
// NullRT
//--------------------------------------------------------------------------------------------------


NullRT::NullRT()
{
}


RenderTarget* NullRT::create()
{
    return new NullRT;
}


NullRT::~NullRT()
{
}


bool NullRT::BeginFrameImpl(unsigned /*clearMask*/)
{
    return true;
}


bool NullRT::EndFrameImpl()
{
    return true;
}


bool NullRT::ResizeImpl(int /*w*/, int /*h*/)
{
    return true;
}


bool NullRT::DisplayColorBufferImpl() const
{
    return true;
}


bool NullRT::DownloadColorBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    for (size_t n = 0; n < bufferSize; ++n)
        buffer[n] = (unsigned char)( (255.0 * n) / bufferSize );

    return true;
}


bool NullRT::DownloadDepthBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    for (size_t n = 0; n < bufferSize; ++n)
        buffer[n] = (unsigned char)( 255.0 - (255.0 * n) / bufferSize );

    return true;
}


//--------------------------------------------------------------------------------------------------
// FramebufferObjectRT
//--------------------------------------------------------------------------------------------------


FramebufferObjectRT::FramebufferObjectRT(PixelFormat ColorFormat, PixelFormat DepthFormat)
    : ColorFormat(ColorFormat)
    , DepthFormat(DepthFormat)
{
}


RenderTarget* FramebufferObjectRT::create(PixelFormat ColorFormat, PixelFormat DepthFormat)
{
    return new FramebufferObjectRT(ColorFormat, DepthFormat);
}


FramebufferObjectRT::~FramebufferObjectRT()
{
}


bool FramebufferObjectRT::BeginFrameImpl(unsigned clearMask)
{
    assert( Framebuffer.get() != 0 );

    // Save current viewport
    glPushAttrib(GL_VIEWPORT_BIT);

    // Bind the framebuffer for rendering
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, Framebuffer.get());

    // Set the viewport
    glViewport(0, 0, width(), height());

    // Clear the render targets
    // Always use transparent black here!
    if (clearMask != 0)
    {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth(1.0);
        glClear(clearMask);
    }

    return true;
}


bool FramebufferObjectRT::EndFrameImpl()
{
    // Unbind the framebuffer
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    // Restore the viewport
    glPopAttrib();

    return true;
}


bool FramebufferObjectRT::ResizeImpl(int w, int h)
{
    PixelFormatInfo cf = mapPixelFormat(ColorFormat);
    PixelFormatInfo df = mapPixelFormat(DepthFormat);

    // Delete current color and depth buffers
    ColorBuffer.reset();
    DepthBuffer.reset();

    //
    // Create the framebuffer object (if not already done...)
    //

    if (Framebuffer.get() == 0)
        Framebuffer.reset( gl::createFramebuffer() );

    glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer.get());

    //
    // Create the color-buffer
    //

    ColorBuffer.reset( gl::createTexture() );

    glBindTexture(GL_TEXTURE_2D, ColorBuffer.get());

    // Initialize texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, cf.internalFormat, w, h, 0, cf.format, cf.type, 0);

    // Attach to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ColorBuffer.get(), 0);

    //
    // Create the depth-buffer
    //

    if (DepthFormat != PF_UNSPECIFIED)
    {
        DepthBuffer.reset( gl::createRenderbuffer() );

        glBindRenderbuffer(GL_RENDERBUFFER, DepthBuffer.get());

        glRenderbufferStorage(GL_RENDERBUFFER, df.internalFormat, w, h);

        // Attach as depth (and stencil) target
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, DepthBuffer.get());
    }

    //
    // Check for errors
    //

    GLenum status = VV_GET_FRAMEBUFFER_STATUS(GL_FRAMEBUFFER);

    // Unbind the framebuffer object!!!
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return status == GL_FRAMEBUFFER_COMPLETE;
}


bool FramebufferObjectRT::DisplayColorBufferImpl() const
{
    gl::blendTexture(ColorBuffer.get());
    return true;
}


bool FramebufferObjectRT::DownloadColorBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    PixelFormatInfo f = mapPixelFormat(ColorFormat);

    assert( bufferSize >= width() * height() * f.size );

    glBindFramebuffer(GL_READ_FRAMEBUFFER, Framebuffer.get());
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glReadPixels(0, 0, width(), height(), f.format, f.type, &buffer[0]);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    return true;

    static_cast<void>(bufferSize); // unused
}


bool FramebufferObjectRT::DownloadDepthBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    PixelFormatInfo f = mapPixelFormat(DepthFormat);

    assert( bufferSize >= width() * height() * f.size );

    glBindFramebuffer(GL_READ_FRAMEBUFFER, Framebuffer.get());
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glReadPixels(0, 0, width(), height(), GL_DEPTH_COMPONENT, f.type, &buffer[0]);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    return true;

    static_cast<void>(bufferSize); // unused
}


//--------------------------------------------------------------------------------------------------
// HostBufferRT
//--------------------------------------------------------------------------------------------------


HostBufferRT::HostBufferRT(PixelFormat ColorFormat, PixelFormat DepthFormat)
    : ColorFormat(ColorFormat)
    , DepthFormat(DepthFormat)
{
}


RenderTarget* HostBufferRT::create(PixelFormat ColorFormat, PixelFormat DepthFormat)
{
    return new HostBufferRT(ColorFormat, DepthFormat);
}


HostBufferRT::~HostBufferRT()
{
}


bool HostBufferRT::BeginFrameImpl(unsigned clearMask)
{
    if (clearMask & CLEAR_COLOR)
        std::fill(ColorBuffer.begin(), ColorBuffer.end(), 0);

    if (clearMask & CLEAR_DEPTH)
        std::fill(DepthBuffer.begin(), DepthBuffer.end(), 0);

    return true;
}


bool HostBufferRT::EndFrameImpl()
{
    return true;
}


bool HostBufferRT::ResizeImpl(int w, int h)
{
    ColorBuffer.resize(ComputeBufferSize(w, h, getPixelSize(ColorFormat)));

    if (DepthFormat != PF_UNSPECIFIED)
        DepthBuffer.resize(ComputeBufferSize(w, h, getPixelSize(DepthFormat)));

    return true;
}


bool HostBufferRT::DisplayColorBufferImpl() const
{
    PixelFormatInfo f = mapPixelFormat(ColorFormat);

    gl::blendPixels(width(), height(), f.format, f.type, &ColorBuffer[0]);
    return true;
}


bool HostBufferRT::DownloadColorBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    size_t bytes = width() * height() * getPixelSize(ColorFormat);

    assert( bufferSize >= bytes );

    memcpy(&buffer[0], &ColorBuffer[0], bytes);
    return true;

    static_cast<void>(bufferSize); // unused
}


bool HostBufferRT::DownloadDepthBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    size_t bytes = width() * height() * getPixelSize(DepthFormat);

    assert( bufferSize >= bytes );

    memcpy(&buffer[0], &DepthBuffer[0], bytes);
    return true;

    static_cast<void>(bufferSize); // unused
}
