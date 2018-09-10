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


#ifndef VV_CUDA_RENDER_TARGET_H
#define VV_CUDA_RENDER_TARGET_H


#include "vvrendertarget.h"

#include <boost/shared_ptr.hpp>


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    // PixelUnpackBufferRT
    //----------------------------------------------------------------------------------------------
    class PixelUnpackBufferRT : public RenderTarget
    {
        PixelUnpackBufferRT(PixelFormat ColorFormat, PixelFormat DepthFormat);

    public:
        // Construct a render target
        static VVAPI RenderTarget* create(PixelFormat ColorFormat = PF_RGBA8, PixelFormat DepthFormat = PF_LUMINANCE8);

        // Clean up
        VVAPI virtual ~PixelUnpackBufferRT();

        // Returns the precision of the color buffer
        VVAPI virtual PixelFormat colorFormat() const VV_OVERRIDE;

        // Returns the precision of the depth buffer
        VVAPI virtual PixelFormat depthFormat() const VV_OVERRIDE;

        // Returns a pointer to the device color buffer
        VVAPI virtual void* deviceColor() VV_OVERRIDE;

        // Returns a pointer to the device depth buffer
        VVAPI virtual void* deviceDepth() VV_OVERRIDE;

        // Returns a pointer to the host depth buffer
        VVAPI virtual void const* hostDepth() const VV_OVERRIDE;

        // Returns the pixel-unpack buffer
        VVAPI GLuint buffer() const;

        // Returns the texture
        VVAPI GLuint texture() const;

    private:
        virtual bool BeginFrameImpl(unsigned clearMask) VV_OVERRIDE;
        virtual bool EndFrameImpl() VV_OVERRIDE;
        virtual bool ResizeImpl(int w, int h) VV_OVERRIDE;

        virtual bool DisplayColorBufferImpl() const VV_OVERRIDE;

        virtual bool DownloadColorBufferImpl(unsigned char* buffer, size_t bufferSize) const VV_OVERRIDE;
        virtual bool DownloadDepthBufferImpl(unsigned char* buffer, size_t bufferSize) const VV_OVERRIDE;

        // (Re-)create the render buffers (but not the depth buffer)
        bool CreateGLBuffers(int w, int h, bool linearInterpolation = false);

    private:
        struct Impl;
        boost::shared_ptr<Impl> impl;
    };


    //----------------------------------------------------------------------------------------------
    // DeviceBufferRT
    //----------------------------------------------------------------------------------------------
    class DeviceBufferRT : public RenderTarget
    {
        DeviceBufferRT(PixelFormat ColorFormat, PixelFormat DepthFormat);

    public:
        // Construct a render target
        static VVAPI RenderTarget* create(PixelFormat ColorFormat = PF_RGBA8, PixelFormat DepthFormat = PF_LUMINANCE8);

        // Clean up
        VVAPI virtual ~DeviceBufferRT();

        // Returns the precision of the color buffer
        VVAPI virtual PixelFormat colorFormat() const VV_OVERRIDE;

        // Returns the precision of the depth buffer
        VVAPI virtual PixelFormat depthFormat() const VV_OVERRIDE;

        // Returns a pointer to the device color buffer
        VVAPI virtual void* deviceColor() VV_OVERRIDE;

        // Returns a pointer to the device depth buffer
        VVAPI virtual void* deviceDepth() VV_OVERRIDE;

        // Returns a pointer to the host color buffer
        VVAPI virtual void const* hostColor() const VV_OVERRIDE;

        // Returns a pointer to the host depth buffer
        VVAPI virtual void const* hostDepth() const VV_OVERRIDE;

        // Returns the size of the color buffer in bytes
        VVAPI unsigned getColorBufferSize() const;

        // Returns the size of the depth buffer in bytes
        VVAPI unsigned getDepthBufferSize() const;

    private:
        virtual bool BeginFrameImpl(unsigned clearMask) VV_OVERRIDE;
        virtual bool EndFrameImpl() VV_OVERRIDE;
        virtual bool ResizeImpl(int w, int h) VV_OVERRIDE;

        virtual bool DisplayColorBufferImpl() const VV_OVERRIDE;

        virtual bool DownloadColorBufferImpl(unsigned char* buffer, size_t bufferSize) const VV_OVERRIDE;
        virtual bool DownloadDepthBufferImpl(unsigned char* buffer, size_t bufferSize) const VV_OVERRIDE;

    private:
        struct Impl;
        boost::shared_ptr<Impl> impl;
    };


} // namespace virvo


#endif
