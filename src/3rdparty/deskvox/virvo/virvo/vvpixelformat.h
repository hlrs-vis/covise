// vvpixelformat.h


#ifndef VV_PIXEL_FORMAT_H
#define VV_PIXEL_FORMAT_H


#include "vvexport.h"


namespace virvo
{


    enum PixelFormat
    {
        PF_UNSPECIFIED,

        // Pixel formats for color buffers and images

        PF_R8,
        PF_RG8,
        PF_RGB8,
        PF_RGBA8,
        PF_R16F,
        PF_RG16F,
        PF_RGB16F,
        PF_RGBA16F,
        PF_R32F,
        PF_RG32F,
        PF_RGB32F,
        PF_RGBA32F,
        PF_R16I,
        PF_RG16I,
        PF_RGB16I,
        PF_RGBA16I,
        PF_R32I,
        PF_RG32I,
        PF_RGB32I,
        PF_RGBA32I,
        PF_R16UI,
        PF_RG16UI,
        PF_RGB16UI,
        PF_RGBA16UI,
        PF_R32UI,
        PF_RG32UI,
        PF_RGB32UI,
        PF_RGBA32UI,

        PF_BGR8,
        PF_BGRA8,

        PF_RGB10_A2,

        PF_R11F_G11F_B10F,

        // Pixel formats for depth/stencil buffers

        PF_DEPTH16,
        PF_DEPTH24,
        PF_DEPTH32,
        PF_DEPTH32F,
        PF_DEPTH24_STENCIL8,
        PF_DEPTH32F_STENCIL8,
        PF_LUMINANCE8,      // not an OpenGL format!
        PF_LUMINANCE16,     // not an OpenGL format!
        PF_LUMINANCE32F,    // not an OpenGL format!

        PF_COUNT // Last!!!
    };


    template<class A>
    void serialize(A& a, PixelFormat& pf, unsigned /*version*/)
    {
        a & static_cast<unsigned>(pf);
    }


    struct PixelFormatInfo
    {
        unsigned internalFormat;
        unsigned format;
        unsigned type;
        unsigned components;
        unsigned size; // per pixel in bytes
    };


    // Returns some information about the given color format
    VVAPI PixelFormatInfo mapPixelFormat(PixelFormat format);


    // Returns the size of a single pixel of the given format
    inline unsigned getPixelSize(PixelFormat format)
    {
        PixelFormatInfo f = mapPixelFormat(format);
        return f.size;
    }


} // namespace virvo


#endif
