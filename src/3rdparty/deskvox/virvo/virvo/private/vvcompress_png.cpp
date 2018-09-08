// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
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


#include "vvcompress.h"
#include "vvcompressedvector.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#ifdef HAVE_PNG


#include <assert.h>
#include <memory.h>
#include <png.h>


#ifdef _MSC_VER
#pragma warning(disable: 4324) // structure was padded due to __declspec(align())
#pragma warning(disable: 4611) // interaction between '_setjmp' and C++ object destruction is non-portable
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wclobbered"
#endif


namespace
{


static void PNGErrorCallback(png_structp png_ptr, png_const_charp msg)
{
    fprintf(stderr, "PNG error: \"%s\"\n", msg);

    longjmp(png_jmpbuf(png_ptr), 1);
}


static void PNGWarningCallback(png_structp /*png_ptr*/, png_const_charp /*msg*/)
{
}


struct PNGIO
{
    std::vector<unsigned char>& buffer;
    std::size_t start;

    inline PNGIO(std::vector<unsigned char>& buffer)
        : buffer(buffer)
        , start(0)
    {
    }
};


struct PNGWriteContext
{
    png_structp png;
    png_infop info;

    inline PNGWriteContext()
        : png(0)
        , info(0)
    {
    }

    inline ~PNGWriteContext()
    {
        png_destroy_write_struct(&png, &info);
    }
};


static void PNGWriteFunc(png_structp png, png_bytep data, png_size_t len)
{
    PNGIO* out = (PNGIO*)png_get_io_ptr(png);

    out->buffer.reserve(out->buffer.size() + len);
    out->buffer.insert(out->buffer.end(), data, data + len);
}


struct PNGReadContext
{
    png_structp png;
    png_infop info;

    inline PNGReadContext() : png(0), info(0)
    {
    }

    inline ~PNGReadContext()
    {
        png_destroy_read_struct(&png, &info, 0);
    }
};


static void PNGReadFunc(png_structp png, png_bytep data, png_size_t len)
{
    PNGIO* in = (PNGIO*)png_get_io_ptr(png);

    // Read some bytes
    memcpy(data, &in->buffer[in->start], len);

    // Offset the pointer
    in->start += len;
}


static int PNGMapBitDepth(virvo::PixelFormatInfo const& info)
{
    // Components must all have the same size in bits.
    // NOTE: Might not catch all invalid image formats...
    if ((info.size % info.components) != 0)
        return -1;

    int bit_depth = 8 * info.size / info.components;

    // Bit depth larger than 16 are not handled by libpng
    if (bit_depth > 16)
        return -1;

    return bit_depth;
}


static int PNGMapColorType(virvo::PixelFormatInfo const& info)
{
    switch (info.components)
    {
    case 1:
        return PNG_COLOR_TYPE_GRAY;
    case 2:
        return PNG_COLOR_TYPE_GRAY_ALPHA;
    case 3:
        return PNG_COLOR_TYPE_RGB;
    case 4:
        return PNG_COLOR_TYPE_RGB_ALPHA;
    }

    return -1; // Not handling paletted images...
}


static bool PNGGetImageInfo(virvo::PixelFormat format, int& bit_depth, int& color_type)
{
    virvo::PixelFormatInfo info = mapPixelFormat(format);

    // Get the size in bits of each component
    bit_depth = PNGMapBitDepth(info);

    // Get the color type
    color_type = PNGMapColorType(info);

    return bit_depth > 0 && color_type >= 0;
}


} // namespace


bool virvo::encodePNG(std::vector<unsigned char>& data, PNGOptions const& options)
{
    int bit_depth   = 0;
    int color_type  = 0;

    // Check if its possible to encode the image...
    if (!PNGGetImageInfo(options.format, bit_depth, color_type))
        return false;

    PNGWriteContext context;

    // Allocate and initialize png_ptr struct for writing, and any other memory
    context.png = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, PNGErrorCallback, PNGWarningCallback);

    if (context.png == 0)
        return false;

    // Allocate and initialize the info structure
    context.info = png_create_info_struct(context.png);

    if (context.info == 0)
        return false;

    std::vector<unsigned char> compressed;

    PNGIO out(compressed);

    // Set error handling.
    if (setjmp(png_jmpbuf(context.png)))
        return false;

    // Replace the default data output functions with a user supplied one(s).
    // If buffered output is not used, then output_flush_fn can be set to NULL.
    // If PNG_WRITE_FLUSH_SUPPORTED is not defined at libpng compile time
    // output_flush_fn will be ignored (and thus can be NULL).
    png_set_write_fn(context.png, reinterpret_cast<void*>(&out), PNGWriteFunc, 0);

    // Set the library compression level.
    //
    // Currently, valid values range from 0 - 9, corresponding directly to the zlib compression
    // levels 0 - 9 (0 - no compression, 9 - "maximal" compression).
    // Note that tests have shown that zlib compression levels 3-6 usually perform as well as
    // level 9 for PNG images, and do considerably fewer caclulations. In the future,
    // these values may not correspond directly to the zlib compression levels.
    png_set_compression_level(context.png, options.compression_level);

    // Set the image information here.
    png_set_IHDR(context.png, context.info, options.w, options.h, bit_depth, color_type,
        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // TODO:
    // png_set_sBIT(...)

    // TODO:
    // png_set_gAMA(...)

    // TODO:
    // png_set_bKGD(...)

    // Writes all the PNG information before the image.
    png_write_info_before_PLTE(context.png, context.info);

    // Write the image data
    for (int y = 0; y < options.h; y++)
    {
        png_bytep row_ptr = &data[y * options.pitch];
        png_write_rows(context.png, &row_ptr, 1);
    }

    png_write_end(context.png, context.info);

    // Update the data
    data.swap(compressed);

    return true;
}


bool virvo::decodePNG(std::vector<unsigned char>& data, PNGOptions& options)
{
    PNGReadContext context;

    // Create and initialize the png_struct with the desired error handler
    // functions.  If you want to use the default stderr and longjump method,
    // you can supply NULL for the last three parameters.  We also supply the
    // the compiler header file version, so that we know if the application
    // was compiled with a compatible version of the library.
    context.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0/*user-data*/, PNGErrorCallback, PNGWarningCallback);

    if (context.png == 0)
        return false;

    // Allocate/initialize the memory for image information.
    context.info = png_create_info_struct(context.png);

    if (context.info == 0)
        return false;

    // Create the output buffer
    std::vector<unsigned char> uncompressed;

    // Create a wrapper for the input buffer
    PNGIO in(data);

    // Set error handling if you are using the setjmp/longjmp method (this is
    // the normal method of doing things with libpng).  REQUIRED unless you
    // set up your own error handlers in the png_create_read_struct() earlier.
    if (setjmp(png_jmpbuf(context.png)))
        return false;

    // Set the read function
    png_set_read_fn(context.png, (void*)&in, PNGReadFunc);

    // The call to png_read_info() gives us all of the information from the
    // PNG file before the first IDAT (image data chunk).
    png_read_info(context.png, context.info);

    png_uint_32 w = 0;
    png_uint_32 h = 0;
    int bit_depth = 0;
    int color_type = 0;

    png_get_IHDR(context.png, context.info, &w, &h, &bit_depth, &color_type, 0, 0, 0);

    //
    // TODO:
    // Check against the values given in the options or simply modify the options???
    //

    //
    // From here down we are concerned with colortables and pixels
    //
    // Expand palette images to RGB, low-bit-depth grayscale images to 8 bits,
    // transparency chunks to alpha channel; strip 16-bit-per-sample
    // images to 8 bits per sample; and convert grayscale to RGB[A]
    //

    // paletted images are expanded to RGB
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_expand(context.png);
    }

    // grayscale images of bit-depth less than 8 are expanded
    // to 8-bit images and tRNS chunks are expanded to alpha channels
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    {
        png_set_expand(context.png);
    }

    // expand transparency chunks to alpha channel
    if (png_get_valid(context.png, context.info, PNG_INFO_tRNS))
    {
        png_set_tRNS_to_alpha(context.png);
    }

    // TODO:
    // png_set_strip_16

    // TODO:
    // png_set_gray_to_rgb

    // TODO:
    // png_set_bgr

    // TODO:
    // png_[s,g]et_gAMA

    // TODO:
    // png_[s,g]et_bKGD

    //
    // After the transformations have been registered update png_info_ptr data
    // Get again width, height and channels
    //

    png_read_update_info(context.png, context.info);

    w           = png_get_image_width(context.png, context.info);
    h           = png_get_image_height(context.png, context.info);
    bit_depth   = png_get_bit_depth(context.png, context.info);
    color_type  = png_get_color_type(context.png, context.info);

    //
    // TODO:
    // - Check for correct image dimension/format...
    // - Store the correct pixel format in options...
    //

    assert( options.w >= 0 && static_cast<png_uint_32>(options.w) == w );
    assert( options.h >= 0 && static_cast<png_uint_32>(options.h) == h );

    // Create the image buffer
    uncompressed.resize(options.pitch * options.h);

    // Read the image
    for (int y = 0; y < options.h; y++)
    {
        png_bytep row = &uncompressed[y * options.pitch];
        png_read_rows(context.png, &row, &row, 1);
    }

    png_read_end(context.png, context.info);

    // Update the image data.
    data.swap(uncompressed);

    return true;
}


#else // HAVE_PNG


bool virvo::encodePNG(std::vector<unsigned char>& /*data*/, PNGOptions const& /*options*/)
{
    return false;
}


bool virvo::decodePNG(std::vector<unsigned char>& /*data*/, PNGOptions& /*options*/)
{
    return false;
}


#endif // !HAVE_PNG


bool virvo::encodePNG(CompressedVector& data, PNGOptions const& options)
{
    if (data.getCompressionType() != Compress_None)
        return false;

    if (encodePNG(data.vector(), options))
    {
        data.setCompressionType(Compress_PNG);
        return true;
    }

    return false;
}


bool virvo::decodePNG(CompressedVector& data, PNGOptions& options)
{
    if (data.getCompressionType() != Compress_PNG)
        return false;

    if (decodePNG(data.vector(), options))
    {
        data.setCompressionType(Compress_None);
        return true;
    }

    return false;
}
