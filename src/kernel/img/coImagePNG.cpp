/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImagePNG
//
// This class @@@
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImagePNG.h"
#include "coImage.h"
#include <covise/covise.h>
#include <assert.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

namespace covise
{
/// taken from Uwe's implementation in COVER - cleanup later
static unsigned char *pngread(FILE *fp, int *w, int *h, int *nc);

static const char *suffixes[] = { "PNG", "png", NULL };

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Factory methods: Initialization and static cTor
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// static initializer
static coImageImpl *createPNG(const char *filename)
{
    return new coImagePNG(filename);
}

/// Registration at factory
static bool registered = coImage::registerImageType(suffixes, &createPNG);
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
using namespace covise;

coImagePNG::coImagePNG(const char *filename)
{
#ifdef WIN32
    FILE *fi = fopen(filename, "rb");
#else
    FILE *fi = fopen(filename, "r");
#endif
    if (!fi)
    {
        setError(strerror(errno));
        return;
    }

    pixmap_ = pngread(fi, &width_, &height_, &numChannels_);

    fclose(fi);

    if (!pixmap_)
    {
        setError("Error reading PNG");
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coImagePNG::~coImagePNG()
{
    if (pixmap_)
        free(pixmap_); // PNG works with malloc
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get pointer to internal data
unsigned char *coImagePNG::getBitmap(int frameno)
{
    (void)frameno;
    return pixmap_;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
coImagePNG::coImagePNG(const coImagePNG &)
    : coImageImpl()
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
coImagePNG &coImagePNG::operator=(const coImagePNG &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
coImagePNG::coImagePNG()
{
    assert(0);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++ Code taken from COVER's vrml97 part
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "png.h"
#include <stdio.h>
#include <stdlib.h>

namespace covise
{

static double get_gamma_exp(void);
static int pngreadstr(FILE *fp,
                      int *w, int *h, int *nc,
                      png_structp png_ptr,
                      png_infop info_ptr,
                      unsigned char **pixels,
                      unsigned char ***rows);

static unsigned char *pngread(FILE *fp, int *w, int *h, int *nc)
{
    png_structp png_ptr;
    png_infop info_ptr;
    unsigned char *pixels = 0, **rows = 0;

    /* Create and initialize the png_struct with the desired error handler
    * functions.  If you want to use the default stderr and longjump method,
    * you can supply NULL for the last three parameters.  We also supply the
    * the compiler header file version, so that we know if the application
    * was compiled with a compatible version of the library.  REQUIRED
    */
    png_ptr = (png_structp)png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);

    if (png_ptr == NULL)
        return 0;

    /* Allocate/initialize the memory for image information.  REQUIRED. */
    info_ptr = (png_infop)png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        return 0;
    }

    /* png_ptr and info_ptr are freed in pngreadstr */
    if (!pngreadstr(fp, w, h, nc, png_ptr, info_ptr, &pixels, &rows))
    {
        /* Free all of the memory associated with the png_ptr and info_ptr */
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);

        if (pixels)
            free(pixels);
        if (rows)
            free(rows);
        return NULL;
    }

    if (rows)
        free(rows);
    return pixels;
}

/* This is broken out into a separate function so the setjmp
 * can't clobber pixels and rows.
 */

static int pngreadstr(FILE *fp,
                      int *w, int *h, int *nc,
                      png_structp png_ptr,
                      png_infop info_ptr,
                      unsigned char **ppixels,
                      unsigned char ***prows)
{
    png_uint_32 width, height;
    int bit_depth, color_type, interlace_type;
    unsigned int bytes_per_row, row;
    int gray_palette;
    unsigned char *pixels, **rows;

/* Set error handling if you are using the setjmp/longjmp method (this is
    * the normal method of doing things with libpng).  REQUIRED unless you
    * set up your own error handlers in the png_create_read_struct() earlier.
    */

#ifndef __hpux
    // HP-UX will crash for failing reads
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        /* If we get here, we had a problem reading the file */
        return 0;
    }
#endif

    /* Set up the input control if you are using standard C streams */
    png_init_io(png_ptr, fp);

    /* The call to png_read_info() gives us all of the information from the
    * PNG file before the first IDAT (image data chunk).  REQUIRED
    */
    png_read_info(png_ptr, info_ptr);

    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
                 &interlace_type, NULL, NULL);

    *nc = png_get_channels(png_ptr, info_ptr);
    /*printf("color_type %d, nc = %d ", color_type, *nc);*/

    /**** Set up the data transformations you want.  Note that these are all
    **** optional.  Only call them if you want/need them.  Many of the
    **** transformations only work on specific types of images, and many
    **** are mutually exclusive.
    ****/

    /* tell libpng to strip 16 bit/color files down to 8 bits/color */
    png_set_strip_16(png_ptr);

    /* Extract multiple pixels with bit depths of 1, 2, and 4 from a single
    * byte into separate bytes (useful for paletted and grayscale images).
    */
    png_set_packing(png_ptr);

    /* Expand paletted colors into true RGB triplets */
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        /* Even gray paletted images will get expanded here */
        png_set_expand(png_ptr);
        *nc = 3;
    }

    /* Expand grayscale images to the full 8 bits from 1, 2, or 4 bits/pixel */
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand(png_ptr);

    /* Expand paletted or RGB images with transparency to full alpha channels
    * so the data will be available as RGBA quartets.
    */
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    {
        png_set_expand(png_ptr);
        ++(*nc);
    }

    gray_palette = 0;
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        int n, num_palette;
        png_colorp palette;
        if (png_get_PLTE(png_ptr, info_ptr, &palette, &num_palette))
        {
            gray_palette = 1;
            for (n = 0; n < num_palette; ++n)
                if (palette[n].red != palette[n].green || palette[n].blue != palette[n].green)
                {
                    gray_palette = 0;
                    break;
                }
        }
    }

    /* set gamma */
    {
        double file_gamma, default_exponent = get_gamma_exp();
        default_exponent = 1.7;
        if (png_get_gAMA(png_ptr, info_ptr, &file_gamma))
        {
            //fprintf(stderr,"default_exponent, FileGamma: %f,%f\n" ,default_exponent,file_gamma);

            png_set_gamma(png_ptr, default_exponent, file_gamma);
            //png_set_gamma(png_ptr, default_exponent, 1.7);
        }
        else
        {
            png_set_gamma(png_ptr, default_exponent, 0.45455);
            //png_set_gamma(png_ptr, default_exponent, 1.7);
            //fprintf(stderr,"DefaultGamma: 1.7\n");
        }
    }

    /* Get updated info */
    png_read_update_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
                 &interlace_type, NULL, NULL);

    /* Allocate the memory to hold the image using the fields of info_ptr. */
    bytes_per_row = *nc * width;

    *ppixels = (unsigned char *)malloc(bytes_per_row * height);
    *prows = (unsigned char **)malloc(height * sizeof(char *));
    if (*ppixels == 0 || *prows == 0)
    {
        /* Free all of the memory associated with the png_ptr and info_ptr */
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
        return 0;
    }

    pixels = *ppixels;
    rows = *prows;
    for (row = 0; row < height; ++row)
        rows[row] = &pixels[row * bytes_per_row];

    /* Now it's time to read the image.  One of these methods is REQUIRED */
    /* Read the entire image in one go */
    png_read_image(png_ptr, rows);

    /* read rest of file, and get additional chunks in info_ptr - REQUIRED */
    png_read_end(png_ptr, info_ptr);

    /* clean up after the read, and free any memory allocated - REQUIRED */
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);

    /* Reduce gray palette images down to intensity or intensity/alpha */
    if (gray_palette)
    {
        int n, np = width * height;
        if (*nc == 3)
        {
            for (n = 1; n < np; ++n)
                pixels[n] = pixels[3 * n];
            *nc = 1;
        }
        else if (*nc == 4)
        {
            for (n = 0; n < np; ++n)
            {
                pixels[2 * n] = pixels[4 * n];
                pixels[2 * n + 1] = pixels[4 * n + 3];
            }
            *nc = 2;
        }
    }

    *w = width;
    *h = height;
    return 1;
}

/* From Greg Roelofs */

static double get_gamma_exp()
{
    static double default_exponent = 2.2;
    static int set = 0;

    if (!set)
    {

#if defined(NeXT)
        default_exponent = 1.0; /* 2.2/next_gamma for 3rd-party utils */

#elif defined(sgi)
        default_exponent = 1.3; /* default == 2.2 / 1.7 */
        /* there doesn't seem to be any documented function to get the
       * "gamma" value, so we do it the hard way */
        FILE *infile;
        if (infile = fopen("/etc/config/system.glGammaVal", "r"))
        {
            double sgi_gamma;
            char fooline[80];
            fgets(fooline, sizeof(fooline), infile);
            fclose(infile);
            sgi_gamma = atof(fooline);
            if (sgi_gamma > 0.0)
                default_exponent = 2.2 / sgi_gamma;
        }

#elif defined(Macintosh)
        default_exponent = 1.5; /* default == (1.8/2.61) * 2.2 */
/*
      if (mac_gamma = some_mac_function_that_returns_gamma())
      default_exponent = (mac_gamma/2.61) * 2.2;
      */
#endif

        set = 1;
    }

    return default_exponent;
}
}
