/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImageTIFF
//
// Class Template for new image formats
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImageTIFF.h"
#include "coImage.h"
#include <covise/covise.h>
#include <assert.h>

namespace covise
{
/// add all suffixes this class might have, NULL-terminated
static const char *suffixes[] = {
    "TIFF", "Tiff", "tiff",
    "TIF", "Tif", "tif", NULL
};

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
unsigned char *tifread(const char *filename, int *w, int *h, int *nc);

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Factory methods: Initialization and static cTor
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// static initializer
static coImageImpl *createTIFF(const char *filename)
{
    return new coImageTIFF(filename);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Registration at factory
static bool registered = coImage::registerImageType(suffixes, &createTIFF);
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
using namespace covise;

coImageTIFF::coImageTIFF(const char *filename)
{
    (void)registered;
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
    fclose(fi);

    pixmap_ = tifread(filename, &width_, &height_, &numChannels_);

    if (!pixmap_)
    {
        setError("Error reading TIFF");
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coImageTIFF::~coImageTIFF()
{
    if (pixmap_)
        free(pixmap_); // libTIFF works with malloc
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get pointer to internal data
unsigned char *
coImageTIFF::getBitmap(int frameno)
{
    (void)frameno;
    return pixmap_;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
coImageTIFF::coImageTIFF(const coImageTIFF &)
    : coImageImpl()
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
coImageTIFF &coImageTIFF::operator=(const coImageTIFF &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
coImageTIFF::coImageTIFF()
{
    assert(0);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tiffio.h"

// we need a patched libtiff to be able to read rgba tiffs from photoshop
//you can find this in extern_libs/linux/tiff
namespace covise
{
unsigned char *tifread(const char *filename, int *w, int *h, int *nc)
{
    TIFF *tif;
    tif = TIFFOpen(filename, "r");
    if (tif)
    {
        size_t npixels;
        size_t widthbytes;
        int i;
        uint32 *raster;
        unsigned char *raster2;
        unsigned char *image;
        int samples;
        samples = 4;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, h);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
        npixels = *w * *h;
        raster = (uint32 *)malloc(npixels * sizeof(uint32));
        if (raster != NULL)
        {
            if (TIFFReadRGBAImage(tif, *w, *h, raster, 0))
            {
                *nc = 4;

                raster2 = (unsigned char *)malloc(npixels * sizeof(uint32));
                image = (unsigned char *)raster;
                widthbytes = *w * sizeof(uint32);
                for (i = 0; i < *h; i++)
                {
                    memcpy(raster2 + (npixels * sizeof(uint32)) - ((i + 1) * widthbytes), image + (i * widthbytes), widthbytes);
                }
                free(raster);
                return (unsigned char *)raster2;
            }
        }
        TIFFClose(tif);
    }
    return NULL;
}
}
