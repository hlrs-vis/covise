/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  Image.cpp

#include "config.h"
#include "Image.h"
#include "Doc.h"
#include "System.h"

#define HAVE_LIBTIF 1

#include <stdlib.h> // free()
#include <string.h>

// gif support is builtin
#include "gifread.h"

#if HAVE_LIBJPEG
#include "jpgread.h"
#include "mpgread.h"
#endif
#if HAVE_LIBPNG
#include "pngread.h"
#endif
#if HAVE_LIBTIF
#include "tifread.h"
#endif

using namespace vrml;

typedef enum
{
    ImageFile_UNKNOWN,

    ImageFile_GIF,
    ImageFile_JPG,
    ImageFile_MOVIE,
    ImageFile_PNG,
    ImageFile_TIF

} ImageFileType;

static ImageFileType imageFileType(const char *, FILE *);

Image::Image(const char *url, Doc *relative)
    : d_url(0)
    , d_w(0)
    , d_h(0)
    , d_nc(0)
    , d_pixels(0)
    , d_frame(0)
{
    static bool firstTime = true;
    static bool sMovieSupport;
    if (firstTime)
    {
        sMovieSupport = System::the->getConfigState("COVER.Plugin.Vrml97.NewMovieSupport", true);
        firstTime = false;
    }
    newMovies = sMovieSupport;
    if (url)
        (void)setURL(url, relative);
}

Image::~Image()
{
    delete d_url;
    if (d_pixels)
        free(d_pixels); // assumes file readers use malloc...
    if (d_frame)
        free(d_frame);
}

bool Image::setURL(const char *url, Doc *relative)
{
    if (d_url)
        delete d_url;
    if (d_pixels)
        free(d_pixels); // assumes file readers use malloc...
    if (d_frame)
        free(d_frame);
    d_pixels = 0;
    d_frame = 0;
    d_w = d_h = d_nc = d_nFrames = 0;
    if (!url)
        return true;

    d_url = new Doc(url, relative);
    auto fileName = d_url->localName();
    //  System::the->debug("Image: trying to create Doc(%s, %s)\n",
    //		   url, relative ? relative->url() : "");

    FILE *fp = d_url->fopen("rb");

    if (fp)
    {
        switch (imageFileType(url, fp))
        {
        case ImageFile_GIF:
            d_pixels = gifread(fp, &d_w, &d_h, &d_nc, &d_nFrames, &d_frame);
            fprintf(stderr, "GIFread %d\n", d_nc);
            break;

        case ImageFile_MOVIE:
            // change here to use new movie code
            if (newMovies)
            {
                d_pixels = (unsigned char *)malloc(fileName.length() + 1);
                strcpy((char *)d_pixels, fileName.c_str());
                d_w = d_h = 0;
                d_nc = -1;
                d_nFrames = 0;
            }
            else
            {
                d_pixels = mpgread(fp, &d_w, &d_h, &d_nc, &d_nFrames, &d_frame);
            }
            break;

#if HAVE_LIBJPEG
        case ImageFile_JPG:
            d_pixels = jpgread(fp, &d_w, &d_h, &d_nc);
            break;
#endif
#if HAVE_LIBPNG
        case ImageFile_PNG:
            d_pixels = pngread(fp, &d_w, &d_h, &d_nc);
            break;
#endif
#if HAVE_LIBTIF
        case ImageFile_TIF:
            d_pixels = tifread(fp, url, &d_w, &d_h, &d_nc);
            break;
#endif

        default:
            d_pixels = (unsigned char *)malloc(fileName.length() + 1);
            strcpy((char *)d_pixels, fileName.c_str());
            d_w = d_h = d_nc = 0;
//fprintf(stderr,"Error: could not open (%s).\n", url);
        break;
        }

        if (!d_pixels)
            fprintf(stderr, "Error: unable to read image file (%s).\n", url);

        d_url->fclose();
    }
    return (d_pixels != 0);
}

bool Image::tryURLs(int nUrls, char **urls, Doc *relative)
{
    int i;
    for (i = 0; i < nUrls; ++i) // Try each url until one succeeds
        if (urls[i] && setURL(urls[i], relative))
            break;

    return i < nUrls;
}

const char *Image::url() { return d_url ? d_url->url().c_str() : 0; }

// Could peek at file header...

static ImageFileType imageFileType(const char *url, FILE *)
{
    char *suffix = strrchr((char *)url, '.');
    if (suffix)
        ++suffix;
    else
        return ImageFile_UNKNOWN;

    if (strcmp(suffix, "gif") == 0 || strcmp(suffix, "GIF") == 0)
        return ImageFile_GIF;

    else if (strcmp(suffix, "jpg") == 0 || strcmp(suffix, "JPG") == 0 || strcmp(suffix, "jpeg") == 0 || strcmp(suffix, "JPEG") == 0)
        return ImageFile_JPG;

    else if (strcmp(suffix, "avi") == 0 || strcmp(suffix, "AVI") == 0 || strcmp(suffix, "mov") == 0 || strcmp(suffix, "MOV") == 0 || strcmp(suffix, "mpg") == 0 || strcmp(suffix, "mp4") == 0 || strcmp(suffix, "MPG") == 0 || strcmp(suffix, "mpeg") == 0 || strcmp(suffix, "MPEG") == 0)
        return ImageFile_MOVIE;

    else if (strcmp(suffix, "tif") == 0 || strcmp(suffix, "TIF") == 0 || strcmp(suffix, "tiff") == 0 || strcmp(suffix, "TIFF") == 0)
        return ImageFile_TIF;

    else if (strcmp(suffix, "png") == 0 || strcmp(suffix, "PNG") == 0)
        return ImageFile_PNG;

    else
        return ImageFile_UNKNOWN;
}

unsigned char *Image::pixels(int frame)
{
    return (frame >= 0 && frame < d_nFrames) ? d_frame[frame] : 0;
}
