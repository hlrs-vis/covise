/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* The Graphics Interchange Format(c) is the Copyright property of
** CompuServe Incorporated.  GIF(sm) is a Service Mark property of
** CompuServe Incorporated.
*/

#define _XOPEN_SOURCE

#ifdef __sgi
/* sgi uses ifl */
#else
#define HAVE_LIBTIF 1
#endif
#include "tifread.h"
#ifdef HAVE_LIBTIF
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>
#ifndef _WIN32
#include <inttypes.h>
#endif
/* we need a patched libtiff to be able to read rgba tiffs from photoshop
   you can find this in extern_libs/linux/tiff */
/*extern int tiffDoRGBA; */
/* probably not needed with newer version 3.7.3 of the tiff-library
 has to be tested though*/
/* not needed with 3.7.3 uwe 1.2006 */

void myWarn(const char *c, const char *c2, va_list list)
{
    (void)c;
    (void)c2;
    (void)list;
}

unsigned char *tifread(FILE *fp, const char *url, int *w, int *h, int *nc)
{
    static int firstTime = 1;
    TIFF *tif;
    if (firstTime)
    {
        firstTime = 0;
        TIFFSetWarningHandler(myWarn);
    }

/*tiffDoRGBA = 1;*/
#ifdef _WIN32
#if _MSC_VER < 1900
    tif = TIFFFdOpen(fp->_file, url, "r");
#else
	tif = TIFFFdOpen(fileno(fp), url, "r");
#endif
#else
    tif = TIFFFdOpen(fileno(fp), url, "r");
#endif
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
                if (samples < 4)
                {
                    /* ugly hack by Uwe for grey scale/b/w images */
                    *nc = 40;
                }

                raster2 = (unsigned char *)malloc(npixels * sizeof(uint32));
                image = (unsigned char *)raster;
                widthbytes = *w * sizeof(uint32);
                for (i = 0; i < *h; i++)
                {
                    memcpy(raster2 + (npixels * sizeof(uint32)) - ((i + 1) * widthbytes), image + (i * widthbytes), widthbytes);
                }
                free(raster);

/* We have to byteswap on SGI ! */
#ifndef BYTESWAP
                {
                    int i;
                    uint32_t *iPtr = (uint32_t *)raster2;
                    for (i = 0; i < npixels; i++, iPtr++)
                        *iPtr = ((*iPtr & 0x000000ff) << 24) | ((*iPtr & 0x0000ff00) << 8) | ((*iPtr & 0x00ff0000) >> 8) | ((*iPtr & 0xff000000) >> 24);
                }
#endif

                /*TIFFClose(tif);*/
                return (unsigned char *)raster2;
            }
        }
    }
    return NULL;
}

#endif
