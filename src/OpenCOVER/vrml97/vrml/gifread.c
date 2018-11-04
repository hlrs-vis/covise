/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* The Graphics Interchange Format(c) is the Copyright property of
** CompuServe Incorporated.  GIF(sm) is a Service Mark property of
** CompuServe Incorporated.
*/

#include "gifread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/************************************************************************
	GIF File Reader
************************************************************************/
/* +-------------------------------------------------------------------+ */
/* | Copyright 1990, David Koblas.                                     | */
/* |   Permission to use, copy, modify, and distribute this software   | */
/* |   and its documentation for any purpose and without fee is hereby | */
/* |   granted, provided that the above copyright notice appear in all | */
/* |   copies and that both that copyright notice and this permission  | */
/* |   notice appear in supporting documentation.  This software is    | */
/* |   provided "as is" without express or implied warranty.           | */
/* +-------------------------------------------------------------------+ */

#define USE_GIF89 1

#define MAXIMAGE 256
#define MAXCOLORMAPSIZE 256
#define CM_RED 0
#define CM_GREEN 1
#define CM_BLUE 2
#define CM_USED 3
#define MAX_LWZ_BITS 12
#define INTERLACE 0x40
#define LOCALCOLORMAP 0x80

#define BitSet(byte, bit) (((byte) & (bit)) == (bit))
#define ReadOK(file, buffer, len) (fread(buffer, len, 1, file) != 0)
#define LM_to_uint(a, b) (((b) << 8) | (a))

static struct
{
    unsigned int Width;
    unsigned int Height;
    unsigned char ColorMap[3][MAXCOLORMAPSIZE];
    unsigned int BitPixel;
    unsigned int ColorResolution;
    unsigned int Background;
    unsigned int AspectRatio;
    unsigned int Grayscale;
    unsigned int Components;
} GifScreen;

#if USE_GIF89
static struct
{
    int transparent;
    int delayTime;
    int inputFlag;
    int disposal;
} Gif89;
#endif

#define TRUE 1
#define FALSE 0

static int verbose = FALSE;
static int showComment = FALSE;
static int ZeroDataBlock = FALSE;
static int usedEntry[MAXCOLORMAPSIZE];

static int ReadGIF(FILE *fp, unsigned char *frame[]);
static int ReadColorMap(FILE *fd, int number, unsigned char buffer[3][MAXCOLORMAPSIZE], unsigned int *grayscale);
static int DoExtension(FILE *fd, int label);
static int GetDataBlock(FILE *fd, unsigned char *buf);
static int GetCode(FILE *fd, int code_size, int flag);
static int LWZReadByte(FILE *fd, int flag, int input_code_size);
static unsigned char *ReadImage(FILE *fd, int len, int height, unsigned char cmap[3][MAXCOLORMAPSIZE], unsigned int grayscale, int interlace, int ignore, int xs, int ys);

static void pm_message(const char *format, ...);
static int pm_error(const char *format, ...);

static int error = FALSE;

unsigned char *gifread(FILE *fp, int *w, int *h, int *nc, int *nf,
                       unsigned char ***frames)
{
    unsigned char *image[MAXIMAGE];
    int i;

    error = FALSE;

    *nf = ReadGIF(fp, image);

    /* Allocate and store frames */
    *frames = (unsigned char **)malloc(*nf * sizeof(unsigned char *));
    if (error || *frames == 0)
    {
        for (i = 0; i < *nf; ++i)
            if (image[i])
                free(image[i]);
        *nf = 0;
        return 0;
    }

    for (i = 0; i < *nf; ++i)
        (*frames)[i] = image[i];

    *nc = GifScreen.Components;
    *w = GifScreen.Width;
    *h = GifScreen.Height;

    return image[0];
}

static int
ReadGIF(FILE *fd, unsigned char *frame[])
{
    unsigned char buf[16];
    unsigned char c;
    int useGlobalColormap;
    int bitPixel;
    char version[4];
    int i, j, p, nc, image_size;
    int left, top, width, height;
    int m, n, k;
    unsigned char *tmp;

    if (!ReadOK(fd, buf, 6))
        return pm_error("error reading magic number");

    if (strncmp((char *)buf, "GIF", 3) != 0)
        return pm_error("not a GIF file");

    strncpy(version, (char *)buf + 3, 3);
    version[3] = '\0';

    if ((strcmp(version, "87a") != 0) && (strcmp(version, "89a") != 0))
        return pm_error("bad version number, not '87a' or '89a'");

    if (!ReadOK(fd, buf, 7))
        return pm_error("failed to read screen descriptor");

    GifScreen.Width = LM_to_uint(buf[0], buf[1]);
    GifScreen.Height = LM_to_uint(buf[2], buf[3]);
    GifScreen.BitPixel = 2 << (buf[4] & 0x07);
    GifScreen.ColorResolution = (((unsigned char)(buf[4] & 0x70) >> 3) + 1);
    GifScreen.Background = buf[5];
    GifScreen.AspectRatio = buf[6];
    GifScreen.Grayscale = FALSE;
    GifScreen.Components = 0;

#if USE_GIF89
    Gif89.transparent = -1;
    Gif89.delayTime = -1;
    Gif89.inputFlag = -1;
    Gif89.disposal = 0;
#endif

    if (BitSet(buf[4], LOCALCOLORMAP))
    { /* Global Colormap */
        if (ReadColorMap(fd, GifScreen.BitPixel, GifScreen.ColorMap,
                         &GifScreen.Grayscale))
            return pm_error("error reading global colormap");
    }

    if (GifScreen.AspectRatio != 0 && GifScreen.AspectRatio != 49)
    {
        double r;
        r = (GifScreen.AspectRatio + 15.0) / 64.0;
        pm_message("warning - non-square pixels; to fix do a 'pnmscale -%cscale %g'",
                   r < 1.0 ? 'x' : 'y',
                   r < 1.0 ? 1.0 / r : r);
    }

    for (i = 0; i < MAXIMAGE;)
    {
        if (!ReadOK(fd, &c, 1))
            return pm_error("EOF / read error on image data");

        if (c == ';')
        { /* GIF terminator */
            return i;
        }

        if (c == '!')
        { /* Extension */
            if (!ReadOK(fd, &c, 1))
                return pm_error("OF / read error on extention function code");
            DoExtension(fd, c);
            continue;
        }

        if (c != ',')
        { /* Not a valid start character */
            pm_message("bogus character 0x%02x, ignoring", (int)c);
            continue;
        }

        if (!ReadOK(fd, buf, 9))
            return pm_error("couldn't read left/top/width/height");

        useGlobalColormap = !BitSet(buf[8], LOCALCOLORMAP);

        bitPixel = 1 << ((buf[8] & 0x07) + 1);

        if (!useGlobalColormap)
            if (ReadColorMap(fd, bitPixel, GifScreen.ColorMap,
                             &GifScreen.Grayscale))
                return pm_error("error reading local colormap");

        nc = GifScreen.Grayscale ? 1 : 3;
        if (Gif89.transparent >= 0)
            ++nc;
        if (GifScreen.Components != 0 && GifScreen.Components != (unsigned)nc)
        {
            pm_message("multiple colormap formats encounted at frame %d",
                       i);
            return i;
        }
        GifScreen.Components = nc;
        left = LM_to_uint(buf[0], buf[1]);
        top = LM_to_uint(buf[2], buf[3]);
        width = LM_to_uint(buf[4], buf[5]);
        height = LM_to_uint(buf[6], buf[7]);
        frame[i] = ReadImage(fd, LM_to_uint(buf[4], buf[5]),
                             LM_to_uint(buf[6], buf[7]),
                             GifScreen.ColorMap,
                             GifScreen.Grayscale,
                             BitSet(buf[8], INTERLACE), 0, GifScreen.Width, GifScreen.Height);

        /* Convert to I, IA, RGB, or RGBA. */
        image_size = GifScreen.Width * GifScreen.Height;
        if ((width != GifScreen.Width) || (height != GifScreen.Height))
        {
            if (i == 0)
            {
                fprintf(stderr, "Error, image sizes don't match\n");
                return pm_error("error reading image frame 0: size mismatch");
            }
            tmp = frame[i];
            frame[i] = (unsigned char *)malloc(nc * GifScreen.Width * GifScreen.Height);
            switch (nc)
            {
            case 4: /* RGBA */
                memcpy(frame[i], frame[i - 1], 4 * image_size);
                for (k = 0; k < width; k++)
                {
                    for (j = 0; j < height; j++)
                    {
                        m = tmp[j * width + k];
                        n = (top + j) * GifScreen.Width + k + left;
                        frame[i][n * 4] = GifScreen.ColorMap[CM_RED][m];
                        frame[i][n * 4 + 1] = GifScreen.ColorMap[CM_GREEN][m];
                        frame[i][n * 4 + 2] = GifScreen.ColorMap[CM_BLUE][m];
                        frame[i][n * 4 + 3] = (m == Gif89.transparent) ? 0 : 255;
                    }
                }
                break;
            case 3: /* RGB */
                memcpy(frame[i], frame[i - 1], 3 * image_size);
                for (k = 0; k < width; k++)
                {
                    for (j = 0; j < height; j++)
                    {
                        m = tmp[j * width + k];
                        n = (top + j) * GifScreen.Width + k + left;
                        frame[i][n * 3] = GifScreen.ColorMap[CM_RED][m];
                        frame[i][n * 3 + 1] = GifScreen.ColorMap[CM_GREEN][m];
                        frame[i][n * 3 + 2] = GifScreen.ColorMap[CM_BLUE][m];
                    }
                }
                break;
            case 2: /* IA (Intensity, Alpha) */
                memcpy(frame[i], frame[i - 1], 2 * image_size);
                for (k = 0; k < width; k++)
                {
                    for (j = 0; j < height; j++)
                    {
                        m = tmp[j * width + k];
                        n = (top + j) * GifScreen.Width + k + left;
                        frame[i][n * 2] = GifScreen.ColorMap[CM_RED][m];
                        frame[i][n * 2 + 1] = (m == Gif89.transparent) ? 0 : 255;
                    }
                }
                break;
            case 1: /* I (Intensity) */
                memcpy(frame[i], frame[i - 1], image_size);
                for (k = 0; k < width; k++)
                {
                    for (j = 0; j < height; j++)
                    {
                        m = tmp[j * width + k];
                        n = (top + j) * GifScreen.Width + k + left;
                        frame[i][n] = GifScreen.ColorMap[CM_RED][m];
                    }
                }
                break;
            }
        }
        else
        {
            switch (nc)
            {
            case 4: /* RGBA */
                for (p = image_size - 1; p >= 0; --p)
                {
                    j = frame[i][p];
                    frame[i][p * 4] = GifScreen.ColorMap[CM_RED][j];
                    frame[i][p * 4 + 1] = GifScreen.ColorMap[CM_GREEN][j];
                    frame[i][p * 4 + 2] = GifScreen.ColorMap[CM_BLUE][j];
                    frame[i][p * 4 + 3] = (j == Gif89.transparent) ? 0 : 255;
                }
                break;
            case 3: /* RGB */
                for (p = image_size - 1; p >= 0; --p)
                {
                    j = frame[i][p];
                    frame[i][p * 3] = GifScreen.ColorMap[CM_RED][j];
                    frame[i][p * 3 + 1] = GifScreen.ColorMap[CM_GREEN][j];
                    frame[i][p * 3 + 2] = GifScreen.ColorMap[CM_BLUE][j];
                }
                break;
            case 2: /* IA (Intensity, Alpha) */
                for (p = image_size - 1; p >= 0; --p)
                {
                    j = frame[i][p];
                    frame[i][p * 2] = GifScreen.ColorMap[CM_RED][j];
                    frame[i][p * 2 + 1] = (j == Gif89.transparent) ? 0 : 255;
                }
                break;
            case 1: /* I (Intensity) */
                for (p = image_size - 1; p >= 0; --p)
                {
                    frame[i][p] = GifScreen.ColorMap[CM_RED][frame[i][p]];
                }
                break;
            }
        }
        ++i;
    }
    return MAXIMAGE;
}

static int
ReadColorMap(FILE *fd, int number,
             unsigned char buffer[3][MAXCOLORMAPSIZE],
             unsigned int *grayscale)
{
    int i;
    unsigned char rgb[3];

    *grayscale = TRUE;
    for (i = 0; i < number; ++i)
    {
        if (!ReadOK(fd, rgb, sizeof(rgb)))
            return pm_error("bad colormap");

        buffer[CM_RED][i] = rgb[0];
        buffer[CM_GREEN][i] = rgb[1];
        buffer[CM_BLUE][i] = rgb[2];
        usedEntry[i] = FALSE;
        if (rgb[0] != rgb[1] || rgb[1] != rgb[2])
            *grayscale = FALSE;
    }

    return FALSE;
}

static int
DoExtension(FILE *fd, int label)
{
    static char buf[256];
    char *str;

    switch (label)
    {
    case 0x01: /* Plain Text Extension */
        str = "Plain Text Extension";
#ifdef notdef
        if (GetDataBlock(fd, (unsigned char *)buf) == 0)
            ;

        lpos = LM_to_uint(buf[0], buf[1]);
        tpos = LM_to_uint(buf[2], buf[3]);
        width = LM_to_uint(buf[4], buf[5]);
        height = LM_to_uint(buf[6], buf[7]);
        cellw = buf[8];
        cellh = buf[9];
        foreground = buf[10];
        background = buf[11];

        while (GetDataBlock(fd, (unsigned char *)buf) != 0)
        {
            PPM_ASSIGN(image[ypos][xpos],
                       cmap[CM_RED][v],
                       cmap[CM_GREEN][v],
                       cmap[CM_BLUE][v]);
            ++index;
        }

        return FALSE;
#else
        break;
#endif
    case 0xff: /* Application Extension */
        str = "Application Extension";
        break;
    case 0xfe: /* Comment Extension */
        str = "Comment Extension";
        while (GetDataBlock(fd, (unsigned char *)buf) != 0)
        {
            if (showComment)
                pm_message("gif comment: %s", buf);
        }
        return FALSE;
    case 0xf9: /* Graphic Control Extension */
        str = "Graphic Control Extension";
        (void)GetDataBlock(fd, (unsigned char *)buf);
#if USE_GIF89
        Gif89.disposal = (buf[0] >> 2) & 0x7;
        Gif89.inputFlag = (buf[0] >> 1) & 0x1;
        Gif89.delayTime = LM_to_uint(buf[1], buf[2]);
        if ((buf[0] & 0x1) != 0)
            Gif89.transparent = buf[3];
#endif
        while (GetDataBlock(fd, (unsigned char *)buf) != 0)
            ;

        return FALSE;
    default:
        str = buf;
        sprintf(buf, "UNKNOWN (0x%02x)", label);
        break;
    }

#if 0
	pm_message("got a '%s' extension - please report this to koblas@mips.com",
					str );
#else
    (void)str;
#endif

    while (GetDataBlock(fd, (unsigned char *)buf) != 0)
        ;
    return FALSE;
}

static int
GetDataBlock(FILE *fd, unsigned char *buf)
{
    unsigned char count;

    if (!ReadOK(fd, &count, 1))
    {
        pm_message("error in getting DataBlock size");
        return -1;
    }

    ZeroDataBlock = count == 0;

    if ((count != 0) && (!ReadOK(fd, buf, count)))
    {
        pm_message("error in reading DataBlock");
        return -1;
    }

    return count;
}

static int
GetCode(FILE *fd, int code_size, int flag)
{
    static unsigned char buf[280];
    static int curbit, lastbit, done, last_byte;
    int i, j, ret;
    unsigned char count;

    if (flag)
    {
        curbit = 0;
        lastbit = 0;
        done = FALSE;
        return 0;
    }

    if ((curbit + code_size) >= lastbit)
    {
        if (done)
        {
            if (curbit >= lastbit)
                pm_error("ran off the end of my bits");
            return -1;
        }
        if (last_byte > 1)
        {
            buf[0] = buf[last_byte - 2];
            buf[1] = buf[last_byte - 1];
        }

        if ((count = GetDataBlock(fd, &buf[2])) == 0)
            done = TRUE;

        last_byte = 2 + count;
        curbit = (curbit - lastbit) + 16;
        lastbit = (2 + count) * 8;
    }

    ret = 0;
    for (i = curbit, j = 0; j < code_size; ++i, ++j)
        ret |= ((buf[i / 8] & (1 << (i % 8))) != 0) << j;

    curbit += code_size;

    return ret;
}

static int
LWZReadByte(FILE *fd, int flag, int input_code_size)
{
    static int fresh = FALSE;
    int code, incode;
    static int code_size, set_code_size;
    static int max_code, max_code_size;
    static int firstcode, oldcode;
    static int clear_code, end_code;
    static int table[2][(1 << MAX_LWZ_BITS)];
    static int stack[(1 << (MAX_LWZ_BITS)) * 2], *sp;
    register int i;

    if (flag)
    {
        set_code_size = input_code_size;
        code_size = set_code_size + 1;
        clear_code = 1 << set_code_size;
        end_code = clear_code + 1;
        max_code_size = 2 * clear_code;
        max_code = clear_code + 2;

        GetCode(fd, 0, TRUE);
        if (error)
            return -1;

        fresh = TRUE;

        for (i = 0; i < clear_code; ++i)
        {
            table[0][i] = 0;
            table[1][i] = i;
        }
        for (; i < (1 << MAX_LWZ_BITS); ++i)
            table[0][i] = table[1][0] = 0;

        sp = stack;

        return 0;
    }
    else if (fresh)
    {
        fresh = FALSE;
        do
        {
            firstcode = oldcode = GetCode(fd, code_size, FALSE);
            if (error)
                return -1;
        } while (firstcode == clear_code);
        return firstcode;
    }

    if (sp > stack)
        return *--sp;

    while ((code = GetCode(fd, code_size, FALSE)) >= 0)
    {
        if (error)
            return -1;
        if (code == clear_code)
        {
            for (i = 0; i < clear_code; ++i)
            {
                table[0][i] = 0;
                table[1][i] = i;
            }
            for (; i < (1 << MAX_LWZ_BITS); ++i)
                table[0][i] = table[1][i] = 0;
            code_size = set_code_size + 1;
            max_code_size = 2 * clear_code;
            max_code = clear_code + 2;
            sp = stack;
            firstcode = oldcode = GetCode(fd, code_size, FALSE);
            return firstcode;
        }
        else if (code == end_code)
        {
            int count;
            unsigned char buf[260];

            if (ZeroDataBlock)
                return -2;

            while ((count = GetDataBlock(fd, buf)) > 0)
                ;

            if (count != 0)
                pm_message("missing EOD in data stream (common occurence)");
            return -2;
        }

        incode = code;

        if (code >= max_code)
        {
            *sp++ = firstcode;
            code = oldcode;
        }

        while (code >= clear_code)
        {
            *sp++ = table[1][code];
            if (code == table[0][code])
                return pm_error("circular table entry BIG ERROR");
            code = table[0][code];
        }

        *sp++ = firstcode = table[1][code];

        if ((code = max_code) < (1 << MAX_LWZ_BITS))
        {
            table[0][code] = oldcode;
            table[1][code] = firstcode;
            ++max_code;
            if ((max_code >= max_code_size) && (max_code_size < (1 << MAX_LWZ_BITS)))
            {
                max_code_size *= 2;
                ++code_size;
            }
        }

        oldcode = incode;

        if (sp > stack)
            return *--sp;
    }
    return code;
}

static unsigned char *
ReadImage(FILE *fd, int len, int height,
          unsigned char cmap[3][MAXCOLORMAPSIZE],
          unsigned int grayscale,
          int interlace,
          int ignore, int xs, int ys)
{
    unsigned char c;
    int v;
    int xpos = 0, ypos = 0, pass = 0;
    unsigned char *image;
    unsigned char *image_ptr;
    int nc = 3; /* Number of components (RGB) */
    cmap = cmap;
    /*
	**  Initialize the Compression routines
	*/
    if (!ReadOK(fd, &c, 1))
    {
        pm_error("EOF / read error on image data");
        return NULL;
    }

    if (LWZReadByte(fd, TRUE, c) < 0)
    {
        pm_error("error reading image");
        return NULL;
    }

    /*
	**  If this is an "uninteresting picture" ignore it.
	*/
    if (ignore)
    {
        if (verbose)
            pm_message("skipping image...");

        while (LWZReadByte(fd, FALSE, c) >= 0)
            ;
        return NULL;
    }

    /* Allocate enough space for nc components.
	 */

    if (grayscale)
        nc = 1;
    if (Gif89.transparent >= 0)
        ++nc;

    image = (unsigned char *)malloc(nc * xs * ys);
    image_ptr = image;

    if (verbose)
        pm_message("reading %d by %d%s GIF image",
                   len, height, interlace ? " interlaced" : "");

    while ((v = LWZReadByte(fd, FALSE, c)) >= 0)
    {

        if (error)
        {
            free(image);
            return NULL;
        }

        *(image_ptr + (ypos * len + xpos)) = (unsigned char)v;
        if (v < MAXCOLORMAPSIZE)
            usedEntry[v] = TRUE;
        else
            fprintf(stderr, "ERROR in readgif\n");

        ++xpos;
        if (xpos == len)
        {
            xpos = 0;
            if (interlace)
            {
                switch (pass)
                {
                case 0:
                case 1:
                    ypos += 8;
                    break;
                case 2:
                    ypos += 4;
                    break;
                case 3:
                    ypos += 2;
                    break;
                }

                if (ypos >= height)
                {
                    ++pass;
                    switch (pass)
                    {
                    case 1:
                        ypos = 4;
                        break;
                    case 2:
                        ypos = 2;
                        break;
                    case 3:
                        ypos = 1;
                        break;
                    default:
                        goto fini;
                    }
                }
            }
            else
            {
                ++ypos;
            }
        }
        if (ypos >= height)
            break;
    }

fini:
    if (LWZReadByte(fd, FALSE, c) >= 0)
        pm_message("too much input data, ignoring extra...");

    if (verbose)
        pm_message("writing output");

    return image;
}

/************************************************************************/
/*
** Copyright (C) 1988 by Jef Poskanzer.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
*/

#include <stdarg.h>

static int showmessages = 1;

static void
pm_message(const char *format, ...)
{
    va_list args;

    va_start(args, format);
    if (showmessages)
    {
        fprintf(stderr, "Image reader: ");
        (void)vfprintf(stderr, format, args);
        fputc('\n', stderr);
    }
    va_end(args);
}

static int
pm_error(const char *format, ...)
{
    va_list args;

    va_start(args, format);

    fprintf(stderr, "Image reader: ");
    (void)vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
    error = TRUE;
    return 0;
}
