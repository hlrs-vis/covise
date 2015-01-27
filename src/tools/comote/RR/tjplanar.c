/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* Copyright (C)2004 Landmark Graphics Corporation
 * Copyright (C)2005 Sun Microsystems, Inc.
 * Copyright (C)2009-2011 D. R. Commander
 *
 * This library is free software and may be redistributed and/or modified under
 * the terms of the wxWindows Library License, Version 3.1 or (at your option)
 * any later version.  The full license is in the LICENSE.txt file included
 * with this distribution.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * wxWindows Library License for more details.
 */

// This implements a JPEG compressor/decompressor using the libjpeg API

#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>
#include "turbojpeg.h"

#include "tjplanar.h"

#ifndef CSTATE_START
#define CSTATE_START 100
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

/***********************************************************
 * copied from turbojpegl.c
 ***********************************************************/

// Error handling

static char lasterror[JMSG_LENGTH_MAX] = "No error";

typedef struct _error_mgr
{
    struct jpeg_error_mgr pub;
    jmp_buf jb;
} error_mgr;

// Global structures, macros, etc.

typedef struct _jpgstruct
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_decompress_struct dinfo;
    struct jpeg_destination_mgr jdms;
    struct jpeg_source_mgr jsms;
    error_mgr jerr;
    int initc, initd;
} jpgstruct;

#define _throw(c)                                      \
    {                                                  \
        snprintf(lasterror, JMSG_LENGTH_MAX, "%s", c); \
        retval = -1;                                   \
        goto bailout;                                  \
    }
#define checkhandle(hnd)                                        \
    jpgstruct *j = (jpgstruct *)hnd;                            \
    if (!j)                                                     \
    {                                                           \
        snprintf(lasterror, JMSG_LENGTH_MAX, "Invalid handle"); \
        return -1;                                              \
    }

/******************************************************
 * end of code copied from turbojpegl.c
 ******************************************************/

#define PAD16(n) (((n) + 15) & ~15)
#define ROUNDUP(i, n) ((((i) + (n)-1) / (n)) * (n))
DLLEXPORT int DLLCALL tjCompressPlanar(tjhandle h,
                                       unsigned char **srcplane,
                                       int width, int pitch, int height, int ps,
                                       unsigned char *dstbuf,
                                       unsigned long *size,
                                       int jpegsub, int qual, int flags)
{
    unsigned long retval = 0;
    int i;
    JSAMPROW *row_pointer[4] = { NULL, NULL, NULL, NULL };
    unsigned char *pad = NULL;

    checkhandle(h);

    if (ps != 3 && ps != 4)
        _throw("This compressor can only handle 24-bit and 32-bit RGB input");
    if (srcplane == NULL || width <= 0 || pitch < 0 || height <= 0
        || dstbuf == NULL || size == NULL
        || jpegsub < 0 || jpegsub >= NUMSUBOPT || qual < 0 || qual > 100
        || srcplane[0] == NULL || srcplane[1] == NULL || srcplane[2] == NULL || srcplane[ps - 1] == NULL)
        _throw("Invalid argument in tjCompress()");
    if (!j->initc)
        _throw("Instance has not been initialized for compression");

    if (pitch == 0)
        pitch = width;

    if (flags & TJ_FORCEMMX)
        putenv("JSIMD_FORCEMMX=1");
    else if (flags & TJ_FORCESSE)
        putenv("JSIMD_FORCESSE=1");
    else if (flags & TJ_FORCESSE2)
        putenv("JSIMD_FORCESSE2=1");

    if (setjmp(j->jerr.jb))
    { // this will execute if LIBJPEG has an error
        fprintf(stderr, "setjmp: ERROR\n");

        retval = -1;
        goto bailout;
    }

    j->cinfo.image_width = width;
    j->cinfo.image_height = height;
    j->cinfo.input_components = ps;

    j->cinfo.in_color_space = JCS_YCbCr;

    jpeg_set_defaults(&j->cinfo);
    jpeg_set_quality(&j->cinfo, qual, TRUE);
    if (qual >= 96)
        j->cinfo.dct_method = JDCT_ISLOW;
    else
        j->cinfo.dct_method = JDCT_FASTEST;

    jpeg_set_colorspace(&j->cinfo, JCS_YCbCr);

    j->cinfo.image_width = width;
    j->cinfo.image_height = height;
    j->cinfo.input_components = ps;
    j->cinfo.raw_data_in = TRUE;

    j->cinfo.comp_info[0].h_samp_factor = tjMCUWidth[jpegsub] / 8;
    j->cinfo.comp_info[1].h_samp_factor = 1;
    j->cinfo.comp_info[2].h_samp_factor = 1;
    j->cinfo.comp_info[0].v_samp_factor = tjMCUHeight[jpegsub] / 8;
    j->cinfo.comp_info[1].v_samp_factor = 1;
    j->cinfo.comp_info[2].v_samp_factor = 1;

    j->jdms.next_output_byte = dstbuf;
    j->jdms.free_in_buffer = TJBUFSIZE(j->cinfo.image_width, j->cinfo.image_height);

    for (int plane = 0; plane < ps; ++plane)
    {
        if ((row_pointer[plane] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * tjMCUHeight[jpegsub])) == NULL)
            _throw("Memory allocation failed in tjCompressPlanar()");
    }

    jpeg_start_compress(&j->cinfo, TRUE);

    if (height % tjMCUHeight[jpegsub] != 0)
    {
        pad = malloc(pitch);
        memset(pad, '\0', pitch);
    }

    while (j->cinfo.next_scanline < j->cinfo.image_height)
    {
        for (int plane = 0; plane < ps; ++plane)
        {
            const int hs = plane == 0 ? 1 : tjMCUWidth[jpegsub] / 8;
            const int vs = plane == 0 ? 1 : tjMCUHeight[jpegsub] / 8;
            const int pitchsub = (pitch + hs - 1) / hs;
            const int hsub = (height + vs - 1) / vs;
            const int line = j->cinfo.next_scanline;
            for (i = 0; i < tjMCUHeight[jpegsub] / vs; i++)
            {
                if (flags & TJ_BOTTOMUP)
                {
                    if (line / vs + i >= (height + vs - 1) / vs)
                        row_pointer[plane][i] = pad;
                    else
                        row_pointer[plane][i] = &srcplane[plane][(hsub - line / vs - i - 1) * pitchsub];
                }
                else
                {
                    if (line / vs + i >= (height + vs - 1) / vs)
                        row_pointer[plane][i] = pad;
                    else
                        row_pointer[plane][i] = &srcplane[plane][(line / vs + i) * pitchsub];
                }
            }
        }

        jpeg_write_raw_data(&j->cinfo, row_pointer, tjMCUHeight[jpegsub]);
    }
    jpeg_finish_compress(&j->cinfo);
    *size = TJBUFSIZE(j->cinfo.image_width, j->cinfo.image_height)
            - (unsigned long)(j->jdms.free_in_buffer);

bailout:
    if (j->cinfo.global_state > CSTATE_START)
        jpeg_abort_compress(&j->cinfo);
    for (int plane = 0; plane < ps; ++plane)
        free(row_pointer[plane]);
    free(pad);

    return retval;
}

DLLEXPORT int DLLCALL tjDecompressPlanar(tjhandle h,
                                         unsigned char *srcbuf, unsigned long size,
                                         unsigned char **dstbuf, int width, int pitch, int height, int ncomp, int subsamp,
                                         int flags)
{
    unsigned long retval = 0;
    JSAMPROW *row_pointer[4] = { NULL, NULL, NULL, NULL };

    checkhandle(h);

    if (ncomp != 3 && ncomp != 4)
        _throw("This compressor can only take 24-bit or 32-bit RGB input");
    if (srcbuf == NULL || size <= 0
        || dstbuf == NULL || width <= 0 || pitch < 0 || height <= 0
        || dstbuf[0] == NULL || dstbuf[1] == NULL || dstbuf[2] == NULL || dstbuf[ncomp - 1] == NULL)
        _throw("Invalid argument in tjDecompressPlanar()");

    if (!j->initd)
        _throw("Instance has not been initialized for decompression");

    if (pitch == 0)
        pitch = width;

    if (flags & TJ_FORCEMMX)
        putenv("JSIMD_FORCEMMX=1");
    else if (flags & TJ_FORCESSE)
        putenv("JSIMD_FORCESSE=1");
    else if (flags & TJ_FORCESSE2)
        putenv("JSIMD_FORCESSE2=1");

    for (int plane = 0; plane < ncomp; ++plane)
    {
        const int hs = plane == 0 ? 1 : tjMCUWidth[jpegsub(subsamp)] / 8;
        const int vs = plane == 0 ? 1 : tjMCUHeight[jpegsub(subsamp)] / 8;
        const int pitchsub = (pitch + hs - 1) / hs;
        const int hsub = (height + vs - 1) / vs;
        const int hroundup = ROUNDUP(hsub, tjMCUHeight[jpegsub(subsamp)] / vs);

        if ((row_pointer[plane] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * hroundup)) == NULL)
            _throw("Memory allocation failed in tjDecompressPlanar()");

        for (int line = 0; line < hroundup; ++line)
        {
            if (flags & TJ_BOTTOMUP)
            {
                row_pointer[plane][line] = &dstbuf[plane][(hroundup - line - 1) * pitchsub];
            }
            else
            {
                row_pointer[plane][line] = &dstbuf[plane][line * pitchsub];
            }
        }
    }

    if (setjmp(j->jerr.jb))
    { // this will execute if LIBJPEG has an error
        fprintf(stderr, "setjmp: ERROR\n");
        fprintf(stderr, "%d %s", j->jerr.pub.msg_code, j->jerr.pub.msg_parm.s);

        jpeg_finish_decompress(&j->dinfo);

        retval = -1;
        goto bailout;
    }

    j->jsms.bytes_in_buffer = size;
    j->jsms.next_input_byte = srcbuf;

    jpeg_read_header(&j->dinfo, TRUE);

    if (flags & TJ_FASTUPSAMPLE)
        j->dinfo.do_fancy_upsampling = FALSE;

    j->dinfo.dct_method = JDCT_FASTEST;
    j->dinfo.raw_data_out = TRUE;
    j->dinfo.out_color_space = JCS_YCbCr;
    jpeg_start_decompress(&j->dinfo);
    while (j->dinfo.output_scanline < j->dinfo.output_height)
    {
        JSAMPROW *row[4] = { NULL, NULL, NULL, NULL };
        for (int plane = 0; plane < ncomp; ++plane)
        {
            const int vs = plane == 0 ? 1 : tjMCUHeight[jpegsub(subsamp)] / 8;
            row[plane] = &row_pointer[plane][j->dinfo.output_scanline / vs];
        }
        jpeg_read_raw_data(&j->dinfo, row,
                           ROUNDUP(j->dinfo.output_height - j->dinfo.output_scanline, tjMCUHeight[jpegsub(subsamp)]));
    }
    jpeg_finish_decompress(&j->dinfo);

bailout:
    for (int plane = 0; plane < ncomp; ++plane)
        free(row_pointer[plane]);

    return retval;
}
