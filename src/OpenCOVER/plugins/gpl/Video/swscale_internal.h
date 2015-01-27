/*
 * Copyright (C) 2001-2003 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef FFMPEG_SWSCALE_INTERNAL_H
#define FFMPEG_SWSCALE_INTERNAL_H

//#include "config.h"

#ifdef HAVE_ALTIVEC_H
#include <altivec.h>
#endif

//#include "libavutil/avutil.h"

#define MAX_FILTER_SIZE 256

#define VOFW 2048
#define VOF (VOFW * 2)

typedef unsigned char uint8_t;
typedef short int16_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;

uint16_t bswap_16(uint16_t x);

typedef int (*SwsFunc)(struct SwsContext *context, uint8_t *src[], int srcStride[], int srcSliceY,
                       int srcSliceH, uint8_t *dst[], int dstStride[]);

/**
 * Pixel format. Notes:
 *
 * PIX_FMT_RGB32 is handled in an endian-specific manner. A RGBA
 * color is put together as:
 *  (A << 24) | (R << 16) | (G << 8) | B
 * This is stored as BGRA on little endian CPU architectures and ARGB on
 * big endian CPUs.
 *
 * When the pixel format is palettized RGB (PIX_FMT_PAL8), the palettized
 * image data is stored in AVFrame.data[0]. The palette is transported in
 * AVFrame.data[1] and, is 1024 bytes long (256 4-byte entries) and is
 * formatted the same as in PIX_FMT_RGB32 described above (i.e., it is
 * also endian-specific). Note also that the individual RGB palette
 * components stored in AVFrame.data[1] should be in the range 0..255.
 * This is important as many custom PAL8 video codecs that were designed
 * to run on the IBM VGA graphics adapter use 6-bit palette components.
 */
typedef enum
{
    PIX_FMT_NONE = -1,
    PIX_FMT_YUV420P, ///< Planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
    PIX_FMT_YUYV422, ///< Packed YUV 4:2:2, 16bpp, Y0 Cb Y1 Cr
    PIX_FMT_RGB24, ///< Packed RGB 8:8:8, 24bpp, RGBRGB...
    PIX_FMT_BGR24, ///< Packed RGB 8:8:8, 24bpp, BGRBGR...
    PIX_FMT_YUV422P, ///< Planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
    PIX_FMT_YUV444P, ///< Planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
    PIX_FMT_RGB32, ///< Packed RGB 8:8:8, 32bpp, (msb)8A 8R 8G 8B(lsb), in cpu endianness
    PIX_FMT_YUV410P, ///< Planar YUV 4:1:0,  9bpp, (1 Cr & Cb sample per 4x4 Y samples)
    PIX_FMT_YUV411P, ///< Planar YUV 4:1:1, 12bpp, (1 Cr & Cb sample per 4x1 Y samples)
    PIX_FMT_RGB565, ///< Packed RGB 5:6:5, 16bpp, (msb)   5R 6G 5B(lsb), in cpu endianness
    PIX_FMT_RGB555, ///< Packed RGB 5:5:5, 16bpp, (msb)1A 5R 5G 5B(lsb), in cpu endianness most significant
    /// bit to 0
    PIX_FMT_GRAY8, ///<        Y        ,  8bpp
    PIX_FMT_MONOWHITE, ///<        Y        ,  1bpp, 0 is white, 1 is black
    PIX_FMT_MONOBLACK, ///<        Y        ,  1bpp, 0 is black, 1 is white
    PIX_FMT_PAL8, ///< 8 bit with PIX_FMT_RGB32 palette
    PIX_FMT_YUVJ420P, ///< Planar YUV 4:2:0, 12bpp, full scale (jpeg)
    PIX_FMT_YUVJ422P, ///< Planar YUV 4:2:2, 16bpp, full scale (jpeg)
    PIX_FMT_YUVJ444P, ///< Planar YUV 4:4:4, 24bpp, full scale (jpeg)
    PIX_FMT_XVMC_MPEG2_MC, ///< XVideo Motion Acceleration via common packet passing(xvmc_render.h)
    PIX_FMT_XVMC_MPEG2_IDCT,
    PIX_FMT_UYVY422, ///< Packed YUV 4:2:2, 16bpp, Cb Y0 Cr Y1
    PIX_FMT_UYYVYY411, ///< Packed YUV 4:1:1, 12bpp, Cb Y0 Y1 Cr Y2 Y3
    PIX_FMT_BGR32, ///< Packed RGB 8:8:8, 32bpp, (msb)8A 8B 8G 8R(lsb), in cpu endianness
    PIX_FMT_BGR565, ///< Packed RGB 5:6:5, 16bpp, (msb)   5B 6G 5R(lsb), in cpu endianness
    PIX_FMT_BGR555, ///< Packed RGB 5:5:5, 16bpp, (msb)1A 5B 5G 5R(lsb), in cpu endianness most significant
    /// bit to 1
    PIX_FMT_BGR8, ///< Packed RGB 3:3:2,  8bpp, (msb)2B 3G 3R(lsb)
    PIX_FMT_BGR4, ///< Packed RGB 1:2:1,  4bpp, (msb)1B 2G 1R(lsb)
    PIX_FMT_BGR4_BYTE, ///< Packed RGB 1:2:1,  8bpp, (msb)1B 2G 1R(lsb)
    PIX_FMT_RGB8, ///< Packed RGB 3:3:2,  8bpp, (msb)2R 3G 3B(lsb)
    PIX_FMT_RGB4, ///< Packed RGB 1:2:1,  4bpp, (msb)2R 3G 3B(lsb)
    PIX_FMT_RGB4_BYTE, ///< Packed RGB 1:2:1,  8bpp, (msb)2R 3G 3B(lsb)
    PIX_FMT_NV12, ///< Planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 for UV
    PIX_FMT_NV21, ///< as above, but U and V bytes are swapped
    PIX_FMT_RGB32_1, ///< Packed RGB 8:8:8, 32bpp, (msb)8R 8G 8B 8A(lsb), in cpu endianness
    PIX_FMT_BGR32_1, ///< Packed RGB 8:8:8, 32bpp, (msb)8B 8G 8R 8A(lsb), in cpu endianness
    PIX_FMT_GRAY16BE, ///<        Y        , 16bpp, big-endian
    PIX_FMT_GRAY16LE, ///<        Y        , 16bpp, little-endian
    PIX_FMT_YUV440P, ///< Planar YUV 4:4:0 (1 Cr & Cb sample per 1x2 Y samples)
    PIX_FMT_YUVJ440P, ///< Planar YUV 4:4:0 full scale (jpeg)
    PIX_FMT_YUVA420P, ///< Planar YUV 4:2:0, 20bpp, (1 Cr & Cb sample per 2x2 Y & A samples)
    PIX_FMT_NB, ///< number of pixel formats, DO NOT USE THIS if you want to link with shared libav* because
    /// the number of formats might differ between versions
} PixelFormat;

typedef struct
{
    /**
    * The name of the class; usually it is the same name as the
    * context structure type to which the AVClass is associated.
    */
    const char *class_name;
} AVClass;

/* This struct should be aligned on at least a 32-byte boundary. */
typedef struct SwsContext
{
    /**
    * info on struct for av_log
    */
    const AVClass *av_class;

    /**
    * Note that src, dst, srcStride, dstStride will be copied in the
    * sws_scale() wrapper so they can be freely modified here.
    */
    SwsFunc swScale;
    int srcW, srcH, dstH;
    int chrSrcW, chrSrcH, chrDstW, chrDstH;
    int lumXInc, chrXInc;
    int lumYInc, chrYInc;
    int dstFormat, srcFormat; ///< format 4:2:0 type is always YV12
    int origDstFormat, origSrcFormat; ///< format
    int chrSrcHSubSample, chrSrcVSubSample;
    int chrIntHSubSample, chrIntVSubSample;
    int chrDstHSubSample, chrDstVSubSample;
    int vChrDrop;
    int sliceDir;
    double param[2];

    int16_t **lumPixBuf;
    int16_t **chrPixBuf;
    int16_t *hLumFilter;
    int16_t *hLumFilterPos;
    int16_t *hChrFilter;
    int16_t *hChrFilterPos;
    int16_t *vLumFilter;
    int16_t *vLumFilterPos;
    int16_t *vChrFilter;
    int16_t *vChrFilterPos;

    uint8_t formatConvBuffer[VOF]; // FIXME dynamic allocation, but we have to change a lot of code for this
    // to be useful

    int hLumFilterSize;
    int hChrFilterSize;
    int vLumFilterSize;
    int vChrFilterSize;
    int vLumBufSize;
    int vChrBufSize;

    uint8_t *funnyYCode;
    uint8_t *funnyUVCode;
    int32_t *lumMmx2FilterPos;
    int32_t *chrMmx2FilterPos;
    int16_t *lumMmx2Filter;
    int16_t *chrMmx2Filter;

    int canMMX2BeUsed;

    int lastInLumBuf;
    int lastInChrBuf;
    int lumBufIndex;
    int chrBufIndex;
    int dstY;
    int flags;
    void *yuvTable; // pointer to the yuv->rgb table start so it can be freed()
    uint8_t *table_rV[256];
    uint8_t *table_gU[256];
    int table_gV[256];
    uint8_t *table_bU[256];

    // Colorspace stuff
    int contrast, brightness, saturation; // for sws_getColorspaceDetails
    int srcColorspaceTable[4];
    int dstColorspaceTable[4];
    int srcRange, dstRange;

#define RED_DITHER "0*8"
#define GREEN_DITHER "1*8"
#define BLUE_DITHER "2*8"
#define Y_COEFF "3*8"
#define VR_COEFF "4*8"
#define UB_COEFF "5*8"
#define VG_COEFF "6*8"
#define UG_COEFF "7*8"
#define Y_OFFSET "8*8"
#define U_OFFSET "9*8"
#define V_OFFSET "10*8"
#define LUM_MMX_FILTER_OFFSET "11*8"
#define CHR_MMX_FILTER_OFFSET "11*8+4*4*256"
#define DSTW_OFFSET "11*8+4*4*256*2" // do not change, it is hardcoded in the ASM
#define ESP_OFFSET "11*8+4*4*256*2+8"
#define VROUNDER_OFFSET "11*8+4*4*256*2+16"
#define U_TEMP "11*8+4*4*256*2+24"
#define V_TEMP "11*8+4*4*256*2+32"

#ifdef WORDS_BIGENDIAN
#define PIX_FMT_RGBA PIX_FMT_RGB32_1
#define PIX_FMT_BGRA PIX_FMT_BGR32_1
#define PIX_FMT_ARGB PIX_FMT_RGB32
#define PIX_FMT_ABGR PIX_FMT_BGR32
#define PIX_FMT_GRAY16 PIX_FMT_GRAY16BE
#else
#define PIX_FMT_RGBA PIX_FMT_BGR32
#define PIX_FMT_BGRA PIX_FMT_RGB32
#define PIX_FMT_ARGB PIX_FMT_BGR32_1
#define PIX_FMT_ABGR PIX_FMT_RGB32_1
#define PIX_FMT_GRAY16 PIX_FMT_GRAY16LE
#endif

    uint64_t redDither /*__attribute__((aligned(8)))*/;
    uint64_t greenDither /*__attribute__((aligned(8)))*/;
    uint64_t blueDither /*__attribute__((aligned(8)))*/;

    uint64_t yCoeff /*__attribute__((aligned(8)))*/;
    uint64_t vrCoeff /*__attribute__((aligned(8)))*/;
    uint64_t ubCoeff /*__attribute__((aligned(8)))*/;
    uint64_t vgCoeff /*__attribute__((aligned(8)))*/;
    uint64_t ugCoeff /*__attribute__((aligned(8)))*/;
    uint64_t yOffset /*__attribute__((aligned(8)))*/;
    uint64_t uOffset /*__attribute__((aligned(8)))*/;
    uint64_t vOffset /*__attribute__((aligned(8)))*/;
    int32_t lumMmxFilter[4 * MAX_FILTER_SIZE];
    int32_t chrMmxFilter[4 * MAX_FILTER_SIZE];
    int dstW;
    uint64_t esp /*__attribute__((aligned(8)))*/;
    uint64_t vRounder /*__attribute__((aligned(8)))*/;
    uint64_t u_temp /*__attribute__((aligned(8)))*/;
    uint64_t v_temp /*__attribute__((aligned(8)))*/;

#ifdef HAVE_ALTIVEC

    vector signed short CY;
    vector signed short CRV;
    vector signed short CBU;
    vector signed short CGU;
    vector signed short CGV;
    vector signed short OY;
    vector unsigned short CSHIFT;
    vector signed short *vYCoeffsBank, *vCCoeffsBank;

#endif

#ifdef ARCH_BFIN
    uint32_t oy /*__attribute__((aligned(4)))*/;
    uint32_t oc /*__attribute__((aligned(4)))*/;
    uint32_t zero /*__attribute__((aligned(4)))*/;
    uint32_t cy /*__attribute__((aligned(4)))*/;
    uint32_t crv /*__attribute__((aligned(4)))*/;
    uint32_t rmask /*__attribute__((aligned(4)))*/;
    uint32_t cbu /*__attribute__((aligned(4)))*/;
    uint32_t bmask /*__attribute__((aligned(4)))*/;
    uint32_t cgu /*__attribute__((aligned(4)))*/;
    uint32_t cgv /*__attribute__((aligned(4)))*/;
    uint32_t gmask /*__attribute__((aligned(4)))*/;
#endif

#ifdef HAVE_VIS
    uint64_t sparc_coeffs[10] /*__attribute__((aligned(8)))*/;
#endif

} SwsContext;
// FIXME check init (where 0)

SwsFunc yuv2rgb_get_func_ptr(SwsContext *c);
int yuv2rgb_c_init_tables(SwsContext *c, const int inv_table[4], int fullRange, int brightness, int contrast,
                          int saturation);

void yuv2rgb_altivec_init_tables(SwsContext *c, const int inv_table[4], int brightness, int contrast,
                                 int saturation);
SwsFunc yuv2rgb_init_altivec(SwsContext *c);
void altivec_yuv2packedX(SwsContext *c, int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,
                         int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize, uint8_t *dest, int dstW,
                         int dstY);

const char *sws_format_name(int format);

// FIXME replace this with something faster
#define isPlanarYUV(x) \
    ((x) == PIX_FMT_YUV410P || (x) == PIX_FMT_YUV420P || (x) == PIX_FMT_YUV411P || (x) == PIX_FMT_YUV422P || (x) == PIX_FMT_YUV444P || (x) == PIX_FMT_YUV440P || (x) == PIX_FMT_NV12 || (x) == PIX_FMT_NV21)
#define isYUV(x) ((x) == PIX_FMT_UYVY422 || (x) == PIX_FMT_YUYV422 || isPlanarYUV(x))
#define isGray(x) ((x) == PIX_FMT_GRAY8 || (x) == PIX_FMT_GRAY16BE || (x) == PIX_FMT_GRAY16LE)
#define isGray16(x) ((x) == PIX_FMT_GRAY16BE || (x) == PIX_FMT_GRAY16LE)
#define isRGB(x) \
    ((x) == PIX_FMT_BGR32 || (x) == PIX_FMT_RGB24 || (x) == PIX_FMT_RGB565 || (x) == PIX_FMT_RGB555 || (x) == PIX_FMT_RGB8 || (x) == PIX_FMT_RGB4 || (x) == PIX_FMT_RGB4_BYTE || (x) == PIX_FMT_MONOBLACK)
#define isBGR(x) \
    ((x) == PIX_FMT_RGB32 || (x) == PIX_FMT_BGR24 || (x) == PIX_FMT_BGR565 || (x) == PIX_FMT_BGR555 || (x) == PIX_FMT_BGR8 || (x) == PIX_FMT_BGR4 || (x) == PIX_FMT_BGR4_BYTE || (x) == PIX_FMT_MONOBLACK)

// static inline int fmt_depth(int fmt)
static int fmt_depth(int fmt)
{
    switch (fmt)
    {
    case PIX_FMT_BGRA:
    case PIX_FMT_ABGR:
    case PIX_FMT_RGBA:
    case PIX_FMT_ARGB:
        return 32;
    case PIX_FMT_BGR24:
    case PIX_FMT_RGB24:
        return 24;
    case PIX_FMT_BGR565:
    case PIX_FMT_RGB565:
    case PIX_FMT_GRAY16BE:
    case PIX_FMT_GRAY16LE:
        return 16;
    case PIX_FMT_BGR555:
    case PIX_FMT_RGB555:
        return 15;
    case PIX_FMT_BGR8:
    case PIX_FMT_RGB8:
        return 8;
    case PIX_FMT_BGR4:
    case PIX_FMT_RGB4:
    case PIX_FMT_BGR4_BYTE:
    case PIX_FMT_RGB4_BYTE:
        return 4;
    case PIX_FMT_MONOBLACK:
        return 1;
    default:
        return 0;
    }
}

// extern const DECLARE_ALIGNED(8, uint64_t, ff_dither4[2]);
// extern const DECLARE_ALIGNED(8, uint64_t, ff_dither8[2]);

const AVClass sws_context_class;

#endif /* FFMPEG_SWSCALE_INTERNAL_H */
