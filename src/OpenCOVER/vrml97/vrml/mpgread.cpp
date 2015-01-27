/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* Derived from mpeg_play v2 */
/*
 * Copyright (c) 1992 The Regents of the University of California.
 * All rights reserved.
 * 
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written agreement is
 * hereby granted, provided that the above copyright notice and the following
 * two paragraphs appear in all copies of this software.
 * 
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF
 * CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#include <stdlib.h>
#include <string.h>
#include "mpgread.h"

#ifndef macintosh
#ifdef _WIN32
#include <Winsock2.h>
#include <windows.h>
#else
#include <sys/types.h> /* to make netinet/in.h happy */
#include <netinet/in.h> /* for htonl */
#endif
#else
#define htonl(x) (x)
#endif

static const int FRAMES_PER_ALLOC = 100;

/* mpeg.h */

/* Number of images stored at one time. */

#define RING_BUF_SIZE 5

/* Structure with reconstructed pixel values. */

typedef struct yuv_image
{
    unsigned char *luminance; /* Luminance plane.   */
    unsigned char *Cr; /* Cr plane.          */
    unsigned char *Cb; /* Cb plane.          */
    int locked; /* Lock flag.         */
} YUVImage;

/* Picture structure. */

typedef struct pict
{
    unsigned int code_type; /* Frame type: P, B, I             */
    int full_pel_forw_vector; /* Forw. vectors specified in full
					    pixel values flag.              */
    unsigned int forw_r_size; /* Used for vector decoding.       */
    unsigned int forw_f; /* Used for vector decoding.       */
    int full_pel_back_vector; /* Back vectors specified in full 
					    pixel values flag.              */
    unsigned int back_r_size; /* Used in decoding.               */
    unsigned int back_f; /* Used in decoding.               */
} Pict;

/* Slice structure. */

typedef struct slice
{
    unsigned int vert_pos; /* Vertical position of slice. */
    unsigned int quant_scale; /* Quantization scale.         */
} Slice;

/* Macroblock structure. */

typedef struct macroblock
{
    int mb_address; /* Macroblock address.              */
    int past_mb_addr; /* Previous mblock address.         */
    int motion_h_forw_code; /* Forw. horiz. motion vector code. */
    unsigned int motion_h_forw_r; /* Used in decoding vectors.        */
    int motion_v_forw_code; /* Forw. vert. motion vector code.  */
    unsigned int motion_v_forw_r; /* Used in decdoinge vectors.       */
    int motion_h_back_code; /* Back horiz. motion vector code.  */
    unsigned int motion_h_back_r; /* Used in decoding vectors.        */
    int motion_v_back_code; /* Back vert. motion vector code.   */
    unsigned int motion_v_back_r; /* Used in decoding vectors.        */
    unsigned int cbp; /* Coded block pattern.             */
    int mb_intra; /* Intracoded mblock flag.          */
    int bpict_past_forw; /* Past B frame forw. vector flag.  */
    int bpict_past_back; /* Past B frame back vector flag.   */
    int past_intra_addr; /* Addr of last intracoded mblock.  */
    int recon_right_for_prev; /* Past right forw. vector.         */
    int recon_down_for_prev; /* Past down forw. vector.          */
    int recon_right_back_prev; /* Past right back vector.          */
    int recon_down_back_prev; /* Past down back vector.           */
} Macroblock;

/* Block structure. */

typedef struct block
{
    short int dct_recon[8][8]; /* Reconstructed dct coeff matrix. */
    short int dct_dc_y_past; /* Past lum. dc dct coefficient.   */
    short int dct_dc_cr_past; /* Past cr dc dct coefficient.     */
    short int dct_dc_cb_past; /* Past cb dc dct coefficient.     */
} Block;

/* Video stream structure. */

struct mpeg_struct
{
    unsigned int cur_bits; /* Current bits.              */
    int buf_length; /* Length of remaining buffer.*/
    int bit_offset; /* Bit offset in stream.      */
    unsigned int *buffer; /* Pointer to next byte in
						  buffer.                    */
    unsigned int *buf_start; /* Pointer to buffer start.   */
    int max_buf_length; /* Max length of buffer.      */
    int state; /* State of decoding.         */
    FILE *fp; /* Input stream.              */
    Pict picture; /* Current picture.           */
    Slice slice; /* Current slice.             */
    Macroblock mblock; /* Current macroblock.        */
    Block block; /* Current block.             */
    YUVImage *past; /* Past predictive frame.     */
    YUVImage *future; /* Future predictive frame.   */
    YUVImage *current; /* Current frame.             */
    YUVImage *ring[RING_BUF_SIZE]; /* Ring buffer of frames.     */
    unsigned int h_size; /* Horiz. size in pixels.     */
    unsigned int v_size; /* Vert. size in pixels.      */
    unsigned int mb_height; /* Vert. size in mblocks.     */
    unsigned int mb_width; /* Horiz. size in mblocks.    */
    unsigned char aspect_ratio; /* Code for aspect ratio.     */
    unsigned char picture_rate; /* Code for picture rate.     */
    unsigned int bit_rate; /* Bit rate.                  */
    unsigned int vbv_buffer_size; /* Minimum buffer size.       */
    int const_param_flag; /* Constrained parameter flag. */
    unsigned char intra_quant_matrix[8][8]; /* Quantization matrix for
						  intracoded frames.         */
    unsigned char non_intra_quant_matrix[8][8]; /* Quanitization matrix for 
						  non intracoded frames.     */
};

#define MPEG_NOFILE 1
#define MPEG_NOMEM 2
#define MPEG_NOTMPEG 3
#define MPEG_READERR 4

#define MPEG_CONV_RGB 0
#define MPEG_CONV_RGBA 1

typedef struct mpeg_struct MPEG;

#define MPEGImageHeight(_m) ((_m)->mb_height * 16)
#define MPEGImageWidth(_m) ((_m)->mb_width * 16)

static MPEG *MPEGOpen(FILE *, int);
static void MPEGClose(MPEG *);
static int MPEGAdvanceFrame(MPEG *);
static int MPEGConvertImage(MPEG *, int, unsigned char *);
/*static int   MPEGRewind (MPEG *); Not used */

unsigned char *mpgread(FILE *fp, int *w, int *h, int *nc, int *nf,
                       unsigned char ***frames)
{
    MPEG *m = MPEGOpen(fp, 0);
    if (m == 0)
        return 0;

    *w = MPEGImageWidth(m);
    *h = MPEGImageHeight(m);
    *nc = 3;

    int nfalloc = FRAMES_PER_ALLOC;
    *frames = (unsigned char **)malloc(nfalloc * sizeof(unsigned char *));
    if (!*frames)
        return 0;

    while (MPEGAdvanceFrame(m))
    {
        unsigned char *pixels = (unsigned char *)malloc(MPEGImageWidth(m) * MPEGImageHeight(m) * 3);

        MPEGConvertImage(m, MPEG_CONV_RGB, pixels);
        if (*nf >= nfalloc - 1)
        {
            nfalloc += FRAMES_PER_ALLOC;
            *frames = (unsigned char **)realloc(*frames, nfalloc * sizeof(unsigned char *));
            if (!*frames)
                return 0;
        }

        (*frames)[*nf] = pixels;
        *nf = *nf + 1;
    }

    MPEGClose(m);
    return (*frames)[0];
}

/*
 * decoders.h
 *
 * This file contains the declarations of structures required for Huffman
 * decoding
 *
 */

typedef int INT32;
typedef short INT16;
#ifndef _WIN32
typedef char INT8;
#endif

/* Code for unbound values in decoding tables */
#define MPG_ERROR ((unsigned)-1)
#define DCT_MPG_ERROR 63

#define MACRO_BLOCK_STUFFING 34
#define MACRO_BLOCK_ESCAPE 35

/* Two types of DCT Coefficients */
#define DCT_COEFF_FIRST 0
#define DCT_COEFF_NEXT 1

/* Special values for DCT Coefficients */
#define END_OF_BLOCK 62
#define ESCAPE 61

/* Structure for an entry in the decoding table of 
 * macroblock_address_increment */
typedef struct
{
    unsigned int value; /* value for macroblock_address_increment */
    int num_bits; /* length of the Huffman code */
} vb_entry;

/* Structure for an entry in the decoding table of macroblock_type */
typedef struct
{
    unsigned int mb_quant; /* macroblock_quant */
    unsigned int mb_motion_forward; /* macroblock_motion_forward */
    unsigned int mb_motion_backward; /* macroblock_motion_backward */
    unsigned int mb_pattern; /* macroblock_pattern */
    unsigned int mb_intra; /* macroblock_intra */
    int num_bits; /* length of the Huffman code */
} mb_type_entry;

/* Structures for an entry in the decoding table of coded_block_pattern */
typedef struct
{
    unsigned int cbp; /* coded_block_pattern */
    int num_bits; /* length of the Huffman code */
} coded_block_pattern_entry;

/* Structure for an entry in the decoding table of motion vectors */
typedef struct
{
    int code; /* value for motion_horizontal_forward_code,
			  * motion_vertical_forward_code, 
			  * motion_horizontal_backward_code, or
			  * motion_vertical_backward_code.
			  */
    int num_bits; /* length of the Huffman code */
} motion_vectors_entry;

/* decoders.c */
static void MPEGInitTables(vb_entry *, mb_type_entry *, mb_type_entry *,
                           motion_vectors_entry *);

/* Definition of Contant integer scale factor. */

#define CONST_BITS 13

/*
 * This routine is specialized to the case DCTSIZE = 8.
 */

#define DCTSIZE 8
#define DCTSIZE2 64 /* DCTSIZE squared; # of elements in a block */

typedef short DCTELEM;
typedef DCTELEM DCTBLOCK[DCTSIZE2];

/* jrevdct.c */
static void init_pre_idct(void);
static void j_rev_dct_sparse(DCTBLOCK, int);
static void j_rev_dct(DCTBLOCK);

/*
 * decoders.c
 *
 * This file contains all the routines for Huffman decoding required in 
 * MPEG
 *
 */

/* Macro for filling up the decoding table for mb_addr_inc */
#define ASSIGN1(start, end, step, val, num)    \
    for (i = start; i < end; i += step)        \
    {                                          \
        for (j = 0; j < step; j++)             \
        {                                      \
            mb_addr_inc[i + j].value = val;    \
            mb_addr_inc[i + j].num_bits = num; \
        }                                      \
        val--;                                 \
    }

/*
 *--------------------------------------------------------------
 *
 * init_mb_addr_inc --
 *
 *	Initialize the VLC decoding table for macro_block_address_increment
 *
 * Results:
 *	The decoding table for macro_block_address_increment will
 *      be filled; illegal values will be filled as MPG_ERROR.
 *
 * Side effects:
 *	The array mb_addr_inc will be filled.
 *
 *--------------------------------------------------------------
 */
static void init_mb_addr_inc(vb_entry *mb_addr_inc)
{
    int i, j, val;

    for (i = 0; i < 8; i++)
    {
        mb_addr_inc[i].value = MPG_ERROR;
        mb_addr_inc[i].num_bits = 0;
    }

    mb_addr_inc[8].value = MACRO_BLOCK_ESCAPE;
    mb_addr_inc[8].num_bits = 11;

    for (i = 9; i < 15; i++)
    {
        mb_addr_inc[i].value = MPG_ERROR;
        mb_addr_inc[i].num_bits = 0;
    }

    mb_addr_inc[15].value = MACRO_BLOCK_STUFFING;
    mb_addr_inc[15].num_bits = 11;

    for (i = 16; i < 24; i++)
    {
        mb_addr_inc[i].value = MPG_ERROR;
        mb_addr_inc[i].num_bits = 0;
    }

    val = 33;

    ASSIGN1(24, 36, 1, val, 11);
    ASSIGN1(36, 48, 2, val, 10);
    ASSIGN1(48, 96, 8, val, 8);
    ASSIGN1(96, 128, 16, val, 7);
    ASSIGN1(128, 256, 64, val, 5);
    ASSIGN1(256, 512, 128, val, 4);
    ASSIGN1(512, 1024, 256, val, 3);
    ASSIGN1(1024, 2048, 1024, val, 1);
}

/* Macro for filling up the decoding table for mb_type */
#define ASSIGN2(start, end, quant, motion_forward, motion_backward, pattern, intra, num, mb_type) \
    for (i = start; i < end; i++)                                                                 \
    {                                                                                             \
        mb_type[i].mb_quant = quant;                                                              \
        mb_type[i].mb_motion_forward = motion_forward;                                            \
        mb_type[i].mb_motion_backward = motion_backward;                                          \
        mb_type[i].mb_pattern = pattern;                                                          \
        mb_type[i].mb_intra = intra;                                                              \
        mb_type[i].num_bits = num;                                                                \
    }

/*
 *--------------------------------------------------------------
 *
 * init_mb_type_P --
 *
 *	Initialize the VLC decoding table for macro_block_type in
 *      predictive-coded pictures.
 *
 * Results:
 *	The decoding table for macro_block_type in predictive-coded
 *      pictures will be filled; illegal values will be filled as MPG_ERROR.
 *
 * Side effects:
 *	The array mb_type_P will be filled.
 *
 *--------------------------------------------------------------
 */
static void init_mb_type_P(mb_type_entry *mb_type_P)
{
    int i;

    mb_type_P[0].mb_quant = mb_type_P[0].mb_motion_forward
        = mb_type_P[0].mb_motion_backward = mb_type_P[0].mb_pattern
        = mb_type_P[0].mb_intra = MPG_ERROR;
    mb_type_P[0].num_bits = 0;

    ASSIGN2(1, 2, 1, 0, 0, 0, 1, 6, mb_type_P)
    ASSIGN2(2, 4, 1, 0, 0, 1, 0, 5, mb_type_P)
    ASSIGN2(4, 6, 1, 1, 0, 1, 0, 5, mb_type_P);
    ASSIGN2(6, 8, 0, 0, 0, 0, 1, 5, mb_type_P);
    ASSIGN2(8, 16, 0, 1, 0, 0, 0, 3, mb_type_P);
    ASSIGN2(16, 32, 0, 0, 0, 1, 0, 2, mb_type_P);
    ASSIGN2(32, 64, 0, 1, 0, 1, 0, 1, mb_type_P);
}

/*
 *--------------------------------------------------------------
 *
 * init_mb_type_B --
 *
 *	Initialize the VLC decoding table for macro_block_type in
 *      bidirectionally-coded pictures.
 *
 * Results:
 *	The decoding table for macro_block_type in bidirectionally-coded
 *      pictures will be filled; illegal values will be filled as MPG_ERROR.
 *
 * Side effects:
 *	The array mb_type_B will be filled.
 *
 *--------------------------------------------------------------
 */
static void init_mb_type_B(mb_type_entry *mb_type_B)
{
    int i;

    mb_type_B[0].mb_quant = mb_type_B[0].mb_motion_forward
        = mb_type_B[0].mb_motion_backward = mb_type_B[0].mb_pattern
        = mb_type_B[0].mb_intra = MPG_ERROR;
    mb_type_B[0].num_bits = 0;

    ASSIGN2(1, 2, 1, 0, 0, 0, 1, 6, mb_type_B);
    ASSIGN2(2, 3, 1, 0, 1, 1, 0, 6, mb_type_B);
    ASSIGN2(3, 4, 1, 1, 0, 1, 0, 6, mb_type_B);
    ASSIGN2(4, 6, 1, 1, 1, 1, 0, 5, mb_type_B);
    ASSIGN2(6, 8, 0, 0, 0, 0, 1, 5, mb_type_B);
    ASSIGN2(8, 12, 0, 1, 0, 0, 0, 4, mb_type_B);
    ASSIGN2(12, 16, 0, 1, 0, 1, 0, 4, mb_type_B);
    ASSIGN2(16, 24, 0, 0, 1, 0, 0, 3, mb_type_B);
    ASSIGN2(24, 32, 0, 0, 1, 1, 0, 3, mb_type_B);
    ASSIGN2(32, 48, 0, 1, 1, 0, 0, 2, mb_type_B);
    ASSIGN2(48, 64, 0, 1, 1, 1, 0, 2, mb_type_B);
}

/* Macro for filling up the decoding tables for motion_vectors */
#define ASSIGN3(start, end, step, val, num)       \
    for (i = start; i < end; i += step)           \
    {                                             \
        for (j = 0; j < step / 2; j++)            \
        {                                         \
            motion_vectors[i + j].code = val;     \
            motion_vectors[i + j].num_bits = num; \
        }                                         \
        for (j = step / 2; j < step; j++)         \
        {                                         \
            motion_vectors[i + j].code = -val;    \
            motion_vectors[i + j].num_bits = num; \
        }                                         \
        val--;                                    \
    }

/*
 *--------------------------------------------------------------
 *
 * init_motion_vectors --
 *
 *	Initialize the VLC decoding table for the various motion
 *      vectors, including motion_horizontal_forward_code, 
 *      motion_vertical_forward_code, motion_horizontal_backward_code,
 *      and motion_vertical_backward_code.
 *
 * Results:
 *	The decoding table for the motion vectors will be filled;
 *      illegal values will be filled as MPG_ERROR.
 *
 * Side effects:
 *	The array motion_vectors will be filled.
 *
 *--------------------------------------------------------------
 */
static void init_motion_vectors(motion_vectors_entry *motion_vectors)
{
    int i, j, val = 16;

    for (i = 0; i < 24; i++)
    {
        motion_vectors[i].code = (int)MPG_ERROR;
        motion_vectors[i].num_bits = 0;
    }

    ASSIGN3(24, 36, 2, val, 11);
    ASSIGN3(36, 48, 4, val, 10);
    ASSIGN3(48, 96, 16, val, 8);
    ASSIGN3(96, 128, 32, val, 7);
    ASSIGN3(128, 256, 128, val, 5);
    ASSIGN3(256, 512, 256, val, 4);
    ASSIGN3(512, 1024, 512, val, 3);
    ASSIGN3(1024, 2048, 1024, val, 1);
}

/*
 *--------------------------------------------------------------
 *
 * init_tables --
 *
 *	Initialize all the tables for VLC decoding; this must be
 *      called when the system is set up before any decoding can
 *      take place.
 *
 * Results:
 *	All the decoding tables will be filled accordingly.
 *
 * Side effects:
 *	The corresponding global array for each decoding table 
 *      will be filled.
 *
 *--------------------------------------------------------------
 */
static void MPEGInitTables(vb_entry *a, mb_type_entry *p, mb_type_entry *b,
                           motion_vectors_entry *m)
{
    init_mb_addr_inc(a);
    init_mb_type_P(p);
    init_mb_type_B(b);
    init_motion_vectors(m);
    init_pre_idct();
}

/*
 * jrevdct.c
 *
 * Copyright (C) 1991, 1992, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains the basic inverse-DCT transformation subroutine.
 *
 * This implementation is based on an algorithm described in
 *   C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 * The primary algorithm described there uses 11 multiplies and 29 adds.
 * We use their alternate method with 12 multiplies and 32 adds.
 * The advantage of this method is that no data path contains more than one
 * multiplication; this allows a very simple and accurate implementation in
 * scaled fixed-point arithmetic, with a minimal number of shifts.
 * 
 * I've made lots of modifications to attempt to take advantage of the
 * sparse nature of the DCT matrices we're getting.  Although the logic
 * is cumbersome, it's straightforward and the resulting code is much
 * faster.
 *
 * A better way to do this would be to pass in the DCT block as a sparse
 * matrix, perhaps with the difference cases encoded.
 */

/* We assume that right shift corresponds to signed division by 2 with
 * rounding towards minus infinity.  This is correct for typical "arithmetic
 * shift" instructions that shift in copies of the sign bit.  But some
 * C compilers implement >> with an unsigned shift.  For these machines you
 * must define RIGHT_SHIFT_IS_UNSIGNED.
 * RIGHT_SHIFT provides a proper signed right shift of an INT32 quantity.
 * It is only applied with constant shift counts.  SHIFT_TEMPS must be
 * included in the variables of any routine using RIGHT_SHIFT.
 */

#ifdef RIGHT_SHIFT_IS_UNSIGNED
#define SHIFT_TEMPS INT32 shift_temp;
#define RIGHT_SHIFT(x, shft) \
    ((shift_temp = (x)) < 0 ? (shift_temp >> (shft)) | ((~((INT32)0)) << (32 - (shft))) : (shift_temp >> (shft)))
#else
#define SHIFT_TEMPS
#define RIGHT_SHIFT(x, shft) ((x) >> (shft))
#endif

/*
 * A 2-D IDCT can be done by 1-D IDCT on each row followed by 1-D IDCT
 * on each column.  Direct algorithms are also available, but they are
 * much more complex and seem not to be any faster when reduced to code.
 *
 * The poop on this scaling stuff is as follows:
 *
 * Each 1-D IDCT step produces outputs which are a factor of sqrt(N)
 * larger than the true IDCT outputs.  The final outputs are therefore
 * a factor of N larger than desired; since N=8 this can be cured by
 * a simple right shift at the end of the algorithm.  The advantage of
 * this arrangement is that we save two multiplications per 1-D IDCT,
 * because the y0 and y4 inputs need not be divided by sqrt(N).
 *
 * We have to do addition and subtraction of the integer inputs, which
 * is no problem, and multiplication by fractional constants, which is
 * a problem to do in integer arithmetic.  We multiply all the constants
 * by CONST_SCALE and convert them to integer constants (thus retaining
 * CONST_BITS bits of precision in the constants).  After doing a
 * multiplication we have to divide the product by CONST_SCALE, with proper
 * rounding, to produce the correct output.  This division can be done
 * cheaply as a right shift of CONST_BITS bits.  We postpone shifting
 * as long as possible so that partial sums can be added together with
 * full fractional precision.
 *
 * The outputs of the first pass are scaled up by PASS1_BITS bits so that
 * they are represented to better-than-integral precision.  These outputs
 * require BITS_IN_JSAMPLE + PASS1_BITS + 3 bits; this fits in a 16-bit word
 * with the recommended scaling.  (To scale up 12-bit sample data further, an
 * intermediate INT32 array would be needed.)
 *
 * To avoid overflow of the 32-bit intermediate results in pass 2, we must
 * have BITS_IN_JSAMPLE + CONST_BITS + PASS1_BITS <= 26.  MPG_ERROR analysis
 * shows that the values given below are the most effective.
 */

#ifdef EIGHT_BIT_SAMPLES
#define PASS1_BITS 2
#else
#define PASS1_BITS 1 /* lose a little precision to avoid overflow */
#endif

#define ONE ((INT32)1)

#define CONST_SCALE (ONE << CONST_BITS)

/* Convert a positive real constant to an integer scaled by CONST_SCALE.
 * IMPORTANT: if your compiler doesn't do this arithmetic at compile time,
 * you will pay a significant penalty in run time.  In that case, figure
 * the correct integer constant values and insert them by hand.
 */

#define FIX(x) ((INT32)((x)*CONST_SCALE + 0.5))

/* Descale and correctly round an INT32 value that's scaled by N bits.
 * We assume RIGHT_SHIFT rounds towards minus infinity, so adding
 * the fudge factor is correct for either sign of X.
 */

#define DESCALE(x, n) RIGHT_SHIFT((x) + (ONE << ((n)-1)), n)

/* Multiply an INT32 variable by an INT32 constant to yield an INT32 result.
 * For 8-bit samples with the recommended scaling, all the variable
 * and constant values involved are no more than 16 bits wide, so a
 * 16x16->32 bit multiply can be used instead of a full 32x32 multiply;
 * this provides a useful speedup on many machines.
 * There is no way to specify a 16x16->32 multiply in portable C, but
 * some C compilers will do the right thing if you provide the correct
 * combination of casts.
 * NB: for 12-bit samples, a full 32-bit multiplication will be needed.
 */

#ifdef EIGHT_BIT_SAMPLES
#ifdef SHORTxSHORT_32 /* may work if 'int' is 32 bits */
#define MULTIPLY(var, const) (((INT16)(var)) * ((INT16)(const)))
#endif
#ifdef SHORTxLCONST_32 /* known to work with Microsoft C 6.0 */
#define MULTIPLY(var, const) (((INT16)(var)) * ((INT32)(const)))
#endif
#endif

#ifndef MULTIPLY /* default definition */
#define MULTIPLY(var, const) ((var) * (const))
#endif

/* Precomputed idct value arrays. */

static DCTELEM PreIDCT[64][64];

/* Pre compute singleton coefficient IDCT values. */
static void init_pre_idct()
{
    int i;

    for (i = 0; i < 64; i++)
    {
        memset((char *)PreIDCT[i], 0, 64 * sizeof(DCTELEM));
        PreIDCT[i][i] = 2048;
        j_rev_dct(PreIDCT[i]);
    }
}

#ifndef ORIG_DCT

/*
 * Perform the inverse DCT on one block of coefficients.
 */

static void
j_rev_dct_sparse(DCTBLOCK data, int pos)
{
    register DCTELEM *dataptr;
    short int val;
    DCTELEM *ndataptr;
    int coeff, rr;
    register int *dp;
    register int v;

    /* If DC Coefficient. */

    if (pos == 0)
    {
        dp = (int *)data;
        v = *data;
        /* Compute 32 bit value to assign.  This speeds things up a bit */
        if (v < 0)
            val = (v - 3) >> 3;
        else
            val = (v + 4) >> 3;
        v = val | (val << 16);
        dp[0] = v;
        dp[1] = v;
        dp[2] = v;
        dp[3] = v;
        dp[4] = v;
        dp[5] = v;
        dp[6] = v;
        dp[7] = v;
        dp[8] = v;
        dp[9] = v;
        dp[10] = v;
        dp[11] = v;
        dp[12] = v;
        dp[13] = v;
        dp[14] = v;
        dp[15] = v;
        dp[16] = v;
        dp[17] = v;
        dp[18] = v;
        dp[19] = v;
        dp[20] = v;
        dp[21] = v;
        dp[22] = v;
        dp[23] = v;
        dp[24] = v;
        dp[25] = v;
        dp[26] = v;
        dp[27] = v;
        dp[28] = v;
        dp[29] = v;
        dp[30] = v;
        dp[31] = v;
        return;
    }

    /* Some other coefficient. */
    dataptr = (DCTELEM *)data;
    coeff = dataptr[pos];
    ndataptr = PreIDCT[pos];

    for (rr = 0; rr < 4; rr++)
    {
        dataptr[0] = (ndataptr[0] * coeff) >> (CONST_BITS - 2);
        dataptr[1] = (ndataptr[1] * coeff) >> (CONST_BITS - 2);
        dataptr[2] = (ndataptr[2] * coeff) >> (CONST_BITS - 2);
        dataptr[3] = (ndataptr[3] * coeff) >> (CONST_BITS - 2);
        dataptr[4] = (ndataptr[4] * coeff) >> (CONST_BITS - 2);
        dataptr[5] = (ndataptr[5] * coeff) >> (CONST_BITS - 2);
        dataptr[6] = (ndataptr[6] * coeff) >> (CONST_BITS - 2);
        dataptr[7] = (ndataptr[7] * coeff) >> (CONST_BITS - 2);
        dataptr[8] = (ndataptr[8] * coeff) >> (CONST_BITS - 2);
        dataptr[9] = (ndataptr[9] * coeff) >> (CONST_BITS - 2);
        dataptr[10] = (ndataptr[10] * coeff) >> (CONST_BITS - 2);
        dataptr[11] = (ndataptr[11] * coeff) >> (CONST_BITS - 2);
        dataptr[12] = (ndataptr[12] * coeff) >> (CONST_BITS - 2);
        dataptr[13] = (ndataptr[13] * coeff) >> (CONST_BITS - 2);
        dataptr[14] = (ndataptr[14] * coeff) >> (CONST_BITS - 2);
        dataptr[15] = (ndataptr[15] * coeff) >> (CONST_BITS - 2);
        dataptr += 16;
        ndataptr += 16;
    }
    return;
}

static void
j_rev_dct(DCTBLOCK data)
{
    INT32 tmp0, tmp1, tmp2, tmp3;
    INT32 tmp10, tmp11, tmp12, tmp13;
    INT32 z1, z2, z3, z4, z5;
    INT32 d0, d1, d2, d3, d4, d5, d6, d7;
    register DCTELEM *dataptr;
    int rowctr;
    SHIFT_TEMPS

    /* Pass 1: process rows. */
    /* Note results are scaled up by sqrt(8) compared to a true IDCT; */
    /* furthermore, we scale the results by 2**PASS1_BITS. */

    dataptr = data;

    for (rowctr = DCTSIZE - 1; rowctr >= 0; rowctr--)
    {
        /* Due to quantization, we will usually find that many of the input
     * coefficients are zero, especially the AC terms.  We can exploit this
     * by short-circuiting the IDCT calculation for any row in which all
     * the AC terms are zero.  In that case each output is equal to the
     * DC coefficient (with scale factor as needed).
     * With typical images and quantization tables, half or more of the
     * row DCT calculations can be simplified this way.
     */

        register int *idataptr = (int *)dataptr;
        d0 = dataptr[0];
        d1 = dataptr[1];
        if ((d1 == 0) && (idataptr[1] | idataptr[2] | idataptr[3]) == 0)
        {
            /* AC terms all zero */
            if (d0)
            {
                /* Compute a 32 bit value to assign. */
                DCTELEM dcval = (DCTELEM)(d0 << PASS1_BITS);
                register int v = (dcval & 0xffff) | ((dcval << 16) & 0xffff0000);

                idataptr[0] = v;
                idataptr[1] = v;
                idataptr[2] = v;
                idataptr[3] = v;
            }

            dataptr += DCTSIZE; /* advance pointer to next row */
            continue;
        }
        d2 = dataptr[2];
        d3 = dataptr[3];
        d4 = dataptr[4];
        d5 = dataptr[5];
        d6 = dataptr[6];
        d7 = dataptr[7];

        /* Even part: reverse the even part of the forward DCT. */
        /* The rotator is sqrt(2)*c(-6). */
        if (d6)
        {
            if (d4)
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 != 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp0 = (d0 + d4) << CONST_BITS;
                        tmp1 = (d0 - d4) << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp1 + tmp2;
                        tmp12 = tmp1 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 != 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp0 = d4 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp2 - tmp0;
                        tmp12 = -(tmp0 + tmp2);
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 != 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp0 = (d0 + d4) << CONST_BITS;
                        tmp1 = (d0 - d4) << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp1 + tmp2;
                        tmp12 = tmp1 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 != 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp0 = d4 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp2 - tmp0;
                        tmp12 = -(tmp0 + tmp2);
                    }
                }
            }
            else
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 == 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp0 = d0 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp0 + tmp2;
                        tmp12 = tmp0 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 == 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp10 = tmp3;
                        tmp13 = -tmp3;
                        tmp11 = tmp2;
                        tmp12 = -tmp2;
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 == 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp0 = d0 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp0 + tmp2;
                        tmp12 = tmp0 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 == 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp10 = tmp3;
                        tmp13 = -tmp3;
                        tmp11 = tmp2;
                        tmp12 = -tmp2;
                    }
                }
            }
        }
        else
        {
            if (d4)
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 != 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp0 = (d0 + d4) << CONST_BITS;
                        tmp1 = (d0 - d4) << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp1 + tmp2;
                        tmp12 = tmp1 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 != 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp0 = d4 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp2 - tmp0;
                        tmp12 = -(tmp0 + tmp2);
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 != 0, d6 == 0 */
                        tmp10 = tmp13 = (d0 + d4) << CONST_BITS;
                        tmp11 = tmp12 = (d0 - d4) << CONST_BITS;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 != 0, d6 == 0 */
                        tmp10 = tmp13 = d4 << CONST_BITS;
                        tmp11 = tmp12 = -tmp10;
                    }
                }
            }
            else
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 == 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp0 = d0 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp0 + tmp2;
                        tmp12 = tmp0 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 == 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp10 = tmp3;
                        tmp13 = -tmp3;
                        tmp11 = tmp2;
                        tmp12 = -tmp2;
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 == 0, d6 == 0 */
                        tmp10 = tmp13 = tmp11 = tmp12 = d0 << CONST_BITS;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 == 0, d6 == 0 */
                        tmp10 = tmp13 = tmp11 = tmp12 = 0;
                    }
                }
            }
        }

        /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

        if (d7)
        {
            if (d5)
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 != 0, d7 != 0 */
                        z1 = d7 + d1;
                        z2 = d5 + d3;
                        z3 = d7 + d3;
                        z4 = d5 + d1;
                        z5 = MULTIPLY(z3 + z4, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(z1, -FIX(0.899976223));
                        z2 = MULTIPLY(z2, -FIX(2.562915447));
                        z3 = MULTIPLY(z3, -FIX(1.961570560));
                        z4 = MULTIPLY(z4, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 != 0, d7 != 0 */
                        z1 = d7;
                        z2 = d5 + d3;
                        z3 = d7 + d3;
                        z5 = MULTIPLY(z3 + d5, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        z1 = MULTIPLY(d7, -FIX(0.899976223));
                        z2 = MULTIPLY(z2, -FIX(2.562915447));
                        z3 = MULTIPLY(z3, -FIX(1.961570560));
                        z4 = MULTIPLY(d5, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 = z1 + z4;
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 != 0, d7 != 0 */
                        z1 = d7 + d1;
                        z2 = d5;
                        z3 = d7;
                        z4 = d5 + d1;
                        z5 = MULTIPLY(z3 + z4, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(z1, -FIX(0.899976223));
                        z2 = MULTIPLY(d5, -FIX(2.562915447));
                        z3 = MULTIPLY(d7, -FIX(1.961570560));
                        z4 = MULTIPLY(z4, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 = z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 != 0, d7 != 0 */
                        tmp0 = MULTIPLY(d7, -FIX(0.601344887));
                        z1 = MULTIPLY(d7, -FIX(0.899976223));
                        z3 = MULTIPLY(d7, -FIX(1.961570560));
                        tmp1 = MULTIPLY(d5, -FIX(0.509795578));
                        z2 = MULTIPLY(d5, -FIX(2.562915447));
                        z4 = MULTIPLY(d5, -FIX(0.390180644));
                        z5 = MULTIPLY(d5 + d7, FIX(1.175875602));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z3;
                        tmp1 += z4;
                        tmp2 = z2 + z3;
                        tmp3 = z1 + z4;
                    }
                }
            }
            else
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 == 0, d7 != 0 */
                        z1 = d7 + d1;
                        z3 = d7 + d3;
                        z5 = MULTIPLY(z3 + d1, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(z1, -FIX(0.899976223));
                        z2 = MULTIPLY(d3, -FIX(2.562915447));
                        z3 = MULTIPLY(z3, -FIX(1.961570560));
                        z4 = MULTIPLY(d1, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 = z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 == 0, d7 != 0 */
                        z3 = d7 + d3;

                        tmp0 = MULTIPLY(d7, -FIX(0.601344887));
                        z1 = MULTIPLY(d7, -FIX(0.899976223));
                        tmp2 = MULTIPLY(d3, FIX(0.509795579));
                        z2 = MULTIPLY(d3, -FIX(2.562915447));
                        z5 = MULTIPLY(z3, FIX(1.175875602));
                        z3 = MULTIPLY(z3, -FIX(0.785694958));

                        tmp0 += z3;
                        tmp1 = z2 + z5;
                        tmp2 += z3;
                        tmp3 = z1 + z5;
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 == 0, d7 != 0 */
                        z1 = d7 + d1;
                        z5 = MULTIPLY(z1, FIX(1.175875602));

                        z1 = MULTIPLY(z1, FIX(0.275899379));
                        z3 = MULTIPLY(d7, -FIX(1.961570560));
                        tmp0 = MULTIPLY(d7, -FIX(1.662939224));
                        z4 = MULTIPLY(d1, -FIX(0.390180644));
                        tmp3 = MULTIPLY(d1, FIX(1.111140466));

                        tmp0 += z1;
                        tmp1 = z4 + z5;
                        tmp2 = z3 + z5;
                        tmp3 += z1;
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 == 0, d7 != 0 */
                        tmp0 = MULTIPLY(d7, -FIX(1.387039845));
                        tmp1 = MULTIPLY(d7, FIX(1.175875602));
                        tmp2 = MULTIPLY(d7, -FIX(0.785694958));
                        tmp3 = MULTIPLY(d7, FIX(0.275899379));
                    }
                }
            }
        }
        else
        {
            if (d5)
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 != 0, d7 == 0 */
                        z2 = d5 + d3;
                        z4 = d5 + d1;
                        z5 = MULTIPLY(d3 + z4, FIX(1.175875602));

                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(d1, -FIX(0.899976223));
                        z2 = MULTIPLY(z2, -FIX(2.562915447));
                        z3 = MULTIPLY(d3, -FIX(1.961570560));
                        z4 = MULTIPLY(z4, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 = z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 != 0, d7 == 0 */
                        z2 = d5 + d3;

                        z5 = MULTIPLY(z2, FIX(1.175875602));
                        tmp1 = MULTIPLY(d5, FIX(1.662939225));
                        z4 = MULTIPLY(d5, -FIX(0.390180644));
                        z2 = MULTIPLY(z2, -FIX(1.387039845));
                        tmp2 = MULTIPLY(d3, FIX(1.111140466));
                        z3 = MULTIPLY(d3, -FIX(1.961570560));

                        tmp0 = z3 + z5;
                        tmp1 += z2;
                        tmp2 += z2;
                        tmp3 = z4 + z5;
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 != 0, d7 == 0 */
                        z4 = d5 + d1;

                        z5 = MULTIPLY(z4, FIX(1.175875602));
                        z1 = MULTIPLY(d1, -FIX(0.899976223));
                        tmp3 = MULTIPLY(d1, FIX(0.601344887));
                        tmp1 = MULTIPLY(d5, -FIX(0.509795578));
                        z2 = MULTIPLY(d5, -FIX(2.562915447));
                        z4 = MULTIPLY(z4, FIX(0.785694958));

                        tmp0 = z1 + z5;
                        tmp1 += z4;
                        tmp2 = z2 + z5;
                        tmp3 += z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 != 0, d7 == 0 */
                        tmp0 = MULTIPLY(d5, FIX(1.175875602));
                        tmp1 = MULTIPLY(d5, FIX(0.275899380));
                        tmp2 = MULTIPLY(d5, -FIX(1.387039845));
                        tmp3 = MULTIPLY(d5, FIX(0.785694958));
                    }
                }
            }
            else
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 == 0, d7 == 0 */
                        z5 = d1 + d3;
                        tmp3 = MULTIPLY(d1, FIX(0.211164243));
                        tmp2 = MULTIPLY(d3, -FIX(1.451774981));
                        z1 = MULTIPLY(d1, FIX(1.061594337));
                        z2 = MULTIPLY(d3, -FIX(2.172734803));
                        z4 = MULTIPLY(z5, FIX(0.785694958));
                        z5 = MULTIPLY(z5, FIX(1.175875602));

                        tmp0 = z1 - z4;
                        tmp1 = z2 + z4;
                        tmp2 += z5;
                        tmp3 += z5;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 == 0, d7 == 0 */
                        tmp0 = MULTIPLY(d3, -FIX(0.785694958));
                        tmp1 = MULTIPLY(d3, -FIX(1.387039845));
                        tmp2 = MULTIPLY(d3, -FIX(0.275899379));
                        tmp3 = MULTIPLY(d3, FIX(1.175875602));
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 == 0, d7 == 0 */
                        tmp0 = MULTIPLY(d1, FIX(0.275899379));
                        tmp1 = MULTIPLY(d1, FIX(0.785694958));
                        tmp2 = MULTIPLY(d1, FIX(1.175875602));
                        tmp3 = MULTIPLY(d1, FIX(1.387039845));
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 == 0, d7 == 0 */
                        tmp0 = tmp1 = tmp2 = tmp3 = 0;
                    }
                }
            }
        }

        /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

        dataptr[0] = (DCTELEM)DESCALE(tmp10 + tmp3, CONST_BITS - PASS1_BITS);
        dataptr[7] = (DCTELEM)DESCALE(tmp10 - tmp3, CONST_BITS - PASS1_BITS);
        dataptr[1] = (DCTELEM)DESCALE(tmp11 + tmp2, CONST_BITS - PASS1_BITS);
        dataptr[6] = (DCTELEM)DESCALE(tmp11 - tmp2, CONST_BITS - PASS1_BITS);
        dataptr[2] = (DCTELEM)DESCALE(tmp12 + tmp1, CONST_BITS - PASS1_BITS);
        dataptr[5] = (DCTELEM)DESCALE(tmp12 - tmp1, CONST_BITS - PASS1_BITS);
        dataptr[3] = (DCTELEM)DESCALE(tmp13 + tmp0, CONST_BITS - PASS1_BITS);
        dataptr[4] = (DCTELEM)DESCALE(tmp13 - tmp0, CONST_BITS - PASS1_BITS);

        dataptr += DCTSIZE; /* advance pointer to next row */
    }

    /* Pass 2: process columns. */
    /* Note that we must descale the results by a factor of 8 == 2**3, */
    /* and also undo the PASS1_BITS scaling. */

    dataptr = data;
    for (rowctr = DCTSIZE - 1; rowctr >= 0; rowctr--)
    {
        /* Columns of zeroes can be exploited in the same way as we did with rows.
     * However, the row calculation has created many nonzero AC terms, so the
     * simplification applies less often (typically 5% to 10% of the time).
     * On machines with very fast multiplication, it's possible that the
     * test takes more time than it's worth.  In that case this section
     * may be commented out.
     */

        d0 = dataptr[DCTSIZE * 0];
        d1 = dataptr[DCTSIZE * 1];
        d2 = dataptr[DCTSIZE * 2];
        d3 = dataptr[DCTSIZE * 3];
        d4 = dataptr[DCTSIZE * 4];
        d5 = dataptr[DCTSIZE * 5];
        d6 = dataptr[DCTSIZE * 6];
        d7 = dataptr[DCTSIZE * 7];

        /* Even part: reverse the even part of the forward DCT. */
        /* The rotator is sqrt(2)*c(-6). */
        if (d6)
        {
            if (d4)
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 != 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp0 = (d0 + d4) << CONST_BITS;
                        tmp1 = (d0 - d4) << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp1 + tmp2;
                        tmp12 = tmp1 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 != 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp0 = d4 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp2 - tmp0;
                        tmp12 = -(tmp0 + tmp2);
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 != 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp0 = (d0 + d4) << CONST_BITS;
                        tmp1 = (d0 - d4) << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp1 + tmp2;
                        tmp12 = tmp1 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 != 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp0 = d4 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp2 - tmp0;
                        tmp12 = -(tmp0 + tmp2);
                    }
                }
            }
            else
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 == 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp0 = d0 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp0 + tmp2;
                        tmp12 = tmp0 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 == 0, d6 != 0 */
                        z1 = MULTIPLY(d2 + d6, FIX(0.541196100));
                        tmp2 = z1 + MULTIPLY(d6, -FIX(1.847759065));
                        tmp3 = z1 + MULTIPLY(d2, FIX(0.765366865));

                        tmp10 = tmp3;
                        tmp13 = -tmp3;
                        tmp11 = tmp2;
                        tmp12 = -tmp2;
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 == 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp0 = d0 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp0 + tmp2;
                        tmp12 = tmp0 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 == 0, d6 != 0 */
                        tmp2 = MULTIPLY(d6, -FIX(1.306562965));
                        tmp3 = MULTIPLY(d6, FIX(0.541196100));

                        tmp10 = tmp3;
                        tmp13 = -tmp3;
                        tmp11 = tmp2;
                        tmp12 = -tmp2;
                    }
                }
            }
        }
        else
        {
            if (d4)
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 != 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp0 = (d0 + d4) << CONST_BITS;
                        tmp1 = (d0 - d4) << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp1 + tmp2;
                        tmp12 = tmp1 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 != 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp0 = d4 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp2 - tmp0;
                        tmp12 = -(tmp0 + tmp2);
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 != 0, d6 == 0 */
                        tmp10 = tmp13 = (d0 + d4) << CONST_BITS;
                        tmp11 = tmp12 = (d0 - d4) << CONST_BITS;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 != 0, d6 == 0 */
                        tmp10 = tmp13 = d4 << CONST_BITS;
                        tmp11 = tmp12 = -tmp10;
                    }
                }
            }
            else
            {
                if (d2)
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 != 0, d4 == 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp0 = d0 << CONST_BITS;

                        tmp10 = tmp0 + tmp3;
                        tmp13 = tmp0 - tmp3;
                        tmp11 = tmp0 + tmp2;
                        tmp12 = tmp0 - tmp2;
                    }
                    else
                    {
                        /* d0 == 0, d2 != 0, d4 == 0, d6 == 0 */
                        tmp2 = MULTIPLY(d2, FIX(0.541196100));
                        tmp3 = MULTIPLY(d2, FIX(1.306562965));

                        tmp10 = tmp3;
                        tmp13 = -tmp3;
                        tmp11 = tmp2;
                        tmp12 = -tmp2;
                    }
                }
                else
                {
                    if (d0)
                    {
                        /* d0 != 0, d2 == 0, d4 == 0, d6 == 0 */
                        tmp10 = tmp13 = tmp11 = tmp12 = d0 << CONST_BITS;
                    }
                    else
                    {
                        /* d0 == 0, d2 == 0, d4 == 0, d6 == 0 */
                        tmp10 = tmp13 = tmp11 = tmp12 = 0;
                    }
                }
            }
        }

        /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */
        if (d7)
        {
            if (d5)
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 != 0, d7 != 0 */
                        z1 = d7 + d1;
                        z2 = d5 + d3;
                        z3 = d7 + d3;
                        z4 = d5 + d1;
                        z5 = MULTIPLY(z3 + z4, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(z1, -FIX(0.899976223));
                        z2 = MULTIPLY(z2, -FIX(2.562915447));
                        z3 = MULTIPLY(z3, -FIX(1.961570560));
                        z4 = MULTIPLY(z4, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 != 0, d7 != 0 */
                        z1 = d7;
                        z2 = d5 + d3;
                        z3 = d7 + d3;
                        z5 = MULTIPLY(z3 + d5, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        z1 = MULTIPLY(d7, -FIX(0.899976223));
                        z2 = MULTIPLY(z2, -FIX(2.562915447));
                        z3 = MULTIPLY(z3, -FIX(1.961570560));
                        z4 = MULTIPLY(d5, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 = z1 + z4;
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 != 0, d7 != 0 */
                        z1 = d7 + d1;
                        z2 = d5;
                        z3 = d7;
                        z4 = d5 + d1;
                        z5 = MULTIPLY(z3 + z4, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(z1, -FIX(0.899976223));
                        z2 = MULTIPLY(d5, -FIX(2.562915447));
                        z3 = MULTIPLY(d7, -FIX(1.961570560));
                        z4 = MULTIPLY(z4, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 = z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 != 0, d7 != 0 */
                        tmp0 = MULTIPLY(d7, -FIX(0.601344887));
                        z1 = MULTIPLY(d7, -FIX(0.899976223));
                        z3 = MULTIPLY(d7, -FIX(1.961570560));
                        tmp1 = MULTIPLY(d5, -FIX(0.509795578));
                        z2 = MULTIPLY(d5, -FIX(2.562915447));
                        z4 = MULTIPLY(d5, -FIX(0.390180644));
                        z5 = MULTIPLY(d5 + d7, FIX(1.175875602));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z3;
                        tmp1 += z4;
                        tmp2 = z2 + z3;
                        tmp3 = z1 + z4;
                    }
                }
            }
            else
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 == 0, d7 != 0 */
                        z1 = d7 + d1;
                        z3 = d7 + d3;
                        z5 = MULTIPLY(z3 + d1, FIX(1.175875602));

                        tmp0 = MULTIPLY(d7, FIX(0.298631336));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(z1, -FIX(0.899976223));
                        z2 = MULTIPLY(d3, -FIX(2.562915447));
                        z3 = MULTIPLY(z3, -FIX(1.961570560));
                        z4 = MULTIPLY(d1, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 += z1 + z3;
                        tmp1 = z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 == 0, d7 != 0 */
                        z3 = d7 + d3;

                        tmp0 = MULTIPLY(d7, -FIX(0.601344887));
                        z1 = MULTIPLY(d7, -FIX(0.899976223));
                        tmp2 = MULTIPLY(d3, FIX(0.509795579));
                        z2 = MULTIPLY(d3, -FIX(2.562915447));
                        z5 = MULTIPLY(z3, FIX(1.175875602));
                        z3 = MULTIPLY(z3, -FIX(0.785694958));

                        tmp0 += z3;
                        tmp1 = z2 + z5;
                        tmp2 += z3;
                        tmp3 = z1 + z5;
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 == 0, d7 != 0 */
                        z1 = d7 + d1;
                        z5 = MULTIPLY(z1, FIX(1.175875602));

                        z1 = MULTIPLY(z1, FIX(0.275899379));
                        z3 = MULTIPLY(d7, -FIX(1.961570560));
                        tmp0 = MULTIPLY(d7, -FIX(1.662939224));
                        z4 = MULTIPLY(d1, -FIX(0.390180644));
                        tmp3 = MULTIPLY(d1, FIX(1.111140466));

                        tmp0 += z1;
                        tmp1 = z4 + z5;
                        tmp2 = z3 + z5;
                        tmp3 += z1;
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 == 0, d7 != 0 */
                        tmp0 = MULTIPLY(d7, -FIX(1.387039845));
                        tmp1 = MULTIPLY(d7, FIX(1.175875602));
                        tmp2 = MULTIPLY(d7, -FIX(0.785694958));
                        tmp3 = MULTIPLY(d7, FIX(0.275899379));
                    }
                }
            }
        }
        else
        {
            if (d5)
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 != 0, d7 == 0 */
                        z2 = d5 + d3;
                        z4 = d5 + d1;
                        z5 = MULTIPLY(d3 + z4, FIX(1.175875602));

                        tmp1 = MULTIPLY(d5, FIX(2.053119869));
                        tmp2 = MULTIPLY(d3, FIX(3.072711026));
                        tmp3 = MULTIPLY(d1, FIX(1.501321110));
                        z1 = MULTIPLY(d1, -FIX(0.899976223));
                        z2 = MULTIPLY(z2, -FIX(2.562915447));
                        z3 = MULTIPLY(d3, -FIX(1.961570560));
                        z4 = MULTIPLY(z4, -FIX(0.390180644));

                        z3 += z5;
                        z4 += z5;

                        tmp0 = z1 + z3;
                        tmp1 += z2 + z4;
                        tmp2 += z2 + z3;
                        tmp3 += z1 + z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 != 0, d7 == 0 */
                        z2 = d5 + d3;

                        z5 = MULTIPLY(z2, FIX(1.175875602));
                        tmp1 = MULTIPLY(d5, FIX(1.662939225));
                        z4 = MULTIPLY(d5, -FIX(0.390180644));
                        z2 = MULTIPLY(z2, -FIX(1.387039845));
                        tmp2 = MULTIPLY(d3, FIX(1.111140466));
                        z3 = MULTIPLY(d3, -FIX(1.961570560));

                        tmp0 = z3 + z5;
                        tmp1 += z2;
                        tmp2 += z2;
                        tmp3 = z4 + z5;
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 != 0, d7 == 0 */
                        z4 = d5 + d1;

                        z5 = MULTIPLY(z4, FIX(1.175875602));
                        z1 = MULTIPLY(d1, -FIX(0.899976223));
                        tmp3 = MULTIPLY(d1, FIX(0.601344887));
                        tmp1 = MULTIPLY(d5, -FIX(0.509795578));
                        z2 = MULTIPLY(d5, -FIX(2.562915447));
                        z4 = MULTIPLY(z4, FIX(0.785694958));

                        tmp0 = z1 + z5;
                        tmp1 += z4;
                        tmp2 = z2 + z5;
                        tmp3 += z4;
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 != 0, d7 == 0 */
                        tmp0 = MULTIPLY(d5, FIX(1.175875602));
                        tmp1 = MULTIPLY(d5, FIX(0.275899380));
                        tmp2 = MULTIPLY(d5, -FIX(1.387039845));
                        tmp3 = MULTIPLY(d5, FIX(0.785694958));
                    }
                }
            }
            else
            {
                if (d3)
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 != 0, d5 == 0, d7 == 0 */
                        z5 = d1 + d3;
                        tmp3 = MULTIPLY(d1, FIX(0.211164243));
                        tmp2 = MULTIPLY(d3, -FIX(1.451774981));
                        z1 = MULTIPLY(d1, FIX(1.061594337));
                        z2 = MULTIPLY(d3, -FIX(2.172734803));
                        z4 = MULTIPLY(z5, FIX(0.785694958));
                        z5 = MULTIPLY(z5, FIX(1.175875602));

                        tmp0 = z1 - z4;
                        tmp1 = z2 + z4;
                        tmp2 += z5;
                        tmp3 += z5;
                    }
                    else
                    {
                        /* d1 == 0, d3 != 0, d5 == 0, d7 == 0 */
                        tmp0 = MULTIPLY(d3, -FIX(0.785694958));
                        tmp1 = MULTIPLY(d3, -FIX(1.387039845));
                        tmp2 = MULTIPLY(d3, -FIX(0.275899379));
                        tmp3 = MULTIPLY(d3, FIX(1.175875602));
                    }
                }
                else
                {
                    if (d1)
                    {
                        /* d1 != 0, d3 == 0, d5 == 0, d7 == 0 */
                        tmp0 = MULTIPLY(d1, FIX(0.275899379));
                        tmp1 = MULTIPLY(d1, FIX(0.785694958));
                        tmp2 = MULTIPLY(d1, FIX(1.175875602));
                        tmp3 = MULTIPLY(d1, FIX(1.387039845));
                    }
                    else
                    {
                        /* d1 == 0, d3 == 0, d5 == 0, d7 == 0 */
                        tmp0 = tmp1 = tmp2 = tmp3 = 0;
                    }
                }
            }
        }

        /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

        dataptr[DCTSIZE * 0] = (DCTELEM)DESCALE(tmp10 + tmp3,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 7] = (DCTELEM)DESCALE(tmp10 - tmp3,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 1] = (DCTELEM)DESCALE(tmp11 + tmp2,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 6] = (DCTELEM)DESCALE(tmp11 - tmp2,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 2] = (DCTELEM)DESCALE(tmp12 + tmp1,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 5] = (DCTELEM)DESCALE(tmp12 - tmp1,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 3] = (DCTELEM)DESCALE(tmp13 + tmp0,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 4] = (DCTELEM)DESCALE(tmp13 - tmp0,
                                                CONST_BITS + PASS1_BITS + 3);

        dataptr++; /* advance pointer to next column */
    }
}

#else

static void
    j_rev_dct_sparse(data, pos)
        DCTBLOCK data;
int pos;
{
    j_rev_dct(data);
}

static void
    j_rev_dct(data)
        DCTBLOCK data;
{
    INT32 tmp0, tmp1, tmp2, tmp3;
    INT32 tmp10, tmp11, tmp12, tmp13;
    INT32 z1, z2, z3, z4, z5;
    register DCTELEM *dataptr;
    int rowctr;
    SHIFT_TEMPS

    /* Pass 1: process rows. */
    /* Note results are scaled up by sqrt(8) compared to a true IDCT; */
    /* furthermore, we scale the results by 2**PASS1_BITS. */

    dataptr = data;
    for (rowctr = DCTSIZE - 1; rowctr >= 0; rowctr--)
    {
        /* Due to quantization, we will usually find that many of the input
     * coefficients are zero, especially the AC terms.  We can exploit this
     * by short-circuiting the IDCT calculation for any row in which all
     * the AC terms are zero.  In that case each output is equal to the
     * DC coefficient (with scale factor as needed).
     * With typical images and quantization tables, half or more of the
     * row DCT calculations can be simplified this way.
     */

        if ((dataptr[1] | dataptr[2] | dataptr[3] | dataptr[4] | dataptr[5] | dataptr[6] | dataptr[7]) == 0)
        {
            /* AC terms all zero */
            DCTELEM dcval = (DCTELEM)(dataptr[0] << PASS1_BITS);

            dataptr[0] = dcval;
            dataptr[1] = dcval;
            dataptr[2] = dcval;
            dataptr[3] = dcval;
            dataptr[4] = dcval;
            dataptr[5] = dcval;
            dataptr[6] = dcval;
            dataptr[7] = dcval;

            dataptr += DCTSIZE; /* advance pointer to next row */
            continue;
        }

        /* Even part: reverse the even part of the forward DCT. */
        /* The rotator is sqrt(2)*c(-6). */

        z2 = (INT32)dataptr[2];
        z3 = (INT32)dataptr[6];

        z1 = MULTIPLY(z2 + z3, FIX(0.541196100));
        tmp2 = z1 + MULTIPLY(z3, -FIX(1.847759065));
        tmp3 = z1 + MULTIPLY(z2, FIX(0.765366865));

        tmp0 = ((INT32)dataptr[0] + (INT32)dataptr[4]) << CONST_BITS;
        tmp1 = ((INT32)dataptr[0] - (INT32)dataptr[4]) << CONST_BITS;

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

        tmp0 = (INT32)dataptr[7];
        tmp1 = (INT32)dataptr[5];
        tmp2 = (INT32)dataptr[3];
        tmp3 = (INT32)dataptr[1];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = MULTIPLY(z3 + z4, FIX(1.175875602)); /* sqrt(2) * c3 */

        tmp0 = MULTIPLY(tmp0, FIX(0.298631336)); /* sqrt(2) * (-c1+c3+c5-c7) */
        tmp1 = MULTIPLY(tmp1, FIX(2.053119869)); /* sqrt(2) * ( c1+c3-c5+c7) */
        tmp2 = MULTIPLY(tmp2, FIX(3.072711026)); /* sqrt(2) * ( c1+c3+c5-c7) */
        tmp3 = MULTIPLY(tmp3, FIX(1.501321110)); /* sqrt(2) * ( c1+c3-c5-c7) */
        z1 = MULTIPLY(z1, -FIX(0.899976223)); /* sqrt(2) * (c7-c3) */
        z2 = MULTIPLY(z2, -FIX(2.562915447)); /* sqrt(2) * (-c1-c3) */
        z3 = MULTIPLY(z3, -FIX(1.961570560)); /* sqrt(2) * (-c3-c5) */
        z4 = MULTIPLY(z4, -FIX(0.390180644)); /* sqrt(2) * (c5-c3) */

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

        dataptr[0] = (DCTELEM)DESCALE(tmp10 + tmp3, CONST_BITS - PASS1_BITS);
        dataptr[7] = (DCTELEM)DESCALE(tmp10 - tmp3, CONST_BITS - PASS1_BITS);
        dataptr[1] = (DCTELEM)DESCALE(tmp11 + tmp2, CONST_BITS - PASS1_BITS);
        dataptr[6] = (DCTELEM)DESCALE(tmp11 - tmp2, CONST_BITS - PASS1_BITS);
        dataptr[2] = (DCTELEM)DESCALE(tmp12 + tmp1, CONST_BITS - PASS1_BITS);
        dataptr[5] = (DCTELEM)DESCALE(tmp12 - tmp1, CONST_BITS - PASS1_BITS);
        dataptr[3] = (DCTELEM)DESCALE(tmp13 + tmp0, CONST_BITS - PASS1_BITS);
        dataptr[4] = (DCTELEM)DESCALE(tmp13 - tmp0, CONST_BITS - PASS1_BITS);

        dataptr += DCTSIZE; /* advance pointer to next row */
    }

    /* Pass 2: process columns. */
    /* Note that we must descale the results by a factor of 8 == 2**3, */
    /* and also undo the PASS1_BITS scaling. */

    dataptr = data;
    for (rowctr = DCTSIZE - 1; rowctr >= 0; rowctr--)
    {
/* Columns of zeroes can be exploited in the same way as we did with rows.
     * However, the row calculation has created many nonzero AC terms, so the
     * simplification applies less often (typically 5% to 10% of the time).
     * On machines with very fast multiplication, it's possible that the
     * test takes more time than it's worth.  In that case this section
     * may be commented out.
     */

#ifndef NO_ZERO_COLUMN_TEST
        if ((dataptr[DCTSIZE * 1] | dataptr[DCTSIZE * 2] | dataptr[DCTSIZE * 3] | dataptr[DCTSIZE * 4] | dataptr[DCTSIZE * 5] | dataptr[DCTSIZE * 6] | dataptr[DCTSIZE * 7]) == 0)
        {
            /* AC terms all zero */
            DCTELEM dcval = (DCTELEM)DESCALE((INT32)dataptr[0], PASS1_BITS + 3);

            dataptr[DCTSIZE * 0] = dcval;
            dataptr[DCTSIZE * 1] = dcval;
            dataptr[DCTSIZE * 2] = dcval;
            dataptr[DCTSIZE * 3] = dcval;
            dataptr[DCTSIZE * 4] = dcval;
            dataptr[DCTSIZE * 5] = dcval;
            dataptr[DCTSIZE * 6] = dcval;
            dataptr[DCTSIZE * 7] = dcval;

            dataptr++; /* advance pointer to next column */
            continue;
        }
#endif

        /* Even part: reverse the even part of the forward DCT. */
        /* The rotator is sqrt(2)*c(-6). */

        z2 = (INT32)dataptr[DCTSIZE * 2];
        z3 = (INT32)dataptr[DCTSIZE * 6];

        z1 = MULTIPLY(z2 + z3, FIX(0.541196100));
        tmp2 = z1 + MULTIPLY(z3, -FIX(1.847759065));
        tmp3 = z1 + MULTIPLY(z2, FIX(0.765366865));

        tmp0 = ((INT32)dataptr[DCTSIZE * 0] + (INT32)dataptr[DCTSIZE * 4]) << CONST_BITS;
        tmp1 = ((INT32)dataptr[DCTSIZE * 0] - (INT32)dataptr[DCTSIZE * 4]) << CONST_BITS;

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

        tmp0 = (INT32)dataptr[DCTSIZE * 7];
        tmp1 = (INT32)dataptr[DCTSIZE * 5];
        tmp2 = (INT32)dataptr[DCTSIZE * 3];
        tmp3 = (INT32)dataptr[DCTSIZE * 1];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = MULTIPLY(z3 + z4, FIX(1.175875602)); /* sqrt(2) * c3 */

        tmp0 = MULTIPLY(tmp0, FIX(0.298631336)); /* sqrt(2) * (-c1+c3+c5-c7) */
        tmp1 = MULTIPLY(tmp1, FIX(2.053119869)); /* sqrt(2) * ( c1+c3-c5+c7) */
        tmp2 = MULTIPLY(tmp2, FIX(3.072711026)); /* sqrt(2) * ( c1+c3+c5-c7) */
        tmp3 = MULTIPLY(tmp3, FIX(1.501321110)); /* sqrt(2) * ( c1+c3-c5-c7) */
        z1 = MULTIPLY(z1, -FIX(0.899976223)); /* sqrt(2) * (c7-c3) */
        z2 = MULTIPLY(z2, -FIX(2.562915447)); /* sqrt(2) * (-c1-c3) */
        z3 = MULTIPLY(z3, -FIX(1.961570560)); /* sqrt(2) * (-c3-c5) */
        z4 = MULTIPLY(z4, -FIX(0.390180644)); /* sqrt(2) * (c5-c3) */

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

        dataptr[DCTSIZE * 0] = (DCTELEM)DESCALE(tmp10 + tmp3,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 7] = (DCTELEM)DESCALE(tmp10 - tmp3,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 1] = (DCTELEM)DESCALE(tmp11 + tmp2,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 6] = (DCTELEM)DESCALE(tmp11 - tmp2,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 2] = (DCTELEM)DESCALE(tmp12 + tmp1,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 5] = (DCTELEM)DESCALE(tmp12 - tmp1,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 3] = (DCTELEM)DESCALE(tmp13 + tmp0,
                                                CONST_BITS + PASS1_BITS + 3);
        dataptr[DCTSIZE * 4] = (DCTELEM)DESCALE(tmp13 - tmp0,
                                                CONST_BITS + PASS1_BITS + 3);

        dataptr++; /* advance pointer to next column */
    }
}

#endif

#ifndef DEFAULT_BUFSIZE
/* 80000 in berkeley code, then multiplied by 4??? when allocating... */
#define DEFAULT_BUFSIZE 2000
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif
#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

/* Macros for picture code type. */

#define I_TYPE 1
#define P_TYPE 2
#define B_TYPE 3

/* Start codes. */

#define SEQ_END_CODE 0x000001b7
#define SEQ_START_CODE 0x000001b3
#define GOP_START_CODE 0x000001b8
#define PICTURE_START_CODE 0x00000100
#define SLICE_MIN_START_CODE 0x00000101
#define SLICE_MAX_START_CODE 0x000001af
#define EXT_START_CODE 0x000001b5
#define USER_START_CODE 0x000001b2

/* Macros used with macroblock address decoding. */

#define MB_STUFFING 34
#define MB_ESCAPE 35

/* Lock flags for ring buffer images. */

#define DISPLAY_LOCK 0x01
#define PAST_LOCK 0x02
#define FUTURE_LOCK 0x04

/*
 * We use a lookup table to make sure values stay in the 0..255 range.
 * Since this is cropping (ie, x = (x < 0)?0:(x>255)?255:x; ), we call this
 * table the "crop table".
 * MAX_NEG_CROP is the maximum neg/pos value we can handle.
 */

#define MAX_NEG_CROP 384
#define NUM_CROP_ENTRIES (256 + 2 * MAX_NEG_CROP)
static unsigned char cropTbl[NUM_CROP_ENTRIES];

/* Bit masks used by bit i/o operations. */
static unsigned int nBitMask[] = { 0x00000000, 0x80000000, 0xc0000000, 0xe0000000,
                                   0xf0000000, 0xf8000000, 0xfc000000, 0xfe000000,
                                   0xff000000, 0xff800000, 0xffc00000, 0xffe00000,
                                   0xfff00000, 0xfff80000, 0xfffc0000, 0xfffe0000,
                                   0xffff0000, 0xffff8000, 0xffffc000, 0xffffe000,
                                   0xfffff000, 0xfffff800, 0xfffffc00, 0xfffffe00,
                                   0xffffff00, 0xffffff80, 0xffffffc0, 0xffffffe0,
                                   0xfffffff0, 0xfffffff8, 0xfffffffc, 0xfffffffe };

static unsigned int bitMask[] = { 0xffffffff, 0x7fffffff, 0x3fffffff, 0x1fffffff,
                                  0x0fffffff, 0x07ffffff, 0x03ffffff, 0x01ffffff,
                                  0x00ffffff, 0x007fffff, 0x003fffff, 0x001fffff,
                                  0x000fffff, 0x0007ffff, 0x0003ffff, 0x0001ffff,
                                  0x0000ffff, 0x00007fff, 0x00003fff, 0x00001fff,
                                  0x00000fff, 0x000007ff, 0x000003ff, 0x000001ff,
                                  0x000000ff, 0x0000007f, 0x0000003f, 0x0000001f,
                                  0x0000000f, 0x00000007, 0x00000003, 0x00000001 };

static unsigned int rBitMask[] = { 0xffffffff, 0xfffffffe, 0xfffffffc, 0xfffffff8,
                                   0xfffffff0, 0xffffffe0, 0xffffffc0, 0xffffff80,
                                   0xffffff00, 0xfffffe00, 0xfffffc00, 0xfffff800,
                                   0xfffff000, 0xffffe000, 0xffffc000, 0xffff8000,
                                   0xffff0000, 0xfffe0000, 0xfffc0000, 0xfff80000,
                                   0xfff00000, 0xffe00000, 0xffc00000, 0xff800000,
                                   0xff000000, 0xfe000000, 0xfc000000, 0xf8000000,
                                   0xf0000000, 0xe0000000, 0xc0000000, 0x80000000 };

static unsigned int bitTest[] = { 0x80000000, 0x40000000, 0x20000000, 0x10000000,
                                  0x08000000, 0x04000000, 0x02000000, 0x01000000,
                                  0x00800000, 0x00400000, 0x00200000, 0x00100000,
                                  0x00080000, 0x00040000, 0x00020000, 0x00010000,
                                  0x00008000, 0x00004000, 0x00002000, 0x00001000,
                                  0x00000800, 0x00000400, 0x00000200, 0x00000100,
                                  0x00000080, 0x00000040, 0x00000020, 0x00000010,
                                  0x00000008, 0x00000004, 0x00000002, 0x00000001 };

/* Decoding table for macroblock_address_increment */
static vb_entry mb_addr_inc[2048];

/* Decoding table for macroblock_type in predictive-coded pictures */
static mb_type_entry mb_type_P[64];

/* Decoding table for macroblock_type in bidirectionally-coded pictures */
static mb_type_entry mb_type_B[64];

/* Decoding table for motion vectors */
static motion_vectors_entry motion_vectors[2048];

/* Decoding table for coded_block_pattern */
static coded_block_pattern_entry coded_block_pattern[512] = { { (unsigned int)MPG_ERROR, 0 }, { (unsigned int)MPG_ERROR, 0 }, { 39, 9 }, { 27, 9 }, { 59, 9 }, { 55, 9 }, { 47, 9 }, { 31, 9 }, { 58, 8 }, { 58, 8 }, { 54, 8 }, { 54, 8 }, { 46, 8 }, { 46, 8 }, { 30, 8 }, { 30, 8 }, { 57, 8 }, { 57, 8 }, { 53, 8 }, { 53, 8 }, { 45, 8 }, { 45, 8 }, { 29, 8 }, { 29, 8 }, { 38, 8 }, { 38, 8 }, { 26, 8 }, { 26, 8 }, { 37, 8 }, { 37, 8 }, { 25, 8 }, { 25, 8 }, { 43, 8 }, { 43, 8 }, { 23, 8 }, { 23, 8 }, { 51, 8 }, { 51, 8 }, { 15, 8 }, { 15, 8 }, { 42, 8 }, { 42, 8 }, { 22, 8 }, { 22, 8 }, { 50, 8 }, { 50, 8 }, { 14, 8 }, { 14, 8 }, { 41, 8 }, { 41, 8 }, { 21, 8 }, { 21, 8 }, { 49, 8 }, { 49, 8 }, { 13, 8 }, { 13, 8 }, { 35, 8 }, { 35, 8 }, { 19, 8 }, { 19, 8 }, { 11, 8 }, { 11, 8 }, { 7, 8 }, { 7, 8 }, { 34, 7 }, { 34, 7 }, { 34, 7 }, { 34, 7 }, { 18, 7 }, { 18, 7 }, { 18, 7 }, { 18, 7 }, { 10, 7 }, { 10, 7 }, { 10, 7 }, { 10, 7 }, { 6, 7 }, { 6, 7 }, { 6, 7 }, { 6, 7 }, { 33, 7 }, { 33, 7 }, { 33, 7 }, { 33, 7 }, { 17, 7 }, { 17, 7 }, { 17, 7 }, { 17, 7 }, { 9, 7 }, { 9, 7 }, { 9, 7 }, { 9, 7 }, { 5, 7 }, { 5, 7 }, { 5, 7 }, { 5, 7 }, { 63, 6 }, { 63, 6 }, { 63, 6 }, { 63, 6 }, { 63, 6 }, { 63, 6 }, { 63, 6 }, { 63, 6 }, { 3, 6 }, { 3, 6 }, { 3, 6 }, { 3, 6 }, { 3, 6 }, { 3, 6 }, { 3, 6 }, { 3, 6 }, { 36, 6 }, { 36, 6 }, { 36, 6 }, { 36, 6 }, { 36, 6 }, { 36, 6 }, { 36, 6 }, { 36, 6 }, { 24, 6 }, { 24, 6 }, { 24, 6 }, { 24, 6 }, { 24, 6 }, { 24, 6 }, { 24, 6 }, { 24, 6 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 62, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 2, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 61, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 1, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 56, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 52, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 44, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 28, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 40, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 20, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 48, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 12, 5 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 32, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 16, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 8, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 }, { 60, 3 } };

/* Decoding table for dct_dc_size_luminance */
static vb_entry dct_dc_size_luminance[128] = { { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 0, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 4, 3 }, { 5, 4 }, { 5, 4 }, { 5, 4 }, { 5, 4 }, { 5, 4 }, { 5, 4 }, { 5, 4 }, { 5, 4 }, { 6, 5 }, { 6, 5 }, { 6, 5 }, { 6, 5 }, { 7, 6 }, { 7, 6 }, { 8, 7 }, { (unsigned int)MPG_ERROR, 0 } };

/* Decoding table for dct_dc_size_chrominance */
static vb_entry dct_dc_size_chrominance[256] = { { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 0, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 1, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 3, 3 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 4, 4 }, { 5, 5 }, { 5, 5 }, { 5, 5 }, { 5, 5 }, { 5, 5 }, { 5, 5 }, { 5, 5 }, { 5, 5 }, { 6, 6 }, { 6, 6 }, { 6, 6 }, { 6, 6 }, { 7, 7 }, { 7, 7 }, { 8, 8 }, { (unsigned int)MPG_ERROR, 0 } };

int MPEGerrno = 0;

/*
 *--------------------------------------------------------------
 *
 * get_more_data
 *
 *	Called when buffer does not have sufficient data to 
 *      satisfy request for bits.
 *
 * Results:
 *      None really.
 *  
 * Side effects:
 *	buf_length and buffer fields in curVidStream structure
 *      may be changed.
 *
 *--------------------------------------------------------------
 */

static int get_more_data(MPEG *m)
{
    int num_read, i, request;
    unsigned char *mark;
    unsigned int *lmark;

    if (m->buf_length > 0)
    {
        memcpy(m->buf_start, m->buffer, m->buf_length * 4);
        mark = (unsigned char *)(m->buf_start + m->buf_length);
    }
    else
    {
        mark = (unsigned char *)(m->buf_start);
        m->buf_length = 0;
    }

    request = (m->max_buf_length - m->buf_length) * 4;
    num_read = (int)fread(mark, 1, request, m->fp);

    if (num_read < 0)
    {
        MPEGerrno = MPEG_READERR;
        return 0;
    }

    else if (num_read == 0)
    {
        /* Make 32 bits after end equal to 0 and 32 bits after
       * that equal to seq end code in order to prevent messy
       * data from infinite recursion.
       */
        *(m->buf_start + m->buf_length) = 0x0;
        *(m->buf_start + m->buf_length + 1) = SEQ_END_CODE;
        /* Unexpected EOF? */
    }

    /* Paulo Villegas - 26/1/1993: Correction for 4-byte alignment */
    else if (num_read < request)
    {
        int num_read_rounded;
        unsigned char *index;

        num_read_rounded = 4 * (num_read / 4);

        /* this can happen only if num_read<request; i.e. end of file reached */
        if (num_read_rounded < num_read)
        {
            num_read_rounded = 4 * (num_read / 4 + 1);
            /* fill in with zeros */
            for (index = mark + num_read; index < mark + num_read_rounded; *(index++) = 0)
                ;
            /* advance to the next 4-byte boundary */
            num_read = num_read_rounded;
        }
    }

    num_read /= 4;
    for (i = 0, lmark = (unsigned int *)mark; i < num_read; i++, lmark++)
        *lmark = htonl(*lmark);

    m->buffer = m->buf_start;
    m->buf_length += num_read;
    m->cur_bits = *m->buffer << m->bit_offset;

    return 1;
}

static void flush_bits(MPEG *m, int num)
{
    /* If insufficient data exists, correct underflow. */
    if (m->buf_length < 2)
        if (!get_more_data(m))
            return;

    m->bit_offset += num;

    if (m->bit_offset & 0x20)
    {
        m->buf_length--;
        m->bit_offset -= 32;
        m->buffer++;
        m->cur_bits = *(m->buffer) << m->bit_offset;
    }
    else
    {
        m->cur_bits <<= num;
    }
}

static unsigned int get_bits1(MPEG *m)
{
    unsigned int result;

    /* If insufficient data exists, correct underflow. */
    if (m->buf_length < 2)
        if (!get_more_data(m))
            return 0;

    result = ((m->cur_bits & 0x80000000) != 0);

    m->cur_bits <<= 1;
    m->bit_offset++;

    if (m->bit_offset & 0x20)
    {
        m->bit_offset = 0;
        m->buffer++;
        m->cur_bits = *(m->buffer);
        m->buf_length--;
    }
    return result;
}

#define get_bits3(m) get_bitsX(m, 3, 0xe0000000, 29)
#define get_bits4(m) get_bitsX(m, 4, 0xf0000000, 28)
#define get_bits5(m) get_bitsX(m, 5, 0xf8000000, 27)
#define get_bits8(m) get_bitsX(m, 8, 0xff000000, 24)
#define get_bits10(m) get_bitsX(m, 10, 0xffc00000, 22)
#define get_bits12(m) get_bitsX(m, 12, 0xfff00000, 20)
#define get_bits18(m) get_bitsX(m, 18, 0xffffc000, 14)
#define get_bitsn(m, n) get_bitsX(m, n, nBitMask[n], 32 - (n))

static unsigned int get_bitsX(MPEG *m, int num, unsigned int mask, int shift)
{
    unsigned int result;

    /* If insufficient data exists, correct underflow. */
    if (m->buf_length < 2)
        if (!get_more_data(m))
            return 0;

    m->bit_offset += num;

    if (m->bit_offset & 0x20)
    {
        m->bit_offset -= 32;
        m->buffer++;
        m->buf_length--;
        if (m->bit_offset)
            m->cur_bits |= (*(m->buffer) >> (num - m->bit_offset));
        result = ((m->cur_bits & mask) >> shift);
        m->cur_bits = *(m->buffer) << m->bit_offset;
    }
    else
    {
        result = ((m->cur_bits & mask) >> shift);
        m->cur_bits <<= num;
    }

    return result;
}

#define show_bits2(_m) show_bitsX(_m, 2, 0xc0000000, 30)
#define show_bits6(_m) show_bitsX(_m, 6, 0xfc000000, 26)
#define show_bits9(_m) show_bitsX(_m, 9, 0xff800000, 23)
#define show_bits11(_m) show_bitsX(_m, 11, 0xffe00000, 21)
#define show_bits16(_m) show_bitsX(_m, 16, 0xffff0000, 16)
#define show_bits24(_m) show_bitsX(_m, 24, 0xffffff00, 8)

#define show_bits32(_m) \
    ((_m)->bit_offset ? (_m)->cur_bits | (*((_m)->buffer + 1) >> (32 - (_m)->bit_offset)) : (_m)->cur_bits)

static unsigned int show_bitsX(MPEG *m, int num, unsigned int mask, int shift)
{
    unsigned int result = 0;
    int bO;

    /* If insufficient data exists, correct underflow. */
    if (m->buf_length < 2)
        if (!get_more_data(m))
            return 0;

    bO = m->bit_offset + num;
    if (bO > 32)
    {
        bO -= 32;
        result = (((m->cur_bits & mask) >> shift) | (*(m->buffer + 1) >> (shift + (num - bO))));
    }
    else
    {
        result = ((m->cur_bits & mask) >> shift);
    }
    return result;
}

/*
 *--------------------------------------------------------------
 *
 * next_start_code --
 *
 *	Parses off bitstream until start code reached. When done
 *      next 4 bytes of bitstream will be start code. Bit offset
 *      reset to 0.
 *
 * Results:
 *	Status code.
 *
 * Side effects:
 *	Bit stream irreversibly parsed.
 *
 *--------------------------------------------------------------
 */

static void next_start_code(MPEG *m)
{
    int state;
    int byteoff;
    unsigned int data;

    /* If insufficient buffer length, correct underflow. */
    if (m->buf_length < 2)
        if (!get_more_data(m))
            return;

    /* If bit offset not zero, reset and advance buffer pointer. */
    byteoff = m->bit_offset % 8;
    if (byteoff != 0)
        flush_bits(m, 8 - byteoff);

    state = 0;

    /* While buffer has data ... */
    while (m->buf_length > 0)
    {

        /* If insufficient data exists, correct underflow. */
        if (m->buf_length < 2)
            if (!get_more_data(m))
                return;

        /* If next byte is zero... */
        data = get_bits8(m);
        switch (data)
        {
        case 0: /* If state < 2, advance state. */
            if (state < 2)
                state++;
            break;
        case 1: /* If state == 2, advance state (i.e. start code found). */
            if (state == 2)
                state++;
            else
                state = 0;
            break;
        default: /* Otherwise byte is neither 1 or 0, reset state to 0. */
            state = 0;
            break;
        }

        /* If state == 3 (i.e. start code found)... */
        if (state == 3)
        {
            /* Set buffer pointer back and reset length & bit offsets so
	 * next bytes will be beginning of start code. 
	 */

            m->bit_offset -= 24;

            if (m->bit_offset < 0)
            {
                m->bit_offset += 32;
                m->buf_length++;
                m->buffer--;
            }
            m->cur_bits = *(m->buffer) << m->bit_offset;

            break;
        }
    }
}

/*
 *--------------------------------------------------------------
 *
 * NewPictImage --
 *
 *	Allocates and initializes a YUVImage structure.
 *      The width and height of the image space are passed in
 *      as parameters.
 *
 * Results:
 *	A pointer to the new YUVImage structure.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

static YUVImage *alloc_image(unsigned int width, unsigned int height)
{
    YUVImage *newimg;

    if ((newimg = (YUVImage *)malloc(sizeof(YUVImage))) == NULL)
        return NULL;

    newimg->luminance = newimg->Cr = newimg->Cb = NULL;

    /* Allocate memory for image spaces. */
    newimg->luminance = (unsigned char *)malloc(width * height);
    newimg->Cr = (unsigned char *)malloc(width * height / 4);
    newimg->Cb = (unsigned char *)malloc(width * height / 4);

    if ((newimg->luminance == NULL) || (newimg->Cr == NULL) || (newimg->Cb == NULL))
    {
        if (newimg->luminance)
            free(newimg->luminance);
        if (newimg->Cr)
            free(newimg->Cr);
        if (newimg->Cb)
            free(newimg->Cb);
        free(newimg);
        return NULL;
    }

    /* Reset locked flag. */
    newimg->locked = 0;

    return newimg;
}

static void free_image(YUVImage *i)
{
    if (i->luminance)
        free(i->luminance);
    if (i->Cr)
        free(i->Cr);
    if (i->Cb)
        free(i->Cb);
    free(i);
}

static void flush_ext_data(MPEG *m)
{
    flush_bits(m, 32); /* Flush start code. */
    while (show_bits24(m) != 1) /* Read until another start code. */
        flush_bits(m, 8);
}

/*
 *--------------------------------------------------------------
 *
 * ParseSeqHead --
 *
 *      Assumes bit stream is at the begining of the sequence
 *      header start code. Parses off the sequence header.
 *
 * Results:
 *      Fills the m structure with values derived and
 *      decoded from the sequence header. Allocates the pict image
 *      structures based on the dimensions of the image space
 *      found in the sequence header.
 *
 * Side effects:
 *      Bit stream irreversibly parsed off.
 *
 *--------------------------------------------------------------
 */

static int parse_seq_header(MPEG *m)
{
    /* Set up array for fast conversion from zig zag order to row/column coords. */
    static int zigzag[64][2] = {
        { 0, 0 }, { 1, 0 }, { 0, 1 }, { 0, 2 }, { 1, 1 }, { 2, 0 }, { 3, 0 }, { 2, 1 }, { 1, 2 }, { 0, 3 }, { 0, 4 }, { 1, 3 }, { 2, 2 }, { 3, 1 }, { 4, 0 }, { 5, 0 }, { 4, 1 }, { 3, 2 }, { 2, 3 }, { 1, 4 }, { 0, 5 }, { 0, 6 }, { 1, 5 }, { 2, 4 }, { 3, 3 }, { 4, 2 }, { 5, 1 }, { 6, 0 }, { 7, 0 }, { 6, 1 }, { 5, 2 }, { 4, 3 }, { 3, 4 }, { 2, 5 }, { 1, 6 }, { 0, 7 }, { 1, 7 }, { 2, 6 }, { 3, 5 }, { 4, 4 }, { 5, 3 }, { 6, 2 }, { 7, 1 }, { 7, 2 }, { 6, 3 }, { 5, 4 }, { 4, 5 }, { 3, 6 }, { 2, 7 }, { 3, 7 }, { 4, 6 }, { 5, 5 }, { 6, 4 }, { 7, 3 }, { 7, 4 }, { 6, 5 }, { 5, 6 }, { 4, 7 }, { 5, 7 }, { 6, 6 }, { 7, 5 }, { 7, 6 }, { 6, 7 }, { 7, 7 }
    };

    int i;
#ifdef DEBUG
    fprintf(stderr, " parse_seq_header\n");
#endif

    flush_bits(m, 32); /* Flush off sequence start code. */
    m->h_size = get_bits12(m); /* Horizontal image size. */
    m->v_size = get_bits12(m); /* Vertical image size. */

    /* Calculate macroblock width and height of image space. */
    m->mb_width = (m->h_size + 15) / 16;
    m->mb_height = (m->v_size + 15) / 16;

    /*
   * Initialize ring buffer of images now that dimensions of image space
   * are known.
   */
    for (i = 0; i < RING_BUF_SIZE; i++)
    {
        if (m->ring[i] == NULL)
            m->ring[i] = alloc_image(m->mb_width * 16, m->mb_height * 16);
        else
            m->ring[i]->locked = 0;
    }

    m->aspect_ratio = (unsigned char)get_bits4(m); /* Aspect ratio. */
    m->picture_rate = (unsigned char)get_bits4(m); /* Picture rate. */
    m->bit_rate = get_bits18(m); /* Bit rate. */
    flush_bits(m, 1); /* Flush marker bit. */
    m->vbv_buffer_size = get_bits10(m); /* vbv bufffer size. */
    m->const_param_flag = get_bits1(m); /* Constrained parameter flag. */

    /* Parse off intra quant matrix values if present. */
    if (get_bits1(m))
        for (i = 0; i < 64; i++)
            m->intra_quant_matrix[zigzag[i][1]][zigzag[i][0]] = (unsigned char)get_bits8(m);

    /* Parse off non intra quant matrix values if present. */
    if (get_bits1(m))
        for (i = 0; i < 64; i++)
            m->non_intra_quant_matrix[zigzag[i][1]][zigzag[i][0]] = (unsigned char)get_bits8(m);

    /* Go to next start code. */
    next_start_code(m);

    /* If next start code is extension start code, parse off extension data. */
    if (show_bits32(m) == EXT_START_CODE)
        flush_ext_data(m); /* Anyone care? */

    /* If next start code is user start code, parse off user data. */
    if (show_bits32(m) == USER_START_CODE)
        flush_ext_data(m); /* Anyone care? */

    return 1;
}

/*--------------------------------------------------------------
 *
 * parse_GOP -- Flushes group of pictures header from bit stream.
 *
 */

static int parse_GOP(MPEG *m)
{
    flush_bits(m, 32); /* Start code */
    next_start_code(m); /* Skip flags, time code, etc. */

    /* If next start code is extension start code, parse off extension data. */
    if (show_bits32(m) == EXT_START_CODE)
        flush_ext_data(m); /* Anyone care? */

    /* If next start code is user start code, parse off user data. */
    if (show_bits32(m) == USER_START_CODE)
        flush_ext_data(m); /* Anyone care? */

    return 1;
}

static void flush_extra_bit_info(MPEG *m)
{
    while (get_bits1(m))
        flush_bits(m, 8);
}

/*--------------------------------------------------------------
 *
 * parse_picture -- Parses picture header.
 *
 * Side effects:
 *      Bit stream irreversibly parsed.
 *
 */

static int parse_picture(MPEG *m)
{
    unsigned int data;
    int i;

    flush_bits(m, 32); /* Flush header start code. */
    flush_bits(m, 10); /* Flush temporal reference. */
    m->picture.code_type = get_bits3(m); /* Picture type. */

    /* Skip the picture if the reference frames aren't available. */
    if (((m->picture.code_type == B_TYPE) && ((m->past == NULL) || (m->future == NULL))) || ((m->picture.code_type == P_TYPE) && (m->future == NULL)))
    {
#ifdef DEBUG
        fprintf(stderr, "Skipping picture (%c type)...",
                m->picture.code_type == B_TYPE ? 'B' : 'P');
#endif
        next_start_code(m);
        while (((data = show_bits32(m)) != PICTURE_START_CODE) && (data != GOP_START_CODE) && (data != SEQ_END_CODE))
        {
            flush_bits(m, 24);
            next_start_code(m);
        }
#ifdef DEBUG
        fprintf(stderr, "Done.\n");
#endif
        return 1;
    }

    flush_bits(m, 16); /* vbv buffer delay value. */

    /* If P or B type frame... */
    if ((m->picture.code_type == 2) || (m->picture.code_type == 3))
    {
        /* Parse off forward vector full pixel flag. */
        m->picture.full_pel_forw_vector = get_bits1(m);
        data = get_bits3(m); /* forw_r_code. */

        /* Decode forw_r_code into forw_r_size and forw_f. */
        m->picture.forw_r_size = data - 1;
        m->picture.forw_f = (1 << m->picture.forw_r_size);
    }

    /* If B type frame... */
    if (m->picture.code_type == 3)
    {
        /* Parse off back vector full pixel flag. */
        m->picture.full_pel_back_vector = get_bits1(m);

        data = get_bits3(m); /* back_r_code. */

        /* Decode back_r_code into back_r_size and back_f. */
        m->picture.back_r_size = data - 1;
        m->picture.back_f = (1 << m->picture.back_r_size);
    }

    /* Go to next start code, flushing extra bit picture info... */
    flush_extra_bit_info(m);
    next_start_code(m);

    /* If next start code is extension start code, parse off extension data. */
    if (show_bits32(m) == EXT_START_CODE)
        flush_ext_data(m); /* Anyone care? */

    /* If next start code is user start code, parse off user data. */
    if (show_bits32(m) == USER_START_CODE)
        flush_ext_data(m); /* Anyone care? */

    /* Find an image structure in ring buffer not currently locked. */
    for (i = 0; i < RING_BUF_SIZE; ++i)
        if (!m->ring[i]->locked)
            break;
    if (i == RING_BUF_SIZE)
    {
        fprintf(stderr, "Ring buffer full.\n");
        return 0;
    }

    /* Set current image structure to the one just found in ring. */
    m->current = m->ring[i];

    /* Reset past macroblock address field. */
    m->mblock.past_mb_addr = -1;

    return 1;
}

/*--------------------------------------------------------------
 *
 * parse_slice -- Parses off slice header.
 *
 * Results:
 *      Values found in slice header put into video stream structure.
 *
 * Side effects:
 *      Bit stream irreversibly parsed.
 *
 */

static int parse_slice(MPEG *m)
{
    flush_bits(m, 24); /* Flush slice start code. */
    m->slice.vert_pos = get_bits8(m); /* Slice vertical position. */
    m->slice.quant_scale = get_bits5(m); /* Quantization scale. */
    flush_extra_bit_info(m);

    /* Reset past intrablock address. */
    m->mblock.past_intra_addr = -2;

    /* Reset previous recon motion vectors. */
    m->mblock.recon_right_for_prev = 0;
    m->mblock.recon_down_for_prev = 0;
    m->mblock.recon_right_back_prev = 0;
    m->mblock.recon_down_back_prev = 0;

    /* Reset macroblock address. */
    m->mblock.mb_address = ((m->slice.vert_pos - 1) * m->mb_width) - 1;

    /* Reset past dct dc y, cr, and cb values. */
    m->block.dct_dc_y_past = 1024;
    m->block.dct_dc_cr_past = 1024;
    m->block.dct_dc_cb_past = 1024;

    return 1;
}

/*
 *--------------------------------------------------------------
 *
 * ProcessSkippedPFrameMBlocks --
 *
 *	Processes skipped macroblocks in P frames.
 *
 * Results:
 *	Calculates pixel values for luminance, Cr, and Cb planes
 *      in current pict image for skipped macroblocks.
 *
 * Side effects:
 *	Pixel values in pict image changed.
 *
 *--------------------------------------------------------------
 */

static void
ProcessSkippedPFrameMBlocks(MPEG *m)
{
    int row_size, half_row, mb_row, mb_col, row, col, rr;
    int addr, row_incr, half_row_incr, crow, ccol;
    int *dest, *src, *dest1, *src1;

    /* Calculate row sizes for luminance and Cr/Cb macroblock areas. */
    row_size = m->mb_width << 4;
    half_row = (row_size >> 1);
    row_incr = row_size >> 2;
    half_row_incr = half_row >> 2;

    /* For each skipped macroblock, do... */

    for (addr = m->mblock.past_mb_addr + 1;
         addr < m->mblock.mb_address; addr++)
    {

        /* Calculate macroblock row and col. */

        mb_row = addr / m->mb_width;
        mb_col = addr % m->mb_width;

        /* Calculate upper left pixel row,col for luminance plane. */

        row = mb_row << 4;
        col = mb_col << 4;

        /* For each row in macroblock luminance plane... */

        dest = (int *)(m->current->luminance + (row * row_size) + col);
        src = (int *)(m->future->luminance + (row * row_size) + col);

        for (rr = 0; rr < 8; rr++)
        {

            /* Copy pixel values from last I or P picture. */

            dest[0] = src[0];
            dest[1] = src[1];
            dest[2] = src[2];
            dest[3] = src[3];
            dest += row_incr;
            src += row_incr;

            dest[0] = src[0];
            dest[1] = src[1];
            dest[2] = src[2];
            dest[3] = src[3];
            dest += row_incr;
            src += row_incr;
        }

        /*
     * Divide row,col to get upper left pixel of macroblock in Cr and Cb
     * planes.
     */

        crow = row >> 1;
        ccol = col >> 1;

        /* For each row in Cr, and Cb planes... */

        dest = (int *)(m->current->Cr + (crow * half_row) + ccol);
        src = (int *)(m->future->Cr + (crow * half_row) + ccol);
        dest1 = (int *)(m->current->Cb + (crow * half_row) + ccol);
        src1 = (int *)(m->future->Cb + (crow * half_row) + ccol);

        for (rr = 0; rr < 4; rr++)
        {

            /* Copy pixel values from last I or P picture. */

            dest[0] = src[0];
            dest[1] = src[1];

            dest1[0] = src1[0];
            dest1[1] = src1[1];

            dest += half_row_incr;
            src += half_row_incr;
            dest1 += half_row_incr;
            src1 += half_row_incr;

            dest[0] = src[0];
            dest[1] = src[1];

            dest1[0] = src1[0];
            dest1[1] = src1[1];

            dest += half_row_incr;
            src += half_row_incr;
            dest1 += half_row_incr;
            src1 += half_row_incr;
        }
    }

    m->mblock.recon_right_for_prev = 0;
    m->mblock.recon_down_for_prev = 0;
}

/*
 *--------------------------------------------------------------
 *
 * ReconSkippedBlock --
 *
 *	Reconstructs predictive block for skipped macroblocks
 *      in B Frames.
 *
 * Results:
 *	No return values.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

static void
ReconSkippedBlock(unsigned char *source,
                  unsigned char *dest,
                  int row, int col, int row_size,
                  int right, int down, int right_half, int down_half, int width)
{
    int rr;
    unsigned char *source2;

    source += ((row + down) * row_size) + col + right;

    if (width == 16)
    {
        if ((!right_half) && (!down_half))
        {
            if (right & 0x1)
            {
                /* No alignment, use bye copy */
                for (rr = 0; rr < 16; rr++)
                {
                    dest[0] = source[0];
                    dest[1] = source[1];
                    dest[2] = source[2];
                    dest[3] = source[3];
                    dest[4] = source[4];
                    dest[5] = source[5];
                    dest[6] = source[6];
                    dest[7] = source[7];
                    dest[8] = source[8];
                    dest[9] = source[9];
                    dest[10] = source[10];
                    dest[11] = source[11];
                    dest[12] = source[12];
                    dest[13] = source[13];
                    dest[14] = source[14];
                    dest[15] = source[15];
                    dest += 16;
                    source += row_size;
                }
            }
            else if (right & 0x2)
            {
                /* Half-word bit aligned, use 16 bit copy */
                short *src = (short *)source;
                short *d = (short *)dest;
                row_size >>= 1;
                for (rr = 0; rr < 16; rr++)
                {
                    d[0] = src[0];
                    d[1] = src[1];
                    d[2] = src[2];
                    d[3] = src[3];
                    d[4] = src[4];
                    d[5] = src[5];
                    d[6] = src[6];
                    d[7] = src[7];
                    d += 8;
                    src += row_size;
                }
            }
            else
            {
                /* Word aligned, use 32 bit copy */
                int *src = (int *)source;
                int *d = (int *)dest;
                row_size >>= 2;
                for (rr = 0; rr < 16; rr++)
                {
                    d[0] = src[0];
                    d[1] = src[1];
                    d[2] = src[2];
                    d[3] = src[3];
                    d += 4;
                    src += row_size;
                }
            }
        }
        else
        {
            source2 = source + right_half + (row_size * down_half);
            for (rr = 0; rr < width; rr++)
            {
                dest[0] = (int)(source[0] + source2[0]) >> 1;
                dest[1] = (int)(source[1] + source2[1]) >> 1;
                dest[2] = (int)(source[2] + source2[2]) >> 1;
                dest[3] = (int)(source[3] + source2[3]) >> 1;
                dest[4] = (int)(source[4] + source2[4]) >> 1;
                dest[5] = (int)(source[5] + source2[5]) >> 1;
                dest[6] = (int)(source[6] + source2[6]) >> 1;
                dest[7] = (int)(source[7] + source2[7]) >> 1;
                dest[8] = (int)(source[8] + source2[8]) >> 1;
                dest[9] = (int)(source[9] + source2[9]) >> 1;
                dest[10] = (int)(source[10] + source2[10]) >> 1;
                dest[11] = (int)(source[11] + source2[11]) >> 1;
                dest[12] = (int)(source[12] + source2[12]) >> 1;
                dest[13] = (int)(source[13] + source2[13]) >> 1;
                dest[14] = (int)(source[14] + source2[14]) >> 1;
                dest[15] = (int)(source[15] + source2[15]) >> 1;
                dest += width;
                source += row_size;
                source2 += row_size;
            }
        }
    }
    else
    { /* (width == 8) */
        /*assert(width == 8);*/
        if ((!right_half) && (!down_half))
        {
            if (right & 0x1)
            {
                for (rr = 0; rr < width; rr++)
                {
                    dest[0] = source[0];
                    dest[1] = source[1];
                    dest[2] = source[2];
                    dest[3] = source[3];
                    dest[4] = source[4];
                    dest[5] = source[5];
                    dest[6] = source[6];
                    dest[7] = source[7];
                    dest += 8;
                    source += row_size;
                }
            }
            else if (right & 0x02)
            {
                short *d = (short *)dest;
                short *src = (short *)source;
                row_size >>= 1;
                for (rr = 0; rr < width; rr++)
                {
                    d[0] = src[0];
                    d[1] = src[1];
                    d[2] = src[2];
                    d[3] = src[3];
                    d += 4;
                    src += row_size;
                }
            }
            else
            {
                int *d = (int *)dest;
                int *src = (int *)source;
                row_size >>= 2;
                for (rr = 0; rr < width; rr++)
                {
                    d[0] = src[0];
                    d[1] = src[1];
                    d += 2;
                    src += row_size;
                }
            }
        }
        else
        {
            source2 = source + right_half + (row_size * down_half);
            for (rr = 0; rr < width; rr++)
            {
                dest[0] = (int)(source[0] + source2[0]) >> 1;
                dest[1] = (int)(source[1] + source2[1]) >> 1;
                dest[2] = (int)(source[2] + source2[2]) >> 1;
                dest[3] = (int)(source[3] + source2[3]) >> 1;
                dest[4] = (int)(source[4] + source2[4]) >> 1;
                dest[5] = (int)(source[5] + source2[5]) >> 1;
                dest[6] = (int)(source[6] + source2[6]) >> 1;
                dest[7] = (int)(source[7] + source2[7]) >> 1;
                dest += width;
                source += row_size;
                source2 += row_size;
            }
        }
    }
}

/*
 *--------------------------------------------------------------
 *
 * ProcessSkippedBFrameMBlocks --
 *
 *	Processes skipped macroblocks in B frames.
 *
 * Results:
 *	Calculates pixel values for luminance, Cr, and Cb planes
 *      in current pict image for skipped macroblocks.
 *
 * Side effects:
 *	Pixel values in pict image changed.
 *
 *--------------------------------------------------------------
 */

static void
ProcessSkippedBFrameMBlocks(MPEG *m)
{
    int row_size, half_row, mb_row, mb_col, row, col, rr;
    int right_half_for = 0, down_half_for = 0,
        c_right_half_for = 0, c_down_half_for = 0;
    int right_half_back = 0, down_half_back = 0,
        c_right_half_back = 0, c_down_half_back = 0;
    int addr, right_for = 0, down_for = 0;
    int recon_right_for, recon_down_for;
    int recon_right_back, recon_down_back;
    int right_back = 0, down_back = 0;
    int c_right_for = 0, c_down_for = 0;
    int c_right_back = 0, c_down_back = 0;
    unsigned char forw_lum[256];
    unsigned char forw_cr[64], forw_cb[64];
    unsigned char back_lum[256], back_cr[64], back_cb[64];
    int row_incr, half_row_incr;
    int ccol, crow;

    /* Calculate row sizes for luminance and Cr/Cb macroblock areas. */

    row_size = m->mb_width << 4;
    half_row = (row_size >> 1);
    row_incr = row_size >> 2;
    half_row_incr = half_row >> 2;

    /* Establish motion vector codes based on full pixel flag. */

    if (m->picture.full_pel_forw_vector)
    {
        recon_right_for = m->mblock.recon_right_for_prev << 1;
        recon_down_for = m->mblock.recon_down_for_prev << 1;
    }
    else
    {
        recon_right_for = m->mblock.recon_right_for_prev;
        recon_down_for = m->mblock.recon_down_for_prev;
    }

    if (m->picture.full_pel_back_vector)
    {
        recon_right_back = m->mblock.recon_right_back_prev << 1;
        recon_down_back = m->mblock.recon_down_back_prev << 1;
    }
    else
    {
        recon_right_back = m->mblock.recon_right_back_prev;
        recon_down_back = m->mblock.recon_down_back_prev;
    }

    /* Calculate motion vectors. */

    if (m->mblock.bpict_past_forw)
    {
        right_for = recon_right_for >> 1;
        down_for = recon_down_for >> 1;
        right_half_for = recon_right_for & 0x1;
        down_half_for = recon_down_for & 0x1;

        recon_right_for /= 2;
        recon_down_for /= 2;
        c_right_for = recon_right_for >> 1;
        c_down_for = recon_down_for >> 1;
        c_right_half_for = recon_right_for & 0x1;
        c_down_half_for = recon_down_for & 0x1;
    }
    if (m->mblock.bpict_past_back)
    {
        right_back = recon_right_back >> 1;
        down_back = recon_down_back >> 1;
        right_half_back = recon_right_back & 0x1;
        down_half_back = recon_down_back & 0x1;

        recon_right_back /= 2;
        recon_down_back /= 2;
        c_right_back = recon_right_back >> 1;
        c_down_back = recon_down_back >> 1;
        c_right_half_back = recon_right_back & 0x1;
        c_down_half_back = recon_down_back & 0x1;
    }
    /* For each skipped macroblock, do... */

    for (addr = m->mblock.past_mb_addr + 1;
         addr < m->mblock.mb_address; addr++)
    {

        /* Calculate macroblock row and col. */

        mb_row = addr / m->mb_width;
        mb_col = addr % m->mb_width;

        /* Calculate upper left pixel row,col for luminance plane. */

        row = mb_row << 4;
        col = mb_col << 4;
        crow = row / 2;
        ccol = col / 2;

        /* If forward predicted, calculate prediction values. */

        if (m->mblock.bpict_past_forw)
        {

            ReconSkippedBlock(m->past->luminance, forw_lum,
                              row, col, row_size, right_for, down_for,
                              right_half_for, down_half_for, 16);
            ReconSkippedBlock(m->past->Cr, forw_cr, crow,
                              ccol, half_row,
                              c_right_for, c_down_for, c_right_half_for, c_down_half_for, 8);
            ReconSkippedBlock(m->past->Cb, forw_cb, crow,
                              ccol, half_row,
                              c_right_for, c_down_for, c_right_half_for, c_down_half_for, 8);
        }
        /* If back predicted, calculate prediction values. */

        if (m->mblock.bpict_past_back)
        {
            ReconSkippedBlock(m->future->luminance, back_lum,
                              row, col, row_size, right_back, down_back,
                              right_half_back, down_half_back, 16);
            ReconSkippedBlock(m->future->Cr, back_cr, crow,
                              ccol, half_row,
                              c_right_back, c_down_back,
                              c_right_half_back, c_down_half_back, 8);
            ReconSkippedBlock(m->future->Cb, back_cb, crow,
                              ccol, half_row,
                              c_right_back, c_down_back,
                              c_right_half_back, c_down_half_back, 8);
        }
        if (m->mblock.bpict_past_forw && !m->mblock.bpict_past_back)
        {

            int *dest, *dest1;
            int *src, *src1;
            dest = (int *)(m->current->luminance + (row * row_size) + col);
            src = (int *)forw_lum;

            for (rr = 0; rr < 16; rr++)
            {

                /* memcpy(dest, forw_lum+(rr<<4), 16);  */
                dest[0] = src[0];
                dest[1] = src[1];
                dest[2] = src[2];
                dest[3] = src[3];
                dest += row_incr;
                src += 4;
            }

            dest = (int *)(m->current->Cr + (crow * half_row) + ccol);
            dest1 = (int *)(m->current->Cb + (crow * half_row) + ccol);
            src = (int *)forw_cr;
            src1 = (int *)forw_cb;

            for (rr = 0; rr < 8; rr++)
            {
                /*
	 * memcpy(dest, forw_cr+(rr<<3), 8); memcpy(dest1, forw_cb+(rr<<3),
	 * 8);
	 */

                dest[0] = src[0];
                dest[1] = src[1];

                dest1[0] = src1[0];
                dest1[1] = src1[1];

                dest += half_row_incr;
                dest1 += half_row_incr;
                src += 2;
                src1 += 2;
            }
        }
        else if (m->mblock.bpict_past_back && !m->mblock.bpict_past_forw)
        {

            int *src, *src1;
            int *dest, *dest1;
            dest = (int *)(m->current->luminance + (row * row_size) + col);
            src = (int *)back_lum;

            for (rr = 0; rr < 16; rr++)
            {
                dest[0] = src[0];
                dest[1] = src[1];
                dest[2] = src[2];
                dest[3] = src[3];
                dest += row_incr;
                src += 4;
            }

            dest = (int *)(m->current->Cr + (crow * half_row) + ccol);
            dest1 = (int *)(m->current->Cb + (crow * half_row) + ccol);
            src = (int *)back_cr;
            src1 = (int *)back_cb;

            for (rr = 0; rr < 8; rr++)
            {
                /*
	 * memcpy(dest, back_cr+(rr<<3), 8); memcpy(dest1, back_cb+(rr<<3),
	 * 8);
	 */

                dest[0] = src[0];
                dest[1] = src[1];

                dest1[0] = src1[0];
                dest1[1] = src1[1];

                dest += half_row_incr;
                dest1 += half_row_incr;
                src += 2;
                src1 += 2;
            }
        }
        else
        {

            unsigned char *src1, *src2, *src1a, *src2a;
            unsigned char *dest, *dest1;
            dest = m->current->luminance + (row * row_size) + col;
            src1 = forw_lum;
            src2 = back_lum;

            for (rr = 0; rr < 16; rr++)
            {
                dest[0] = (int)(src1[0] + src2[0]) >> 1;
                dest[1] = (int)(src1[1] + src2[1]) >> 1;
                dest[2] = (int)(src1[2] + src2[2]) >> 1;
                dest[3] = (int)(src1[3] + src2[3]) >> 1;
                dest[4] = (int)(src1[4] + src2[4]) >> 1;
                dest[5] = (int)(src1[5] + src2[5]) >> 1;
                dest[6] = (int)(src1[6] + src2[6]) >> 1;
                dest[7] = (int)(src1[7] + src2[7]) >> 1;
                dest[8] = (int)(src1[8] + src2[8]) >> 1;
                dest[9] = (int)(src1[9] + src2[9]) >> 1;
                dest[10] = (int)(src1[10] + src2[10]) >> 1;
                dest[11] = (int)(src1[11] + src2[11]) >> 1;
                dest[12] = (int)(src1[12] + src2[12]) >> 1;
                dest[13] = (int)(src1[13] + src2[13]) >> 1;
                dest[14] = (int)(src1[14] + src2[14]) >> 1;
                dest[15] = (int)(src1[15] + src2[15]) >> 1;
                dest += row_size;
                src1 += 16;
                src2 += 16;
            }

            dest = m->current->Cr + (crow * half_row) + ccol;
            dest1 = m->current->Cb + (crow * half_row) + ccol;
            src1 = forw_cr;
            src2 = back_cr;
            src1a = forw_cb;
            src2a = back_cb;

            for (rr = 0; rr < 8; rr++)
            {
                dest[0] = (int)(src1[0] + src2[0]) >> 1;
                dest[1] = (int)(src1[1] + src2[1]) >> 1;
                dest[2] = (int)(src1[2] + src2[2]) >> 1;
                dest[3] = (int)(src1[3] + src2[3]) >> 1;
                dest[4] = (int)(src1[4] + src2[4]) >> 1;
                dest[5] = (int)(src1[5] + src2[5]) >> 1;
                dest[6] = (int)(src1[6] + src2[6]) >> 1;
                dest[7] = (int)(src1[7] + src2[7]) >> 1;
                dest += half_row;
                src1 += 8;
                src2 += 8;

                dest1[0] = (int)(src1a[0] + src2a[0]) >> 1;
                dest1[1] = (int)(src1a[1] + src2a[1]) >> 1;
                dest1[2] = (int)(src1a[2] + src2a[2]) >> 1;
                dest1[3] = (int)(src1a[3] + src2a[3]) >> 1;
                dest1[4] = (int)(src1a[4] + src2a[4]) >> 1;
                dest1[5] = (int)(src1a[5] + src2a[5]) >> 1;
                dest1[6] = (int)(src1a[6] + src2a[6]) >> 1;
                dest1[7] = (int)(src1a[7] + src2a[7]) >> 1;
                dest1 += half_row;
                src1a += 8;
                src2a += 8;
            }
        }
    }
}

/*
 *--------------------------------------------------------------
 *
 * ComputeVector --
 *
 *	Computes motion vector given parameters previously parsed
 *      and reconstructed.
 *
 * Results:
 *      Reconstructed motion vector info is put into recon_* parameters
 *      passed to this function. Also updated previous motion vector
 *      information.
 *
 * Side effects:
 *      None.
 *
 *--------------------------------------------------------------
 */

#define ComputeVector(recon_right_ptr, recon_down_ptr, recon_right_prev, recon_down_prev, f, full_pel_vector, motion_h_code, motion_v_code, motion_h_r, motion_v_r) \
                                                                                                                                                                    \
    {                                                                                                                                                               \
        int comp_h_r, comp_v_r;                                                                                                                                     \
        int right_little, right_big, down_little, down_big;                                                                                                         \
        int max, min, new_vector;                                                                                                                                   \
                                                                                                                                                                    \
        /* The following procedure for the reconstruction of motion vectors                                                                                         \
           is a direct and simple implementation of the instructions given                                                                                          \
           in the mpeg December 1991 standard draft.                                                                                                                \
        */                                                                                                                                                          \
                                                                                                                                                                    \
        if (f == 1 || motion_h_code == 0)                                                                                                                           \
            comp_h_r = 0;                                                                                                                                           \
        else                                                                                                                                                        \
            comp_h_r = f - 1 - motion_h_r;                                                                                                                          \
                                                                                                                                                                    \
        if (f == 1 || motion_v_code == 0)                                                                                                                           \
            comp_v_r = 0;                                                                                                                                           \
        else                                                                                                                                                        \
            comp_v_r = f - 1 - motion_v_r;                                                                                                                          \
                                                                                                                                                                    \
        right_little = motion_h_code * f;                                                                                                                           \
        if (right_little == 0)                                                                                                                                      \
            right_big = 0;                                                                                                                                          \
        else                                                                                                                                                        \
        {                                                                                                                                                           \
            if (right_little > 0)                                                                                                                                   \
            {                                                                                                                                                       \
                right_little = right_little - comp_h_r;                                                                                                             \
                right_big = right_little - 32 * f;                                                                                                                  \
            }                                                                                                                                                       \
            else                                                                                                                                                    \
            {                                                                                                                                                       \
                right_little = right_little + comp_h_r;                                                                                                             \
                right_big = right_little + 32 * f;                                                                                                                  \
            }                                                                                                                                                       \
        }                                                                                                                                                           \
                                                                                                                                                                    \
        down_little = motion_v_code * f;                                                                                                                            \
        if (down_little == 0)                                                                                                                                       \
            down_big = 0;                                                                                                                                           \
        else                                                                                                                                                        \
        {                                                                                                                                                           \
            if (down_little > 0)                                                                                                                                    \
            {                                                                                                                                                       \
                down_little = down_little - comp_v_r;                                                                                                               \
                down_big = down_little - 32 * f;                                                                                                                    \
            }                                                                                                                                                       \
            else                                                                                                                                                    \
            {                                                                                                                                                       \
                down_little = down_little + comp_v_r;                                                                                                               \
                down_big = down_little + 32 * f;                                                                                                                    \
            }                                                                                                                                                       \
        }                                                                                                                                                           \
                                                                                                                                                                    \
        max = 16 * f - 1;                                                                                                                                           \
        min = -16 * f;                                                                                                                                              \
                                                                                                                                                                    \
        new_vector = recon_right_prev + right_little;                                                                                                               \
                                                                                                                                                                    \
        if (new_vector <= max && new_vector >= min)                                                                                                                 \
            *recon_right_ptr = recon_right_prev + right_little;                                                                                                     \
        /* just new_vector */                                                                                                                                       \
        else                                                                                                                                                        \
            *recon_right_ptr = recon_right_prev + right_big;                                                                                                        \
        recon_right_prev = *recon_right_ptr;                                                                                                                        \
        if (full_pel_vector)                                                                                                                                        \
            *recon_right_ptr = *recon_right_ptr << 1;                                                                                                               \
                                                                                                                                                                    \
        new_vector = recon_down_prev + down_little;                                                                                                                 \
        if (new_vector <= max && new_vector >= min)                                                                                                                 \
            *recon_down_ptr = recon_down_prev + down_little;                                                                                                        \
        /* just new_vector */                                                                                                                                       \
        else                                                                                                                                                        \
            *recon_down_ptr = recon_down_prev + down_big;                                                                                                           \
        recon_down_prev = *recon_down_ptr;                                                                                                                          \
        if (full_pel_vector)                                                                                                                                        \
            *recon_down_ptr = *recon_down_ptr << 1;                                                                                                                 \
    }

/*
 *--------------------------------------------------------------
 *
 * ComputeForwVector --
 *
 *	Computes forward motion vector by calling ComputeVector
 *      with appropriate parameters.
 *
 * Results:
 *	Reconstructed motion vector placed in recon_right_for_ptr and
 *      recon_down_for_ptr.
 *
 * Side effects:
 *      None.
 *
 *--------------------------------------------------------------
 */

void
ComputeForwVector(MPEG *m, int *recon_right_for_ptr, int *recon_down_for_ptr)
{
    Pict *picture;
    Macroblock *mblock;

    picture = &(m->picture);
    mblock = &(m->mblock);

    ComputeVector(recon_right_for_ptr, recon_down_for_ptr,
                  mblock->recon_right_for_prev,
                  mblock->recon_down_for_prev,
                  picture->forw_f, picture->full_pel_forw_vector,
                  mblock->motion_h_forw_code, mblock->motion_v_forw_code,
                  mblock->motion_h_forw_r, mblock->motion_v_forw_r);
}

/*
 *--------------------------------------------------------------
 *
 * ComputeBackVector --
 *
 *	Computes backward motion vector by calling ComputeVector
 *      with appropriate parameters.
 *
 * Results:
 *	Reconstructed motion vector placed in recon_right_back_ptr and
 *      recon_down_back_ptr.
 *
 * Side effects:
 *      None.
 *
 *--------------------------------------------------------------
 */

void
ComputeBackVector(MPEG *m, int *recon_right_back_ptr, int *recon_down_back_ptr)
{
    Pict *picture;
    Macroblock *mblock;

    picture = &(m->picture);
    mblock = &(m->mblock);

    ComputeVector(recon_right_back_ptr, recon_down_back_ptr,
                  mblock->recon_right_back_prev,
                  mblock->recon_down_back_prev,
                  picture->back_f, picture->full_pel_back_vector,
                  mblock->motion_h_back_code, mblock->motion_v_back_code,
                  mblock->motion_h_back_r, mblock->motion_v_back_r);
}

/*
 *--------------------------------------------------------------
 *
 * ReconIMBlock --
 *
 *	Reconstructs intra coded macroblock.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

static void
ReconIMBlock(MPEG *m, int bnum)
{
    int mb_row, mb_col, row, col, row_size, rr;
    unsigned char *dest;

    /* Calculate macroblock row and column from address. */

    mb_row = m->mblock.mb_address / m->mb_width;
    mb_col = m->mblock.mb_address % m->mb_width;

    /* If block is luminance block... */

    if (bnum < 4)
    {

        /* Calculate row and col values for upper left pixel of block. */

        row = mb_row * 16;
        col = mb_col * 16;
        if (bnum > 1)
            row += 8;
        if (bnum % 2)
            col += 8;

        /* Set dest to luminance plane of current pict image. */

        dest = m->current->luminance;

        /* Establish row size. */

        row_size = m->mb_width * 16;
    }
    /* Otherwise if block is Cr block... */

    else if (bnum == 4)
    {

        /* Set dest to Cr plane of current pict image. */

        dest = m->current->Cr;

        /* Establish row size. */

        row_size = m->mb_width * 8;

        /* Calculate row,col for upper left pixel of block. */

        row = mb_row * 8;
        col = mb_col * 8;
    }
    /* Otherwise block is Cb block, and ... */

    else
    {

        /* Set dest to Cb plane of current pict image. */

        dest = m->current->Cb;

        /* Establish row size. */

        row_size = m->mb_width * 8;

        /* Calculate row,col for upper left pixel value of block. */

        row = mb_row * 8;
        col = mb_col * 8;
    }

    /*
   * For each pixel in block, set to cropped reconstructed value from inverse
   * dct.
   */
    {
        short *sp = &m->block.dct_recon[0][0];
        unsigned char *cm = cropTbl + MAX_NEG_CROP;
        dest += row * row_size + col;
        for (rr = 0; rr < 4; rr++, sp += 16, dest += row_size)
        {
            dest[0] = cm[sp[0]];
            dest[1] = cm[sp[1]];
            dest[2] = cm[sp[2]];
            dest[3] = cm[sp[3]];
            dest[4] = cm[sp[4]];
            dest[5] = cm[sp[5]];
            dest[6] = cm[sp[6]];
            dest[7] = cm[sp[7]];

            dest += row_size;
            dest[0] = cm[sp[8]];
            dest[1] = cm[sp[9]];
            dest[2] = cm[sp[10]];
            dest[3] = cm[sp[11]];
            dest[4] = cm[sp[12]];
            dest[5] = cm[sp[13]];
            dest[6] = cm[sp[14]];
            dest[7] = cm[sp[15]];
        }
    }
}

/*
 *--------------------------------------------------------------
 *
 * ReconPMBlock --
 *
 *	Reconstructs forward predicted macroblocks.
 *
 * Results:
 *      None.
 *
 * Side effects:
 *      None.
 *
 *--------------------------------------------------------------
 */

static void
ReconPMBlock(MPEG *m, int bnum, int recon_right_for, int recon_down_for, int zflag)
{
    int mb_row, mb_col, row, col, row_size, rr;
    unsigned char *dest, *past = 0;
    static int right_for, down_for, right_half_for, down_half_for;
    unsigned char *rindex1, *rindex2;
    unsigned char *index;
    short int *blockvals;

    /* Calculate macroblock row and column from address. */

    mb_row = m->mblock.mb_address / m->mb_width;
    mb_col = m->mblock.mb_address % m->mb_width;

    if (bnum < 4)
    {

        /* Calculate right_for, down_for motion vectors. */

        right_for = recon_right_for >> 1;
        down_for = recon_down_for >> 1;
        right_half_for = recon_right_for & 0x1;
        down_half_for = recon_down_for & 0x1;

        /* Set dest to luminance plane of current pict image. */

        dest = m->current->luminance;

        if (m->picture.code_type == B_TYPE)
        {
            if (m->past != NULL)
                past = m->past->luminance;
        }
        else
        {

            /* Set predicitive frame to current future frame. */

            if (m->future != NULL)
                past = m->future->luminance;
        }

        /* Establish row size. */

        row_size = m->mb_width << 4;

        /* Calculate row,col of upper left pixel in block. */

        row = mb_row << 4;
        col = mb_col << 4;
        if (bnum > 1)
            row += 8;
        if (bnum % 2)
            col += 8;
    }
    /* Otherwise, block is NOT luminance block, ... */

    else
    {

        /* Construct motion vectors. */

        recon_right_for /= 2;
        recon_down_for /= 2;
        right_for = recon_right_for >> 1;
        down_for = recon_down_for >> 1;
        right_half_for = recon_right_for & 0x1;
        down_half_for = recon_down_for & 0x1;

        /* Establish row size. */

        row_size = m->mb_width << 3;

        /* Calculate row,col of upper left pixel in block. */

        row = mb_row << 3;
        col = mb_col << 3;

        /* If block is Cr block... */

        if (bnum == 4)
        {

            /* Set dest to Cr plane of current pict image. */

            dest = m->current->Cr;

            if (m->picture.code_type == B_TYPE)
            {

                if (m->past != NULL)
                    past = m->past->Cr;
            }
            else
            {
                if (m->future != NULL)
                    past = m->future->Cr;
            }
        }
        /* Otherwise, block is Cb block... */

        else
        {

            /* Set dest to Cb plane of current pict image. */

            dest = m->current->Cb;

            if (m->picture.code_type == B_TYPE)
            {
                if (m->past != NULL)
                    past = m->past->Cb;
            }
            else
            {
                if (m->future != NULL)
                    past = m->future->Cb;
            }
        }
    }

    /* For each pixel in block... */

    index = dest + (row * row_size) + col;
    rindex1 = past + (row + down_for) * row_size + col + right_for;

    blockvals = &(m->block.dct_recon[0][0]);

    /*
     * Calculate predictive pixel value based on motion vectors and copy to
     * dest plane.
     */

    if ((!down_half_for) && (!right_half_for))
    {
        unsigned char *cm = cropTbl + MAX_NEG_CROP;
        if (!zflag)
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = cm[(int)rindex1[0] + (int)blockvals[0]];
                index[1] = cm[(int)rindex1[1] + (int)blockvals[1]];
                index[2] = cm[(int)rindex1[2] + (int)blockvals[2]];
                index[3] = cm[(int)rindex1[3] + (int)blockvals[3]];
                index[4] = cm[(int)rindex1[4] + (int)blockvals[4]];
                index[5] = cm[(int)rindex1[5] + (int)blockvals[5]];
                index[6] = cm[(int)rindex1[6] + (int)blockvals[6]];
                index[7] = cm[(int)rindex1[7] + (int)blockvals[7]];
                index += row_size;
                rindex1 += row_size;

                index[0] = cm[(int)rindex1[0] + (int)blockvals[8]];
                index[1] = cm[(int)rindex1[1] + (int)blockvals[9]];
                index[2] = cm[(int)rindex1[2] + (int)blockvals[10]];
                index[3] = cm[(int)rindex1[3] + (int)blockvals[11]];
                index[4] = cm[(int)rindex1[4] + (int)blockvals[12]];
                index[5] = cm[(int)rindex1[5] + (int)blockvals[13]];
                index[6] = cm[(int)rindex1[6] + (int)blockvals[14]];
                index[7] = cm[(int)rindex1[7] + (int)blockvals[15]];
                blockvals += 16;
                index += row_size;
                rindex1 += row_size;
            }
        else
        {
            if (right_for & 0x1)
            {
                /* No alignment, use bye copy */
                for (rr = 0; rr < 4; rr++)
                {
                    index[0] = rindex1[0];
                    index[1] = rindex1[1];
                    index[2] = rindex1[2];
                    index[3] = rindex1[3];
                    index[4] = rindex1[4];
                    index[5] = rindex1[5];
                    index[6] = rindex1[6];
                    index[7] = rindex1[7];
                    index += row_size;
                    rindex1 += row_size;

                    index[0] = rindex1[0];
                    index[1] = rindex1[1];
                    index[2] = rindex1[2];
                    index[3] = rindex1[3];
                    index[4] = rindex1[4];
                    index[5] = rindex1[5];
                    index[6] = rindex1[6];
                    index[7] = rindex1[7];
                    index += row_size;
                    rindex1 += row_size;
                }
            }
            else if (right_for & 0x2)
            {
                /* Half-word bit aligned, use 16 bit copy */
                short *src = (short *)rindex1;
                short *dest = (short *)index;
                row_size >>= 1;
                for (rr = 0; rr < 4; rr++)
                {
                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest[2] = src[2];
                    dest[3] = src[3];
                    dest += row_size;
                    src += row_size;

                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest[2] = src[2];
                    dest[3] = src[3];
                    dest += row_size;
                    src += row_size;
                }
            }
            else
            {
                /* Word aligned, use 32 bit copy */
                int *src = (int *)rindex1;
                int *dest = (int *)index;
                row_size >>= 2;
                for (rr = 0; rr < 4; rr++)
                {
                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest += row_size;
                    src += row_size;

                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest += row_size;
                    src += row_size;
                }
            }
        }
    }
    else
    {
        unsigned char *cm = cropTbl + MAX_NEG_CROP;
        rindex2 = rindex1 + right_half_for + (down_half_for * row_size);
        if (!zflag)
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = cm[((int)(rindex1[0] + rindex2[0]) >> 1) + blockvals[0]];
                index[1] = cm[((int)(rindex1[1] + rindex2[1]) >> 1) + blockvals[1]];
                index[2] = cm[((int)(rindex1[2] + rindex2[2]) >> 1) + blockvals[2]];
                index[3] = cm[((int)(rindex1[3] + rindex2[3]) >> 1) + blockvals[3]];
                index[4] = cm[((int)(rindex1[4] + rindex2[4]) >> 1) + blockvals[4]];
                index[5] = cm[((int)(rindex1[5] + rindex2[5]) >> 1) + blockvals[5]];
                index[6] = cm[((int)(rindex1[6] + rindex2[6]) >> 1) + blockvals[6]];
                index[7] = cm[((int)(rindex1[7] + rindex2[7]) >> 1) + blockvals[7]];
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;

                index[0] = cm[((int)(rindex1[0] + rindex2[0]) >> 1) + blockvals[8]];
                index[1] = cm[((int)(rindex1[1] + rindex2[1]) >> 1) + blockvals[9]];
                index[2] = cm[((int)(rindex1[2] + rindex2[2]) >> 1) + blockvals[10]];
                index[3] = cm[((int)(rindex1[3] + rindex2[3]) >> 1) + blockvals[11]];
                index[4] = cm[((int)(rindex1[4] + rindex2[4]) >> 1) + blockvals[12]];
                index[5] = cm[((int)(rindex1[5] + rindex2[5]) >> 1) + blockvals[13]];
                index[6] = cm[((int)(rindex1[6] + rindex2[6]) >> 1) + blockvals[14]];
                index[7] = cm[((int)(rindex1[7] + rindex2[7]) >> 1) + blockvals[15]];
                blockvals += 16;
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;
            }
        else
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = (int)(rindex1[0] + rindex2[0]) >> 1;
                index[1] = (int)(rindex1[1] + rindex2[1]) >> 1;
                index[2] = (int)(rindex1[2] + rindex2[2]) >> 1;
                index[3] = (int)(rindex1[3] + rindex2[3]) >> 1;
                index[4] = (int)(rindex1[4] + rindex2[4]) >> 1;
                index[5] = (int)(rindex1[5] + rindex2[5]) >> 1;
                index[6] = (int)(rindex1[6] + rindex2[6]) >> 1;
                index[7] = (int)(rindex1[7] + rindex2[7]) >> 1;
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;

                index[0] = (int)(rindex1[0] + rindex2[0]) >> 1;
                index[1] = (int)(rindex1[1] + rindex2[1]) >> 1;
                index[2] = (int)(rindex1[2] + rindex2[2]) >> 1;
                index[3] = (int)(rindex1[3] + rindex2[3]) >> 1;
                index[4] = (int)(rindex1[4] + rindex2[4]) >> 1;
                index[5] = (int)(rindex1[5] + rindex2[5]) >> 1;
                index[6] = (int)(rindex1[6] + rindex2[6]) >> 1;
                index[7] = (int)(rindex1[7] + rindex2[7]) >> 1;
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;
            }
    }
}

/*
 *--------------------------------------------------------------
 *
 * ReconBMBlock --
 *
 *	Reconstructs back predicted macroblocks.
 *
 * Results:
 *      None.
 *
 * Side effects:
 *      None.
 *
 *--------------------------------------------------------------
 */

static void
ReconBMBlock(MPEG *m, int bnum, int recon_right_back, int recon_down_back, int zflag)
{
    int mb_row, mb_col, row, col, row_size, rr;
    unsigned char *dest, *future = 0;
    int right_back, down_back, right_half_back, down_half_back;
    unsigned char *rindex1, *rindex2;
    unsigned char *index;
    short int *blockvals;

    /* Calculate macroblock row and column from address. */

    mb_row = m->mblock.mb_address / m->mb_width;
    mb_col = m->mblock.mb_address % m->mb_width;

    /* If block is luminance block... */

    if (bnum < 4)
    {

        /* Calculate right_back, down_bakc motion vectors. */

        right_back = recon_right_back >> 1;
        down_back = recon_down_back >> 1;
        right_half_back = recon_right_back & 0x1;
        down_half_back = recon_down_back & 0x1;

        /* Set dest to luminance plane of current pict image. */

        dest = m->current->luminance;

        /*
     * If future frame exists, set future to luminance plane of future frame.
     */

        if (m->future != NULL)
            future = m->future->luminance;

        /* Establish row size. */

        row_size = m->mb_width << 4;

        /* Calculate row,col of upper left pixel in block. */

        row = mb_row << 4;
        col = mb_col << 4;
        if (bnum > 1)
            row += 8;
        if (bnum % 2)
            col += 8;
    }
    /* Otherwise, block is NOT luminance block, ... */

    else
    {

        /* Construct motion vectors. */

        recon_right_back /= 2;
        recon_down_back /= 2;
        right_back = recon_right_back >> 1;
        down_back = recon_down_back >> 1;
        right_half_back = recon_right_back & 0x1;
        down_half_back = recon_down_back & 0x1;

        /* Establish row size. */

        row_size = m->mb_width << 3;

        /* Calculate row,col of upper left pixel in block. */

        row = mb_row << 3;
        col = mb_col << 3;

        /* If block is Cr block... */

        if (bnum == 4)
        {

            /* Set dest to Cr plane of current pict image. */

            dest = m->current->Cr;

            /*
       * If future frame exists, set future to Cr plane of future image.
       */

            if (m->future != NULL)
                future = m->future->Cr;
        }
        /* Otherwise, block is Cb block... */

        else
        {

            /* Set dest to Cb plane of current pict image. */

            dest = m->current->Cb;

            /*
       * If future frame exists, set future to Cb plane of future frame.
       */

            if (m->future != NULL)
                future = m->future->Cb;
        }
    }

    /* For each pixel in block do... */

    index = dest + (row * row_size) + col;
    rindex1 = future + (row + down_back) * row_size + col + right_back;

    blockvals = &(m->block.dct_recon[0][0]);

    if ((!right_half_back) && (!down_half_back))
    {
        unsigned char *cm = cropTbl + MAX_NEG_CROP;
        if (!zflag)
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = cm[(int)rindex1[0] + (int)blockvals[0]];
                index[1] = cm[(int)rindex1[1] + (int)blockvals[1]];
                index[2] = cm[(int)rindex1[2] + (int)blockvals[2]];
                index[3] = cm[(int)rindex1[3] + (int)blockvals[3]];
                index[4] = cm[(int)rindex1[4] + (int)blockvals[4]];
                index[5] = cm[(int)rindex1[5] + (int)blockvals[5]];
                index[6] = cm[(int)rindex1[6] + (int)blockvals[6]];
                index[7] = cm[(int)rindex1[7] + (int)blockvals[7]];
                index += row_size;
                rindex1 += row_size;

                index[0] = cm[(int)rindex1[0] + (int)blockvals[8]];
                index[1] = cm[(int)rindex1[1] + (int)blockvals[9]];
                index[2] = cm[(int)rindex1[2] + (int)blockvals[10]];
                index[3] = cm[(int)rindex1[3] + (int)blockvals[11]];
                index[4] = cm[(int)rindex1[4] + (int)blockvals[12]];
                index[5] = cm[(int)rindex1[5] + (int)blockvals[13]];
                index[6] = cm[(int)rindex1[6] + (int)blockvals[14]];
                index[7] = cm[(int)rindex1[7] + (int)blockvals[15]];
                blockvals += 16;
                index += row_size;
                rindex1 += row_size;
            }
        else
        {
            if (right_back & 0x1)
            {
                /* No alignment, use bye copy */
                for (rr = 0; rr < 4; rr++)
                {
                    index[0] = rindex1[0];
                    index[1] = rindex1[1];
                    index[2] = rindex1[2];
                    index[3] = rindex1[3];
                    index[4] = rindex1[4];
                    index[5] = rindex1[5];
                    index[6] = rindex1[6];
                    index[7] = rindex1[7];
                    index += row_size;
                    rindex1 += row_size;

                    index[0] = rindex1[0];
                    index[1] = rindex1[1];
                    index[2] = rindex1[2];
                    index[3] = rindex1[3];
                    index[4] = rindex1[4];
                    index[5] = rindex1[5];
                    index[6] = rindex1[6];
                    index[7] = rindex1[7];
                    index += row_size;
                    rindex1 += row_size;
                }
            }
            else if (right_back & 0x2)
            {
                /* Half-word bit aligned, use 16 bit copy */
                short *src = (short *)rindex1;
                short *dest = (short *)index;
                row_size >>= 1;
                for (rr = 0; rr < 4; rr++)
                {
                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest[2] = src[2];
                    dest[3] = src[3];
                    dest += row_size;
                    src += row_size;

                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest[2] = src[2];
                    dest[3] = src[3];
                    dest += row_size;
                    src += row_size;
                }
            }
            else
            {
                /* Word aligned, use 32 bit copy */
                int *src = (int *)rindex1;
                int *dest = (int *)index;
                row_size >>= 2;
                for (rr = 0; rr < 4; rr++)
                {
                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest += row_size;
                    src += row_size;

                    dest[0] = src[0];
                    dest[1] = src[1];
                    dest += row_size;
                    src += row_size;
                }
            }
        }
    }
    else
    {
        unsigned char *cm = cropTbl + MAX_NEG_CROP;
        rindex2 = rindex1 + right_half_back + (down_half_back * row_size);
        if (!zflag)
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = cm[((int)(rindex1[0] + rindex2[0]) >> 1) + blockvals[0]];
                index[1] = cm[((int)(rindex1[1] + rindex2[1]) >> 1) + blockvals[1]];
                index[2] = cm[((int)(rindex1[2] + rindex2[2]) >> 1) + blockvals[2]];
                index[3] = cm[((int)(rindex1[3] + rindex2[3]) >> 1) + blockvals[3]];
                index[4] = cm[((int)(rindex1[4] + rindex2[4]) >> 1) + blockvals[4]];
                index[5] = cm[((int)(rindex1[5] + rindex2[5]) >> 1) + blockvals[5]];
                index[6] = cm[((int)(rindex1[6] + rindex2[6]) >> 1) + blockvals[6]];
                index[7] = cm[((int)(rindex1[7] + rindex2[7]) >> 1) + blockvals[7]];
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;

                index[0] = cm[((int)(rindex1[0] + rindex2[0]) >> 1) + blockvals[8]];
                index[1] = cm[((int)(rindex1[1] + rindex2[1]) >> 1) + blockvals[9]];
                index[2] = cm[((int)(rindex1[2] + rindex2[2]) >> 1) + blockvals[10]];
                index[3] = cm[((int)(rindex1[3] + rindex2[3]) >> 1) + blockvals[11]];
                index[4] = cm[((int)(rindex1[4] + rindex2[4]) >> 1) + blockvals[12]];
                index[5] = cm[((int)(rindex1[5] + rindex2[5]) >> 1) + blockvals[13]];
                index[6] = cm[((int)(rindex1[6] + rindex2[6]) >> 1) + blockvals[14]];
                index[7] = cm[((int)(rindex1[7] + rindex2[7]) >> 1) + blockvals[15]];
                blockvals += 16;
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;
            }
        else
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = (int)(rindex1[0] + rindex2[0]) >> 1;
                index[1] = (int)(rindex1[1] + rindex2[1]) >> 1;
                index[2] = (int)(rindex1[2] + rindex2[2]) >> 1;
                index[3] = (int)(rindex1[3] + rindex2[3]) >> 1;
                index[4] = (int)(rindex1[4] + rindex2[4]) >> 1;
                index[5] = (int)(rindex1[5] + rindex2[5]) >> 1;
                index[6] = (int)(rindex1[6] + rindex2[6]) >> 1;
                index[7] = (int)(rindex1[7] + rindex2[7]) >> 1;
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;

                index[0] = (int)(rindex1[0] + rindex2[0]) >> 1;
                index[1] = (int)(rindex1[1] + rindex2[1]) >> 1;
                index[2] = (int)(rindex1[2] + rindex2[2]) >> 1;
                index[3] = (int)(rindex1[3] + rindex2[3]) >> 1;
                index[4] = (int)(rindex1[4] + rindex2[4]) >> 1;
                index[5] = (int)(rindex1[5] + rindex2[5]) >> 1;
                index[6] = (int)(rindex1[6] + rindex2[6]) >> 1;
                index[7] = (int)(rindex1[7] + rindex2[7]) >> 1;
                index += row_size;
                rindex1 += row_size;
                rindex2 += row_size;
            }
    }
}

/*
 *--------------------------------------------------------------
 *
 * ReconBiMBlock --
 *
 *	Reconstructs bidirectionally predicted macroblocks.
 *
 * Results:
 *      None.
 *
 * Side effects:
 *      None.
 *
 *--------------------------------------------------------------
 */

static void
ReconBiMBlock(MPEG *m, int bnum, int recon_right_for, int recon_down_for,
              int recon_right_back, int recon_down_back, int zflag)
{
    int mb_row, mb_col, row, col, row_size, rr;
    unsigned char *dest, *past = 0, *future = 0;
    int right_for, down_for; //, right_half_for, down_half_for;
    int right_back, down_back; //, right_half_back, down_half_back;
    unsigned char *index, *rindex1, *bindex1;
    short int *blockvals;
    int forw_row_start, back_row_start, forw_col_start, back_col_start;

    /* Calculate macroblock row and column from address. */

    mb_row = m->mblock.mb_address / m->mb_width;
    mb_col = m->mblock.mb_address % m->mb_width;

    /* If block is luminance block... */

    if (bnum < 4)
    {

        /*
     * Calculate right_for, down_for, right_half_for, down_half_for,
     * right_back, down_bakc, right_half_back, and down_half_back, motion
     * vectors.
     */

        right_for = recon_right_for >> 1;
        down_for = recon_down_for >> 1;
        //right_half_for = recon_right_for & 0x1;
        //down_half_for = recon_down_for & 0x1;

        right_back = recon_right_back >> 1;
        down_back = recon_down_back >> 1;
        //right_half_back = recon_right_back & 0x1;
        //down_half_back = recon_down_back & 0x1;

        /* Set dest to luminance plane of current pict image. */

        dest = m->current->luminance;

        /* If past frame exists, set past to luminance plane of past frame. */

        if (m->past != NULL)
            past = m->past->luminance;

        /*
     * If future frame exists, set future to luminance plane of future frame.
     */

        if (m->future != NULL)
            future = m->future->luminance;

        /* Establish row size. */

        row_size = (m->mb_width << 4);

        /* Calculate row,col of upper left pixel in block. */

        row = (mb_row << 4);
        col = (mb_col << 4);
        if (bnum > 1)
            row += 8;
        if (bnum & 0x01)
            col += 8;

        forw_col_start = col + right_for;
        forw_row_start = row + down_for;

        back_col_start = col + right_back;
        back_row_start = row + down_back;
    }
    /* Otherwise, block is NOT luminance block, ... */

    else
    {

        /* Construct motion vectors. */

        recon_right_for /= 2;
        recon_down_for /= 2;
        right_for = recon_right_for >> 1;
        down_for = recon_down_for >> 1;
        //right_half_for = recon_right_for & 0x1;
        //down_half_for = recon_down_for & 0x1;

        recon_right_back /= 2;
        recon_down_back /= 2;
        right_back = recon_right_back >> 1;
        down_back = recon_down_back >> 1;
        //right_half_back = recon_right_back & 0x1;
        //down_half_back = recon_down_back & 0x1;

        /* Establish row size. */

        row_size = (m->mb_width << 3);

        /* Calculate row,col of upper left pixel in block. */

        row = (mb_row << 3);
        col = (mb_col << 3);

        forw_col_start = col + right_for;
        forw_row_start = row + down_for;

        back_col_start = col + right_back;
        back_row_start = row + down_back;

        /* If block is Cr block... */

        if (bnum == 4)
        {

            /* Set dest to Cr plane of current pict image. */

            dest = m->current->Cr;

            /* If past frame exists, set past to Cr plane of past image. */

            if (m->past != NULL)
                past = m->past->Cr;

            /*
       * If future frame exists, set future to Cr plane of future image.
       */

            if (m->future != NULL)
                future = m->future->Cr;
        }
        /* Otherwise, block is Cb block... */

        else
        {

            /* Set dest to Cb plane of current pict image. */

            dest = m->current->Cb;

            /* If past frame exists, set past to Cb plane of past frame. */

            if (m->past != NULL)
                past = m->past->Cb;

            /*
       * If future frame exists, set future to Cb plane of future frame.
       */

            if (m->future != NULL)
                future = m->future->Cb;
        }
    }

    /* For each pixel in block... */

    index = dest + (row * row_size) + col;
    rindex1 = past + forw_row_start * row_size + forw_col_start;
    bindex1 = future + back_row_start * row_size + back_col_start;

    blockvals = (short int *)&(m->block.dct_recon[0][0]);

    {
        unsigned char *cm = cropTbl + MAX_NEG_CROP;
        if (!zflag)
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = cm[((int)(rindex1[0] + bindex1[0]) >> 1) + blockvals[0]];
                index[1] = cm[((int)(rindex1[1] + bindex1[1]) >> 1) + blockvals[1]];
                index[2] = cm[((int)(rindex1[2] + bindex1[2]) >> 1) + blockvals[2]];
                index[3] = cm[((int)(rindex1[3] + bindex1[3]) >> 1) + blockvals[3]];
                index[4] = cm[((int)(rindex1[4] + bindex1[4]) >> 1) + blockvals[4]];
                index[5] = cm[((int)(rindex1[5] + bindex1[5]) >> 1) + blockvals[5]];
                index[6] = cm[((int)(rindex1[6] + bindex1[6]) >> 1) + blockvals[6]];
                index[7] = cm[((int)(rindex1[7] + bindex1[7]) >> 1) + blockvals[7]];
                index += row_size;
                rindex1 += row_size;
                bindex1 += row_size;

                index[0] = cm[((int)(rindex1[0] + bindex1[0]) >> 1) + blockvals[8]];
                index[1] = cm[((int)(rindex1[1] + bindex1[1]) >> 1) + blockvals[9]];
                index[2] = cm[((int)(rindex1[2] + bindex1[2]) >> 1) + blockvals[10]];
                index[3] = cm[((int)(rindex1[3] + bindex1[3]) >> 1) + blockvals[11]];
                index[4] = cm[((int)(rindex1[4] + bindex1[4]) >> 1) + blockvals[12]];
                index[5] = cm[((int)(rindex1[5] + bindex1[5]) >> 1) + blockvals[13]];
                index[6] = cm[((int)(rindex1[6] + bindex1[6]) >> 1) + blockvals[14]];
                index[7] = cm[((int)(rindex1[7] + bindex1[7]) >> 1) + blockvals[15]];
                blockvals += 16;
                index += row_size;
                rindex1 += row_size;
                bindex1 += row_size;
            }

        else
            for (rr = 0; rr < 4; rr++)
            {
                index[0] = (int)(rindex1[0] + bindex1[0]) >> 1;
                index[1] = (int)(rindex1[1] + bindex1[1]) >> 1;
                index[2] = (int)(rindex1[2] + bindex1[2]) >> 1;
                index[3] = (int)(rindex1[3] + bindex1[3]) >> 1;
                index[4] = (int)(rindex1[4] + bindex1[4]) >> 1;
                index[5] = (int)(rindex1[5] + bindex1[5]) >> 1;
                index[6] = (int)(rindex1[6] + bindex1[6]) >> 1;
                index[7] = (int)(rindex1[7] + bindex1[7]) >> 1;
                index += row_size;
                rindex1 += row_size;
                bindex1 += row_size;

                index[0] = (int)(rindex1[0] + bindex1[0]) >> 1;
                index[1] = (int)(rindex1[1] + bindex1[1]) >> 1;
                index[2] = (int)(rindex1[2] + bindex1[2]) >> 1;
                index[3] = (int)(rindex1[3] + bindex1[3]) >> 1;
                index[4] = (int)(rindex1[4] + bindex1[4]) >> 1;
                index[5] = (int)(rindex1[5] + bindex1[5]) >> 1;
                index[6] = (int)(rindex1[6] + bindex1[6]) >> 1;
                index[7] = (int)(rindex1[7] + bindex1[7]) >> 1;
                index += row_size;
                rindex1 += row_size;
                bindex1 += row_size;
            }
    }
}

/* DCT coeff tables. */

static unsigned short int dct_coeff_tbl_0[256] = {
    0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff,
    0x052f, 0x051f, 0x050f, 0x04ff,
    0x183f, 0x402f, 0x3c2f, 0x382f,
    0x342f, 0x302f, 0x2c2f, 0x7c1f,
    0x781f, 0x741f, 0x701f, 0x6c1f,
    0x028e, 0x028e, 0x027e, 0x027e,
    0x026e, 0x026e, 0x025e, 0x025e,
    0x024e, 0x024e, 0x023e, 0x023e,
    0x022e, 0x022e, 0x021e, 0x021e,
    0x020e, 0x020e, 0x04ee, 0x04ee,
    0x04de, 0x04de, 0x04ce, 0x04ce,
    0x04be, 0x04be, 0x04ae, 0x04ae,
    0x049e, 0x049e, 0x048e, 0x048e,
    0x01fd, 0x01fd, 0x01fd, 0x01fd,
    0x01ed, 0x01ed, 0x01ed, 0x01ed,
    0x01dd, 0x01dd, 0x01dd, 0x01dd,
    0x01cd, 0x01cd, 0x01cd, 0x01cd,
    0x01bd, 0x01bd, 0x01bd, 0x01bd,
    0x01ad, 0x01ad, 0x01ad, 0x01ad,
    0x019d, 0x019d, 0x019d, 0x019d,
    0x018d, 0x018d, 0x018d, 0x018d,
    0x017d, 0x017d, 0x017d, 0x017d,
    0x016d, 0x016d, 0x016d, 0x016d,
    0x015d, 0x015d, 0x015d, 0x015d,
    0x014d, 0x014d, 0x014d, 0x014d,
    0x013d, 0x013d, 0x013d, 0x013d,
    0x012d, 0x012d, 0x012d, 0x012d,
    0x011d, 0x011d, 0x011d, 0x011d,
    0x010d, 0x010d, 0x010d, 0x010d,
    0x282c, 0x282c, 0x282c, 0x282c,
    0x282c, 0x282c, 0x282c, 0x282c,
    0x242c, 0x242c, 0x242c, 0x242c,
    0x242c, 0x242c, 0x242c, 0x242c,
    0x143c, 0x143c, 0x143c, 0x143c,
    0x143c, 0x143c, 0x143c, 0x143c,
    0x0c4c, 0x0c4c, 0x0c4c, 0x0c4c,
    0x0c4c, 0x0c4c, 0x0c4c, 0x0c4c,
    0x085c, 0x085c, 0x085c, 0x085c,
    0x085c, 0x085c, 0x085c, 0x085c,
    0x047c, 0x047c, 0x047c, 0x047c,
    0x047c, 0x047c, 0x047c, 0x047c,
    0x046c, 0x046c, 0x046c, 0x046c,
    0x046c, 0x046c, 0x046c, 0x046c,
    0x00fc, 0x00fc, 0x00fc, 0x00fc,
    0x00fc, 0x00fc, 0x00fc, 0x00fc,
    0x00ec, 0x00ec, 0x00ec, 0x00ec,
    0x00ec, 0x00ec, 0x00ec, 0x00ec,
    0x00dc, 0x00dc, 0x00dc, 0x00dc,
    0x00dc, 0x00dc, 0x00dc, 0x00dc,
    0x00cc, 0x00cc, 0x00cc, 0x00cc,
    0x00cc, 0x00cc, 0x00cc, 0x00cc,
    0x681c, 0x681c, 0x681c, 0x681c,
    0x681c, 0x681c, 0x681c, 0x681c,
    0x641c, 0x641c, 0x641c, 0x641c,
    0x641c, 0x641c, 0x641c, 0x641c,
    0x601c, 0x601c, 0x601c, 0x601c,
    0x601c, 0x601c, 0x601c, 0x601c,
    0x5c1c, 0x5c1c, 0x5c1c, 0x5c1c,
    0x5c1c, 0x5c1c, 0x5c1c, 0x5c1c,
    0x581c, 0x581c, 0x581c, 0x581c,
    0x581c, 0x581c, 0x581c, 0x581c,
};

static unsigned short int dct_coeff_tbl_1[16] = {
    0x00bb, 0x202b, 0x103b, 0x00ab,
    0x084b, 0x1c2b, 0x541b, 0x501b,
    0x009b, 0x4c1b, 0x481b, 0x045b,
    0x0c3b, 0x008b, 0x182b, 0x441b,
};

static unsigned short int dct_coeff_tbl_2[4] = {
    0x4019, 0x1429, 0x0079, 0x0839,
};

static unsigned short int dct_coeff_tbl_3[4] = {
    0x0449, 0x3c19, 0x3819, 0x1029,
};

static unsigned short int dct_coeff_next[256] = {
    0xffff, 0xffff, 0xffff, 0xffff,
    0xf7d5, 0xf7d5, 0xf7d5, 0xf7d5,
    0x0826, 0x0826, 0x2416, 0x2416,
    0x0046, 0x0046, 0x2016, 0x2016,
    0x1c15, 0x1c15, 0x1c15, 0x1c15,
    0x1815, 0x1815, 0x1815, 0x1815,
    0x0425, 0x0425, 0x0425, 0x0425,
    0x1415, 0x1415, 0x1415, 0x1415,
    0x3417, 0x0067, 0x3017, 0x2c17,
    0x0c27, 0x0437, 0x0057, 0x2817,
    0x0034, 0x0034, 0x0034, 0x0034,
    0x0034, 0x0034, 0x0034, 0x0034,
    0x1014, 0x1014, 0x1014, 0x1014,
    0x1014, 0x1014, 0x1014, 0x1014,
    0x0c14, 0x0c14, 0x0c14, 0x0c14,
    0x0c14, 0x0c14, 0x0c14, 0x0c14,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0xfbe1, 0xfbe1, 0xfbe1, 0xfbe1,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
    0x0011, 0x0011, 0x0011, 0x0011,
};

static unsigned short int dct_coeff_first[256] = {
    0xffff, 0xffff, 0xffff, 0xffff,
    0xf7d5, 0xf7d5, 0xf7d5, 0xf7d5,
    0x0826, 0x0826, 0x2416, 0x2416,
    0x0046, 0x0046, 0x2016, 0x2016,
    0x1c15, 0x1c15, 0x1c15, 0x1c15,
    0x1815, 0x1815, 0x1815, 0x1815,
    0x0425, 0x0425, 0x0425, 0x0425,
    0x1415, 0x1415, 0x1415, 0x1415,
    0x3417, 0x0067, 0x3017, 0x2c17,
    0x0c27, 0x0437, 0x0057, 0x2817,
    0x0034, 0x0034, 0x0034, 0x0034,
    0x0034, 0x0034, 0x0034, 0x0034,
    0x1014, 0x1014, 0x1014, 0x1014,
    0x1014, 0x1014, 0x1014, 0x1014,
    0x0c14, 0x0c14, 0x0c14, 0x0c14,
    0x0c14, 0x0c14, 0x0c14, 0x0c14,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0023, 0x0023, 0x0023, 0x0023,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0813, 0x0813, 0x0813, 0x0813,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0412, 0x0412, 0x0412, 0x0412,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010,
};

#define RUN_MASK 0xfc00
#define LEVEL_MASK 0x03f0
#define NUM_MASK 0x000f
#define RUN_SHIFT 10
#define LEVEL_SHIFT 4

#define DECODE_DCT_COEFF(m, dct_coeff_tbl, run, level)                            \
    {                                                                             \
        unsigned int temp, index;                                                 \
        unsigned int value, next32bits, flushed;                                  \
                                                                                  \
        /*                                                                        \
         * Grab the next 32 bits and use it to improve performance of             \
         * getting the bits to parse. Thus, calls are translated as:              \
         *                                                                        \
         *	show_bitsX  <-->   next32bits >> (32-X)                                \
         *	get_bitsX   <-->   val = next32bits >> (32-flushed-X);                 \
         *			   flushed += X;                                                     \
         *			   next32bits &= bitMask[flushed];                                   \
         *	flush_bitsX <-->   flushed += X;                                       \
         *			   next32bits &= bitMask[flushed];                                   \
         *                                                                        \
         * I've streamlined the code a lot, so that we don't have to mask         \
         * out the low order bits and a few of the extra adds are removed.        \
         */                                                                       \
        next32bits = show_bits32(m);                                              \
                                                                                  \
        /* show_bits8(index); */                                                  \
        index = next32bits >> 24;                                                 \
                                                                                  \
        if (index > 3)                                                            \
        {                                                                         \
            value = dct_coeff_tbl[index];                                         \
            run = value >> RUN_SHIFT;                                             \
            if (run != END_OF_BLOCK)                                              \
            {                                                                     \
                /* num_bits = (value & NUM_MASK) + 1; */                          \
                /* flush_bits(num_bits); */                                       \
                if (run != ESCAPE)                                                \
                {                                                                 \
                    /* get_bits1(value); */                                       \
                    /* if (value) level = -level; */                              \
                    flushed = (value & NUM_MASK) + 2;                             \
                    level = (value & LEVEL_MASK) >> LEVEL_SHIFT;                  \
                    value = next32bits >> (32 - flushed);                         \
                    value &= 0x1;                                                 \
                    if (value)                                                    \
                        level = -level;                                           \
                    /* next32bits &= ((~0) >> flushed);  last op before update */ \
                }                                                                 \
                else                                                              \
                { /* run == ESCAPE */                                             \
                    /* Get the next six into run, and next 8 into temp */         \
                    /* get_bits14(temp); */                                       \
                    flushed = (value & NUM_MASK) + 1;                             \
                    temp = next32bits >> (18 - flushed);                          \
                    /* Normally, we'd ad 14 to flushed, but I've saved a few      \
                     * instr by moving the add below */                           \
                    temp &= 0x3fff;                                               \
                    run = temp >> 8;                                              \
                    temp &= 0xff;                                                 \
                    if (temp == 0)                                                \
                    {                                                             \
                        /* get_bits8(level); */                                   \
                        level = next32bits >> (10 - flushed);                     \
                        level &= 0xff;                                            \
                        flushed += 22;                                            \
                    }                                                             \
                    else if (temp != 128)                                         \
                    {                                                             \
                        /* Grab sign bit */                                       \
                        flushed += 14;                                            \
                        level = ((int)(temp << 24)) >> 24;                        \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        /* get_bits8(level); */                                   \
                        level = next32bits >> (10 - flushed);                     \
                        level &= 0xff;                                            \
                        flushed += 22;                                            \
                        level = level - 256;                                      \
                    }                                                             \
                }                                                                 \
                /* Update bitstream... */                                         \
                flush_bits(m, flushed);                                           \
            }                                                                     \
        }                                                                         \
        else                                                                      \
        {                                                                         \
            if (index == 2)                                                       \
            {                                                                     \
                /* show_bits10(index); */                                         \
                index = next32bits >> 22;                                         \
                value = dct_coeff_tbl_2[index & 3];                               \
            }                                                                     \
            else if (index == 3)                                                  \
            {                                                                     \
                /* show_bits10(index); */                                         \
                index = next32bits >> 22;                                         \
                value = dct_coeff_tbl_3[index & 3];                               \
            }                                                                     \
            else if (index)                                                       \
            { /* index == 1 */                                                    \
                /* show_bits12(index); */                                         \
                index = next32bits >> 20;                                         \
                value = dct_coeff_tbl_1[index & 15];                              \
            }                                                                     \
            else                                                                  \
            { /* index == 0 */                                                    \
                /* show_bits16(index); */                                         \
                index = next32bits >> 16;                                         \
                value = dct_coeff_tbl_0[index & 255];                             \
            }                                                                     \
            run = value >> RUN_SHIFT;                                             \
            level = (value & LEVEL_MASK) >> LEVEL_SHIFT;                          \
                                                                                  \
            /*                                                                    \
             * Fold these operations together to make it fast...                  \
             */                                                                   \
            /* num_bits = (value & NUM_MASK) + 1; */                              \
            /* flush_bits(num_bits); */                                           \
            /* get_bits1(value); */                                               \
            /* if (value) level = -level; */                                      \
                                                                                  \
            flushed = (value & NUM_MASK) + 2;                                     \
            value = next32bits >> (32 - flushed);                                 \
            value &= 0x1;                                                         \
            if (value)                                                            \
                level = -level;                                                   \
                                                                                  \
            /* Update bitstream ... */                                            \
            flush_bits(m, flushed);                                               \
        }                                                                         \
    }

/*
 *--------------------------------------------------------------
 *
 * ParseReconBlock --
 *
 *	Parse values for block structure from bitstream.
 *      n is an indication of the position of the block within
 *      the macroblock (i.e. 0-5) and indicates the type of 
 *      block (i.e. luminance or chrominance). Reconstructs
 *      coefficients from values parsed and puts in 
 *      block.dct_recon array in vid stream structure.
 *      sparseFlag is set when the block contains only one
 *      coeffictient and is used by the IDCT.
 *
 * Results:
 *	
 *
 * Side effects:
 *      Bit stream irreversibly parsed.
 *
 *--------------------------------------------------------------
 */

#define DCT_recon blockPtr->dct_recon
#define DCT_dc_y_past blockPtr->dct_dc_y_past
#define DCT_dc_cr_past blockPtr->dct_dc_cr_past
#define DCT_dc_cb_past blockPtr->dct_dc_cb_past

static void ParseReconBlock(MPEG *m, int n)
{
    /* Array mapping zigzag to array pointer offset. */
    static int zigzag_direct[64] = {
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12,
        19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
        42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    };

    int coeffCount = 0;
    Block *blockPtr = &m->block;

    if (m->buf_length < 100)
        if (!get_more_data(m))
            return;

    {
        int diff;
        int size, level = 0, i, run, pos, coeff;
        short int *reconptr;
        unsigned char *iqmatrixptr, *niqmatrixptr;
        int qscale;

        reconptr = DCT_recon[0];

        /* 
     * Hand coded version of memset that's a little faster...
     * Old call:
     *	memset((char *) DCT_recon, 0, 64*sizeof(short int));
     */
        {
            INT32 *p;
            p = (INT32 *)reconptr;

            p[0] = p[1] = p[2] = p[3] = p[4] = p[5] = p[6] = p[7] = p[8] = p[9] = p[10] = p[11] = p[12] = p[13] = p[14] = p[15] = p[16] = p[17] = p[18] = p[19] = p[20] = p[21] = p[22] = p[23] = p[24] = p[25] = p[26] = p[27] = p[28] = p[29] = p[30] = p[31] = 0;
        }

        if (m->mblock.mb_intra)
        {

            if (n < 4)
            {

                /*
	 * Get the luminance bits.  This code has been hand optimized to
	 * get by the normal bit parsing routines.  We get some speedup
	 * by grabbing the next 16 bits and parsing things locally.
	 * Thus, calls are translated as:
	 *
	 *	show_bitsX  <-->   next16bits >> (16-X)
	 *	get_bitsX   <-->   val = next16bits >> (16-flushed-X);
	 *			   flushed += X;
	 *			   next16bits &= bitMask[flushed];
	 *	flush_bitsX <-->   flushed += X;
	 *			   next16bits &= bitMask[flushed];
	 *
	 * I've streamlined the code a lot, so that we don't have to mask
	 * out the low order bits and a few of the extra adds are removed.
	 *	bsmith
	 */
                /* And I've messed it all up by removing the globals. cmm */
                unsigned int next16bits, index, flushed;

                next16bits = show_bits16(m);
                index = next16bits >> (16 - 7);
                size = dct_dc_size_luminance[index].value;
                flushed = dct_dc_size_luminance[index].num_bits;
                next16bits &= bitMask[16 + flushed];

                if (size != 0)
                {
                    flushed += size;
                    diff = next16bits >> (16 - flushed);
                    if (!(diff & bitTest[32 - size]))
                    {
                        diff = rBitMask[size] | (diff + 1);
                    }
                }
                else
                {
                    diff = 0;
                }
                flush_bits(m, flushed);

                if (n == 0)
                {
                    coeff = diff << 3;
                    if (m->mblock.mb_address - m->mblock.past_intra_addr > 1)
                        coeff += 1024;
                    else
                        coeff += DCT_dc_y_past;
                    DCT_dc_y_past = coeff;
                }
                else
                {
                    coeff = DCT_dc_y_past + (diff << 3);
                    DCT_dc_y_past = coeff;
                }
            }
            else
            {

                /*
	 * Get the chrominance bits.  This code has been hand optimized to
	 * as described above
	 */
                unsigned int next16bits, index, flushed;

                next16bits = show_bits16(m);
                index = next16bits >> (16 - 8);
                size = dct_dc_size_chrominance[index].value;
                flushed = dct_dc_size_chrominance[index].num_bits;
                next16bits &= bitMask[16 + flushed];

                if (size != 0)
                {
                    flushed += size;
                    diff = next16bits >> (16 - flushed);
                    if (!(diff & bitTest[32 - size]))
                    {
                        diff = rBitMask[size] | (diff + 1);
                    }
                }
                else
                {
                    diff = 0;
                }
                flush_bits(m, flushed);

                if (n == 4)
                {
                    coeff = diff << 3;
                    if (m->mblock.mb_address - m->mblock.past_intra_addr > 1)
                        coeff += 1024;
                    else
                        coeff += DCT_dc_cr_past;
                    DCT_dc_cr_past = coeff;
                }
                else
                {
                    coeff = diff << 3;
                    if (m->mblock.mb_address - m->mblock.past_intra_addr > 1)
                        coeff += 1024;
                    else
                        coeff += DCT_dc_cb_past;
                    DCT_dc_cb_past = coeff;
                }
            }

            *reconptr = coeff;
            i = 0;
            pos = 0;
            coeffCount = (coeff != 0);

            if (m->picture.code_type != 4)
            {

                qscale = m->slice.quant_scale;
                iqmatrixptr = m->intra_quant_matrix[0];

                while (1)
                {

                    DECODE_DCT_COEFF(m, dct_coeff_next, run, level);

                    if (run == END_OF_BLOCK)
                        break;

                    i = i + run + 1;
                    pos = zigzag_direct[i];
                    coeff = (level * qscale * ((int)iqmatrixptr[pos])) >> 3;
                    if (level < 0)
                    {
                        coeff += (coeff & 1);
                    }
                    else
                    {
                        coeff -= (coeff & 1);
                    }

                    reconptr[pos] = coeff;
                    if (coeff)
                    {
                        coeffCount++;
                    }
                }

                flush_bits(m, 2);

                goto end;
            }
        }

        else
        {

            niqmatrixptr = m->non_intra_quant_matrix[0];
            qscale = m->slice.quant_scale;

            DECODE_DCT_COEFF(m, dct_coeff_first, run, level);
            i = run;

            pos = zigzag_direct[i];
            if (level < 0)
            {
                coeff = (((level << 1) - 1) * qscale * ((int)(niqmatrixptr[pos]))) >> 4;
                coeff += (coeff & 1);
            }
            else
            {
                coeff = (((level << 1) + 1) * qscale * ((int)(*(niqmatrixptr + pos)))) >> 4;
                coeff -= (coeff & 1);
            }
            reconptr[pos] = coeff;
            if (coeff)
            {
                coeffCount = 1;
            }

            if (m->picture.code_type != 4)
            {

                while (1)
                {

                    DECODE_DCT_COEFF(m, dct_coeff_next, run, level);
                    if (run == END_OF_BLOCK)
                        break;

                    i = i + run + 1;
                    pos = zigzag_direct[i];
                    if (level < 0)
                    {
                        coeff = (((level << 1) - 1) * qscale * ((int)(niqmatrixptr[pos]))) >> 4;
                        coeff += (coeff & 1);
                    }
                    else
                    {
                        coeff = (((level << 1) + 1) * qscale * ((int)(*(niqmatrixptr + pos)))) >> 4;
                        coeff -= (coeff & 1);
                    }
                    reconptr[pos] = coeff;
                    if (coeff)
                    {
                        coeffCount++;
                    }
                }

                flush_bits(m, 2);

                goto end;
            }
        }

    end:

        if (coeffCount == 1)
            j_rev_dct_sparse(reconptr, pos);
        else
            j_rev_dct(reconptr);
    }
}

#undef DCT_recon
#undef DCT_dc_y_past
#undef DCT_dc_cr_past
#undef DCT_dc_cb_past

/*
 *--------------------------------------------------------------
 *
 * ParseMacroBlock --
 *
 *      Parseoff macroblock. Reconstructs DCT values. Applies
 *      inverse DCT, reconstructs motion vectors, calculates and
 *      set pixel values for macroblock in current pict image
 *      structure.
 *
 * Results:
 *      Here's where everything really happens. Welcome to the
 *      heart of darkness.
 *
 * Side effects:
 *      Bit stream irreversibly parsed off.
 *
 *--------------------------------------------------------------
 */

static int parse_macro_block(MPEG *m)
{
    int addr_incr;
    unsigned int data;
    int mask, i, recon_right_for, recon_down_for, recon_right_back,
        recon_down_back;
    int zero_block_flag;
    int mb_quant, mb_motion_forw, mb_motion_back, mb_pattern;

    /*
   * Parse off macroblock address increment and add to macroblock address.
   */
    do
    {
        data = show_bits11(m);
        addr_incr = mb_addr_inc[data].value;
        flush_bits(m, mb_addr_inc[data].num_bits);
        if (addr_incr == MB_ESCAPE)
        {
            m->mblock.mb_address += 33;
            addr_incr = MB_STUFFING;
        }
    } while (addr_incr == MB_STUFFING);
    m->mblock.mb_address += addr_incr;

    if (m->mblock.mb_address > ((int)m->mb_height * (int)m->mb_width - 1))
        return 0;

    /*
   * If macroblocks have been skipped, process skipped macroblocks.
   */
    if (m->mblock.mb_address - m->mblock.past_mb_addr > 1)
    {
        if (m->picture.code_type == P_TYPE)
            ProcessSkippedPFrameMBlocks(m);
        else if (m->picture.code_type == B_TYPE)
            ProcessSkippedBFrameMBlocks(m);
    }
    /* Set past macroblock address to current macroblock address. */
    m->mblock.past_mb_addr = m->mblock.mb_address;

    /* Based on picture type decode macroblock type. */
    switch (m->picture.code_type)
    {
    default:
    case I_TYPE: /* Intra coded */
    {
        static int quantTbl[4] = { -1, 1, 0, 0 };
        unsigned int index = show_bits2(m);

        mb_motion_forw = 0;
        mb_motion_back = 0;
        mb_pattern = 0;
        m->mblock.mb_intra = 1;
        mb_quant = quantTbl[index];
        if (index)
            flush_bits(m, 1 + mb_quant);
    }
    break;

    case P_TYPE: /* Predictive coded */
    {
        unsigned int index = show_bits6(m);

        mb_quant = mb_type_P[index].mb_quant;
        mb_motion_forw = mb_type_P[index].mb_motion_forward;
        mb_motion_back = mb_type_P[index].mb_motion_backward;
        mb_pattern = mb_type_P[index].mb_pattern;
        m->mblock.mb_intra = mb_type_P[index].mb_intra;
        flush_bits(m, mb_type_P[index].num_bits);
    }
    break;

    case B_TYPE: /* Bidirectionally coded */
    {
        unsigned int index = show_bits6(m);

        mb_quant = mb_type_B[index].mb_quant;
        mb_motion_forw = mb_type_B[index].mb_motion_forward;
        mb_motion_back = mb_type_B[index].mb_motion_backward;
        mb_pattern = mb_type_B[index].mb_pattern;
        m->mblock.mb_intra = mb_type_B[index].mb_intra;
        flush_bits(m, mb_type_B[index].num_bits);
    }
    break;
    }

    /* If quantization flag set, parse off new quantization scale. */
    if (mb_quant)
        m->slice.quant_scale = get_bits5(m);

    /* If forward motion vectors exist... */
    if (mb_motion_forw)
    {

        /* Parse off and decode horizontal forward motion vector. */
        data = show_bits11(m);
        m->mblock.motion_h_forw_code = motion_vectors[data].code;
        flush_bits(m, motion_vectors[data].num_bits);

        /* If horiz. forward r data exists, parse off. */
        if ((m->picture.forw_f != 1) && (m->mblock.motion_h_forw_code != 0))
            m->mblock.motion_h_forw_r = get_bitsn(m, m->picture.forw_r_size);

        /* Parse off and decode vertical forward motion vector. */
        data = show_bits11(m);
        m->mblock.motion_v_forw_code = motion_vectors[data].code;
        flush_bits(m, motion_vectors[data].num_bits);

        /* If vert. forw. r data exists, parse off. */
        if ((m->picture.forw_f != 1) && (m->mblock.motion_v_forw_code != 0))
            m->mblock.motion_v_forw_r = get_bitsn(m, m->picture.forw_r_size);
    }
    /* If back motion vectors exist... */
    if (mb_motion_back)
    {

        /* Parse off and decode horiz. back motion vector. */
        data = show_bits11(m);
        m->mblock.motion_h_back_code = motion_vectors[data].code;
        flush_bits(m, motion_vectors[data].num_bits);

        /* If horiz. back r data exists, parse off. */
        if ((m->picture.back_f != 1) && (m->mblock.motion_h_back_code != 0))
            m->mblock.motion_h_back_r = get_bitsn(m, m->picture.back_r_size);

        /* Parse off and decode vert. back motion vector. */
        data = show_bits11(m);
        m->mblock.motion_v_back_code = motion_vectors[data].code;
        flush_bits(m, motion_vectors[data].num_bits);

        /* If vert. back r data exists, parse off. */
        if ((m->picture.back_f != 1) && (m->mblock.motion_v_back_code != 0))
            m->mblock.motion_v_back_r = get_bitsn(m, m->picture.back_r_size);
    }

    /* If mblock pattern flag set, parse and decode CBP (code block pattern). */
    if (mb_pattern)
    {
        data = show_bits9(m);
        m->mblock.cbp = coded_block_pattern[data].cbp;
        flush_bits(m, coded_block_pattern[data].num_bits);
    }
    /* Otherwise, set CBP to zero. */
    else
        m->mblock.cbp = 0;

    /* Reconstruct motion vectors depending on picture type. */
    if (m->picture.code_type == P_TYPE)
    {

        /*
     * If no forw motion vectors, reset previous and current vectors to 0.
     */
        if (!mb_motion_forw)
        {
            recon_right_for = 0;
            recon_down_for = 0;
            m->mblock.recon_right_for_prev = 0;
            m->mblock.recon_down_for_prev = 0;
        }
        /*
     * Otherwise, compute new forw motion vectors. Reset previous vectors to
     * current vectors.
     */
        else
        {
            ComputeForwVector(m, &recon_right_for, &recon_down_for);
        }
    }
    if (m->picture.code_type == B_TYPE)
    {

        /* Reset prev. and current vectors to zero if mblock is intracoded. */

        if (m->mblock.mb_intra)
        {
            m->mblock.recon_right_for_prev = 0;
            m->mblock.recon_down_for_prev = 0;
            m->mblock.recon_right_back_prev = 0;
            m->mblock.recon_down_back_prev = 0;
        }
        else
        {

            /* If no forw vectors, current vectors equal prev. vectors. */

            if (!mb_motion_forw)
            {
                recon_right_for = m->mblock.recon_right_for_prev;
                recon_down_for = m->mblock.recon_down_for_prev;
            }
            /*
       * Otherwise compute forw. vectors. Reset prev vectors to new values.
       */

            else
            {
                ComputeForwVector(m, &recon_right_for, &recon_down_for);
            }

            /* If no back vectors, set back vectors to prev back vectors. */

            if (!mb_motion_back)
            {
                recon_right_back = m->mblock.recon_right_back_prev;
                recon_down_back = m->mblock.recon_down_back_prev;
            }
            /* Otherwise compute new vectors and reset prev. back vectors. */
            else
            {
                ComputeBackVector(m, &recon_right_back, &recon_down_back);
            }

            /*
       * Store vector existance flags in structure for possible skipped
       * macroblocks to follow.
       */

            m->mblock.bpict_past_forw = mb_motion_forw;
            m->mblock.bpict_past_back = mb_motion_back;
        }
    }

    for (mask = 32, i = 0; i < 6; mask >>= 1, i++)
    {

        /* If block exists... */
        if ((m->mblock.mb_intra) || (m->mblock.cbp & mask))
        {
            zero_block_flag = 0;
            ParseReconBlock(m, i);
        }
        else
        {
            zero_block_flag = 1;
        }

        /* If macroblock is intra coded... */
        if (m->mblock.mb_intra)
        {
            ReconIMBlock(m, i);
        }
        else if (mb_motion_forw && mb_motion_back)
        {
            ReconBiMBlock(m, i, recon_right_for, recon_down_for,
                          recon_right_back, recon_down_back, zero_block_flag);
        }
        else if (mb_motion_forw || (m->picture.code_type == P_TYPE))
        {
            ReconPMBlock(m, i, recon_right_for, recon_down_for,
                         zero_block_flag);
        }
        else if (mb_motion_back)
        {
            ReconBMBlock(m, i, recon_right_back, recon_down_back,
                         zero_block_flag);
        }
    }

    /* If D Type picture, flush marker bit. */
    if (m->picture.code_type == 4)
        flush_bits(m, 1);

    /* If macroblock was intracoded, set macroblock past intra address. */
    if (m->mblock.mb_intra)
        m->mblock.past_intra_addr = m->mblock.mb_address;

    return 1;
}

static int init_mpeg(MPEG *m)
{
    static unsigned char default_intra_matrix[64] = {
        8, 16, 19, 22, 26, 27, 29, 34,
        16, 16, 22, 24, 27, 29, 34, 37,
        19, 22, 26, 27, 29, 34, 34, 38,
        22, 22, 26, 27, 29, 34, 37, 40,
        22, 26, 27, 29, 32, 35, 40, 48,
        26, 27, 29, 32, 35, 40, 48, 58,
        26, 27, 29, 34, 38, 46, 56, 69,
        27, 29, 35, 38, 46, 56, 69, 83
    };
    int i, j;

    /* Copy default intra matrix. */
    for (i = 0; i < 8; i++)
        for (j = 0; j < 8; j++)
            m->intra_quant_matrix[j][i] = default_intra_matrix[i * 8 + j];

    /* Initialize non intra quantization matrix. */
    for (i = 0; i < 8; i++)
        for (j = 0; j < 8; j++)
            m->non_intra_quant_matrix[j][i] = 16;

    /* Initialize bitstream i/o fields. */
    m->bit_offset = 0;
    m->buf_length = 0;
    m->buffer = m->buf_start;

    /* Find start code, make sure it is a sequence start code, then
   * parse the header and allocate image space.
   */
    next_start_code(m);
    return ((show_bits32(m) == SEQ_START_CODE) && parse_seq_header(m));
}

/*--------------------------------------------------------------
 *
 * MPEGOpen -- Open a file, return an MPEG stream object.
 * Modified to take an open FILE*.
 */

MPEG *MPEGOpen(/*char *filename*/ FILE *fp, int bufsize)
{
    static int first = 1;
    int i;
    MPEG *m;
    /*FILE *fp = fopen(filename,"r");*/

    MPEGerrno = 0;

    /* One-time only initializations. */
    if (first)
    {
        /* Initialize crop table. Is this really worthwhile? ... */
        for (i = (-MAX_NEG_CROP); i < NUM_CROP_ENTRIES - MAX_NEG_CROP; i++)
            cropTbl[i + MAX_NEG_CROP] = MIN(255, MAX(0, i));

        /* Initialize decoding tables. */
        MPEGInitTables(mb_addr_inc, mb_type_P, mb_type_B, motion_vectors);

        first = 0;
    }

    if (fp == NULL)
    {
        MPEGerrno = MPEG_NOFILE;
        return NULL;
    }

    if ((m = (MPEG *)malloc(sizeof(MPEG))) == NULL)
    {
        MPEGerrno = MPEG_NOMEM;
        /*fclose(fp);*/
        return NULL;
    }

    m->fp = fp;

    /* Make buffer length multiple of 4. */
    if (bufsize <= 0)
        bufsize = DEFAULT_BUFSIZE;
    bufsize = (bufsize + 3) >> 2;

    /* Create input buffer. Ought to look at mmap'ping the file... */
    /* Should this really be bufsize * 4? Looks like a mistake... */
    if ((m->buf_start = (unsigned int *)malloc(bufsize * 4)) == NULL)
    {
        MPEGerrno = MPEG_NOMEM;
        free(m);
        /*fclose(fp);*/
        return NULL;
    }

    /*
   * Set max_buf_length to one less than actual length to deal with messy
   * data without proper seq. end codes.
   */
    m->max_buf_length = bufsize - 1;

    /* Initialize pointers to image spaces. */
    m->current = m->past = m->future = NULL;
    for (i = 0; i < RING_BUF_SIZE; i++)
        m->ring[i] = NULL;

    if (!init_mpeg(m))
    {
        MPEGerrno = MPEG_NOTMPEG;
        free(m->buf_start);
        free(m);
        /*fclose(fp);*/
        return NULL;
    }

    return m;
}

void MPEGClose(MPEG *m)
{
    int i;

    for (i = 0; i < RING_BUF_SIZE; i++)
        if (m->ring[i] != NULL)
            free_image(m->ring[i]);

    free(m->buf_start);
    /*fclose(m->fp);*/
    free(m);
}

/*
 *  Returns 0 if no frame available.
 */

int MPEGAdvanceFrame(MPEG *m)
{
    unsigned int data;

#ifdef DEBUG
    extern long ftell();
    fprintf(stderr, "MPEGAdvanceFrame: offset %6ld: found ", ftell(m->fp));
#endif

    /* Process according to start code. */
    for (;;)
    {

        next_start_code(m);
        switch (data = show_bits32(m))
        {

        case SEQ_END_CODE: /* Return last frame if available. */
#ifdef DEBUG
            fprintf(stderr, "SEQ_END_CODE\n");
#endif
            if (m->future != NULL)
            {
                m->current = m->future;
                m->future = NULL;
                return 1;
            }
            return 0;
            break;

        case SEQ_START_CODE: /* Parse sequence header. */
#ifdef DEBUG
            fprintf(stderr, "SEQ_START_CODE\n");
#endif
            if (!parse_seq_header(m))
                goto error;
            break;

        case GOP_START_CODE: /* Parse Group of Pictures header. */
#ifdef DEBUG
            fprintf(stderr, "GOP_START_CODE\n");
#endif
            if (!parse_GOP(m))
                goto error;
        /*FALLTHROUGH*/

        case PICTURE_START_CODE: /* Parse picture header and first slice header. */
#ifdef DEBUG
            fprintf(stderr, "PICTURE_START_CODE\n");
#endif
            if (!parse_picture(m))
                goto error;
            if (!parse_slice(m))
                goto error;
            break;

        default: /* Check for slice start code. */
#ifdef DEBUG
            fprintf(stderr, "something else (%08x)\n", data);
#endif
            if ((data >= SLICE_MIN_START_CODE) && (data <= SLICE_MAX_START_CODE))
            { /* Slice start code. Parse slice header. */
                if (!parse_slice(m))
                    goto error;
            }
            else
                goto error;
            break;
        }

        /* Parse macroblocks until the next start code is seen. */
        for (data = show_bits32(m); (data & 0xfffffe00); data = show_bits32(m))
            if (!parse_macro_block(m))
                goto error;

        next_start_code(m);
        data = show_bits32(m);

        /* If start code is outside range of slice start codes, frame is done. */
        if ((data < SLICE_MIN_START_CODE) || (data > SLICE_MAX_START_CODE))
            break;
    }

    /* Update past and future references if needed. */
    if ((m->picture.code_type == I_TYPE) || (m->picture.code_type == P_TYPE))
    {
        if (m->future == NULL)
        {
            m->future = m->current;
            m->future->locked |= FUTURE_LOCK;
        }
        else
        {
            if (m->past != NULL)
            {
                m->past->locked &= ~PAST_LOCK;
            }
            m->past = m->future;
            m->past->locked &= ~FUTURE_LOCK;
            m->past->locked |= PAST_LOCK;
            m->future = m->current;
            m->future->locked |= FUTURE_LOCK;
            m->current = m->past;
        }
    }

    return 1;

error:
    fprintf(stderr, "MPG_ERROR decoding MPEG stream.\n");
    return 0;
}

#if 0
/* Not used */
int MPEGRewind(MPEG *m)
{
  rewind(m->fp);
  return init_mpeg(m);
}
#endif

/*
 * We'll define the "ConvertColor" macro here to do fixed point arithmetic
 * that'll convert from YCrCb to RGB using:
 *	R = L + 1.40200*Cr;
 *	G = L - 0.34414*Cb - 0.71414*Cr
 *	B = L + 1.77200*Cb;
 *
 */

#define CLAMP(ll, x, ul) (((x) < (ll)) ? (ll) : (((x) > (ul)) ? (ul) : (x)))

/*
 *--------------------------------------------------------------
 *
 * ColorDitherImage --
 *
 *	Converts image into 24 bit color.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

static void
ColorDitherImage(unsigned char *lum,
                 unsigned char *cr,
                 unsigned char *cb,
                 unsigned char *out,
                 int rows, int cols, int alpha)
{
    unsigned char *row1, *row2;
    unsigned char *lum2;
    int x, y;
    int row_size;

    if (alpha)
        row_size = 4 * cols;
    else
        row_size = 3 * cols;

    row1 = out;
    row2 = row1 + row_size;
    lum2 = lum + cols;

    for (y = 0; y < rows; y += 2)
    {
        for (x = 0; x < cols; x += 2)
        {
            float fl, fcr, fcb;
            float fr, fg, fb;

            fcr = ((float)*cr++) - 128.0f;
            fcb = ((float)*cb++) - 128.0f;

            fl = (float)*lum++;
            fr = fl + (1.40200f * fcb);
            fg = fl - (0.71414f * fcb) - (0.34414f * fcr);
            fb = fl + (1.77200f * fcr);
            *row1++ = (unsigned char)CLAMP(0., fr, 255.);
            *row1++ = (unsigned char)CLAMP(0., fg, 255.);
            *row1++ = (unsigned char)CLAMP(0., fb, 255.);
            if (alpha)
                *row1++ = 255;

            fl = (float)*lum++;
            fr = fl + (1.40200f * fcb);
            fg = fl - (0.71414f * fcb) - (0.34414f * fcr);
            fb = fl + (1.77200f * fcr);
            *row1++ = (unsigned char)CLAMP(0., fr, 255.);
            *row1++ = (unsigned char)CLAMP(0., fg, 255.);
            *row1++ = (unsigned char)CLAMP(0., fb, 255.);
            if (alpha)
                *row1++ = 255;

            /*
	 * Now, do second row.
	 */
            fl = (float)*lum2++;
            fr = fl + (1.40200f * fcb);
            fg = fl - (0.71414f * fcb) - (0.34414f * fcr);
            fb = fl + (1.77200f * fcr);
            *row2++ = (unsigned char)CLAMP(0., fr, 255.);
            *row2++ = (unsigned char)CLAMP(0., fg, 255.);
            *row2++ = (unsigned char)CLAMP(0., fb, 255.);
            if (alpha)
                *row2++ = 255;

            fl = (float)*lum2++;
            fr = fl + (1.40200f * fcb);
            fg = fl - (0.71414f * fcb) - (0.34414f * fcr);
            fb = fl + (1.77200f * fcr);
            *row2++ = (unsigned char)CLAMP(0., fr, 255.);
            *row2++ = (unsigned char)CLAMP(0., fg, 255.);
            *row2++ = (unsigned char)CLAMP(0., fb, 255.);
            if (alpha)
                *row2++ = 255;
        }
        lum += cols;
        lum2 += cols;
        row1 += row_size;
        row2 += row_size;
    }
}

int MPEGConvertImage(MPEG *m, int to, unsigned char *pixels)
{
    switch (to)
    {
    case MPEG_CONV_RGB:
    case MPEG_CONV_RGBA:
        ColorDitherImage(m->current->luminance,
                         m->current->Cr,
                         m->current->Cb,
                         pixels,
                         m->mb_height * 16, m->mb_width * 16,
                         (to == MPEG_CONV_RGBA));
        break;
    default:
        return 0;
    }
    return 1;
}
