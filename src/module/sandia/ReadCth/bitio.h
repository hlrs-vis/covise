/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 *  bitio.h include file.
 */

/*  Include File SCCS header
 *  "@(#)SCCSID: bitio.h 1.1"
 *  "@(#)SCCSID: Version Created: 11/18/92 20:43:01"
 */

#ifndef _BITIO_H
#define _BITIO_H

#include <stdio.h>

typedef struct bit_file
{
    FILE *file;
    unsigned char mask;
    int current_byte;
} BIT_FILE;

#ifdef __STDC__

BIT_FILE *AllocInputBitFile(FILE *file);
BIT_FILE *AllocOutputBitFile(FILE *file);
void OutputBit(BIT_FILE *bit_file, int bit);
void OutputBits(BIT_FILE *bit_file,
                unsigned long code, int count);
int InputBit(BIT_FILE *bit_file);
unsigned long InputBits(BIT_FILE *bit_file, int bit_count);
void FreeInputBitFile(BIT_FILE *bit_file);
void FreeOutputBitFile(BIT_FILE *bit_file);
void FilePrintBinary(FILE *file, unsigned int code, int bits);

#else /* __STDC__ */

BIT_FILE *AllocInputBitFile();
BIT_FILE *AllocOutputBitFile();
void OutputBit();
void OutputBits();
int InputBit();
unsigned long InputBits();
void FreeInputBitFile();
void FreeOutputBitFile();
void FilePrintBinary();
#endif /* __STDC__ */
#endif /* _BITIO_H */
