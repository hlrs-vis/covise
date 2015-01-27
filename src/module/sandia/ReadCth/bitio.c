/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef lint
static char *sccsid1[] = {
    "@(#)SCCSID: SCCS/s.bitio.c 1.1",
    "@(#)SCCSID: Version Created: 11/18/92 20:51:42"
};
#endif
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * NAME
 *      bitio - bit reading/writing routines
 *
 * FILE
 *      bitio.c
 *
 * SECURITY CLASSIFICATION
 *      Unclassified
 *
 * DESCRIPTION
 *      This utility file contains all of the routines needed to impement
 *	bit oriented routines under either ANSI or K&R C.
 *
 * DIAGNOSTICS
 *
 * LIMITATIONS
 *      None
 *
 * FILES
 *      None
 *
 *
 * SEE ALSO
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/************************** Start of BITIO.C *************************
 *
 * This utility file contains all of the routines needed to impement
 * bit oriented routines under either ANSI or K&R C. 
 *
 */
#include <stdlib.h>
#include "bitio.h"
#include "error_info.h"
#include "local_defs.h"
#define error_info(a, b, c, d) fprintf(stderr, "%d %s %s", a, c, d)

BIT_FILE *
    AllocOutputiBitFile(file)
        FILE *file;
{
    BIT_FILE *bit_file;

    bit_file = (BIT_FILE *)UCALLOC(BIT_FILE, 1);
    if (bit_file == NULL)
        return (bit_file);
    bit_file->file = file;
    bit_file->current_byte = 0;
    bit_file->mask = 0x80;

    return (bit_file);
}

BIT_FILE *
    AllocInputBitFile(file)
        FILE *file;
{
    BIT_FILE *bit_file;

    bit_file = (BIT_FILE *)UCALLOC(BIT_FILE, 1);
    if (bit_file == NULL)
        return (bit_file);
    bit_file->file = file;
    bit_file->current_byte = 0;
    bit_file->mask = 0x80;

    return (bit_file);
}

void
    FreeOutputBitFile(bit_file)
        BIT_FILE *bit_file;
{
    char *routine = "FreeOutputBitFile";

    if (bit_file->mask != 0x80)
        if (putc(bit_file->current_byte, bit_file->file) != bit_file->current_byte)
            error_info(99, FATAL, routine, "Failure to write out last bytes properly");
    UFREE(bit_file);
}

void
    FreeInputBitFile(bit_file)
        BIT_FILE *bit_file;
{
    UFREE(bit_file);
}

void
    OutputBit(bit_file, bit)
        BIT_FILE *bit_file;
int bit;
{
    char *routine = "OutputBit";

    if (bit)
        bit_file->current_byte |= bit_file->mask;
    bit_file->mask >>= 1;
    if (bit_file->mask == 0)
    {
        if (putc(bit_file->current_byte, bit_file->file) != bit_file->current_byte)
            error_info(99, FATAL, routine, "Failure to write out bits properly");
        bit_file->current_byte = 0;
        bit_file->mask = 0x80;
    }
}

void
    OutputBits(bit_file, code, count)
        BIT_FILE *bit_file;
unsigned long code;
int count;
{
    unsigned long mask;
    char *routine = "OutputBits";

    mask = 1L << (count - 1);
    while (mask != 0)
    {
        if (mask & code)
            bit_file->current_byte |= bit_file->mask;
        bit_file->mask >>= 1;
        if (bit_file->mask == 0)
        {
            if (putc(bit_file->current_byte, bit_file->file) != bit_file->current_byte)
                error_info(99, FATAL, routine, "Failure to write out bits properly");
            bit_file->current_byte = 0;
            bit_file->mask = 0x80;
        }
        mask >>= 1;
    }
}

int
    InputBit(bit_file)
        BIT_FILE *bit_file;
{
    int value;
    char *routine = "InputBit";

    if (bit_file->mask == 0x80)
    {
        bit_file->current_byte = getc(bit_file->file);
        if (bit_file->current_byte == EOF)
            error_info(99, FATAL, routine, "EOF encountered while attempting to read bit");
    }
    value = bit_file->current_byte & bit_file->mask;
    bit_file->mask >>= 1;
    if (bit_file->mask == 0)
        bit_file->mask = 0x80;
    return (value ? 1 : 0);
}

unsigned long
    InputBits(bit_file, bit_count)
        BIT_FILE *bit_file;
int bit_count;
{
    unsigned long mask;
    unsigned long return_value;
    char *routine = "InputBits";

    mask = 1L << (bit_count - 1);
    return_value = 0;
    while (mask != 0)
    {
        if (bit_file->mask == 0x80)
        {
            bit_file->current_byte = getc(bit_file->file);
            if (bit_file->current_byte == EOF)
                error_info(99, FATAL, routine, "EOF encountered while attempting to read bits");
        }
        if (bit_file->current_byte & bit_file->mask)
            return_value |= mask;
        mask >>= 1;
        bit_file->mask >>= 1;
        if (bit_file->mask == 0)
            bit_file->mask = 0x80;
    }
    return (return_value);
}

void
    FilePrintBinary(file, code, bits)
        FILE *file;
unsigned int code;
int bits;
{
    unsigned int mask;

    mask = 1 << (bits - 1);
    while (mask != 0)
    {
        if (code & mask)
            fputc('1', file);
        else
            fputc('0', file);
        mask >>= 1;
    }
}
