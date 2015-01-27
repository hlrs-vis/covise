/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/******************************************************************************
 * Class Converter encapsulates the conversion requirements for the exchange
 * data format used between COVISE processes. We have four different kinds of
 * methods in this class:
 *
 *    type_to_exch :       conversion of one element of the mentionend type to
 *                         corresponding type used by the exchange format.
 *    exch_to_type :       the other way round
 *    type_array_to_exch : conversion of a number of elements of the type to
 *                         the exchange format.
 *    exch_to_type_array : again the other way
 *
 * The data given in exchange format will be handled as char*. For array
 * conversion the number of elements describes the number of type elements ---
 * ATTENTION: the user of this class is responsible for checking if there is
 * enough space or enough data available inside all arrays.
 *****************************************************************************/

#include "covise.h"
#ifdef _CRAYT3E // currently implemented and testet vor Cray T3E

#include "net/covise_msg.h"
#include "covise_converter.h"

using namespace covise;

Converter converter;

//------------------------------------------------------------------------------
// This is an implementation of the converter class for the Cray T3E, that is
// conversion from IEEE (32 bit) to IEEE (64 bit) and vice versa
//      Type:         Size (32 bit):        Size (64 bit):
//      short         2                     4
//      int           4                     8
//      long          4                     8
//      float         4                     4
//      double        8                     8
//------------------------------------------------------------------------------
void Converter::short_to_exch(short input, char *output)
{
    output[1] = (char)(input & 0x000000ff);
    input >>= 8;
    output[0] = (char)((input & 0x0000007f) | ((input >> 16) & 0x00000080));
}

void Converter::ushort_to_exch(unsigned short input, char *output)
{
    output[1] = (char)(input & 0x000000ff);
    input >>= 8;
    output[0] = (char)(input & 0x000000ff);
}

void Converter::int_to_exch(int input, char *output)
{
    output[3] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[2] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[1] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[0] = (char)((input & 0x000000000000007f) | ((input >> 32) & 0x0000000000000080));
}

void Converter::uint_to_exch(unsigned int input, char *output)
{
    output[3] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[2] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[1] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[0] = (char)(input & 0x00000000000000ff);
}

void Converter::long_to_exch(long input, char *output)
{
    output[3] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[2] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[1] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[0] = (char)((input & 0x000000000000007f) | ((input >> 32) & 0x0000000000000080));
}

void Converter::ulong_to_exch(unsigned long input, char *output)
{
    output[3] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[2] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[1] = (char)(input & 0x00000000000000ff);
    input >>= 8;
    output[0] = (char)(input & 0x00000000000000ff);
}

void Converter::float_to_exch(float input, char *output)
{
    memcpy(output, (char *)&input, SIZEOF_IEEE_FLOAT);
}

void Converter::double_to_exch(double input, char *output)
{
    memcpy(output, (char *)&input, SIZEOF_IEEE_DOUBLE);
}

/*------------------------------------------------------------------------------
 * Single data, exchange format to internal format
 *-----------------------------------------------------------------------------*/

void Converter::exch_to_short(char *input, short *output)
{
    short tmp = 0;

    tmp = ((input[0] & 0x7f) << 8) | input[1];
    if (input[0] & 0x80)
        tmp = tmp | 0xffff8000;
    *output = tmp;
}

void Converter::exch_to_ushort(char *input, unsigned short *output)
{
    unsigned short tmp = 0;

    tmp = (input[0] << 8) | input[1];
    *output = tmp;
}

void Converter::exch_to_int(char *input, int *output)
{
    int tmp = 0;

    tmp = ((input[0] & 0x7f) << 8) | input[1];
    tmp = (tmp << 8) | input[2];
    tmp = (tmp << 8) | input[3];
    if (input[0] & 0x80)
        tmp = tmp | 0xffffffff80000000;
    *output = tmp;
}

void Converter::exch_to_uint(char *input, unsigned int *output)
{
    unsigned int tmp = 0;

    tmp = (input[0] << 8) | input[1];
    tmp = (tmp << 8) | input[2];
    tmp = (tmp << 8) | input[3];
    *output = tmp;
}

void Converter::exch_to_long(char *input, long *output)
{
    long tmp = 0;

    tmp = ((input[0] & 0x7f) << 8) | input[1];
    tmp = (tmp << 8) | input[2];
    tmp = (tmp << 8) | input[3];
    if (input[0] & 0x80)
        tmp = tmp | 0xffffffff80000000;
    *output = tmp;
}

void Converter::exch_to_ulong(char *input, unsigned long *output)
{
    unsigned long tmp = 0;

    tmp = (input[0] << 8) | input[1];
    tmp = (tmp << 8) | input[2];
    tmp = (tmp << 8) | input[3];
    *output = tmp;
}

void Converter::exch_to_float(char *input, float *output)
{
    memcpy((char *)output, input, SIZEOF_IEEE_FLOAT);
}

void Converter::exch_to_double(char *input, double *output)
{
    memcpy((char *)output, input, SIZEOF_IEEE_DOUBLE);
}

void Converter::short_array_to_exch(short *input, char *output, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        short_to_exch(*input, output);
        input++;
        output += SIZEOF_IEEE_SHORT;
    }
}

void Converter::int_array_to_exch(int *input, char *output, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        int_to_exch(*input, output);
        input++;
        output += SIZEOF_IEEE_INT;
    }
}

void Converter::long_array_to_exch(long *input, char *output, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        long_to_exch(*input, output);
        input++;
        output += SIZEOF_IEEE_LONG;
    }
}

void Converter::float_array_to_exch(float *input, char *output, int n)
{
    memcpy((char *)output, input, SIZEOF_IEEE_FLOAT * n);
}

void Converter::double_array_to_exch(double *input, char *output, int n)
{
    memcpy((char *)output, input, SIZEOF_IEEE_DOUBLE * n);
}

void Converter::exch_to_short_array(char *input, short *output, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        exch_to_short(input, output);
        input += SIZEOF_IEEE_SHORT;
        output++;
    }
}

void Converter::exch_to_int_array(char *input, int *output, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        exch_to_int(input, output);
        input += SIZEOF_IEEE_INT;
        output++;
    }
}

void Converter::exch_to_long_array(char *input, long *output, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        exch_to_long(input, output);
        input += SIZEOF_IEEE_LONG;
        output++;
    }
}

void Converter::exch_to_float_array(char *input, float *output, int n)
{
    memcpy((char *)output, input, SIZEOF_IEEE_FLOAT * n);
}

void Converter::exch_to_double_array(char *input, double *output, int n)
{
    memcpy((char *)output, input, SIZEOF_IEEE_DOUBLE * n);
}
#endif
