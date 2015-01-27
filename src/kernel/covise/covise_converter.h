/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/********************************************************************************
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
 *******************************************************************************/

#ifndef _COVISE_CONVERTER_H_
#define _COVISE_CONVERTER_H_

#include "covise.h"

namespace covise
{

class COVISEEXPORT Converter
{
public:
    void short_to_exch(short input, char *output);
    void int_to_exch(int input, char *output);
    void long_to_exch(long input, char *output);
    void float_to_exch(float input, char *output);
    void double_to_exch(double input, char *output);

    void ushort_to_exch(unsigned short input, char *output);
    void uint_to_exch(unsigned int input, char *output);
    void ulong_to_exch(unsigned long input, char *output);

    void exch_to_short(char *input, short *output);
    void exch_to_int(char *input, int *output);
    void exch_to_long(char *input, long *output);
    void exch_to_float(char *input, float *output);
    void exch_to_double(char *input, double *output);

    void exch_to_ushort(char *input, unsigned short *output);
    void exch_to_uint(char *input, unsigned int *output);
    void exch_to_ulong(char *input, unsigned long *output);

    void short_array_to_exch(short *input, char *output, int n);
    void int_array_to_exch(int *input, char *output, int n);
    void long_array_to_exch(long *input, char *output, int n);
    void float_array_to_exch(float *input, char *output, int n);
    void double_array_to_exch(double *input, char *output, int n);

    void exch_to_short_array(char *input, short *output, int n);
    void exch_to_int_array(char *input, int *output, int n);
    void exch_to_long_array(char *input, long *output, int n);
    void exch_to_float_array(char *input, float *output, int n);
    void exch_to_double_array(char *input, double *output, int n);
};

extern Converter converter;
}
#endif
