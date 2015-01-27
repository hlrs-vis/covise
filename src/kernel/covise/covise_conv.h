/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COVISE_CONV_H_
#define _COVISE_CONV_H_

/******************************************************************************
 * Naming conventions:
 *    Each functions has the prefix "conv_" followed by the number of the
 *    desired conversions: only one item (single) or more than one (array).
 *    The next part determines, what kind of type is to be converted (int or
 *    float). The last part tells the direction and kind of convertion where
 *    'i' means IEEE and 'c' means CRAY. The numbers following each of the
 *    two characters give information about the size of the type.
 *****************************************************************************/

#if defined(CRAY) && !defined(_CRAYT3E)
extern "C" {
void conv_array_float_c8i4(int *input, int *output, int number, int start);
void conv_array_float_c8i8(int *input, int *output, int number);
void conv_array_float_i4c8(int *input, int *output, int number, int start);
void conv_array_float_i8c8(int *input, int *output, int number);
void conv_array_int_i4c8(int *input, int *output, int number, int start);
void conv_array_int_c8i4(int *input, int *output, int number, int start);
void conv_array_int_i2c8(int *input, int *output, int number, int start);
void conv_array_int_c8i2(int *input, int *output, int number, int start);
void conv_single_float_c8i4(int input, int *output);
void conv_single_float_c8i4_second(int input, int *output);
void conv_single_float_c8i8(int input, int *output);
void conv_single_float_i4c8(int input, int *output);
void conv_single_float_i4c8_second(int input, int *output);
void conv_single_float_i8c8(int input, int *output);
void conv_single_int_i4c8(int input, int *output);
void conv_single_int_i4c8_second(int input, int *output);
void conv_single_int_c8i4(int input, int *output);
void conv_single_int_c8i4_second(int input, int *output);
void conv_single_int_i2c8(int input, int *output);
void conv_single_int_i2c8_second(int input, int *output);
void conv_single_int_c8i2(int input, int *output);
void conv_single_int_c8i2_second(int input, int *output);
}
#endif

#ifdef _CRAYT3E
#include "covise_converter.h"

//extern "C" {
//   void conv_array_int_i4t8(int *input, int *output, int number, int start);
//   void conv_array_int_t8i4(int *input, int *output, int number, int start);
//}
#endif
#endif
