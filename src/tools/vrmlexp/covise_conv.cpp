/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTypes.h"

#if defined(CRAY) && !defined(_CRAYT3E)

void hex_out_3(int out)
{
    unsigned int i, j, sh;

    /*    printf("%: ", i); */
    sh = 1 << 31;
    for (j = 0; j<32; j++, sh>> 1)
    {
        printf("%1d", (sh & out) ? 1 : 0);
        printf("sh : %d\n", sh);
    }
    printf("\n");
    ;
}

void hex_out_6(int out)
{
    unsigned int i, j, sh;
    char feld[66];

    sh = 1;
    for (j = 0; j < 64; j++)
    {
        feld[63 - j] = (sh & out) ? '1' : '0';
        sh <<= 1;
    }
    feld[64] = '\n';
    feld[65] = '\0';
    printf("%s", feld);
}

void conv_array_float_c8i4(int *input, int *output, int number, int start)
{
    int i;
    int count = start;
    int index, shift_index;

    for (i = 0; i < number; i++)
    {
        count += 4;
        index = (count - 1) >> 3;
        shift_index = (8 - (1 + ((count - 1) & (8 - 1)))) << 3;
        output[index] = (output[index] & (0xffffffffffffffff ^ (_mask(96) << shift_index))) | (((0xffffffffffffffff ^ _mask(_leadz(input[i]) & _mask(64 - 6))) & (((input[i] >> 32) & (1 << 31))
                                                                                                                                                                  | (((((input[i] >> 48) & 32767) - 16258 << 23)
                                                                                                                                                                      & (255 << 23))
                                                                                                                                                                     | ((input[i] >> 24) & _mask(105)))))
                                                                                               << shift_index);
    }
}

void conv_single_float_c8i4(int input, int *output)
{
    *output = (*output & _mask(96)) | (((0xffffffffffffffff ^ _mask(_leadz(input) & _mask(64 - 6))) & (((input >> 32) & (1 << 31))
                                                                                                       | (((((input >> 48) & 32767) - 16258 << 23)
                                                                                                           & (255 << 23))
                                                                                                          | ((input >> 24) & _mask(105)))))
                                       << 32);
}

void conv_single_float_c8i4_second(int input, int *output)
{
    *output = (*output & _mask(32)) | (((0xffffffffffffffff ^ _mask(_leadz(input) & _mask(64 - 6))) & (((input >> 32) & (1 << 31))
                                                                                                       | (((((input >> 48) & 32767) - 16258 << 23)
                                                                                                           & (255 << 23))
                                                                                                          | ((input >> 24) & _mask(105))))));
}

void conv_array_float_c8i8(int *input, int *output, int number)
{
    int i;
    for (i = 0; i < number; i++)
    {
        output[i] = ((0xffffffffffffffff ^ _mask((_leadz(input[i])
                                                  & _mask(64 - 6))))
                     & ((input[i] & (1 << 63))
                        | (((((input[i]
                               >> 48) & 32767) - 15362
                             << 52) & (2047 << 52))
                           | ((input[i] << 5)
                              & (_mask(81) << 5)))));
    }
}

void conv_single_float_c8i8(int input, int *output)
{
    *output = ((0xffffffffffffffff ^ _mask((_leadz(input)
                                            & _mask(64 - 6))))
               & ((input & (1 << 63))
                  | (((((input
                         >> 48) & 32767) - 15362
                       << 52) & (2047 << 52))
                     | ((input << 5)
                        & (_mask(81) << 5)))));
}

void conv_array_float_i4c8(int *input, int *output, int number, int start)
{
    int i;
    int count = start;
    int index, shift_index;
    int read_val;
    for (i = 0; i < number; i++)
    {
        count += 4;
        index = (count - 1) >> 3;
        shift_index = (8 - (1 + ((count - 1) & (8 - 1)))) << 3;
        read_val = (input[index] >> shift_index) & _mask(96);

        output[i] = ((0xffffffffffffffff ^ _mask((_leadz(read_val) & _mask(64 - 6))))
                     & (((read_val << 32) & (1 << 63))
                        | (((((read_val >> 23) & 255) + 16258 << 48)
                            & (32767 << 48)) | ((1 << 47) | ((read_val << 24) & (_mask(105) << 24))))));
    }
}

void conv_single_float_i4c8(int input, int *output)
{
    int read_val;
    read_val = (input >> 32) & _mask(96);

    *output = ((0xffffffffffffffff ^ _mask((_leadz(read_val) & _mask(64 - 6))))
               & (((read_val << 32) & (1 << 63))
                  | (((((read_val >> 23) & 255) + 16258 << 48)
                      & (32767 << 48)) | ((1 << 47) | ((read_val << 24) & (_mask(105) << 24))))));
}

void conv_single_float_i4c8_second(int input, int *output)
{
    int read_val;
    read_val = input & _mask(96);

    *output = ((0xffffffffffffffff ^ _mask((_leadz(read_val) & _mask(64 - 6))))
               & (((read_val << 32) & (1 << 63))
                  | (((((read_val >> 23) & 255) + 16258 << 48)
                      & (32767 << 48)) | ((1 << 47) | ((read_val << 24) & (_mask(105) << 24))))));
}

void conv_array_float_i8c8(int *input, int *output, int number)
{
    int i;
    for (i = 0; i < number; i++)
    {
        output[i] = ((0xffffffffffffffff ^ _mask((_leadz(input[i]) & _mask(64 - 6))))
                     & ((input[i] & (1 << 63)) | (((((input[i] >> 52) & 2047) + 15362 << 48)
                                                   & (32767 << 48))
                                                  | ((1 << 47) | ((input[i] >> 5) & _mask(81))))));
    }
}

void conv_single_float_i8c8(int input, int *output)
{
    *output = ((0xffffffffffffffff ^ _mask((_leadz(input) & _mask(64 - 6))))
               & ((input & (1 << 63)) | (((((input >> 52) & 2047) + 15362 << 48)
                                          & (32767 << 48))
                                         | ((1 << 47) | ((input >> 5) & _mask(81))))));
}

void conv_array_int_i4c8(int *input, int *output, int number, int start)
{
    int i;
    int count = start;
    int index, shift_index;
    int read_val, bm;

    for (i = 0; i < number; i++)
    {
        count += 4;
        index = (count - 1) >> 3;
        shift_index = (8 - (1 + ((count - 1) & (8 - 1)))) << 3;
        read_val = (input[index] >> shift_index) & _mask(96);
        bm = 0x80000000 & read_val;
        output[i] = read_val | (bm ? _mask(32) : 0x0);
    }
}

void conv_single_int_i4c8(int input, int *output)
{
    int read_val, bm;

    read_val = (input >> 32) & _mask(96);
    bm = 0x80000000 & read_val;
    *output = read_val | (bm ? _mask(32) : 0x0);
}

void conv_single_int_i4c8_second(int input, int *output)
{
    int read_val, bm;

    read_val = input & _mask(96);
    bm = 0x80000000 & read_val;
    *output = read_val | (bm ? _mask(32) : 0x0);
}

void conv_array_int_c8i4(int *input, int *output, int number, int start)
{
    int i;
    int count = start;
    int index, shift_index;
    for (i = 0; i < number; i++)
    {
        count += 4;
        index = (count - 1) >> 3;
        shift_index = (8 - (1 + ((count - 1) & (8 - 1)))) << 3;
        output[index] = (output[index] & (0xffffffffffffffff ^ (_mask(96) << shift_index))) | ((input[i] << shift_index) & (_mask(96) << shift_index));
    }
}

void conv_single_int_c8i4(int input, int *output)
{
    *output = (*output & _mask(96)) | (input << 32);
}

void conv_single_int_c8i4_second(int input, int *output)
{
    *output = (*output & _mask(32)) | input;
}

void conv_array_int_i2c8(int *input, int *output, int number, int start)
{
    int i;
    int count = start;
    int index, shift_index;
    int read_val, bm;

    for (i = 0; i < number; i++)
    {
        count += 2;
        index = (count - 1) >> 3;
        shift_index = (8 - (1 + ((count - 1) & (8 - 1)))) << 3;
        read_val = (input[index] >> shift_index) & _mask(96);
        bm = 0x80000000 & read_val;
        output[i] = read_val | (bm ? _mask(32) : 0x0);
    }
}

void conv_single_int_i2c8(int input, int *output)
{
    int read_val, bm;

    read_val = (input >> 48) & _mask(112);
    bm = 0x8000 & read_val;
    *output = read_val | (bm ? _mask(48) : 0x0);
}

void conv_array_int_c8i2(int *input, int *output, int number, int start)
{
    int i;
    int count = start;
    int index, shift_index;
    for (i = 0; i < number; i++)
    {
        count += 4;
        index = (count - 1) >> 3;
        shift_index = (8 - (1 + ((count - 1) & (8 - 1)))) << 3;
        output[index] = (output[index] & (0xffffffffffffffff ^ (_mask(96) << shift_index))) | ((input[i] << shift_index) & (_mask(96) << shift_index));
    }
}

void conv_single_int_c8i2(int input, int *output)
{
    *output = (*output & _mask(112)) | (input << 48);
}
#endif
