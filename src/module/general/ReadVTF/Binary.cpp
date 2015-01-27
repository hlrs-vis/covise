/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2001    **
 ** Author:                                                              **
 **                            Karin Mueller                             **
 **                                        Vircinity                     **
 **                            Technologiezentrum                        **
 **                            70550 Stuttgart                           **
 ** Date:  01.10.01                                                      **
\**************************************************************************/

#include <ctype.h>
#include "Binary.h"

//==============================================================================
// check: if we find an unprintable char, it is binary
int Binary::isBinary(FILE *fi)
{
    int bin = 0;
    int k = 0;

    while (!feof(fi) && k < 1000 && !bin)
    {
        enum
        {
            CR = 13,
            NL = 10
        };
        char c = getc(fi);

        if (c > 0 && !isprint(c) && c != '\n' && c != NL && c != CR)
            bin = 1;

        k++;
    }

    rewind(fi);

    return bin;
}

//==============================================================================

void Binary::byteswap(int &val)
{
    val = ((val & 0xff000000) >> 24)
          | ((val & 0x00ff0000) >> 8)
          | ((val & 0x0000ff00) << 8)
          | ((val & 0x000000ff) << 24);
}

//==============================================================================

void Binary::byteswap(float &fval)
{
    int &val = (int &)fval;
    val = ((val & 0xff000000) >> 24)
          | ((val & 0x00ff0000) >> 8)
          | ((val & 0x0000ff00) << 8)
          | ((val & 0x000000ff) << 24);
}

//==============================================================================

void Binary::byteswap(int *field, int numElem)
{
    int i;

    for (i = 0; i < numElem; i++)
    {
        int &val = field[i];
        val = ((val & 0xff000000) >> 24)
              | ((val & 0x00ff0000) >> 8)
              | ((val & 0x0000ff00) << 8)
              | ((val & 0x000000ff) << 24);
    }
}

//==============================================================================

void Binary::byteswap(float *field, int numElem)
{
    int i;

    for (i = 0; i < numElem; i++)
    {
        int &val = (int &)field[i];
        val = ((val & 0xff000000) >> 24)
              | ((val & 0x00ff0000) >> 8)
              | ((val & 0x0000ff00) << 8)
              | ((val & 0x000000ff) << 24);
    }
}
