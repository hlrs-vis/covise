/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BINARY_H
#define BINARY_H

#include <stdio.h>

class Binary
{
public:
    int isBinary(FILE *fi);
    void byteswap(int &val);
    void byteswap(float &fval);
    void byteswap(int *field, int numElem);
    void byteswap(float *field, int numElem);

private:
};
#endif
