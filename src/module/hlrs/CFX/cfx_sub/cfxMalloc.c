/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#define CFXMAL cfxmal_

#ifdef __cplusplus
extern "C" {
#endif

extern int *CFXMAL(int size);

#ifdef __cplusplus
}
#endif

int *CFXMAL(int size)
{

    int *x;
    x = (int *)malloc(size * sizeof(int));
    return x;
}
