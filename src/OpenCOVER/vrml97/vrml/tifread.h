/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
//  Copyright (C) 2002 Uwe Wössner
*/

#include <stdio.h>

#if defined(__APPLE__) && defined(__LITTLE_ENDIAN__)
#ifndef BYTESWAP
#define BYTESWAP
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

unsigned char *tifread(FILE *fp, const char *url, int *w, int *h, int *nc);

#ifdef __cplusplus
}
#endif
