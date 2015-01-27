/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __USEFULL_ROUTINES__
#define __USEFULL_ROUTINES__

#ifndef MIN
#define MIN(a, b) ((a > b) ? (b) : (a))
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? (a) : (b))
#endif
#ifndef SIGN
#define SIGN(a) ((a < 0) ? (-1) : (+1))
#endif

// PI definieren, wenn noch nicht da
#ifndef PI
#ifdef __PI
#define PI __PI
#else
#define PI 3.1415926535897932384626433832795f
#endif
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef BOOL
typedef int BOOL;
#endif

#ifndef _WIN32

int strnicmp(const char *, const char *, int);
int stricmp(const char *, const char *);
#endif
int isfloat(char);

#endif
