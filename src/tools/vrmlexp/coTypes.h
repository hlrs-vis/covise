/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TYPES_H
#define CO_TYPES_H

/* ---------------------------------------------------------------------- //
//                                                                        //
//                                                                        //
// Description: DLL EXPORT/IMPORT specification and type definitions      //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                             (C)2003 HLRS                               //
// Author: Uwe Woessner, Ruth Lang                                        //
// Date:  30.10.03  V1.0                                                  */

#if defined(__linux__) && !defined(_POSIX_SOURCE)
#define _POSIX_SOURCE
#endif

#ifdef _WIN32

#if (_MSC_VER > 1310) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
/* windows.h would include winsock.h, so be faster */
#include <winsock2.h>
#include <windows.h>
#include <process.h>
//# include <util/unixcompat.h>
#else
#include <unistd.h>
#endif
#define _USE_MATH_DEFINES

//#include <sysdep/net.h>

//#include <util/coTypes.h>
#include <sys/types.h>

#if defined __cplusplus && defined(_STANDARD_C_PLUS_PLUS)
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#else
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#endif

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#include <sys/select.h>
#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#endif
#endif

#ifdef __cplusplus
#ifdef _STANDARD_C_PLUS_PLUS

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::flush;
using std::ostream;
using std::ofstream;
using std::fstream;
using std::istream;
using std::ios;
using std::ifstream;
using std::istringstream;
using std::ostringstream;
using std::stringstream;

#include <set>
using std::set;

#include <map>
using std::map;
using std::multimap;

#include <list>
using std::list;

#include <vector>
using std::vector;
using std::allocator;

#include <string>
using std::string;

#include <memory>
using std::pair;

#include <stack>
using std::stack;

#include <algorithm>
using std::find;

#if (defined(__linux__) && !defined(CO_linux)) || defined(__APPLE__)
#include <ext/hash_map>
#include <ext/hash_set>
using __gnu_cxx::hash_map;
using __gnu_cxx::hash_set;
#else
#ifdef _WIN32
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS
#include <hash_map>
using stdext::hash_map;
#include <hash_set>
using stdext::hash_set;
#else
#include <hash_map>
using std::hash_map;
#include <hash_set>
using std::hash_set;
#endif
#endif

#else /* _STANDARD_C_PLUS_PLUS */

#include <iostream.h>
#include <strstream.h>
#include <fstream.h>
#include <iomanip.h>
#include <map.h>
#include <set.h>
#include <list.h>
#include <algo.h>
using std::find;
using std::set;
using std::map;
using std::pair;
using std::list;

typedef istrstream istringstream;
typedef ostrstream ostringstream;
typedef strstream stringstream;

#include <vector>
#include <string>

#include <hash_map>
#include <hash_set>

//#endif
#include <stack.h>
#endif /* _STANDARD_C_PLUS_PLUS */
#endif /* __cplusplus */

#ifdef __linux__
#include <stdint.h>
#endif

#ifdef _WIN32
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
/* #define memcpy(a,b,c) memcpy_s(a,c,b,c) */
#endif

#ifdef WIN32

#if (_MSC_VER > 1310)
/* && !(defined(MIDL_PASS) || defined(RC_INVOKED)) */
#define POINTER_64 __ptr64
#endif

#include <winsock2.h>

#endif
/* +++++++++++ System Thread support */

#ifdef __linux__
#define POSIX_THREADS
#define SVR4_DYNAMIC_LINKING
#endif

/* +++++++++++ Alignment: currently constant */

#define CO_SIZEOF_ALIGNMENT 8

#ifdef _WIN32
#ifdef pid_t
#undef pid_t
#endif
typedef int pid_t;
#endif

/* +++++++++++ Windows type definitions */

#ifdef _WIN32

typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
typedef long long coInt64;
typedef unsigned long long coUInt64;
typedef long ssize_t;
typedef unsigned short ushort;

/* define the int types defined in inttypes.h on *ix systems */
/* the ifndef is necessary to avoid conflict with OpenInventor inttypes !
   The Inventor includes should be made before coTypes.h is included!
 */
#ifndef _INVENTOR_INTTYPES_
#define _INVENTOR_INTTYPES_
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
#endif

#define _BOOL_
#endif /* _WIN32 */

/* +++++++++++ Cray-T3E type definitions */

/* +++++++++++ Linux type definitions */

#ifdef __linux__

typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
typedef long long coInt64;
typedef unsigned long long coUInt64;

#define _BOOL_
#endif /* LINUX */

/* +++++++++++ HP-UX type definitions (IA-64, PA-RISC not tested!) */

/* ++++++++++ Mac OS X type definitions */
#ifdef __APPLE__

typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
typedef long long coInt64;
typedef unsigned long long coUInt64;

#define _BOOL_
#endif

/* +++++++++++ Bool definitions */
#ifndef _BOOL_

/* if no bool type is supplied by system, "util/coTypes.h" does it. */
typedef int bool;
#define false 0
#define true(!false)
#endif

/* Our data Data types are int's  ( @@@@ is that ok on T3E ) */
typedef int coDataType;

/* Parallelisation styles */

#if !defined(__linux__) && !defined(__APPLE__)
static char *strsep(char **s, const char *delim)
{
    if (!s)
        return NULL;

    char *start = *s;
    char *p = *s;

    if (p == NULL)
        return NULL;

    while (*p)
    {
        const char *d = delim;
        while (*d)
        {
            if (*p == *d)
            {
                *p = '\0';
                *s = p + 1;
                return start;
            }
            d++;
        }
        p++;
    }

    *s = NULL;
    return start;
}
#endif
#endif
