/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMMON_INCLUDE_FILES_AND_DEFINES_FOR_COVISE_AND_YAC
#define COMMON_INCLUDE_FILES_AND_DEFINES_FOR_COVISE_AND_YAC

#if defined(__linux__) && !defined(_POSIX_SOURCE)
#define _POSIX_SOURCE
#endif

#if defined(__APPLE__) && defined(__LITTLE_ENDIAN__)
#ifndef BYTESWAP
#define BYTESWAP
#endif
#endif

#ifdef _WIN32

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x501 // This specifies WinXP or later - it is needed to access rawmouse from the user32.dll
#endif

#if (_MSC_VER > 1310) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#ifndef _WIN32_WCE
#define POINTER_64 __ptr64
#endif
#else
#define POINTER_64
#endif
/* windows.h would include winsock.h, so be faster */
#include <winsock2.h>
#include <windows.h>
#ifndef _WIN32_WCE
#include <process.h>
#include "unixcompat.h"
#endif
#else
#include <unistd.h>
#endif
#if !defined(_USE_MATH_DEFINES) && !defined(__MINGW32__)
#define _USE_MATH_DEFINES
#endif

#ifndef _WIN32_WCE
#include <sysdep/net.h>
#include "coExport.h"
#include <sys/types.h>
#endif

#if defined __cplusplus && defined(_STANDARD_C_PLUS_PLUS) && !defined(__sgi)
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
#ifndef _WIN32_WCE
#include <errno.h>
#include <signal.h>
#endif
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#endif

#ifdef _WIN32
#ifndef _WIN32_WCE
#include <io.h>
#endif
#else
#include <unistd.h>
#ifndef __hpux
#include <sys/select.h>
#endif
#if defined(__sgi) || defined(__hpux) || defined(_SX) || defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#endif
#endif

#ifdef __cplusplus

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

#endif /* __cplusplus */

#ifdef __hpux
#include "unixcompat.h" /* for strtok_r */
#endif

#if defined(__linux__) || defined(__MINGW32__)
#include <stdint.h>
#endif

#ifdef _MSC_VER
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
/* #define memcpy(a,b,c) memcpy_s(a,c,b,c) */
#endif
#endif /* COVISE_H */
