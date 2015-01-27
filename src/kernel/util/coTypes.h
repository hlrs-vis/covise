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

#define __STDC_FORMAT_MACROS

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif

#if (_MSC_VER > 1310)
/* && !(defined(MIDL_PASS) || defined(RC_INVOKED)) */
#ifndef _WIN32_WCE
#define POINTER_64 __ptr64
#endif
#endif

#include <winsock2.h>

#endif
#ifndef _WIN32_WCE
#include <sys/types.h>
#endif
#if !defined(_WIN32) || defined(__MINGW32__)
#include <stdint.h>
#include <inttypes.h>
#else
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif

/* +++++++++++ System Thread support */

#if defined(CO_sgi) || defined(CO_sgin32) || defined(CO_sgi64)

/* coThreads library using SGI native threads instead of posix */
#define SGI_NATIVE_THREADS
/* #define POSIX_THREADS */

/* comment this out for engine lib when using normal coThreads lib */
/* #define SGI_THREADS_LIBRARY */

/*
#ifndef CERR
#define CERR cerr
#include <fstream.h>
ofstream myout("/dev/null");
#define CERR myout
#endif
*/
#define SGIDLADD
#define SVR4_DYNAMIC_LINKING
#endif

#ifdef __linux__
#define POSIX_THREADS
#define SVR4_DYNAMIC_LINKING
#endif

#ifdef CO_hp
#define DCE_THREADS
#endif

/* +++++++++++ Alignment: currently constant */

#define CO_SIZEOF_ALIGNMENT 8

#ifdef _WIN32
#ifdef pid_t
#undef pid_t
#endif
typedef int pid_t;
#endif

/* +++++++++++ SGI type definitions : insure uses SGI compiler */

#if defined(__sgi) || defined(__insure__)

typedef unsigned char coUByte;

typedef int coInt32;
typedef unsigned int coUInt32;

#if (_MIPS_SZLONG == 64)
typedef long coInt64;
typedef unsigned long coUInt64;
#else
typedef long long coInt64;
typedef unsigned long long coUInt64;
#endif

/* SGI compilers introduce Symbol _BOOL if boolean types available
 * we have to use own symbol to prevent problems on other platforms
 * (HP-UX)
 */
#ifdef _BOOL
#define _BOOL_
#endif
#endif /* CO_sgi    SGI type definitions */

/* +++++++++++ Windows type definitions */

#ifdef _WIN32

typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
typedef long long coInt64;
typedef unsigned long long coUInt64;
typedef unsigned short ushort;
/* typedef long            ssize_t; */

/* code copied from extern_libs/ARCHSUFFIX/python/include/pyconfig.h
to avoid different definition of  ssize_t */
#ifdef _WIN64
typedef __int64 ssize_t;
#else
#if !defined(_WIN32_WCE) && !defined(__MINGW32__)
typedef _W64 int ssize_t;
#endif
#endif
/* end copy */

/* define the int types defined in inttypes.h on *ix systems */
/* the ifndef is necessary to avoid conflict with OpenInventor inttypes !
   The Inventor includes should be made before coTypes.h is included!
 */

/* VisualStudio 2010 finally comes with stdint.h !!! */
#if (_MSC_VER >= 1600)
#include <stdint.h>
#else
#if !defined(_INVENTOR_INTTYPES_) && !defined(__MINGW32__)
#define _INVENTOR_INTTYPES_
typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
#endif
#endif

#define _BOOL_
#endif /* _WIN32 */

/* +++++++++++ Cray-T3E type definitions */

#ifdef CO_t3e

typedef unsigned char coUByte;
typedef short coInt32;
typedef unsigned short coUInt32;
typedef int coInt64;
typedef unsigned int coUInt64;

#define PARALLEL
#endif /* CO_t3e */

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

#ifdef __hpux

typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
typedef long long coInt64;
typedef unsigned long long coUInt64;

#define _BOOL_
#endif /* HP_UX */

/* ++++++++++ Mac OS X type definitions */
#ifdef __APPLE__

#ifdef __LITTLE_ENDIAN__
#ifndef BYTESWAP
#define BYTESWAP
#endif
#endif
typedef unsigned char coUByte;
typedef int coInt32;
typedef unsigned int coUInt32;
typedef long long coInt64;
typedef unsigned long long coUInt64;

#define _BOOL_
#endif

/* Our data Data types are int's  ( @@@@ is that ok on T3E ) */
typedef int coDataType;

/* Parallelisation styles */

#include "coExport.h"
#endif
