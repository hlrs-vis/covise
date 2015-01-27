/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file os.hxx
 * operating system dependent definitions.
 */

// #include "os.h" // operating system dependent definitions

#ifndef __operatingsystem_hxx__
#define __operatingsystem_hxx__

#ifdef __Use_MPI__
#include <mpi.h>
#endif

// #define __Lib_IMSL_F__   // library IMSL in Fortran
#ifdef WIN32
#define __Windows_NT__ // operating system Windows NT
#else
#define __Unix__ // operating system Unix
#endif

struct PointCC
{
    void reset();

    double x; ///< x-coordinate of the point
    double y; ///< y-coordinate of the point
    double z; ///< z-coordinate of the point
};

// ----------------------------------------------------------------------------------------------------------------------
//                                 UNIX
// ----------------------------------------------------------------------------------------------------------------------

#ifdef __Unix__

// integers
typedef unsigned char UCHAR_F; ///< unsigned character
typedef char CHAR_F; ///< signed character
typedef unsigned short USHOR_F; ///< unsigned 16bit integer
typedef short SHOR_F; ///< signed 16bit integer
typedef unsigned int UINT_F; ///< unsigned 32bit integer
typedef int INT_F; ///< signed 32bit integer
typedef int INT; ///< signed 32bit integer
typedef unsigned long ULONG_F; ///< unsigned 64bit integer
typedef long LONG_F; ///< signed 64bit integer

#endif // __Unix__

// ----------------------------------------------------------------------------------------------------------------------
//                                 Windows NT
// ----------------------------------------------------------------------------------------------------------------------
#ifdef __Windows_NT__

#if defined(_MSC_VER)
// integers
typedef unsigned char UCHAR_F; ///< unsigned character
typedef char CHAR_F; ///< signed character
typedef unsigned __int16 USHOR_F; ///< unsigned 16bit integer
typedef __int16 SHOR_F; ///< signed 16bit integer
typedef unsigned __int32 UINT_F; ///< unsigned 32bit integer
typedef __int32 INT_F; ///< signed 32bit integer
typedef unsigned __int64 ULONG_F; ///< unsigned 64bit integer
typedef __int64 LONG_F; ///< signed 64bit integer
#else
// mingw
typedef unsigned char UCHAR_F; ///< unsigned character
typedef char CHAR_F; ///< signed character
typedef unsigned short USHOR_F; ///< unsigned 16bit integer
typedef short SHOR_F; ///< signed 16bit integer
typedef unsigned int UINT_F; ///< unsigned 32bit integer
typedef int INT_F; ///< signed 32bit integer
#if defined(__LP64__)
typedef signed long LONG_F;
typedef unsigned long ULONG_F;
#else
typedef signed long long LONG_F;
typedef unsigned long long ULONG_F;
#endif
#endif

// compiler statements
#pragma warning(disable : 4786) // disable warning, if name of STL is longer than 255 characters

#endif // __Windows_NT__

#endif // __operatingsystem_hxx__
