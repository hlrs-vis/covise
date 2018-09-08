// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_INTTYPES_H
#define VV_INTTYPES_H

//============================================================================
// Type Declarations
//============================================================================

#ifdef _WIN32
/* code copied from Python's pyconfig.h,
 * to avoid different definition of  ssize_t */
#ifdef _WIN64
typedef __int64 ssize_t;
#else
#if !defined(_WIN32_WCE) && !defined(__MINGW32__) 
typedef _W64 int ssize_t;
#endif
#endif
/* end copy */
#endif

#ifndef __sgi
#ifndef _MSC_VER
#include <stdint.h>
#include <sys/types.h>
#else
#ifdef HAVE_GDCM
#include "stdint.h"
#else
#if (_MSC_VER >= 1600)  /* VisualStudio 2010 comes with stdint.h */
#include <stdint.h>
#else
typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif
#endif
#endif
#else
#include <inttypes.h>
#endif

#if defined(__GNUC__)
#  if (__LP64__ == 1)
#    define VV_PRI64_PREFIX "l"
#  else
#    define VV_PRI64_PREFIX ""
#  endif
#  define VV_SCN64_PREFIX VV_PRI64_PREFIX
#else
#  define VV_PRI64_PREFIX ""
#  define VV_SCN64_PREFIX VV_PRI64_PREFIX
#endif

#define VV_PRIdSIZE VV_PRI64_PREFIX "d"
#define VV_PRIiSIZE VV_PRI64_PREFIX "i"
#define VV_PRIoSIZE VV_PRI64_PREFIX "o"
#define VV_PRIuSIZE VV_PRI64_PREFIX "u"
#define VV_PRIxSIZE VV_PRI64_PREFIX "x"
#define VV_PRIXSIZE VV_PRI64_PREFIX "X"

#define VV_SCNdSIZE VV_SCN64_PREFIX "d"
#define VV_SCNiSIZE VV_SCN64_PREFIX "i"
#define VV_SCNoSIZE VV_SCN64_PREFIX "o"
#define VV_SCNuSIZE VV_SCN64_PREFIX "u"
#define VV_SCNxSIZE VV_SCN64_PREFIX "x"
#define VV_SCNXSIZE VV_SCN64_PREFIX "X"

typedef unsigned char   uchar;                    ///< abbreviation for unsigned char
typedef unsigned short  ushort;                   ///< abbreviation for unsigned short
typedef unsigned int    uint;                     ///< abbreviation for unsigned int
typedef unsigned long   ulong;                    ///< abbreviation for unsigned long
typedef signed   char   schar;                    ///< abbreviation for signed char
typedef signed   short  sshort;                   ///< abbreviation for signed short
typedef signed   int    sint;                     ///< abbreviation for signed int
typedef signed   long   slong;                    ///< abbreviation for signed long

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
