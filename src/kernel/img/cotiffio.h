/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// this header is a wrapper around tiffio.h enabling using fixed
// sized int types that have been deprecated in newer versions of the TIFF
// library without warnings

#ifndef IMG_COTIFFIO_H
#define IMG_COTIFFIO_H

//#ifdef HAVE_LIBTIFF

#include <tiffvers.h>

#if TIFFLIB_VERSION >= 20210416
#define TIFF_DISABLE_DEPRECATED /* enable using uint32 instead of uint32_t */
#define uint16 uint16_t
#define uint32 uint32_t
#define uint64 uint64_t
#endif

#include <tiffio.h>
//#endif

#endif
