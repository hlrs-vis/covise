/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __gInclude__
#define __gInclude__

#if SGI
#undef BEOS
#undef MAC
#undef WINDOWS
//
#define ASIO_BIG_ENDIAN 1
#define ASIO_CPU_MIPS 1
#elif defined(_WIN32) || defined(_WIN64)
#undef BEOS
#undef MAC
#undef SGI
#define WINDOWS 1
#define ASIO_LITTLE_ENDIAN 1
#define ASIO_CPU_X86 1
#elif BEOS
#undef MAC
#undef SGI
#undef WINDOWS
#define ASIO_LITTLE_ENDIAN 1
#define ASIO_CPU_X86 1
//
#else
#define MAC 1
#undef BEOS
#undef WINDOWS
#undef SGI
#define ASIO_BIG_ENDIAN 1
#define ASIO_CPU_PPC 1
#endif

// always
#define NATIVE_INT64 0
#define IEEE754_64FLOAT 1

#endif // __gInclude__
