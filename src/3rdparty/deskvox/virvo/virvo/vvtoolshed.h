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

#ifndef VV_TOOLSHED_H
#define VV_TOOLSHED_H

#include <list>
#include <string>
#include <vector>

#include <stdio.h>

#include "vvexport.h"
#include "vvinttypes.h"

//============================================================================
// Constant Definitions
//============================================================================

                                                  ///< compiler independent definition for pi
const float TS_PI = 3.1415926535897932384626433832795028841971693993751058f;

//============================================================================
// Function Templates
//============================================================================

/// @return the maximum of two values
template <class C> inline C ts_max(const C a, const C b)
{
  return (a < b) ? b : a;
}

/// @return the maximum of three values
template <class C> inline C ts_max(const C a, const C b, const C c)
{
  return ts_max(a, ts_max(b, c));
}

/// @return the maximum of six values
template <class C> inline C ts_max(const C a, const C b, const C c,
                                   const C d, const C e, const C f)
{
  return ts_max(ts_max(a, b, c), ts_max(d, e, f));
}

/// @return the minimum of two values
template <class C> inline C ts_min(const C a, const C b)
{
  return (a > b) ? b : a;
}

/// @return the minimum of three values
template <class C> inline C ts_min(const C a, const C b, const C c)
{
  return ts_min(a, ts_min(b, c));
}

/// @return the minimum of six values
template <class C> inline C ts_min(const C a, const C b, const C c,
                                   const C d, const C e, const C f)
{
  return ts_min(ts_max(a, b, c), ts_min(d, e, f));
}

/// @return the absolute value of a value
template <class C> inline C ts_abs(const C a)
{
  return (a < 0) ? (-a) : a;
}

/// @return the sign (-1 or 1) of a value
template <class C> inline C ts_sgn(const C a)
{
  return (a < 0) ? (-1) : 1;
}

/// @return the sign or zero(-1, 0, 1)
template <class C> inline C ts_zsgn(const C a)
{
  return (a < 0) ? (-1) : (a > 0 ? 1 : 0);
}

/// @return the result of value a clamped between left and right
template <class C> inline C ts_clamp(const C a, const C left, const C right)
{
  if (a < left)  return left;
  if (a > right) return right;
  return a;
}

/// Swaps the values a and b
template <class C> inline void ts_swap(C a, C b)
{
  C bak = a;
  a     = b;
  b     = bak;
}

/// @param a a value
/// @return the square of a
template <class C> inline C ts_sqr(C a)
{
  return a * a;
}


namespace virvo { namespace toolshed { namespace serialization {

enum EndianType                               /// endianness
{
  VV_LITTLE_END,                              ///< little endian: low-order byte is stored first
  VV_BIG_END                                  ///< big endian: hight-order byte is stored first
};

VIRVO_FILEIOEXPORT EndianType getEndianness();

}}

namespace serialization = toolshed::serialization;

}



//============================================================================
// Class Definitions
//============================================================================

/** Collection of miscellaneous tools.
    Consists of static helper functions which are project independent.
    @author Juergen Schulze-Doebold

    <B>Terminology for extraction functions:</B><BR>
    Example: c:/test/vfview.exe
    <UL>
      <LI>Pathname  = c:/test/vfview.exe
      <LI>Dirname   = c:/test/
      <LI>Extension = exe
      <LI>Filename  = vfview.exe
<LI>Basename  = vfview
</UL>
*/
class VIRVO_FILEIOEXPORT vvToolshed
{
  private:
    static int progressSteps;                     ///< total number of progress steps

  public:
    enum ErrorType
    {
      VV_OK = 0,
      VV_INVALID_SIZE,
      VV_OUT_OF_MEMORY
    };

    enum Format
    {
      VV_LUMINANCE = 0,
      VV_RGB,
      VV_ARGB,
      VV_RGBA,
      VV_BGRA
    };

    static bool    isWhitespace(const char);
    static int     strCompare(const char*, const char*);
    static int     strCompare(const char*, const char*, size_t n);
    static std::vector<std::string> split(const std::string& str, const std::string& delim);
    static bool    isSuffix(const char*, const char*);
    static void    HSBtoRGB(float*, float*, float*);
    static void    HSBtoRGB(float, float, float, float*, float*, float*);
    static void    RGBtoHSB(float*, float*, float*);
    static void    RGBtoHSB(float, float, float, float*, float*, float*);
    static void    strcpyTail(char*, const char*, char);
    static std::string  strcpyTail(const std::string, char);
    static void    strcpyHead(char*, const char*, char);
    static std::string strTrim(const std::string& str);
    static uint32_t parseNextUint32(const char*, size_t&);
    static void    extractFilename(char*, const char*);
    static std::string  extractFilename(const std::string);
    static void    extractDirname(char*, const char*);
    static std::string  extractDirname(const std::string);
    static void    extractExtension(char*, const char*);
    static std::string  extractExtension(const std::string);
    static void    extractBasename(char*, const char*);
    static std::string  extractBasename(const std::string);
    static void    extractBasePath(char*, const char*);
    static void    replaceExtension(char*, const char*, const char*);
    static bool    increaseFilename(char*);
    static bool    increaseFilename(std::string&);
    static void    draw3DLine(int, int, int, int, int, int, uchar*, uchar*, int, int, int, int);
    static void    draw2DLine(int, int, int, int, uint, uchar*, int, int, int);
    static size_t  getTextureSize(size_t);
    static bool    isFile(const char*);
    static bool    isDirectory(const char*);
    static long    getFileSize(const char*);
    static void    getMinMax(const float*, size_t, float*, float*);
    static void    getMinMax(const uchar*, size_t, int*, int*);
    static void    getMinMax16bitHost(const uchar*, size_t, int*, int*);
    static void    getMinMaxAlpha(const uchar*, size_t, int*, int*);
    static void    getMinMaxIgnore(const float*, size_t, float, float*, float*);
    static void    convertUChar2Float(const uchar*, float*, int);
    static void    convertFloat2UChar(const float*, uchar*, int);
    static void    convertFloat2UCharClamp(const float*, uchar*, int, float, float);
    static void    convertFloat2ShortClamp(const float*, uchar*, int, float, float,
                       virvo::serialization::EndianType = virvo::serialization::VV_BIG_END);
    static void    convertFloat2UCharClampZero(const float*, uchar*, int, float, float, float);
    static int     getLargestPrimeFactor(const int);
    // Rounds x to the nearest integer
    static int     round(float x);
    // Rounds x to the nearest integer
    static int     round(double x);
    static void    initProgress(int);
    static void    printProgress(int);
    static ErrorType encodeRLE(uint8_t*, uint8_t*, size_t, size_t, size_t, size_t* outsize);
    static ErrorType decodeRLE(uint8_t*, uint8_t*, size_t, size_t, size_t, size_t* outsize);
    static size_t  encodeRLEFast(uint8_t*, uint8_t*, size_t, size_t);
    static size_t  decodeRLEFast(uint8_t*, uint8_t*, size_t, size_t);
    static int     getNumProcessors();
    static void    makeColorBoardTexture(int, int, float, uchar*);
    static void    convertXY2HS(float, float, float*, float*);
    static void    convertHS2XY(float, float, float*, float*);
    static int     align(const int i, const int pot = 16);
    static void    makeArraySystemIndependent(int, float*);
    static void    makeArraySystemDependent(int, float*);
    static void    sleep(int);
    static void    resample(uchar*, int, int, int, uchar*, int, int, int);
    static void    blendMIP(uchar*, int, int, int, uchar*);
    static void    getCurrentDirectory(char*, int);
    static void    setCurrentDirectory(const char*);
    static void    getProgramDirectory(char*, int);
    static bool    decodeBase64(const char*, int, uchar*);
    static float   interpolateLinear(float, float, float, float);
    static float   interpolateLinear(float, float, float, float, float);
    static bool    makeFileList(std::string&, std::list<std::string>&, std::list<std::string>&);
    static bool    nextListString(std::list<std::string>&, std::string&, std::string&);
    static void    quickSort(int*, int);
    static void    qSort(int*, int, int);
    static float   meanAbsError(float*, float*, const int);
    static float   meanError(float*, float*, const int);
    static float   meanSqrError(float*, float*, const int);
    static std::string file2string(const std::string& filename);
    static int     string2Int(const char* str);
    static void    pixels2Ppm(uchar* pixels, const int width, const int height,
                              const char* fileName, const Format format = VV_RGBA);
    static void    pixels2Ppm(float* pixels, const int width, const int height,
                              const char* fileName, const Format format = VV_RGBA);
    static int     parsePort(std::string const& url);
    static std::string stripPort(std::string const& url);
    static void    printBacktrace();
};

namespace virvo
{
namespace toolshed
{
template <typename T>
inline T iDivUp(T a, T b)
{
  return (a + b - 1) / b;
}

template <typename T>
inline T clamp(T const& a, T const& left, T const& right)
{
  if (a < left)  return left;
  if (a > right) return right;
  return a;
}

VIRVO_FILEIOEXPORT std::vector<std::string> entryList(std::string const& dir);

VIRVO_FILEIOEXPORT bool startsWith(std::string const& string, std::string const& prefix);
VIRVO_FILEIOEXPORT bool endsWith(std::string const& string, std::string const& suffix);

namespace serialization
{
VIRVO_FILEIOEXPORT size_t  read(uint8_t* src, uint8_t* val);
VIRVO_FILEIOEXPORT size_t  read(uint8_t* src, uint16_t* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(uint8_t* src, uint32_t* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(uint8_t* src, uint64_t* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(uint8_t* src, float* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(FILE* src, uint8_t* val);
VIRVO_FILEIOEXPORT size_t  read(FILE* src, uint16_t* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(FILE* src, uint32_t* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(FILE* src, uint64_t* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(FILE* src, float* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(std::ifstream& src, uint32_t* val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  read(std::ifstream& src, uint64_t* val, EndianType end = VV_BIG_END);

VIRVO_FILEIOEXPORT size_t  write(uint8_t* dst, uint8_t val);
VIRVO_FILEIOEXPORT size_t  write(uint8_t* dst, uint16_t val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  write(uint8_t* dst, uint32_t val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  write(uint8_t* dst, uint64_t val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  write(uint8_t* dst, float val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  write(FILE* dst, uint8_t val);
VIRVO_FILEIOEXPORT size_t  write(FILE* dst, uint16_t val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  write(FILE* dst, uint32_t val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  write(FILE* dst, uint64_t val, EndianType end = VV_BIG_END);
VIRVO_FILEIOEXPORT size_t  write(FILE* dst, float val, EndianType end = VV_BIG_END);

/* legacy functions -- DEPRECATED */
inline uint8_t  read8(uint8_t* src);
inline uint16_t read16(uint8_t* src, EndianType end = VV_BIG_END);
inline uint32_t read32(uint8_t* src, EndianType end = VV_BIG_END);
inline uint64_t read64(uint8_t* src, EndianType end = VV_BIG_END);
inline float    readFloat(uint8_t* src, EndianType end = VV_BIG_END);
inline uint8_t  read8(FILE* src);
inline uint16_t read16(FILE* src, EndianType end = VV_BIG_END);
inline uint32_t read32(FILE* src, EndianType end = VV_BIG_END);
inline uint64_t read64(FILE* src, EndianType end = VV_BIG_END);
inline float    readFloat(FILE* src, EndianType end = VV_BIG_END);
inline uint32_t read32(std::ifstream& src, EndianType end = VV_BIG_END);
inline uint64_t read64(std::ifstream& src, EndianType end = VV_BIG_END);

inline size_t   write8(uint8_t* dst, uint8_t val);
inline size_t   write16(uint8_t* dst, uint16_t val, EndianType end = VV_BIG_END);
inline size_t   write32(uint8_t* dst, uint32_t val, EndianType end = VV_BIG_END);
inline size_t   write64(uint8_t* dst, uint64_t val, EndianType end = VV_BIG_END);
inline size_t   writeFloat(uint8_t* dst, float val, EndianType end = VV_BIG_END);
inline size_t   write8(FILE* dst, uint8_t val);
inline size_t   write16(FILE* dst, uint16_t val, EndianType end = VV_BIG_END);
inline size_t   write32(FILE* dst, uint32_t val, EndianType end = VV_BIG_END);
inline size_t   write64(FILE* dst, uint64_t val, EndianType end = VV_BIG_END);
inline size_t   writeFloat(FILE* dst, float val, EndianType end = VV_BIG_END);

} // serialization

} // toolshed

namespace serialization = toolshed::serialization;

} // virvo

#include "vvtoolshed.impl.h"

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
