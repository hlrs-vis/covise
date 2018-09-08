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

#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <iterator>

#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <boost/filesystem.hpp>

#include <errno.h>
#include "vvplatform.h"

#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvtoolshed.h"

#include "private/vvlog.h"

#ifdef __sun
#define powf pow
#define atanf atan
#define sqrtf sqrt
#define sinf sin
#define cosf cos
#endif

using namespace std;

#ifdef __hpux
# include <sys/pstat.h>
# include <sys/param.h>
# include <sys/unistd.h>
#endif
//#define VV_STANDALONE      // define to perform self test

int vvToolshed::progressSteps = 0;

//============================================================================
//
//============================================================================


virvo::serialization::EndianType virvo::serialization::getEndianness()
{
  const float one = 1.0f;                          // memory representation of 1.0 on big endian machines: 3F 80 00 00

  const uchar* ptr = (uchar*)&one;
  if (*ptr == 0x3f) return VV_BIG_END;
  else
  {
    assert(*ptr == 0);
    return VV_LITTLE_END;
  }
}


//============================================================================
// method definitions
//============================================================================

//----------------------------------------------------------------------------
/** Tells if a given char is a whitespace, i.e. a tab, space or new line
    @param c char to test
    @return true, if whitespace, false otherwise
 */
bool vvToolshed::isWhitespace(const char c)
{
  bool result;

  if ((c == ' ') || (c == '\t') || (c == '\n'))
  {
    result = true;
  }
  else
  {
    result = false;
  }

  return result;
}

//----------------------------------------------------------------------------
/** Case insensitive string comparison
    @param str1,str2 pointers to strings that are being compared
    @return
      <UL>
        <LI> 0 if equal
        <LI>-1 if str1<str2
        <LI> 1 if str1>str2
      </UL>
*/
int vvToolshed::strCompare(const char* str1, const char* str2)
{
#ifdef _WIN32
  return stricmp(str1, str2);
#else
  return strcasecmp(str1, str2);
#endif
}

//----------------------------------------------------------------------------
/** Case insensitive string comparison with a number of characters.
    @param str2,str2 pointers to strings that are being compared
    @param n = number of characters to compare
    @return the same values as in #strCompare(const char*, const char*)
*/
int vvToolshed::strCompare(const char* str1, const char* str2, size_t n)
{
#ifdef _WIN32
  return strnicmp(str1, str2, n);
#else
  return strncasecmp(str1, str2, n);
#endif
}

std::vector<std::string> vvToolshed::split(const std::string& str, const std::string& delim)
{
  std::vector<std::string> result;

  std::string strCopy(str);
  size_t pos = strCopy.find_first_of(delim);
  while (pos != std::string::npos)
  {
    if (pos > 0)
    {
      result.push_back(strCopy.substr(0, pos));
      strCopy = strCopy.substr(pos + 1);
      pos = strCopy.find_first_of(delim);
    }
  }

  if (strCopy.length() > 0)
  {
    result.push_back(strCopy);
  }

  return result;
}

//----------------------------------------------------------------------------
/** Case insensitive string suffix comparison.
    @param str    pointer to string
    @param suffix pointer to suffix
    @return true if suffix is the suffix of str
*/
bool vvToolshed::isSuffix(const char* str, const char* suffix)
{
  if (vvToolshed::strCompare(str + strlen(str) - strlen(suffix), suffix) == 0)
    return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Convert HSB color model to RGB.
    @param a hue (0..360) (becomes red)
    @param b saturation (0..1) (becomes green)
    @param c value = brightness (0..1) (becomes blue)
    @return RGB values in a,b,c
*/
void vvToolshed::HSBtoRGB(float* a, float* b, float* c)
{
  float red, green, blue;

  HSBtoRGB(*a, *b, *c, &red, &green, &blue);
  *a = red;
  *b = green;
  *c = blue;
}

//----------------------------------------------------------------------------
/** Convert HSB color model to RGB.
    @param h hue (0..1)
    @param s saturation (0..1)
    @param v value = brightness (0..1)
    @return RGB values (0..1) in r,g,b
*/
void vvToolshed::HSBtoRGB(float h, float s, float v, float* r, float* g, float* b)
{
  float f, p, q, t;
  int i;

  // Clamp values to their valid ranges:
  h = ts_clamp(h, 0.0f, 1.0f);
  s = ts_clamp(s, 0.0f, 1.0f);
  v = ts_clamp(v, 0.0f, 1.0f);

  // Convert hue:
  if (h == 1.0f) h = 0.0f;
  h *= 360.0f;

  if (s==0.0f)                                    // grayscale value?
  {
    *r = v;
    *g = v;
    *b = v;
  }
  else
  {
    h /= 60.0;
    i = int(h);
    f = h - i;
    p = v * (1.0f - s);
    q = v * (1.0f - (s * f));
    t = v * (1.0f - (s * (1.0f - f)));
    switch (i)
    {
      case 0: *r = v; *g = t; *b = p; break;
      case 1: *r = q; *g = v; *b = p; break;
      case 2: *r = p; *g = v; *b = t; break;
      case 3: *r = p; *g = q; *b = v; break;
      case 4: *r = t; *g = p; *b = v; break;
      case 5: *r = v; *g = p; *b = q; break;
    }
  }
}

//----------------------------------------------------------------------------
/** Convert RGB colors to HSB model
    @param a red [0..360] (becomes hue)
    @param b green [0..1] (becomes saturation)
    @param c blue [0..1]  (becomes brightness)
    @return HSB in a,b,c
*/
void vvToolshed::RGBtoHSB(float* a, float* b, float* c)
{
  float h,s,v;
  RGBtoHSB(*a, *b, *c, &h, &s, &v);
  *a = h;
  *b = s;
  *c = v;
}

//----------------------------------------------------------------------------
/** Convert RGB colors to HSB model.
    @param r,g,b RGB values [0..1]
    @return h = hue [0..1], s = saturation [0..1], v = value = brightness [0..1]
*/
void vvToolshed::RGBtoHSB(float r, float g, float b, float* h, float* s, float* v)
{
  float max, min, delta;

  // Clamp input values to valid range:
  r = ts_clamp(r, 0.0f, 1.0f);
  g = ts_clamp(g, 0.0f, 1.0f);
  b = ts_clamp(b, 0.0f, 1.0f);

  max = ts_max(r, ts_max(g, b));
  min = ts_min(r, ts_min(g, b));
  *v = max;
  *s = (max != 0.0f) ? ((max - min) / max) :0.0f;
  if (*s == 0.0f) *h = 0.0f;
  else
  {
    delta = max - min;
    if (r==max)
      *h = (g - b) / delta;
    else if (g==max)
      *h = 2.0f + (b - r) / delta;
    else if (b==max)
      *h = 4.0f + (r - g) / delta;
    *h *= 60.0f;
    if (*h < 0.0f)
      *h += 360.0f;
  }
  *h /= 360.0f;
}

//----------------------------------------------------------------------------
/** Copies the tail string after the last occurrence of a given character.
    Example: str="c:\ local\ testfile.dat", c='\' => suffix="testfile.dat"
    @param suffix <I>allocated</I> space for the found string
    @param str    source string
    @param c      character after which to copy characters
    @return result in suffix, empty string if c was not found in str
*/
void vvToolshed::strcpyTail(char* suffix, const char* str, char c)
{
  const char *p = strrchr(str, c);
  if (p)
    strcpy(suffix, p+1);
  else
    strcpy(suffix, "");
}

//----------------------------------------------------------------------------
/** Copies the tail string after the last occurrence of a given character.
    Example: str="c:\local\testfile.dat", c='\' => suffix="testfile.dat"
    @param str    source string
    @param c      character after which to copy characters
    @return string after c ("testfile.dat")
*/
string vvToolshed::strcpyTail(const string str, char c)
{
  return str.substr(str.rfind(c) + 1);
}

//----------------------------------------------------------------------------
/** Copies the head string before the first occurrence of a given character.
    Example: str="c:\ local\ testfile.dat", c='.' => head="c:\ local\ testfile"
    @param head  <I>allocated</I> space for the found string
    @param str    source string
    @param c      character before which to copy characters
    @return result in head, empty string if c was not found in str
*/
void vvToolshed::strcpyHead(char* head, const char* str, char c)
{
  int i = 0;

  if (strchr(str, c) == NULL)
  {
    head[0] = '\0';
    return;
  }
  while (str[i] != c)
  {
    head[i] = str[i];
    ++i;
  }
  head[i] = '\0';
}

//----------------------------------------------------------------------------
/** Removes leading and trailing spaces from a string.
    Example: str="  hello " => str="hello"
    @param str    string to trim
    @return result in str
*/
std::string vvToolshed::strTrim(const std::string& str)
{
  std::string result = str;

  // Trim trailing spaces:
  for (size_t i=result.length()-1; i>0; --i)
  {
    if (std::isspace(str[i])) result.erase(i);
    else break;
  }
  if (result.length()==0) return result; // done

  // Trim leading spaces:
  while(std::isspace(result[0]))
  {
    result.erase(0,0);
  }

  return result;
}

//----------------------------------------------------------------------------
/** Parses the next integer from the given position and updates the counter.
    Example: if str="hallihallo3" and iterator=9 ==> return int(3) and
    iterator is incremented (via ref) to 10
    @param str    source string
    @return next int at given position
 */
uint32_t vvToolshed::parseNextUint32(const char* str, size_t& iterator)
{
  uint32_t result = 0;

  // Skip leading white spaces.
  while (isWhitespace(str[iterator]))
  {
    ++iterator;
  }

  // Parse integer value.
  char c = str[iterator];
  while ((c >= '0') && (c <= '9'))
  {
      result *= 10;
      result += static_cast<uint32_t>(c-48);

      iterator++;
      c = str[iterator];
  }

  return result;
}

//----------------------------------------------------------------------------
/** Extracts a filename from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param filename <I>allocated</I> space for filename (e.g. "testfile.dat")
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in filename
*/
void vvToolshed::extractFilename(char* filename, const char* pathname)
{
#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  if (strchr(pathname, delim)) strcpyTail(filename, pathname, delim);
  else strcpy(filename, pathname);
}

//----------------------------------------------------------------------------
/** Extracts a filename from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat")
    @return filename (e.g. "testfile.dat")
*/
string vvToolshed::extractFilename(const string pathname)
{
#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  if (pathname.find(delim, 0) != string::npos) return strcpyTail(pathname, delim);
  else return pathname;
}

//----------------------------------------------------------------------------
/** Extracts a directory name from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param dirname  <I>allocated</I> space for directory name (e.g. "/usr/local/" or "c:\user\")
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat" or "c:\user\testfile.dat")
    @return result in dirname
*/
void vvToolshed::extractDirname(char* dirname, const char* pathname)
{
  int i, j;

#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  // Search for '\' or '/' in pathname:
  i = strlen(pathname) - 1;
  while (i>=0 && pathname[i]!=delim)
    --i;

  // Extract preceding string:
  if (i<0)                                        // delimiter not found?
    strcpy(dirname, "");
  else
  {
    for (j=0; j<=i; ++j)
      dirname[j] = pathname[j];
    dirname[j] = '\0';
  }
}

//----------------------------------------------------------------------------
/** Extracts a directory name from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat" or "c:\user\testfile.dat")
    @return directory namename (e.g. "/usr/local/" or "c:\user\")
*/
string vvToolshed::extractDirname(const string pathname)
{
  string dirname;
  size_t delimPos;

#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  delimPos = pathname.rfind(delim);
  if (delimPos == string::npos) dirname = pathname;
  else dirname.insert(0, pathname, 0, delimPos+1);
  return dirname;
}

//----------------------------------------------------------------------------
/** Extracts an extension from a given path or filename.
    @param extension <I>allocated</I> space for extension (e.g. "dat")
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in extension
*/
void vvToolshed::extractExtension(char* extension, const char* pathname)
{
  char *filename = new char[strlen(pathname)+1];
  extractFilename(filename, pathname);

  strcpyTail(extension, filename, '.');
  delete[] filename;
}

//----------------------------------------------------------------------------
/** Extracts an extension from a given path or filename.
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return extension, e.g., "dat"
*/
string vvToolshed::extractExtension(const string pathname)
{
  std::string result;

  size_t pos = pathname.rfind('.');

  if (pos == string::npos)
  {
    result = "";
  }
  else
  {
    result.insert(0, pathname, pos + 1, pathname.size());
  }

  return result;
}

//----------------------------------------------------------------------------
/** Extracts the base file name from a given path or filename, excluding
    the '.' delimiter.
    @param basename  <I>allocated</I> memory space for basename (e.g. "testfile").
                     Memory must be allocated for at least strlen(pathname)+1 chars!
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in basename
*/
void vvToolshed::extractBasename(char* basename, const char* pathname)
{
  int i;

  extractFilename(basename, pathname);

  // Search for '.' in pathname:
  i = strlen(basename) - 1;
  while (i>=0 && basename[i]!='.')
    --i;

  if (i>0) basename[i] = '\0';                    // convert '.' to '\0' to terminate string
}

//----------------------------------------------------------------------------
/** Extracts the base file name from a given path or filename, excluding
    the '.' delimiter.
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return basename (e.g. "testfile").
*/
string vvToolshed::extractBasename(const string pathname)
{
  string basename;

  basename = extractFilename(pathname);
  basename.erase(basename.rfind('.'));
  return basename;
}

//----------------------------------------------------------------------------
/** Remove the extension from a path string. If no '.' is present in path
    string, the path is removed without changes.
    @param basepath  <I>allocated</I> space for path without extension
                     (e.g., "/usr/local/testfile")
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in basepath
*/
void vvToolshed::extractBasePath(char* basepath, const char* pathname)
{
  int i, j;

  // Search for '.' in pathname:
  i = strlen(pathname) - 1;
  while (i>=0 && pathname[i]!='.')
    --i;

  // Extract tail string:
  if (i<0)                                        // '.' not found?
  {
    strcpy(basepath, pathname);
  }
  else
  {
    for (j=0; j<i; ++j)
      basepath[j] = pathname[j];
    basepath[j] = '\0';
  }
}

//----------------------------------------------------------------------------
/** Replaces a file extension with a new one, overwriting the old one.
    If the pathname does not have an extension yet, the new extension will be
    added.
    @param newPath      _allocated space_ for resulting path name with new
                        extension (e.g. "/usr/local/testfile.txt")
    @param newExtension new extension without '.' (e.g. "txt")
    @param pathname     file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in newPath
*/
void vvToolshed::replaceExtension(char* newPath, const char* newExtension, const char* pathname)
{
  char* pointPos;
  int baseNameLen;                                // length of base file name, including point

  pointPos = (char*)strrchr(pathname, '.');
  if (pointPos==NULL)                             // is there a point in pathname?
  {
    // No point, so just add new extension:
    strcpy(newPath, pathname);
    strcat(newPath, ".");
    strcat(newPath, newExtension);
  }
  else
  {
    baseNameLen = pointPos-pathname+1;
    memcpy(newPath, pathname, baseNameLen);       // copy everything before the point, including the point
    newPath[baseNameLen] = '\0';
    strcat(newPath, newExtension);
  }
}

//----------------------------------------------------------------------------
/** Increases the filename (filename must include an extension!).
  @return true if successful, false if filename couldn't be increased.
          Does not check if the file with the increased name exists.
*/
bool vvToolshed::increaseFilename(char* filename)
{
  bool done = false;
  int i;
  char ext[256];

  extractExtension(ext, filename);
  if (strlen(ext)==0) i=strlen(filename) - 1;
  else i = strlen(filename) - strlen(ext) - 2;
  bool hex = false;

  if(hex)
  {
	  while (!done)
	  {
		  if (i<0 || ((filename[i]<'0' || filename[i]>'9')&& (filename[i]<'A' || filename[i]>'F')))
			  return false;

		  if (filename[i] == 'F')                       // overflow?
		  {
			  filename[i] = '0';
			  --i;
		  }
		  else
		  {
			  if(filename[i]== '9')
				  filename[i] = 'A';
			  else
			  ++filename[i];
			  done = 1;
		  }
	  }
  }
  else
  {
	  while (!done)
	  {
		  if (i<0 || filename[i]<'0' || filename[i]>'9')
			  return false;

		  if (filename[i] == '9')                       // overflow?
		  {
			  filename[i] = '0';
			  --i;
		  }
		  else
		  {
			  ++filename[i];
			  done = 1;
		  }
	  }
  }
  return true;
}

//----------------------------------------------------------------------------
/** Increases the filename (filename must include an extension!).
  @return true if successful, false if filename couldn't be increased.
          Does not check if the file with the increased name exists.
*/
bool vvToolshed::increaseFilename(string& filename)
{
  bool done = false;
  int i;
  string ext;

  ext = extractExtension(filename);
  if (ext.size()==0) i = filename.size() - 1;
  else i = filename.size() - ext.size() - 2;
  while (!done)
  {
    if (i<0 || filename[i]<'0' || filename[i]>'9')
      return false;

    if (filename[i] == '9')                       // overflow?
    {
      filename[i] = '0';
      --i;
    }
    else
    {
      ++filename[i];
      done = 1;
    }
  }
  return true;
}

//----------------------------------------------------------------------------
/** Draws a line in a 3D volume dataset using Bresenham's algorithm.
    Both line end points must lie within the volume. The Coordinate system is:
    <PRE>
           y
           |__ x
          /
         z
    </PRE>
    The volume data is arranged like this:
    <UL>
      <LI>origin is top left front
<LI>width in positive x direction
<LI>height in negative y direction
<LI>slices in negative z direction
</UL>
@param x0,y0,z0  line starting point in voxels
@param x1,y1,z1  line end point in voxels
@param color     array with line color elements (size = bpm * mod)
@param data      pointer to raw volume data
@param bpv       bytes per voxel
@param w,h,s     width/height/slices of volume data array [voxels]
*/
void vvToolshed::draw3DLine(int x0, int y0, int z0, int x1, int y1, int z1,
uchar* color, uchar* data, int bytes, int w, int h, int s)
{
  int xd, yd, zd;
  int x, y, z;
  int ax, ay, az;
  int sx, sy, sz;
  int dx, dy, dz;
  int i;

  x0 = ts_clamp(x0, 0, w-1);
  x1 = ts_clamp(x1, 0, w-1);
  y0 = ts_clamp(y0, 0, h-1);
  y1 = ts_clamp(y1, 0, h-1);
  z0 = ts_clamp(z0, 0, s-1);
  z1 = ts_clamp(z1, 0, s-1);

  dx = x1 - x0;
  dy = y1 - y0;
  dz = z1 - z0;

  ax = ts_abs(dx) << 1;
  ay = ts_abs(dy) << 1;
  az = ts_abs(dz) << 1;

  sx = ts_zsgn(dx);
  sy = ts_zsgn(dy);
  sz = ts_zsgn(dz);

  x = x0;
  y = y0;
  z = z0;

  if (ax >= ts_max(ay, az))                       // x is dominant
  {
    yd = ay - (ax >> 1);
    zd = az - (ax >> 1);
    for (;;)
    {
      for (i=0; i<bytes; ++i)
      {
        data[bytes * (z * w * h + y * w + x) + i] = color[i];
      }
      if (x == x1) return;
      if (yd >= 0)
      {
        y += sy;
        yd -= ax;
      }
      if (zd >= 0)
      {
        z += sz;
        zd -= ax;
      }
      x += sx;
      yd += ay;
      zd += az;
    }
  }
  else if (ay >= ts_max(ax, az))                  // y is dominant
  {
    xd = ax - (ay >> 1);
    zd = az - (ay >> 1);
    for (;;)
    {
      for (i=0; i<bytes; ++i)
      {
        data[bytes * (z * w * h + y * w + x) + i] = color[i];
      }
      if (y == y1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= ay;
      }
      if (zd >= 0)
      {
        z += sz;
        zd -= ay;
      }
      y += sy;
      xd += ax;
      zd += az;
    }
  }
  else if (az >= ts_max(ax, ay))                  // z is dominant
  {
    xd = ax - (az >> 1);
    yd = ay - (az >> 1);
    for (;;)
    {
      for (i=0; i<bytes; ++i)
      {
        data[bytes * (z * w * h + y * w + x) + i] = color[i];
      }
      if (z == z1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= az;
      }
      if (yd >= 0)
      {
        y += sy;
        yd -= az;
      }
      z += sz;
      xd += ax;
      yd += ay;
    }
  }
}

//----------------------------------------------------------------------------
/** Draws a line in a 2D image dataset using Bresenham's algorithm.
    Both line end points must lie within the image. The coordinate system is:
    <PRE>
           y
           |__ x
    </PRE>
    The image data is arranged like this:
    <UL>
      <LI>origin is top left
      <LI>width is in positive x direction
      <LI>height is in negative y direction
</UL>
@param x0/y0  line starting point in pixels
@param x1/y1  line end point in pixels
@param color  line color, 32 bit value: bits 0..7=first color,
8..15=second color etc.
@param data   pointer to raw image data
@param bpp    byte per pixel (e.g. 3 for 24 bit RGB), range: [1..4]
@param w/h    width/height of image data array in pixels
*/
void vvToolshed::draw2DLine(int x0, int y0, int x1, int y1,
uint color, uchar* data, int bpp, int w, int h)
{
  int xd, yd;
  int x, y;
  int ax, ay;
  int sx, sy;
  int dx, dy;
  int i;
  uchar col[4];                                   // color components; 0=most significant byte

  assert(bpp <= 4);

  col[0] = (uchar)((color >> 24) & 0xff);
  col[1] = (uchar)((color >> 16) & 0xff);
  col[2] = (uchar)((color >> 8)  & 0xff);
  col[3] = (uchar)(color & 0xff);

  x0 = ts_clamp(x0, 0, w-1);
  x1 = ts_clamp(x1, 0, w-1);
  y0 = ts_clamp(y0, 0, h-1);
  y1 = ts_clamp(y1, 0, h-1);

  dx = x1 - x0;
  dy = y1 - y0;

  ax = ts_abs(dx) << 1;
  ay = ts_abs(dy) << 1;

  sx = ts_zsgn(dx);
  sy = ts_zsgn(dy);

  x = x0;
  y = y0;

  if (ax >= ay)                                   // x is dominant
  {
    yd = ay - (ax >> 1);
    for (;;)
    {
      for (i=0; i<bpp; ++i)
        data[bpp * (y * w + x) + i] = col[i];
      if (x == x1) return;
      if (yd >= 0)
      {
        y += sy;
        yd -= ax;
      }
      x += sx;
      yd += ay;
    }
  }
  else if (ay >= ax)                              // y is dominant
  {
    xd = ax - (ay >> 1);
    for (;;)
    {
      for (i=0; i<bpp; ++i)
        data[bpp * (y * w + x) + i] = col[i];
      if (y == y1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= ay;
      }
      y += sy;
      xd += ax;
    }
  }
}

//----------------------------------------------------------------------------
/** Compute texture hardware compatible numbers.
    @param imgSize  the image size [pixels]
    @return the closest power-of-2 value that is greater than or equal to imgSize.
*/
size_t vvToolshed::getTextureSize(size_t imgSize)
{
  return (size_t)pow(2.0, (double)ceil(log((double)imgSize) / log(2.0)));
}

//----------------------------------------------------------------------------
/** Checks if a file exists.
    @param filename file name to check for
    @return true if file exists
*/
bool vvToolshed::isFile(const char* filename)
{
#ifdef _WIN32
  FILE* fp = fopen(filename, "rb");
  if (fp==NULL) return false;
  fclose(fp);
  return true;
#else
  struct stat buf;
  if (stat(filename, &buf) == 0)
  {
    if (S_ISREG(buf.st_mode)) return true;
  }
  return false;
#endif
}

//----------------------------------------------------------------------------
/** Checks if a directory exists.
    @param dir directory name to check for
    @return true if directory exists
*/
bool vvToolshed::isDirectory(const char* path)
{
#ifdef _WIN32
  WIN32_FIND_DATAA fileInfo;
  HANDLE found;
  found = FindFirstFileA(path, &fileInfo);
  bool ret;

  if (found == INVALID_HANDLE_VALUE) return false;
  else
  {
    if(fileInfo.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY) ret = true;
    else ret = false;
    FindClose(found);
    return ret;
  }
#else
  struct stat buf;
  if (stat(path, &buf) == 0)
  {
    if (S_ISDIR(buf.st_mode)) return true;
  }
  return false;
#endif
}

//----------------------------------------------------------------------------
/** Figures out the size of a file in bytes.
    @param  filename file name including path
    @return file size in bytes or -1 on error
*/
long vvToolshed::getFileSize(const char* filename)
{
  FILE* fp;
  long size;

  fp = fopen(filename, "rb");
  if (fp==NULL) return -1;
  if (fseek(fp, 0L, SEEK_END) != 0)
  {
    fclose(fp);
    return -1;
  }
  size = ftell(fp);
  fclose(fp);
  return size;
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in an uchar data array.
    @param data        source data array
    @param elements    number of bytes in source array
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMax(const uchar* data, size_t elements, int* min, int* max)
{
  *min = 255;
  *max = 0;

  for (size_t i=0; i<elements; ++i)
  {
    if (data[i] > *max) *max = data[i];
    if (data[i] < *min) *min = data[i];
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in a 16 bit big endian data array.
    @param data        source data array
    @param elements    number of 16 bit elements in source array
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMax16bitHost(const uchar* data, size_t elements, int* min, int* max)
{
  int value;

  *min = 65535;
  *max = 0;
  size_t bytes = 2 * elements;

  for (size_t i=0; i<bytes; i+=2)
  {
    value = *(uint16_t *)(&data[i]);
    if (value > *max) *max = value;
    if (value < *min) *min = value;
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in an RGBA dataset, only
    considering the alpha component.
    @param data        source data array
    @param elements    number of RGBA elements in source array
    @return minimum and maximum of the alpha component in min and max
*/
void vvToolshed::getMinMaxAlpha(const uchar* data, size_t elements, int* min, int* max)
{
  *min = 255;
  *max = 0;
  size_t bytes = 4 * elements;

  for (size_t i=3; i<bytes; i+=4)
  {
    if (data[i] > *max) *max = data[i];
    if (data[i] < *min) *min = data[i];
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in a float data array.
    @param data        source data array
    @param elements    number of elements in source array
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMax(const float* data, size_t elements, float* min, float* max)
{
  *min = FLT_MAX;
  *max = -(*min);

  for (size_t i=0; i<elements; ++i)
  {
    if (data[i] > *max) *max = data[i];
    if (data[i] < *min) *min = data[i];
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in a data array and specify
    a value which is to be ignored, i.e., it does not change the determined
    minimum or maximum values.
    @param data        source data array
    @param elements    number of elements in source array
    @param ignore      value which is to be ignored (e.g, FLT_MAX)
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMaxIgnore(const float* data, size_t elements, float ignore,
float* min, float* max)
{
  *min = FLT_MAX;
  *max = -(*min);

  for (size_t i=0; i<elements; ++i)
  {
    if (data[i] != ignore)
    {
      if (data[i] > *max) *max = data[i];
      if (data[i] < *min) *min = data[i];
    }
  }
}

//----------------------------------------------------------------------------
/** Convert a sequence of uchar values to float.
    Be sure to have the floatArray allocated before this call!
    @param ucharArray  source array
    @param floatArray  destination array
    @param elements    number of uchar array elements to convert
    @return result in floatArray
*/
void vvToolshed::convertUChar2Float(const uchar* ucharArray, float* floatArray, int elements)
{
  int i;
  for (i=0; i<elements; ++i)
    floatArray[i] = (float)((double)ucharArray[i] / 255.0);
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to uchar.
    The uchar values cover a range of 0.0 to 1.0 of the float values.
    Be sure to have the ucharArray allocated before this call!
    @param floatArray  source array
    @param ucharArray  destination array
    @param elements    number of float array elements to convert
    @return result in ucharArray
*/
void vvToolshed::convertFloat2UChar(const float* floatArray,
uchar* ucharArray, int elements)
{
  for (int i=0; i<elements; ++i)
    ucharArray[i] = (uchar)(255.0 * floatArray[i]);
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to uchar.
    The uchar values will cover the range defined by the minimum and
    maximum float values.
    Be sure to have the ucharArray allocated before this call!
    @param floatArray  source array
    @param ucharArray  destination array
    @param elements    number of float array elements to convert
    @param min,max     minimum and maximum float values which will be assigned
                       to 0 and 255 respectively.
    @return result in ucharArray
*/
void vvToolshed::convertFloat2UCharClamp(const float* floatArray,
uchar* ucharArray, int elements, float min, float max)
{
  int i;

  if (min>=max)
    memset(ucharArray, 0, elements);
  else
  {
    for (i=0; i<elements; ++i)
      ucharArray[i] = (uchar)(255.0f * (floatArray[i] - min) / (max - min));
  }
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to 16 bit values.
    The 16 bit values will cover the range defined by the minimum and
    maximum float values.
    Be sure to have the ucharArray allocated before this call (requires
    elements * 2 bytes)!
    @param floatArray  source array
    @param ucharArray  destination array (16 bit values)
    @param elements    number of float array elements to convert
    @param min,max     minimum and maximum float values which will be assigned
                       to 0 and 65535 respectively.
    @return result in ucharArray
*/
void vvToolshed::convertFloat2ShortClamp(const float* floatArray,
uchar* ucharArray, int elements, float min, float max,
virvo::toolshed::serialization::EndianType endianness)
{
  int i;
  int shortValue;

  if (min>=max)
  {
    memset(ucharArray, 0, elements);
  }
  else
  {
    for (i=0; i<elements; ++i)
    {
      shortValue = int(65535.0f * (floatArray[i] - min) / (max - min));
      if (endianness == virvo::toolshed::serialization::VV_BIG_END)
      {
        ucharArray[2*i]   = (uchar)((shortValue >> 8) & 255);
        ucharArray[2*i+1] = (uchar)(shortValue & 255);
      }
      else
      {
        ucharArray[2*i] = (uchar)(shortValue & 255);
        ucharArray[2*i+1]   = (uchar)((shortValue >> 8) & 255);
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to uchar.
    The uchar values will cover the range defined by the maximum
    and minimum float values.
    The uchar value of 0 is set for all float values of the 'zero' value,
    and only for them.
    The minimum float value which is not 'zero' becomes an uchar value of 1.
    Be sure to have the ucharArray allocated before this call!
    @param floatArray  source array
    @param ucharArray  destination array
    @param elements    number of float array elements to convert
    @param min,max     minimum and maximum float values which will be assigned
to 1 and 255 respectively.
@param zero        float value which will become 0 in uchar array
@return result in ucharArray
*/
void vvToolshed::convertFloat2UCharClampZero(const float* floatArray,
uchar* ucharArray, int elements, float min, float max, float zero)
{
  int i;

  if (min>=max)
    memset(ucharArray, 0, elements);
  else
    for (i=0; i<elements; ++i)
  {
    if (floatArray[i] == zero)
      ucharArray[i] = (uchar)0;
    else
    {
      if (min==max)
        ucharArray[i] = (uchar)1;
      else
        ucharArray[i] = (uchar)((uchar)(254.0f * (floatArray[i] - min) / (max - min)) + (uchar)1);
    }
  }
}

//----------------------------------------------------------------------------
/** Compute the largest prime factor which is not the number itself.
    @param number number to examine (>1)
    @return largest prime factor, -1 on error
*/
int vvToolshed::getLargestPrimeFactor(const int number)
{
  int remainder;
  int factor = 2, largest = 1;

  if (number < 2) return -1;
  remainder = number;
  while (factor < remainder/2)
  {
    if ((remainder % factor) == 0)
    {
      remainder /= factor;
      largest = factor;
    }
    else
      ++factor;
  }
  if (largest==1) return 1;
  else return ts_max(remainder, largest);
}

int vvToolshed::round(float x)
{
  return x >= 0.0f ? std::floor(x + 0.5f) : std::ceil(x - 0.5f);
}

int vvToolshed::round(double x)
{
  return x >= 0.0 ? std::floor(x + 0.5) : std::ceil(x - 0.5);
}

//----------------------------------------------------------------------------
/** Initialize progress display.
    @param total total number of progress steps
*/
void vvToolshed::initProgress(int total)
{
  progressSteps = total;
  cerr << "     ";
}

//----------------------------------------------------------------------------
/** Print progress.
  Format: 'xxx %'.
  @param current current progress step
*/
void vvToolshed::printProgress(int current)
{
  int percent, i;

  if (progressSteps<2) percent = 100;
  else
    percent = 100 * current / (progressSteps - 1);
  for (i=0; i<5; ++i)
    cerr << (char)8;                              // ASCII 8 = backspace (BS)
  cerr << setw(3) << percent << " %";
}

//----------------------------------------------------------------------------
/** Run length encode (RLE) a sequence in memory.
  Encoding scheme: X is first data chunk (unsigned char).<UL>
  <LI>if X<128:  copy next X+1 chunks (literal run)</LI>
  <LI>if X>=128: repeat next chunk X-127 times (replicate run)</LI></UL>
  @param out  destination position in memory (must be _allocated_!)
  @param in   source location in memory
  @param size number of bytes to encode
  @param symbol_size  bytes per chunk
  @param space  number of bytes allocated for destination array.
                Encoding process is stopped when this number is reached.
  @param outsize number of bytes written to destination memory
  @return ok if data was written destination memory or an error type if there is not
enough destination memory or if an invalid data size was passed
@see decodeRLE
@author Michael Poehnl
*/
vvToolshed::ErrorType vvToolshed::encodeRLE(uint8_t* out, uint8_t* in, size_t size, size_t symbol_size, size_t space, size_t* outsize)
{
  int same_symbol=1;
  int diff_symbol=0;
  size_t src=0;
  size_t dest=0;
  bool same;
  size_t i;

  if ((size % symbol_size) != 0)
  {
    *outsize = 0;
    return VV_INVALID_SIZE;
  }

  while (src < (size - symbol_size))
  {
    same = true;
    for (i=0; i<symbol_size; i++)
    {
      if (in[src+i] != in[src+symbol_size+i])
      {
        same = false;
        break;
      }
    }
    if (same)
    {
      if (same_symbol == 129)
      {
        assert(dest<space);
        out[dest] = (uint8_t)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        same_symbol++;
        if (diff_symbol > 0)
        {
          assert(dest<space);
          out[dest] = (uint8_t)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
        if (same_symbol == 2)
        {
          if ((dest+1+symbol_size) > space)
          {
            *outsize = 0;
            return VV_OUT_OF_MEMORY;
          }
          memcpy(&out[dest+1], &in[src], symbol_size);
        }
      }
    }
    else
    {
      if (same_symbol > 1)
      {
        assert(dest<space);
        out[dest] = (uint8_t)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
        {
          outsize = 0;
          return VV_OUT_OF_MEMORY;
        }
        memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
        diff_symbol++;
        if (diff_symbol == 128)
        {
          assert(dest<space);
          out[dest] = (uint8_t)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
      }
    }
    src += symbol_size;
  }
  if (same_symbol > 1)
  {
    assert(dest<space);
    out[dest] = (uint8_t)(126+same_symbol);
    dest += symbol_size+1;
  }
  else
  {
    if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
    {
      *outsize = 0;
      return VV_OUT_OF_MEMORY;
    }
    memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
    diff_symbol++;
    out[dest] = (uint8_t)(diff_symbol-1);
    dest += 1+symbol_size*diff_symbol;
  }
  *outsize = dest;
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Decode a run length encoded (RLE) sequence.
  Data chunks of any byte aligned size can be processed.
  @param out  destination position in memory (_allocated_ space for max bytes)
  @param in   source location in memory
  @param size number of bytes in source array to decode
  @param symbol_size  bytes per chunk (e.g., to encode 24 bit RGB data, use bpc=3)
  @param space  number of allocated bytes in destination memory (for range checking)
  @param outsize number of bytes written to destination memory
  @return ok if bytes written to destination memory. If max would
          have been exceeded, an error type is returned
  @see encodeRLE
  @author Michael Poehnl
*/
vvToolshed::ErrorType vvToolshed::decodeRLE(uint8_t* out, uint8_t* in, size_t size, size_t symbol_size, size_t space, size_t* outsize)
{
  size_t src=0;
  size_t dest=0;
  size_t i, length;

  while (src < size)
  {
    length = (size_t)in[src];
    if (length > 127)
    {
      for(i=0; i<(length - 126); i++)
      {
        if ((dest + symbol_size) > space)
        {
          *outsize = 0;
          return VV_INVALID_SIZE;
        }
        memcpy(&out[dest], &in[src+1], symbol_size);
        dest += symbol_size;
      }
      src += 1+symbol_size;
    }
    else
    {
      length++;
      if ((dest + length*symbol_size) > space)
      {
        *outsize = 0;
        return VV_OUT_OF_MEMORY;
      }
      memcpy(&out[dest], &in[src+1], symbol_size*length);
      dest += length*symbol_size;
      src += 1+symbol_size*length;
    }
  }
  *outsize = dest;
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Run length encode (RLE) a sequence of 8 bit values in memory.
  Encoding scheme: X is first data byte (unsigned char).<UL>
  <LI>if X<128:  copy next X+1 bytes (literal run)</LI>
  <LI>if X>=128: repeat next byte X-127 times (replicate run)</LI></UL>
  @param dst  destination position in memory (must be _allocated_!)
  @param src  source location in memory
  @param len  number of bytes to encode
  @param max  number of bytes allocated in destination memory.
              Encoding process is stopped when this number is reached
  @return number of bytes written to destination memory
  @see decodeRLEFast
*/
size_t vvToolshed::encodeRLEFast(uint8_t* dst, uint8_t* src, size_t len, size_t max)
{
  size_t offset;                                  // start position of currently processed run in source array
  size_t index;                                   // index in source array
  size_t out;                                     // index in destination array
  size_t i;                                       // counter
  uint8_t cur;                                    // currently processed data byte

  offset = out = 0;
  while (offset < len)
  {
    index = offset;
    cur = src[index++];                           // load first data byte from source array
    while (index<len && index-offset<128 && src[index]==cur)
      index++;                                    // search for replicate run
    if (index-offset==1)                          // generate literal run
    {
      // Failed to "replicate" the current byte. See how many to copy.
      // Avoid a replicate run of only 2-pixels after a literal run. There
      // is no gain in this, and there is a risk of loss if the run after
      // the two identical pixels is another literal run. So search for
      // 3 identical pixels.
      while (index<len && index-offset<128 &&
          (src[index]!=src[index-1] || (index>1 && src[index]!=src[index-2])))
        index++;
      // Check why this run stopped. If it found two identical pixels, reset
      // the index so we can add a run. Do this twice: the previous run
      // tried to detect a replicate run of at least 3 pixels. So we may be
      // able to back up two pixels if such a replicate run was found.
      while (index<len && src[index]==src[index-1])
        index--;
      if (out < max)
        dst[out++] = (uint8_t)(index - offset - 1);
      for (i=offset; i<index; i++)
        if (out < max) dst[out++] = src[i];
    }
    else                                          // generate replicate run
    {
      if (out < max)
        dst[out++] = (uint8_t)(index - offset + 127);
      if (out < max)
        dst[out++] = cur;
    }
    offset = index;
  }
  return out;
}

//----------------------------------------------------------------------------
/** Decode a run length encoded (RLE) sequence of 8 bit values.
  @param dst  destination position in memory (must be _allocated_!)
  @param src  source location in memory
  @param len  number of bytes to decode
  @param max  number of allocated bytes in destination memory (for range checking)
  @return number of bytes written to destination memory. If max would
          have been exceeded, max+1 is returned
  @see encodeRLEFast
*/
size_t vvToolshed::decodeRLEFast(uint8_t* dst, uint8_t* src, size_t len, size_t max)
{
  size_t count;                                   // RLE counter
  size_t out=0;                                   // counter for written output bytes

  while (len > 0)
  {
    count = (int)*src++;
    if (count > 127)                              // replicate run?
    {
      count -= 127;                               // remove bias
      if (out+count <= max)                       // don't exceed allocated memory array
        memset(dst, *src++, count);
      else
      {
        if (out < max)                            // write as much as possible
          memset(dst, *src++, max-out);
        return max+1;
      }
      len -= 2;
    }
    else                                          // literal run
    {
      ++count;                                    // remove bias
      if (out+count <= max)                       // don't exceed allocated memory array
        memcpy(dst, src, count);
      else
      {
        if (out < max)                            // write as much as possible
          memcpy(dst, src, max-out);
        return max+1;
      }
      src += count;
      len -= count + 1;
    }
    dst += count;
    out += count;
  }
  return out;
}

//----------------------------------------------------------------------------
/** Get the number of (logical) system CPUs.
  @return number of processors, or -1 if unable to determine it
*/
int vvToolshed::getNumProcessors()
{
#ifdef _WIN32
  SYSTEM_INFO sysinfo;
  GetNativeSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
#elif defined(__hpux)
  struct pst_dynamic psd;
  if (pstat_getdynamic(&psd, sizeof(psd), (size_t)1, 0) != -1)
  {
    int nspu = psd.psd_proc_cnt;
    return nspu;
  }
  else
  {
    return 1;
  }
#else
  long numProcs;
#ifdef __sgi
  numProcs = sysconf(_SC_NPROC_CONF);
#else
  numProcs = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  if (numProcs < 1) return -1;
  else return numProcs;
#endif
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for a hue/saturation color chooser.
 Texture values are returned in data as 4 bytes per texel, bottom to top,
 ordered RGBARGBARGBA...
 @param width,height   width and height of texture [pixels]
 @param brightness     brightness of color values [0..1]
 @param data           pointer to allocated memory space providing width * height * 4 bytes
*/
void vvToolshed::makeColorBoardTexture(int width, int height, float brightness,
uchar* data)
{
  float h, s, v;                                  // hue, saturation, value
  float r, g, b;                                  // RGB
  float nx, ny;                                   // x and y normalized to range [0..1]
  float dx, dy;                                   // distance from center
  int   i = 0;                                    // index of current texel element
  int   x, y;                                     // current texel position

  for (y=0; y<height; ++y)
    for (x=0; x<width; ++x)
  {
    nx = (float)x / (float)(width-1);
    ny = (float)y / (float)(height-1);
    dx = 2.0f * nx - 1.0f;
    dy = 2.0f * ny - 1.0f;
    if ( (dx * dx + dy * dy) > 1.0f)
    {
      // Outer area is black:
      data[i++] = (uchar)0;                       // red
      data[i++] = (uchar)0;                       // green
      data[i++] = (uchar)0;                       // blue
      data[i++] = (uchar)255;                     // alpha
    }
    else
    {
      v = brightness;                             // circle area has requested brightness
      convertXY2HS(nx, ny, &h, &s);
      vvToolshed::HSBtoRGB(h, s, v, &r, &g, &b);
      data[i++] = (uchar)(r * 255.0f);            // red
      data[i++] = (uchar)(g * 255.0f);            // green
      data[i++] = (uchar)(b * 255.0f);            // blue
      data[i++] = (uchar)255;                     // alpha
    }
  }
}

//----------------------------------------------------------------------------
/** The given x|y coordinates of the mouse are translated to hue and
    saturation values.
 Mouse coordinate 0|0 is bottom left, 1|1 is top right.
 Hue and saturation values are in range [0..1].
*/
void vvToolshed::convertXY2HS(float x, float y, float* hue, float* saturation)
{
  float dx, dy;                                   // distance from center of x/y area

  // Determine hue:
  dx = x - 0.5f;
  dy = y - 0.5f;

  if (dx==0.0f)
  {
    if (dy>=0.0f) *hue = 0.0f;
    else *hue = 180.0f;
  }
  else if (dy==0.0f)
  {
    if (dx>0.0f) *hue = 90.0f;
    else *hue = 270.0f;
  }
  else
  {
    if      (dx>0.0f && dy>0.0f) *hue = atanf(dx / dy);
    else if (dx>0.0f && dy<0.0f) *hue = TS_PI * 0.5f + atanf(-dy / dx);
    else if (dx<0.0f && dy<0.0f) *hue = TS_PI + atanf(-dx / -dy);
    else                         *hue = TS_PI * 1.5f + atanf(dy / -dx);
  }
  *hue /= (2.0f * TS_PI);
  *hue = ts_clamp(*hue, 0.0f, 1.0f);

  // Determine saturation:
  dx *= 2.0f;
  dy *= 2.0f;
  *saturation = sqrtf(dx * dx + dy * dy);
  *saturation = ts_clamp(*saturation, 0.0f, 1.0f);
}

//----------------------------------------------------------------------------
/** The given hue and saturation values are converted to mouse x|y coordinates.
 Hue and saturation values are in range [0..1].
 Mouse coordinate 0|0 is bottom left, 1|1 is top right.
*/
void vvToolshed::convertHS2XY(float hue, float saturation, float* x, float* y)
{
  float angle;                                    // angle of point xy position within color circle
  float dx, dy;                                   // point position relative to circle midpoint

  angle = hue * 2.0f * TS_PI;
  dx = 0.5f * saturation * sinf(angle);
  dy = 0.5f * saturation * cosf(angle);
  *x = dx + 0.5f;
  *y = dy + 0.5f;
  *x = ts_clamp(*x, 0.0f, 1.0f);
  *y = ts_clamp(*y, 0.0f, 1.0f);
}

//----------------------------------------------------------------------------
/** Return an integer aligned to a power of two (dflt: 16).
 */
int vvToolshed::align(const int i, const int pot)
{
  const int alignment = sizeof(int) * pot;
  return (i + (alignment - 1)) & ~(alignment - 1);
}

//----------------------------------------------------------------------------
/** Make a float array system independent:
  convert each four byte float to unix-style (most significant byte first).
  @param numValues number of float values in array
  @param array     pointer to float array (size of array must be 4*numValues!)
*/
void vvToolshed::makeArraySystemIndependent(int numValues, float* array)
{
  uint8_t* buf;                                   // array pointer in uchar format
  int i;
  uint8_t tmp;                                    // temporary byte value from float array, needed for swapping

  assert(sizeof(float) == 4);
  if (virvo::serialization::getEndianness()==virvo::serialization::VV_BIG_END)  return;       // nothing needs to be done

  buf = (uchar*)array;
  for (i=0; i<numValues; ++i)
  {
    // Reverse byte order:
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
    buf += 4;
  }
}

//----------------------------------------------------------------------------
/** Make a system independent float array system dependent:
  convert each four byte float value back to system style.
  @param numValues number of float values in array
  @param array     pointer to float array (size of array must be 4*numValues!)
*/
void vvToolshed::makeArraySystemDependent(int numValues, float* array)
{
  // Swapping bytes is the same as above, therefore use the same code:
  makeArraySystemIndependent(numValues, array);
}

//----------------------------------------------------------------------------
/** Suspend process for a specific time. If milliseconds are not available
  on a specific system type, seconds are used (e.g., on Cray systems).
  @param msec suspension time [milliseconds]
*/
void vvToolshed::sleep(int msec)
{
#ifdef _WIN32
  Sleep(msec);
#elif CRAY
  sleep(msec / 1000);
#else
  usleep(msec * 1000);
#endif
}

//----------------------------------------------------------------------------
/** Resize a 2D image by resampling with nearest neighbor interpolation.
  Interleaved pixel format is assumed (RGBRGBRGB for 3 byte per pixel (bpp).
  If the pixel formats of source and destination image differ,
  the following approach is taken:
  - srcBPP==1: value will be replicated for first 3 destination pixels, 4th will be 0xFF
  - srcBPP < dstBPP: pixels are padded with 0xFF
  - srcBPP > dstBPP: rightmost components are dropped
  @param dstData must be _pre-allocated_ by the caller with
                 dstWidth*dstHeight*dstBPP bytes!
*/
void vvToolshed::resample(uchar* srcData, int srcWidth, int srcHeight, int srcBPP,
uchar* dstData, int dstWidth, int dstHeight, int dstBPP)
{
  int x, y, i, xsrc, ysrc, minBPP, dstBytes;
  int srcOffset, dstOffset;

  dstBytes = dstWidth * dstHeight * dstBPP;
                                                  // trivial
  if (srcWidth==dstWidth && srcHeight==dstHeight && srcBPP==dstBPP)
  {
    memcpy(dstData, srcData, dstBytes);
    return;
  }

  minBPP = ts_min(srcBPP, dstBPP);

  // Create a black opaque image as basis to work on:
  memset(dstData, 0, dstBytes);                   // black background
  if (dstBPP==4)                                  // make alpha channel opaque
  {
    for (i=0; i<dstBytes; i+=4)
    {
      *(dstData + i + 3) = 255;
    }
  }
  for (y=0; y<dstHeight; ++y)
  {
    for (x=0; x<dstWidth; ++x)
    {
      xsrc = (int)((float)(x * srcWidth)  / (float)dstWidth);
      ysrc = (int)((float)(y * srcHeight) / (float)dstHeight);
      xsrc = ts_clamp(xsrc, 0, srcWidth-1);
      ysrc = ts_clamp(ysrc, 0, srcHeight-1);
      srcOffset = srcBPP * (xsrc + ysrc * srcWidth);
      dstOffset = dstBPP * (x + y * dstWidth);
      if (srcBPP==1)
      {
        memset(dstData + dstOffset, *(srcData + srcOffset), ts_min(dstBPP, 3));
      }
      else
      {
        memcpy(dstData + dstOffset, srcData + srcOffset, minBPP);
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Blend two images together using maximum intensity projection (MIP).
  Both image must be of same width, height, and color depth.
  The color components of the pixels are interleaved.
  @param srcData pointer to source image
  @param width,height image size [pixels]
  @param bpp bytes per pixel
  @param dstData pointer to destination image
*/
void vvToolshed::blendMIP(uchar* srcData, int width, int height, int bpp, uchar* dstData)
{
  uchar* srcPtr;
  uchar* dstPtr;
  int i;

  srcPtr = srcData;
  dstPtr = dstData;
  for (i=0; i<width*height*bpp; ++i)
  {
    *dstPtr = ts_max(*srcPtr, *dstPtr);
    ++srcPtr;
    ++dstPtr;
  }
}

//----------------------------------------------------------------------------
/** @param path will contain the directory that the OS considers current
    @param maxChars specifies the size of the array path provides space for
*/
void vvToolshed::getCurrentDirectory(char* path, int maxChars)
{
  char* buf = new char[maxChars + 64];
#ifdef _WIN32
  GetCurrentDirectoryA(maxChars, buf);
#else
  if(!getcwd(buf, maxChars))
     std::cerr << "vvToolshed::getCurrentDirectory failed: " << strerror(errno) << std::endl;
#endif
  extractDirname(path, buf);
}

//----------------------------------------------------------------------------
/** @param path will contain the directory that the OS should consider
 */
void vvToolshed::setCurrentDirectory(const char* path)
{
#ifdef _WIN32
  SetCurrentDirectoryA(path);
#else
  if(chdir(path) == -1)
     std::cerr << "vvToolshed::setCurrentDirectory failed: " << strerror(errno) << std::endl;
#endif
}

//----------------------------------------------------------------------------
/** @param path will contain the directory that the current executable is
                located at
    @param maxChars specifies the size of the array path provides space for
*/
void vvToolshed::getProgramDirectory(char* path, int maxChars)
{
#ifdef _WIN32
  char* buf = new char[maxChars + 64];
  GetModuleFileNameA(NULL, buf, maxChars);
  extractDirname(path, buf);
#elif _LINUX64BIT                               // This code unfortunately doesn't work under 32 bit
  struct load_module_desc desc;
  dlget(-2, &desc, sizeof(desc));
  strcpy(path, dlgetname(&desc, sizeof(desc), NULL, NULL, NULL));
#else
  // TODO: this is not the correct path if the file was started from somewhere else
  if(!getcwd(path, maxChars))
  {
     std::cerr << "vvToolshed::getProgramDirectory failed: " << strerror(errno) << std::endl;
  }
#endif
}

//----------------------------------------------------------------------------
/** Decode Base64 encoded text to its original binary format.
    Source: http://www.fourmilab.ch/webtools/base64/
    @param src pointer to ASCII source data
    @param numChars number of ASCII characters in source array
    @param dst pointer to _allocated_ binary destination data.
               At least sizeof(src) / 4 * 3 bytes must be allocated.
    @return true on success, false on error
*/
bool vvToolshed::decodeBase64(const char* src, int numChars, uchar* dst)
{
  uchar dtable[256];
  uchar a[4], b[4], o[3];
  int i, j, c, srcIndex=0, dstIndex=0;

  for(i=0;   i<255;  ++i) dtable[i]= uchar(0x80);
  for(i='A'; i<='I'; ++i) dtable[i]= 0+(i-'A');
  for(i='J'; i<='R'; ++i) dtable[i]= 9+(i-'J');
  for(i='S'; i<='Z'; ++i) dtable[i]= 18+(i-'S');
  for(i='a'; i<='i'; ++i) dtable[i]= 26+(i-'a');
  for(i='j'; i<='r'; ++i) dtable[i]= 35+(i-'j');
  for(i='s'; i<='z'; ++i) dtable[i]= 44+(i-'s');
  for(i='0'; i<='9'; ++i) dtable[i]= 52+(i-'0');
  dtable[int('+')]= 62;
  dtable[int('/')]= 63;
  dtable[int('=')]= 0;

  for(;;)
  {
    // Loop over four characters, in which three bytes are encoded:
    for(i=0; i<4; ++i)
    {
      c = src[srcIndex];
      ++srcIndex;

      if (srcIndex==numChars)                     // end of source array reached?
      {
        if (i>0)                                  // end reached in midst of source quadruple?
        {
          cerr << "vvToolshed::decodeBase64: Input array incomplete." << endl;
          return false;
        }
        else return true;                         // finished
      }
      if(dtable[c] & 0x80)
      {
        cerr << "vvToolshed::decodeBase64: Illegal character in input file." << endl;
        return false;
      }
      a[i] = (uchar)c;
      b[i] = (uchar)dtable[c];
    }
    o[0] = (b[0] << 2) | (b[1] >> 4);
    o[1] = (b[1] << 4) | (b[2] >> 2);
    o[2] = (b[2] << 6) |  b[3];
    i = (a[2] == '=') ? 1 : ((a[3] == '=') ? 2 : 3);

    for (j=0; j<i; ++j) dst[dstIndex+j] = o[j];
    if(i<3) return true;                          // finished if incomplete triple has been written
  }
}

//----------------------------------------------------------------------------
/** Interpolate linearly given a x/y value pair and a slope.
 @param x1, y1  point on straight line (x/y value pair)
 @param slope   slope of line
 @param x       x value for which to find interpolated y value
 @return interpolated y value
*/
float vvToolshed::interpolateLinear(float x1, float y1, float slope, float x)
{
  return (y1 + slope * (x - x1));
}

//----------------------------------------------------------------------------
/** Interpolate linearly between two x/y value pairs.
 @param x1, y1  one x/y value pair
 @param x2, y2  another x/y value pair
 @param x       x value for which to find interpolated y value
 @return interpolated y value
*/
float vvToolshed::interpolateLinear(float x1, float y1, float x2, float y2, float x)
{
  if (x1==x2) return ts_max(y1, y2);              // on equal x values: return maximum value
  if (x1 > x2)                                    // make x1 less than x2
  {
    ts_swap(x1, x2);
    ts_swap(y1, y2);
  }
  return (y2 - y1) * (x - x1) / (x2 - x1) + y1;
}

//----------------------------------------------------------------------------
/** Make a list of files and folders in a path.
  @param path location to search in
  @return fileNames and folderNames, alphabetically sorted
*/
bool vvToolshed::makeFileList(std::string& path, std::list<std::string>& fileNames, std::list<std::string>& folderNames)
{
#ifdef _WIN32
  WIN32_FIND_DATAA fileInfo;
  HANDLE fileHandle;
  string searchPath;

  searchPath = path + "\\*";
  fileHandle = FindFirstFileA(searchPath.c_str(), &fileInfo);
  if (fileHandle == INVALID_HANDLE_VALUE)
  {
    cerr << "FindFirstFile failed: " << GetLastError() << endl;
    return false;
  }
  do  // add all files and directory in specified path to lists
  {
    cerr << "file=" << fileInfo.cFileName << endl;
    if(fileInfo.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY)
    {
      folderNames.push_back(fileInfo.cFileName);
    }
    else
    {
      fileNames.push_back(fileInfo.cFileName);
    }
  }
  while (FindNextFileA(fileHandle, &fileInfo));   // was another file found?
  FindClose(fileHandle);
#else
  DIR* dirHandle;
  struct dirent* entry;
  struct stat statbuf;

  dirHandle = opendir(path.c_str());
  if (dirHandle==NULL)
  {
    cerr << "Cannot read directory: " << path << endl;
    return false;
  }
  if (chdir(path.c_str()) != 0)
  {
    const int PATH_SIZE = 256;
    char cwd[PATH_SIZE];
    cerr << "Cannot chdir to " << path <<
      ". Searching for files in " << getcwd(cwd, PATH_SIZE) << endl;
  }
  while ((entry=readdir(dirHandle)) != NULL)
  {
    stat(entry->d_name, &statbuf);
    if (S_ISDIR(statbuf.st_mode))      // found a folder?
    {
      folderNames.push_back(entry->d_name);
    }
    else      // found a file
    {
      fileNames.push_back(entry->d_name);
    }
  }

  fileNames.sort();
  folderNames.sort();

  closedir(dirHandle);
#endif
  return true;
}

//----------------------------------------------------------------------------
/** Return the next string in the list after a given one
  @param listStrings list of strings to search
  @param knownEntry we're looking for the entry after this one
  @param nextEntry returned string, or "" if nothing found
  @return true if next string found, or false if not
*/
bool vvToolshed::nextListString(list<string>& listStrings, string& knownEntry, string& nextEntry)
{
  list<string>::const_iterator iter;
  for (iter=listStrings.begin(); iter != listStrings.end(); ++iter)
  {
    if ((*iter)==knownEntry)
    {
      ++iter;
      if (iter == listStrings.end()) return false;  // end of list reached
      else
      {
        nextEntry = *iter;
        return true;
      }
    }
  }
  nextEntry = "";
  return false;   // knownEntry has not been found
}

void vvToolshed::quickSort(int* numbers, int arraySize)
{
  qSort(numbers, 0, arraySize - 1);
}

void vvToolshed::qSort(int* numbers, int left, int right)
{
  int pivot, l_hold, r_hold;

  l_hold = left;
  r_hold = right;
  pivot  = numbers[left];
  while (left < right)
  {
    while ((numbers[right] >= pivot) && (left < right))
    {
      --right;
    }
    if (left != right)
    {
      numbers[left] = numbers[right];
      ++left;
    }
    while ((numbers[left] <= pivot) && (left < right))
    {
      ++left;
    }
    if (left != right)
    {
      numbers[right] = numbers[left];
      --right;
    }
  }
  numbers[left] = pivot;
  pivot = left;
  left  = l_hold;
  right = r_hold;
  if (left < pivot)  qSort(numbers, left, pivot-1);
  if (right > pivot) qSort(numbers, pivot+1, right);
}

float vvToolshed::meanAbsError(float* val1, float* val2, const int numValues)
{
  float result = 0.0f;
  int i;

  for (i = 0; i < numValues; ++i)
  {
    result += fabs(val1[i] - val2[i]);
  }

  result /= numValues;

  return result;
}

float vvToolshed::meanError(float* val1, float* val2, const int numValues)
{
  float result = 0.0f;
  int i;

  for (i = 0; i < numValues; ++i)
  {
    result += (val1[i] - val2[i]);
  }

  result /= numValues;

  return result;
}

float vvToolshed::meanSqrError(float* val1, float* val2, const int numValues)
{
  float err = meanError(val1, val2, numValues);
  return err * err;
}

std::string vvToolshed::file2string(const std::string& filename)
{
  std::ifstream source(filename.c_str());

  if (!source.is_open())
    throw std::runtime_error("File not found: \"" + filename + "\"");

  return std::string(
    std::istreambuf_iterator<char>(source.rdbuf()),
    std::istreambuf_iterator<char>()
    );
}

int vvToolshed::string2Int(const char* str)
{
  int result = 0;
  size_t i = 0;
  size_t len = strlen(str);
  int sign = 1;

  char tmp = str[i];

  if (tmp == '-')
  {
    sign = -1;
    ++i;
    tmp = str[i];
  }

  while (i < len)
  {
    if ((tmp >= '0') && (tmp <= '9'))
    {
      result *= 10;
      result += static_cast<int>(tmp-'0');
    }
    ++i;
    tmp = str[i];
  }

  result *= sign;
  return result;
}

//----------------------------------------------------------------------------
/** Write pixels (texture, framebuffer content, etc.) to a ppm file
    for debugging purposes.
*/
void vvToolshed::pixels2Ppm(uchar* pixels, const int width, const int height,
                            const char* fileName, const vvToolshed::Format format)
{
  std::ofstream ppm;
  ppm.open(fileName);

  ppm << "P3" << endl;
  ppm << width << " " << height << endl;
  ppm << "255" << endl;

  if (format == VV_RGB)
  {
    int ite = 0;
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        ppm << (int)pixels[ite + 0] << " ";
        ppm << (int)pixels[ite + 1] << " ";
        ppm << (int)pixels[ite + 2] << " ";

        ite += 3;
      }
      ppm << endl;
    }
  }
  else if (format == VV_RGBA)
  {
    int ite = 0;
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        ppm << (int)pixels[ite + 0] << " ";
        ppm << (int)pixels[ite + 1] << " ";
        ppm << (int)pixels[ite + 2] << " ";

        ite += 4;
      }
      ppm << endl;
    }
  }
  else if (format == VV_LUMINANCE)
  {
    int ite = 0;
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        ppm << (int)pixels[ite] << " ";
        ppm << (int)pixels[ite] << " ";
        ppm << (int)pixels[ite] << " ";

        ite += 1;
      }
      ppm << endl;
    }
  }
}

//----------------------------------------------------------------------------
/** Write pixels (texture, framebuffer content, etc.) to a ppm file
    for debugging purposes. Overloaded version for float textures,
    floats will be converted to bytes before rendering.
*/
void vvToolshed::pixels2Ppm(float* pixels, const int width, const int height,
                            const char* fileName, const vvToolshed::Format format)
{
  int size = 0;

  switch (format)
  {
  case VV_RGB:
    size = width * height * 3;
    break;
  case VV_RGBA:
    size = width * height * 4;
    break;
  case VV_LUMINANCE:
    size = width * height;
    break;
  default:
    break;
  }

  uchar* tmp = new uchar[size];

  float max = 0.0f;
  float min = 255.0f;

  for (int i = 0; i < size; ++i)
  {
    tmp[i] = static_cast<uchar>(pixels[i] * 255.0f);

    if ((int)tmp[i] < min)
    {
      min = static_cast<float>(tmp[i]);
    }

    if ((int)tmp[i] > max)
    {
      max = static_cast<float>(tmp[i]);
    }
  }

  // Clamp / scale up to 0..255.
  if (true)
  {
    const float diffInv = 1.0f / (max - min);
    for (int i = 0; i < size; ++i)
    {
      float f = static_cast<float>(tmp[i]) - min;
      tmp[i] = static_cast<uchar>(f * diffInv * max);
    }
  }

  pixels2Ppm(tmp, width, height, fileName, format);
  delete[] tmp;
}

//----------------------------------------------------------------------------
/** Parse port from an url specified like this:
    <hostname>[:port]
    If no port is found, -1 is returned.
    @param url  The url to parse the port from
    @return  The port if specified, -1 otherwise.
*/
int vvToolshed::parsePort(std::string const& url)
{
    std::string::size_type pos = url.find_last_of(":");

    // No colon found, no port!
    if (pos == std::string::npos)
        return -1;

    // Colon found
    // Check if there is a number following the colon

    std::string strPort = url.substr(pos + 1);

    char const* begin = strPort.c_str();
    char* end = 0;

    unsigned long port = strtoul(begin, &end, 10);

    // Note:
    // strtoul doesn't set errno if no valid conversion has been performed and simply returns 0.
    // If the correct value is out of the range of representable values, ULONG_MAX is returned,
    // an the global variable errno is set to ERANGE

    if (port == 0 || port >= 65536)
        return -1;

    return port; // A valid port has been found
}

//----------------------------------------------------------------------------
/** Remove port from an url specified as follows:
    <hostname>[:port]
    No checks will be performed if there is actually
    a port substring. The caller is responsible for
    performing this check.
    @param url  The url to strip.
    @return The stripped result.
*/
std::string vvToolshed::stripPort(std::string const& url)
{
    assert( parsePort(url) != -1 ); // Not checked for a valid port?!

    return url.substr(0, url.find_last_of(":"));
}

void vvToolshed::printBacktrace()
{
#if defined(__linux__) || defined(__APPLE__)
  const int MaxFrames = 16;

  void* buffer[MaxFrames] = { 0 };
  const int count = backtrace(buffer, MaxFrames);

  char** symbols = backtrace_symbols(buffer, count);

  if (symbols)
  {
    for (int n=0; n<count; ++n)
    {
      fprintf(stderr, "%s\n", symbols[n]);
    }
    free(symbols);
  }
#endif
}

std::vector<std::string> virvo::toolshed::entryList(std::string const& dir)
{
    std::vector<std::string> result;

    namespace filesystem = boost::filesystem;

    filesystem::path path(dir);

    if (filesystem::exists(path) && filesystem::is_directory(path))
    {

        filesystem::directory_iterator end;
        for (filesystem::directory_iterator it(path); it != end; ++it)
        {
            if ( filesystem::is_regular_file(it->status()) )
            {
                filesystem::path p = filesystem::canonical(*it);
                result.push_back( p.filename().string() );
            }
        }

    }

    return result;
}

bool virvo::toolshed::startsWith(std::string const& string, std::string const& prefix)
{
    if (string.size() < prefix.size())
        return false;

    if (prefix.empty())
        return true;

    return memcmp(string.c_str(), prefix.c_str(), prefix.size()) == 0;
}

bool virvo::toolshed::endsWith(std::string const& string, std::string const& suffix)
{
    if (string.size() < suffix.size())
        return false;

    if (suffix.empty())
        return true;

    return memcmp(string.c_str() + (string.size() - suffix.size()), suffix.c_str(), suffix.size()) == 0;
}

//============================================================================
// serialization namespace
//============================================================================

size_t virvo::serialization::read(uint8_t* src, uint8_t* val)
{
  *val = *src;
  return 1;
}

size_t virvo::serialization::read(uint8_t* src, uint16_t* val, virvo::serialization::EndianType end)
{
  if (end == VV_LITTLE_END)
  {
    *val = static_cast<uint16_t>((uint16_t)src[0] + (uint16_t)src[1] * (uint16_t)256);
  }
  else
  {
    *val = static_cast<uint16_t>((uint16_t)src[0] * (uint16_t)256 + (uint16_t)src[1]);
  }
  return 2;
}

size_t virvo::serialization::read(uint8_t* src, uint32_t* val, virvo::serialization::EndianType end)
{
  if (end == VV_LITTLE_END)
  {
    *val = (uint32_t)src[3] * (uint32_t)0x1000000
         + (uint32_t)src[2] * (uint32_t)0x10000
         + (uint32_t)src[1] * (uint32_t)0x100
         + (uint32_t)src[0] * (uint32_t)0x1;
  }
  else
  {
    *val = (uint32_t)src[0] * (uint32_t)0x1000000
         + (uint32_t)src[1] * (uint32_t)0x10000
         + (uint32_t)src[2] * (uint32_t)0x100
         + (uint32_t)src[3] * (uint32_t)0x1;
  }
  return 4;
}

size_t virvo::serialization::read(uint8_t* src, uint64_t* val, virvo::serialization::EndianType end)
{
  if (end == VV_LITTLE_END)
  {
    *val = static_cast<uint64_t>(src[7]) * 0x100000000000000
         + static_cast<uint64_t>(src[6]) * 0x1000000000000
         + static_cast<uint64_t>(src[5]) * 0x10000000000
         + static_cast<uint64_t>(src[4]) * 0x100000000
         + static_cast<uint64_t>(src[3]) * 0x1000000
         + static_cast<uint64_t>(src[2]) * 0x10000
         + static_cast<uint64_t>(src[1]) * 0x100
         + static_cast<uint64_t>(src[0]) * 0x1;
  }
  else
  {
    *val = static_cast<uint64_t>(src[0]) * 0x100000000000000
         + static_cast<uint64_t>(src[1]) * 0x1000000000000
         + static_cast<uint64_t>(src[2]) * 0x10000000000
         + static_cast<uint64_t>(src[3]) * 0x100000000
         + static_cast<uint64_t>(src[4]) * 0x1000000
         + static_cast<uint64_t>(src[5]) * 0x10000
         + static_cast<uint64_t>(src[6]) * 0x100
         + static_cast<uint64_t>(src[7]) * 0x1;
  }
  return 8;
}

size_t virvo::serialization::read(uint8_t* src, float* val, virvo::serialization::EndianType end)
{
  uint8_t* ptr;
  uint8_t  tmp;

  assert(sizeof(float) == 4);
  memcpy(val, src, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    ptr = (uint8_t*)val;
    tmp = ptr[0]; ptr[0] = ptr[3]; ptr[3] = tmp;
    tmp = ptr[1]; ptr[1] = ptr[2]; ptr[2] = tmp;
  }
  return 4;
}


size_t virvo::serialization::read(FILE* src, uint8_t* val)
{
  size_t retval;

  retval = fread(val, 1, 1, src);
  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::read(FILE*, uint8_t*) failed";
    return 0;
  }
  return retval;
}

size_t virvo::serialization::read(FILE* src, uint16_t* val, virvo::serialization::EndianType end)
{
  uint8_t buf[2];
  size_t retval;

  retval = fread(buf, 2, 1, src);

  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::read(FILE*, uint16_t*) failed";
    return 0;
  }
  if (end == VV_LITTLE_END)
  {
    *val = (uint16_t)buf[0] + (uint16_t)buf[1] * (uint16_t)256;
  }
  else
  {
    *val = (uint16_t)buf[0] * (uint16_t)256 + (uint16_t)buf[1];
  }
  return 2;
}

size_t virvo::serialization::read(FILE* src, uint32_t* val, virvo::serialization::EndianType end)
{
  uint8_t buf[4];

  size_t retval;
  retval = fread(buf, 4, 1, src);
  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::read(FILE*, uint32_t*) failed";
    return 0;
  }

  if (end == VV_LITTLE_END)
  {
    *val = (uint32_t)buf[3] * 0x1000000
         + (uint32_t)buf[2] * 0x10000 
         + (uint32_t)buf[1] * 0x100
         + (uint32_t)buf[0] * 0x1;
  }
  else
  {
    *val = (uint32_t)buf[0] * 0x1000000
         + (uint32_t)buf[1] * 0x10000
         + (uint32_t)buf[2] * 0x100
         + (uint32_t)buf[3] * 0x1;
  }
  return 4;
}

size_t virvo::serialization::read(FILE* src, uint64_t* val, virvo::serialization::EndianType end)
{
  uint8_t buf[8];

  size_t retval = fread(buf, 8, 1, src);
  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::read(FILE*, uint64_t*) failed";
    return 0;
  }

  if (end == VV_LITTLE_END)
  {
    *val = static_cast<uint64_t>(buf[7]) * 0x100000000000000
         + static_cast<uint64_t>(buf[6]) * 0x1000000000000
         + static_cast<uint64_t>(buf[5]) * 0x10000000000
         + static_cast<uint64_t>(buf[4]) * 0x100000000
         + static_cast<uint64_t>(buf[3]) * 0x1000000
         + static_cast<uint64_t>(buf[2]) * 0x10000
         + static_cast<uint64_t>(buf[1]) * 0x100
         + static_cast<uint64_t>(buf[0]) * 0x1;
  }
  else
  {
    *val = static_cast<uint64_t>(buf[0]) * 0x10000000000000
         + static_cast<uint64_t>(buf[1]) * 0x1000000000000
         + static_cast<uint64_t>(buf[2]) * 0x10000000000
         + static_cast<uint64_t>(buf[3]) * 0x100000000
         + static_cast<uint64_t>(buf[4]) * 0x1000000
         + static_cast<uint64_t>(buf[5]) * 0x10000
         + static_cast<uint64_t>(buf[6]) * 0x100
         + static_cast<uint64_t>(buf[7]) * 0x1;
  }
  return 8;
}

size_t virvo::serialization::read(FILE* src, float* val, virvo::serialization::EndianType end)
{
  uint8_t *buf;
  uint8_t tmp;

  size_t retval;
  retval = fread(val, 4, 1, src);

  if (retval != 1)
  {
    VV_LOG(0) << "vvToolshed::readFloat fread failed";
    return 0;
  }

  if (getEndianness() != end)
  {
    // Reverse byte order:
    buf = (uint8_t*)val;
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
  }
  return 4;
}

size_t virvo::serialization::read(std::ifstream& src, uint32_t* val, virvo::serialization::EndianType end)
{
  uint8_t buf[4];

  src.read(reinterpret_cast< char* >(buf), 4);
  if (!src)
  {
    VV_LOG(0) << "virvo::serialization::read(std::ifstream&, uint32_t*) failed";
    return 0;
  }

  if (end == VV_LITTLE_END)
  {
    *val = (uint32_t)buf[3] * 0x1000000
         + (uint32_t)buf[2] * 0x10000
         + (uint32_t)buf[1] * 0x100
         + (uint32_t)buf[0] * 0x1;
  }
  else
  {
    *val = (uint32_t)buf[0] * 0x1000000
         + (uint32_t)buf[1] * 0x10000
         + (uint32_t)buf[2] * 0x100
         + (uint32_t)buf[3] * 0x1;
  }
  return 4;
}

size_t virvo::serialization::read(std::ifstream& src, uint64_t* val, virvo::serialization::EndianType end)
{
  uint8_t buf[8];

  src.read(reinterpret_cast< char* >(buf), 8);
  if (!src)
  {
    VV_LOG(0) << "virvo::serialization::read(std::ifstream&, uint64_t*) failed";
    return 0;
  }

  if (end == VV_LITTLE_END)
  {
    *val = (uint64_t)buf[7] * 0x100000000000000UL
         + (uint64_t)buf[6] * 0x1000000000000UL
         + (uint64_t)buf[5] * 0x10000000000UL
         + (uint64_t)buf[4] * 0x100000000UL
         + (uint64_t)buf[3] * 0x1000000UL
         + (uint64_t)buf[2] * 0x10000UL
         + (uint64_t)buf[1] * 0x100UL
         + (uint64_t)buf[0] * 0x1UL;
  }
  else
  {
    *val = (uint64_t)buf[0] * 0x100000000000000UL
         + (uint64_t)buf[1] * 0x1000000000000UL
         + (uint64_t)buf[2] * 0x10000000000UL
         + (uint64_t)buf[3] * 0x100000000UL
         + (uint64_t)buf[4] * 0x1000000UL
         + (uint64_t)buf[5] * 0x10000UL
         + (uint64_t)buf[6] * 0x100UL
         + (uint64_t)buf[7] * 0x1UL;
  }
  return 8;
}

size_t virvo::serialization::write(uint8_t* dst, uint8_t val)
{
  *dst = val;
  return sizeof(uint8_t);
}

size_t virvo::serialization::write(uint8_t* dst, uint16_t val, virvo::serialization::EndianType end)
{
  if (end == VV_LITTLE_END)
  {
    dst[0] = (uint8_t)(val & 0xFF);
    dst[1] = (uint8_t)(val >> 8);
  }
  else
  {
    dst[0] = (uint8_t)(val >> 8);
    dst[1] = (uint8_t)(val & 0xFF);
  }
  return 2 * sizeof(uint8_t);
}

size_t virvo::serialization::write(uint8_t* dst, uint32_t val, virvo::serialization::EndianType end)
{
  if (end == VV_LITTLE_END)
  {
    dst[0] = (uint8_t)(val & 0xFF);
    dst[1] = (uint8_t)((val >> 8)  & 0xFF);
    dst[2] = (uint8_t)((val >> 16) & 0xFF);
    dst[3] = (uint8_t)(val  >> 24);
  }
  else
  {
    dst[0] = (uint8_t)(val  >> 24);
    dst[1] = (uint8_t)((val >> 16) & 0xFF);
    dst[2] = (uint8_t)((val >> 8)  & 0xFF);
    dst[3] = (uint8_t)(val & 0xFF);
  }
  return sizeof(uint32_t);
}

size_t virvo::serialization::write(uint8_t* dst, uint64_t val, virvo::serialization::EndianType end)
{
  if (end == VV_LITTLE_END)
  {
    dst[0] = static_cast<uint8_t>(val & 0xFF);
    dst[1] = static_cast<uint8_t>((val >> 8)  & 0xFF);
    dst[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    dst[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    dst[4] = static_cast<uint8_t>((val >> 32) & 0xFF);
    dst[5] = static_cast<uint8_t>((val >> 40) & 0xFF);
    dst[6] = static_cast<uint8_t>((val >> 48) & 0xFF);
    dst[7] = static_cast<uint8_t>((val >> 56) & 0xFF);
  }
  else
  {
    dst[0] = static_cast<uint8_t>(val  >> 56);
    dst[1] = static_cast<uint8_t>((val >> 48) & 0xFF);
    dst[2] = static_cast<uint8_t>((val >> 40) & 0xFF);
    dst[3] = static_cast<uint8_t>((val >> 32) & 0xFF);
    dst[4] = static_cast<uint8_t>((val >> 24) & 0xFF);
    dst[5] = static_cast<uint8_t>((val >> 16) & 0xFF);
    dst[6] = static_cast<uint8_t>((val >>  8) & 0xFF);
    dst[7] = static_cast<uint8_t>(val & 0xFF);
  }
  return sizeof(uint64_t);
}

size_t virvo::serialization::write(uint8_t* dst, float val, virvo::serialization::EndianType end)
{
  uint8_t tmp;

  assert(sizeof(float) == 4);
  memcpy(dst, &val, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    tmp = dst[0]; dst[0] = dst[3]; dst[3] = tmp;
    tmp = dst[1]; dst[1] = dst[2]; dst[2] = tmp;
  }
  return sizeof(float);
}

size_t virvo::serialization::write(FILE* dst, uint8_t val)
{
  size_t retval = fwrite(&val, 1, 1, dst);
  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::write(FILE*, uint8_t) failed";
    return 0;
  }
  return 1;
}

size_t virvo::serialization::write(FILE* dst, uint16_t val, virvo::serialization::EndianType end)
{
  uint8_t buf[2];

  if (end == VV_LITTLE_END)
  {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)(val >> 8);
  }
  else
  {
    buf[0] = (uint8_t)(val >> 8);
    buf[1] = (uint8_t)(val & 0xFF);
  }
  size_t retval = fwrite(buf, 2, 1, dst);
  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::write(FILE*, uint16_t) failed";
    return 0;
  }
  return 2;
}

size_t virvo::serialization::write(FILE* dst, uint32_t val, virvo::serialization::EndianType end)
{
  uint8_t buf[4];

  if (end == VV_LITTLE_END)
  {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8)  & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)(val  >> 24);
  }
  else
  {
    buf[0] = (uint8_t)(val  >> 24);
    buf[1] = (uint8_t)((val >> 16) & 0xFF);
    buf[2] = (uint8_t)((val >> 8)  & 0xFF);
    buf[3] = (uint8_t)(val & 0xFF);
  }
  size_t retval = fwrite(buf, 4, 1, dst);
  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::write(FILE*, uint32_t) failed";
    return 0;
  }
  return 4;
}

size_t virvo::serialization::write(FILE* dst, uint64_t val, virvo::serialization::EndianType end)
{
  uint8_t buf[8];

  if (end == VV_LITTLE_END)
  {
    buf[0] = static_cast<uint8_t>(val & 0xFF);
    buf[1] = static_cast<uint8_t>((val >> 8)  & 0xFF);
    buf[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    buf[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    buf[4] = static_cast<uint8_t>((val >> 32) & 0xFF);
    buf[5] = static_cast<uint8_t>((val >> 40) & 0xFF);
    buf[6] = static_cast<uint8_t>((val >> 48) & 0xFF);
    buf[7] = static_cast<uint8_t>((val >> 56) & 0xFF);
  }
  else
  {
    buf[0] = static_cast<uint8_t>(val  >> 56);
    buf[1] = static_cast<uint8_t>((val >> 48) & 0xFF);
    buf[2] = static_cast<uint8_t>((val >> 40) & 0xFF);
    buf[3] = static_cast<uint8_t>((val >> 32) & 0xFF);
    buf[4] = static_cast<uint8_t>((val >> 24) & 0xFF);
    buf[5] = static_cast<uint8_t>((val >> 16) & 0xFF);
    buf[6] = static_cast<uint8_t>((val >> 8)  & 0xFF);
    buf[7] = static_cast<uint8_t>(val & 0xFF);
  }
  size_t retval = fwrite(buf, 8, 1, dst);
  if (retval != 1)
  {
    VV_LOG(0) << "virvo::serialization::write(FILE*, uint64_t) failed";
    return 0;
  }
  return 8;
}

size_t virvo::serialization::write(FILE* dst, float val, virvo::serialization::EndianType end)
{
  uint8_t* buf;
  uint8_t tmp;

  if (getEndianness() != end)
  {
    // Reverse byte order:
    buf = (uint8_t*)&val;
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
  }

  size_t retval=fwrite(&val, 4, 1, dst);
  if (retval != 1)
  {
    VV_LOG(0) << "vvToolshed::writeFloat fwrite failed";
    return 0;
  }
  return 4;
}


//----------------------------------------------------------------------------
/// Main function for standalone test mode.
#ifdef VV_STANDALONE
int main(int, char**)
{
#ifdef _WIN32
  char* pathname={"c:\\user\\testfile.dat"};
#else
  char* pathname={"/usr/local/testfile.dat"};
#endif
  char  teststring[256];

  cout << "ts_max(2,9)  = " << ts_max(2,9)  << endl;
  cout << "ts_min(2,9)  = " << ts_min(2,9)  << endl;
  cout << "ts_abs(-7)   = " << ts_abs(-7)   << endl;
  cout << "ts_sgn(-9.1) = " << ts_sgn(-9.1) << endl;
  cout << "ts_zsgn(0.0) = " << ts_zsgn(0.0) << endl;
  cout << "ts_zsgn(-2)  = " << ts_zsgn(-2)  << endl;
  cout << "ts_clamp(1.2, 1.0, 2.0)  = " << ts_clamp(1.2f, 1.0f, 2.0f)  << endl;
  cout << "ts_clamp(-0.5, 1.0, 2.0)  = " << ts_clamp(0.5f, 1.0f, 2.0f)  << endl;
  cout << "ts_clamp(2.1, 1.0, 2.0)  = " << ts_clamp(2.1f, 1.0f, 2.0f)  << endl;

  cout << "isSuffix(" << pathname << "), 'Dat' = ";
  if (vvToolshed::isSuffix(pathname, "Dat") == true)
    cout << "true" << endl;
  else
    cout << "false" << endl;

  cout << "isSuffix(" << pathname << "), 'data' = ";
  if (vvToolshed::isSuffix(pathname, "data") == true)
    cout << "true" << endl;
  else
    cout << "false" << endl;

  vvToolshed::extractFilename(teststring, pathname);
  cout << "extractFilename(" << pathname << ") = " << teststring << endl;

  vvToolshed::extractDirname(teststring, pathname);
  cout << "extractDirname(" << pathname << ") = " << teststring << endl;

  vvToolshed::extractExtension(teststring, pathname);
  cout << "extractExtension(" << pathname << ") = " << teststring << endl;

  vvToolshed::extractBasename(teststring, pathname);
  cout << "extractBasename(" << pathname << ") = " << teststring << endl;

  cout << "getTextureSize(84) = " << vvToolshed::getTextureSize(84) << endl;

  char* testData = {"ABABACACACABABABACABCD"};
  char encoded[100];
  char decoded[100];
  size_t len;
  size_t bpc = 2;
  cout << "Unencoded: " << testData << endl;
  bool success = vvToolshed::encodeRLE((uint8_t*)encoded, (uint8_t*)testData, strlen(testData), bpc, 100, &len);
  success &= vvToolshed::decodeRLE((uint8_t*)decoded, (uint8_t*)encoded, len, bpc, 100, &len);
  decoded[len] = '\0';
  cout << "Decoded:   " << decoded << endl;

  return 1;
}
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
