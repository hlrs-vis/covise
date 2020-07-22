///////////////////////////////////////////////////////////////////////////////////////////////////
// LibGizmo
// File Name : 
// Creation : 10/01/2012
// Author : Cedric Guillemet
// Description : LibGizmo
//
///Copyright (C) 2012 Cedric Guillemet
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//of the Software, and to permit persons to whom the Software is furnished to do
///so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
///FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 


#ifndef ZBASEDEFS_H__
#define ZBASEDEFS_H__

#ifdef WIN32
#include <windows.h>
#include <stdlib.h>
#endif


typedef unsigned long tulong;
typedef long tlong;
typedef double tdouble;

typedef char tchar;
typedef unsigned char tuchar;

typedef short tshort;
typedef unsigned short tushort;

typedef unsigned char uint8;
typedef char int8;
typedef unsigned short uint16;
typedef short int16;
typedef unsigned int uint32;
typedef int int32;

typedef unsigned short TClassID;

typedef unsigned int uint;
#define SAFE_DELETE(x) if (x) { delete x; x = NULL; }
#define    FLOAT_EPSILON    float(1.192092896e-07)    // Smallest positive number x, such that x+1.0 is not equal to 1.0
const float        RealEpsilon=FLOAT_EPSILON*16.f;


#define ZMAX(a,b)            (((a) > (b)) ? (a) : (b))
#define ZMIN(a,b)            (((a) < (b)) ? (a) : (b))

inline bool isPowerOf2(unsigned int n)
{
    return n == 1 || (n & (n-1)) == 0;
}

#include <vector>
#include <list>
#include <map>
#include <string>

#define tarray std::vector
#define tlist std::list

// targa header

typedef struct TargaHeader_t {
    tuchar    IDLength;
    tuchar    ColormapType;
    tuchar    ImageType;
    tuchar    ColormapSpecification[5];
    tushort    XOrigin;
    tushort    YOrigin;
    tushort    ImageWidth;
    tushort    ImageHeight;
    tuchar    PixelDepth;
    tuchar    ImageDescriptor;
} TargaHeader_t;

// macros

#define foreach( i, c, type )\
  if (!c.empty()) for( std::vector<type>::iterator i = c.begin(); i != c.end(); ++i )

#define foreach_const( i, c, type )\
  if (!c.empty()) for( std::vector<type>::const_iterator i = c.begin(); i != c.end(); ++i )

#define foreachmap( i, c, type )\
  if (!c.empty()) for( std::map<type>::iterator i = c.begin(); i != c.end(); ++i )

#define foreachmap_const( i, c, type )\
  if (!c.empty()) for( std::map<type>::const_iterator i = c.begin(); i != c.end(); ++i )

void Zexit(int aRet);


#ifdef WIN32
typedef CRITICAL_SECTION ZCriticalSection_t;
typedef HWND WindowHandle_t;
typedef HANDLE ThreadHandle_t;

inline char* GetCurrentDirectory(int bufLength, char *pszDest)
{
    return (char*)GetCurrentDirectoryA(bufLength, pszDest);
}

#endif

#ifdef LINUX
#include <pthread.h>

typedef pthread_mutex_t ZCriticalSection_t;
typedef int WindowHandle_t;
typedef pthread_t ThreadHandle_t;

#define TRUE 1
#define FALSE 0

typedef struct RECT
{
    int left, right, bottom, top;
} RECT;


#define _MAX_PATH 256
#define MAX_PATH _MAX_PATH

#define stricmp strcasecmp
#define _stricmp strcasecmp
#define sscanf_s sscanf
#define sprintf_s snprintf
#define OutputDebugString printf
#define OutputDebugStringA printf

inline void strlwr(char *)
{
}
#include<iostream>

#include<iomanip>

#include<fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

inline char* GetCurrentDirectory(int bufLength, char *pszDest)
{
    return getcwd(pszDest, bufLength);
}

inline void ZeroMemory(void *pDest, int aSize)
{
    memset(pDest, 0, aSize);
}

inline unsigned long GetCurrentThreadId()
{
    return (unsigned long) pthread_self();//getpid();
}

#endif

#ifdef MAC_OS
#import <CoreServices/CoreServices.h>

typedef MPCriticalRegionID ZCriticalSection_t;
typedef int WindowHandle_t;
typedef pthread_t ThreadHandle_t;

#define TRUE 1
#define FALSE 0

typedef struct RECT
	{
		int left, right, bottom, top;
	} RECT;


#define _MAX_PATH 256
#define MAX_PATH _MAX_PATH

#define stricmp strcasecmp
#define _stricmp strcasecmp
#define sscanf_s sscanf
#define sprintf_s snprintf
#define OutputDebugString printf
#define OutputDebugStringA printf

inline void strlwr(char *pszBuf)
{
}
#include<iostream>

#include<iomanip>

#include<fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

inline char* GetCurrentDirectory(int bufLength, char *pszDest)
{
    return getcwd(pszDest, bufLength);
}

inline void ZeroMemory(void *pDest, int aSize)
{
    memset(pDest, 0, aSize);
}

inline unsigned long GetCurrentThreadId()
{
    return (unsigned long) pthread_self();//getpid();
}

// DX compatibility
/*
typedef void IDirect3DDevice9;
typedef void IDirect3DTexture9;
*/


#endif

#endif
