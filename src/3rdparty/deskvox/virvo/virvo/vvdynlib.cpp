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

#define GLX_GLXEXT_LEGACY

#include <iostream>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#undef USE_NEXTSTEP

#ifdef USE_NEXTSTEP
#  include <mach-o/dyld.h>
#elif defined(__linux__) || defined(LINUX) || defined(__hpux) || defined(__APPLE__) || defined(__sun)
#include <dlfcn.h>
#endif

#include "vvopengl.h"

#include "vvdynlib.h"

using namespace std;

int vvDynLib::close(VV_SHLIB_HANDLE handle)
{
#ifdef _WIN32
  FreeLibrary (handle);
  return 1;

#elif __hpux
  // HP-UX 10.x and 32-bit 11.00 do not pay attention to the ref count when
  // unloading a dynamic lib.  So, if the ref count is more than 1, do not
  // unload the lib.  This will cause a library loaded more than once to
  // not be unloaded until the process runs down, but that's life.  It's
  // better than unloading a library that's in use.
  // So far as I know, there's no way to decrement the refcnt that the kernel
  // is looking at - the shl_descriptor is a copy of what the kernel has, not
  // the actual struct.
  // On 64-bit HP-UX using dlopen, this problem has been fixed.
  struct shl_descriptor  desc;
  if (shl_gethandle_r(handle, &desc) == -1)
    return -1;
  if (desc.ref_count > 1)
    return 1;
#if defined(__GNUC__) || __cplusplus >= 199707L
  shl_unload(handle);
#else
  cxxshl_unload(handle);
#endif                                          /* aC++ vs. Hp C++ */
  return 1;

#else

#ifdef __sun4
  // SunOS4 does not automatically call _fini()!
  void *ptr;
  ptr = sym(handle, "_fini");

  if (ptr != 0)
    (*((int (*)(void)) ptr)) ();                  // Call _fini hook explicitly.
#endif

  dlclose(handle);
  return 1;
#endif
}

char* vvDynLib::error()
{
#ifdef __hpux
  return strerror(errno);
#elif _WIN32
  static char buf[128];
  FormatMessageA (FORMAT_MESSAGE_FROM_SYSTEM,
    NULL,
    GetLastError (),
    0,
    buf,
    sizeof buf,
    NULL);
  return buf;
#else
  return dlerror();
#endif
}

VV_SHLIB_HANDLE vvDynLib::open(const char* filename, int mode)
{
  void* handle;

#ifdef _WIN32
  handle = LoadLibraryA (filename);
#elif __hpux
#if defined(__GNUC__) || __cplusplus >= 199707L
  handle = shl_load(filename, mode, 0L);
#else
  handle = cxxshl_load(filename, mode, 0L);
#endif
#else
  handle = dlopen(filename, mode);
#endif

  if(handle == NULL)
  {
    cerr << error() << endl;
  }

#ifdef __sun4
  if (handle != 0)
  {
    void *ptr;
    // Some systems (e.g., SunOS4) do not automatically call _init(), so
    // we'll have to call it manually.

    ptr = sym(handle, "_init");

                                                  // Call _init hook explicitly.
    if (ptr != 0 && (*((int (*)(void)) ptr)) () == -1)
    {
      // Close down the handle to prevent leaks.
      close(handle);
      return 0;
    }
  }
#endif

#ifdef _WIN32
  mode = mode;                                    // prevent warning
#endif

  return (VV_SHLIB_HANDLE)handle;
}

void* vvDynLib::sym(VV_SHLIB_HANDLE handle, const char* symbolname)
{
#ifdef _WIN32
  return (void *)GetProcAddress(handle, symbolname);
#elif __hpux
  void *value;
  int status;
  shl_t _handle = handle;
  status = shl_findsym(&_handle, symbolname, TYPE_UNDEFINED, &value);
  return status == 0 ? value : NULL;
#else
  return dlsym (handle, symbolname);
#endif
}

// #define USE_DYNLIB_FOR_GL // this has problems
void (*vvDynLib::glSym(const char *symbolname))(void)
{
#if defined(_WIN32)
  return (void (*)()) wglGetProcAddress(symbolname);
#elif defined(USE_DYNLIB_FOR_GL) || defined(__sgi)
  VV_SHLIB_HANDLE dynLib = open("libGL.so", RTLD_NOW);
  void (*func)() = (void (*)()) sym(dynLib, symbolname);
  close(dynLib);
  return func;
#elif defined(__APPLE__)
#ifndef USE_NEXTSTEP
  VV_SHLIB_HANDLE dynLib = open(NULL, RTLD_NOW);
  void (*func)() = (void (*)()) sym(dynLib, symbolname);
  close(dynLib);
  return func;
#else
  NSSymbol symbol;
  // Prepend a '_' for the Unix C symbol mangling convention
  char *mangledSymbolName = new char[strlen(symbolname) + 2];
  strcpy(mangledSymbolName + 1, symbolname);
  mangledSymbolName[0] = '_';
  symbol = NULL;
  if (NSIsSymbolNameDefined (mangledSymbolName))
    symbol = NSLookupAndBindSymbol (mangledSymbolName);
  delete[] (mangledSymbolName);
  return symbol ? (void (*)(void))NSAddressOfSymbol (symbol) : NULL;
#endif
#elif defined(GLX_ARB_get_proc_address)
  return (void (*)())  glXGetProcAddressARB((const GLubyte *)symbolname);
#else
  return (void (*)()) glXGetProcAddress((const GLubyte *)symbolname);
#endif
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
