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

#ifndef VV_DYNLIB_H
#define VV_DYNLIB_H

#include "vvplatform.h"

#ifdef __sgi
#include <dlfcn.h>
#endif
#ifdef __hpux
#include <dl.h>
#endif

#ifdef __hpux
typedef shl_t VV_SHLIB_HANDLE;
#elif _WIN32
typedef HINSTANCE VV_SHLIB_HANDLE;
#else
typedef void *VV_SHLIB_HANDLE;
#endif

#include "vvexport.h"

/** This class encapsulates the functionality of dynamic library loading.
  @author Uwe Woessner
*/
class VIRVOEXPORT vvDynLib
{
  public:
    static char* error(void);
    static VV_SHLIB_HANDLE open(const char* filename, int mode);
    static void* sym(VV_SHLIB_HANDLE handle, const char* symbolname);
    static void  (*glSym(const char* symbolname))(void);
    static int close(VV_SHLIB_HANDLE handle);
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
