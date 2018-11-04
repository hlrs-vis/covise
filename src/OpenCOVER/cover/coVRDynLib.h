/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_DYNAMIC_LIB_H
#define _CO_DYNAMIC_LIB_H

/*! \file
 \brief  dynamic library loading

 \author (C) 
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coTypes.h>
#include <string>
#ifdef __hpux
#include <dl.h>
#endif

#ifdef __hpux
typedef shl_t CO_SHLIB_HANDLE;
#elif _WIN32
#include <windows.h>
#define RTLD_LAZY 1
typedef HINSTANCE CO_SHLIB_HANDLE;
#else
typedef void *CO_SHLIB_HANDLE;
#endif

//#define SGIDLADD
#ifndef _WIN32
#define SVR4_DYNAMIC_LINKING
#endif
namespace opencover
{
//
//
//
class COVEREXPORT coVRDynLib
{

public:
    static const char *dlerror(void);
    static CO_SHLIB_HANDLE dlopen(const char *filename, bool showErrors = true);
    static CO_SHLIB_HANDLE dlopen(const std::string &filename, bool showErrors = true);
    static void *dlsym(CO_SHLIB_HANDLE handle, const char *symbolname);
    static int dlclose(CO_SHLIB_HANDLE handle);
    static std::string libName(const std::string &name);

private:
    coVRDynLib()
    {
    }
    ~coVRDynLib()
    {
    }
};
}
#endif
