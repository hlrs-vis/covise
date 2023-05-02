/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <errno.h>
#include <util/common.h>

#ifndef _WIN32
#include <strings.h>
#endif

#include <util/environment.h>

#if defined(__linux__) || defined(__APPLE__) || defined(__sgi) || defined(__FreeBSD__)
#include <dlfcn.h>
#endif

#include "coVRDynLib.h"
#include "coVRPluginSupport.h"
#include <sstream>

//
using namespace covise;
using namespace opencover;
// This class encapsulates the functionality of dynamic library loading
//
//
int coVRDynLib::dlclose(CO_SHLIB_HANDLE handle)
{

#if defined(SVR4_DYNAMIC_LINKING)
    ::dlclose(handle);
    return 1;
#endif

#if defined(CO_sun4)
    // SunOS4 does not automatically call _fini()!
    void *ptr;
    ptr = ::dlsym(handle, "_fini");

    if (ptr != 0)
        (*((int (*)(void))ptr))(); // Call _fini hook explicitly.

#elif defined(_WIN32)
    ::FreeLibrary(handle);
    return 1;

#elif defined(__hpux)
    // HP-UX 10.x and 32-bit 11.00 do not pay attention to the ref count when
    // unloading a dynamic lib.  So, if the ref count is more than 1, do not
    // unload the lib.  This will cause a library loaded more than once to
    // not be unloaded until the process runs down, but that's life.  It's
    // better than unloading a library that's in use.
    // So far as I know, there's no way to decrement the refcnt that the kernel
    // is looking at - the shl_descriptor is a copy of what the kernel has, not
    // the actual struct.
    // On 64-bit HP-UX using dlopen, this problem has been fixed.
    struct shl_descriptor desc;
    if (shl_gethandle_r(handle, &desc) == -1)
        return -1;
    if (desc.ref_count > 1)
        return 1;
#if defined(__GNUC__) || __cplusplus >= 199707L
    ::shl_unload(handle);
#else
    ::cxxshl_unload(handle);
#endif /* aC++ vs. Hp C++ */
    return 1;

#else
    cerr << "Dynamic Linking is not supported on this platform" << endl;
    return 0;
#endif
}

const char *coVRDynLib::dlerror(void)
{

#if defined(SVR4_DYNAMIC_LINKING)
    const char *err = ::dlerror();
    if (err)
    {
        return err;
    }
    else
    {
        return "";
    }

#elif defined(__hpux)
    return ::strerror(errno);

#elif defined(_WIN32)
    static char buf[128];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM,
                   NULL,
                   GetLastError(),
                   0,
                   buf,
                   sizeof buf,
                   NULL);
    return buf;
#else
    return "Dynamic Linking is not supported on this platform";
#endif
}

CO_SHLIB_HANDLE coVRDynLib::dlopen(const std::string &filename, bool showErrors)
{
    return dlopen(filename.c_str(), showErrors);
}

CO_SHLIB_HANDLE try_dlopen(const char *filename, bool showErrors)
{
    const int mode = RTLD_LAZY;

    CO_SHLIB_HANDLE handle = 0;
#if defined(SGIDLADD)
    handle = ::sgidladd(filename, mode);
#elif defined(SVR4_DYNAMIC_LINKING)
    handle = ::dlopen(filename, mode);
#elif defined(_WIN32)
    handle = LoadLibraryA(filename);
#elif defined(__hpux)
#if defined(__GNUC__) || __cplusplus >= 199707L
    handle = shl_load(filename, mode, 0L);
#else
    handle = cxxshl_load(filename, mode, 0L);
#endif
#endif /* SGIDLADD */

    if (handle == NULL)
    {
        if (cover->debugLevel(2) && showErrors)
            cerr << "coVRDynLib::try_dlopen(" << filename << ") failed: " << coVRDynLib::dlerror() << endl;
    }
    else
    {
        if (cover->debugLevel(3))
            cerr << "loaded " << filename << endl;
    }

    return handle;
}



CO_SHLIB_HANDLE coVRDynLib::dlopen(const char *filename, bool showErrors)
{
    CO_SHLIB_HANDLE handle = NULL;
    char buf[800];
    bool absolute = filename[0] == '/';
#ifdef _WIN32
    const char separator[] = ";";
#else
    const char separator[] = ":";
#endif

    std::vector<std::string> tried_files;
#ifdef __APPLE__
    std::string bundlepath = getBundlePath();
    if (!absolute && !bundlepath.empty())
    {
        snprintf(buf, sizeof(buf), "%s/Contents/PlugIns/%s", bundlepath.c_str(), filename);
        handle = try_dlopen(buf, showErrors);
        tried_files.push_back(buf);
    }
#endif

    const char *covisepath = getenv("COVISE_PATH");
    const char *archsuffix = getenv("ARCHSUFFIX");

    if (!absolute && covisepath && archsuffix && !handle)
    {
        char *cPath = new char[strlen(covisepath) + 1];
        strcpy(cPath, covisepath);
        char *dirname = strtok(cPath, separator);
        while (dirname != NULL)
        {
#ifdef _WIN32
            sprintf(buf, "%s\\%s\\lib\\OpenCOVER\\plugins\\%s", dirname, archsuffix, filename);
#else
            snprintf(buf, sizeof(buf), "%s/%s/lib/OpenCOVER/plugins/%s", dirname, archsuffix, filename);
#endif
            handle = try_dlopen(buf, showErrors);
            tried_files.push_back(buf);
            if (handle)
                break;

            dirname = strtok(NULL, separator);
        }
        delete[] cPath;
    }

    if (handle == NULL)
    {
        handle = try_dlopen(filename, showErrors);
        tried_files.push_back(filename);
    }

#ifdef CO_sun4
    if (handle != 0)
    {
        void *ptr;
        // Some systems (e.g., SunOS4) do not automatically call _init(), so
        // we'll have to call it manually.

        ptr = ::dlsym(handle, "_init");

        // Call _init hook explicitly.
        if (ptr != 0 && (*((int (*)(void))ptr))() == -1)
        {
            // Close down the handle to prevent leaks.
            ::dlclose(handle);
            return 0;
        }
    }
#endif /* CO_sun4 */

    if (handle == NULL && showErrors)
    {
        cerr << "coVRDynLib::dlopen() error: " << dlerror() << endl;
        cerr << "tried files:" << endl;
        for (size_t i=0; i<tried_files.size(); ++i)
        {
            cerr << "   " << tried_files[i] << endl;
        }
    }
    return handle;
}

void *coVRDynLib::dlsym(CO_SHLIB_HANDLE handle, const char *symbolname)
{

#if defined(SVR4_DYNAMIC_LINKING)

#if defined(LACKS_POSIX_PROTOTYPES)
    return ::dlsym(handle, (char *)symbolname);
#elif defined(ASM_SYMBOL_IN_DLSYM)
    int l = strlen(symbolname) + 2;
    char *asm_symbolname;
    asm_symbolname = new char[l];
    strcpy(asm_symbolname, "_");
    strcpy(asm_symbolname + 1, symbolname);
    void *_result;
    _result = ::dlsym(handle, asm_symbolname);
    delete[] asm_symbolname;
    return _result;

#else
    return ::dlsym(handle, symbolname);
#endif /* LACKS_POSIX_PROTOTYPES */

#elif defined(_WIN32)
    return (void *)::GetProcAddress(handle, symbolname);

#elif defined(__hpux)

    void *value;
    int status;
    shl_t _handle = handle;
    status = ::shl_findsym(&_handle, symbolname, TYPE_UNDEFINED, &value);
    return status == 0 ? value : NULL;

#else
    cerr << "Platform doesn't support dynamic linking" << endl;
    return 0;
#endif
}

std::string coVRDynLib::libName(const std::string &name)
{

    std::string libname;

    if (name[0] == '/')
    {
        libname = name;
    }
    else
    {
        stringstream str;
#if defined(_WIN32)
        str << name << ".dll";
#elif defined(__APPLE__)
        str << "lib" << name << ".so";
#else
        str << "lib" << name << ".so";
#endif
        libname = str.str();
    }

    return libname;
}
