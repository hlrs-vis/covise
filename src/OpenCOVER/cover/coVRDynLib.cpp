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

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
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

#elif defined(_WIN32)
    ::FreeLibrary(handle);
    return 1;
#endif /* aC++ vs. Hp C++ */
    return 1;

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
#elif defined(__GNUC__) || __cplusplus >= 199707L
    handle = shl_load(filename, mode, 0L);
#else
    handle = cxxshl_load(filename, mode, 0L);
#endif

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
