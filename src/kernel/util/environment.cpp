/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <string>
#include <sstream>
#include <algorithm>

#include "environment.h"
#include "coFileUtil.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#if defined(__APPLE__)
//#include <CoreServices/CoreServices.h>
#include <CoreFoundation/CoreFoundation.h>
//#include <Carbon/Carbon.h>
#endif

//#define DEBUG

namespace covise
{

static void setvar(const char *variable, const std::string &value)
{
    std::stringstream ss;
    ss << variable << "=" << value;
    putenv(strdup(ss.str().c_str()));
#ifdef DEBUG
    std::cerr << ss.str() << std::endl;
#endif
}

static void splitpath(const std::string &value, std::vector<std::string> *components)
{
#ifdef _WIN32
    const char *sep = ";";
#else
    const char *sep = ":";
#endif

    std::string::size_type begin = 0;
    do
    {
        std::string::size_type end = value.find(sep, begin);

        std::string c;
        if (end != std::string::npos)
        {
            c = value.substr(begin, end - begin);
            ++end;
        }
        else
        {
            c = value.substr(begin);
        }
        begin = end;

        if (!c.empty())
            components->push_back(c);
    } while (begin != std::string::npos);
}

static void prependpath(const char *variable, std::string value)
{
#ifdef _WIN32
    const char *sep = ";";
#else
    const char *sep = ":";
#endif

    std::string path = value;
    if (const char *current = getenv(variable))
    {
        std::vector<std::string> components;
        splitpath(current, &components);

        for (int i = 0; i < components.size(); ++i)
        {
            std::string s1 = components[i];
            std::string s2 = value;
            std::replace(s1.begin(), s1.end(), '\\', '/'); // replace all '\\' with '/'
            std::replace(s2.begin(), s2.end(), '\\', '/'); // replace all '\\' with '/'
            if (s1 != s2)
            {
                path += sep;
                path += components[i];
            }
        }
    }
    setvar(variable, path);
}

static void addpath(const char *variable, std::string value)
{
#ifdef _WIN32
    const char *sep = ";";
#else
    const char *sep = ":";
#endif

    bool found = false;
    if (const char *current = getenv(variable))
    {
        std::vector<std::string> components;
        splitpath(current, &components);

        for (int i = 0; i < components.size(); ++i)
        {
            std::string s1 = components[i];
            std::string s2 = value;
            std::replace(s1.begin(), s1.end(), '\\', '/'); // replace all '\\' with '/'
            std::replace(s2.begin(), s2.end(), '\\', '/'); // replace all '\\' with '/'
            if (s1 == s2)
            {
                found = true;
                break;
            }
        }
    }

    if (!found)
    {
        std::string path = value;
        if (const char *current = getenv(variable))
        {
            path += sep;
            path += current;
        }
        setvar(variable, path);
    }
}


#ifdef __APPLE__
std::string getBundlePath()
{
    std::string path;

    // this is inspired by OpenSceneGraph: osgDB/FilePath.cpp

    // Start with the the Bundle PlugIns directory.

    // Get the main bundle first. No need to retain or release it since
    //  we are not keeping a reference
    CFBundleRef myBundle = CFBundleGetMainBundle();

    if (myBundle != NULL)
    {
        // CFBundleGetMainBundle will return a bundle ref even if
        //  the application isn't part of a bundle, so we need to check
        //  if the path to the bundle ends in ".app" to see if it is a
        //  proper application bundle. If it is, the plugins path is added
        CFURLRef urlRef = CFBundleCopyBundleURL(myBundle);
        if (urlRef)
        {
            char bundlePath[1024];
            if (CFURLGetFileSystemRepresentation(urlRef, true, (UInt8 *)bundlePath, sizeof(bundlePath)))
            {
                size_t len = strlen(bundlePath);
                if (len > 4 && !strcmp(bundlePath + len - 4, ".app"))
                {
                    path = bundlePath;
                }
            }
            CFRelease(urlRef); // docs say we are responsible for releasing CFURLRef
        }
    }

    return path;
}
#endif

bool setupEnvironment(int argc, char *argv[])
{
    std::string covisedir;
    char *cdir = getenv("COVISEDIR");
    if (cdir)
        covisedir = cdir;

    setvar("ARCHSUFFIX", ARCHSUFFIX);

    // determine complete path to executable
    std::string executable;
#ifdef _WIN32
    char buf[2000];
    DWORD sz = GetModuleFileName(NULL, buf, sizeof(buf));
    if (sz != 0)
    {
        executable = buf;
    }
    else
    {
        std::cerr << "setupEnvironment(): GetModuleFileName failed - error: " << GetLastError() << std::endl;
    }
#else
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf));
    if (len != -1)
    {
        executable = std::string(buf, len);
    }
    else if (argc >= 1 && coDirectory::current()->path())
    {
        bool found = false;
        std::string cwd = coDirectory::current()->path();
        if (!strchr(argv[0], '/'))
        {
            if (const char *path = getenv("PATH"))
            {
                std::vector<std::string> components;
                splitpath(path, &components);
                for (int i = 0; i < components.size(); ++i)
                {
                    std::string component = components[i];
                    if (component[0] != '/')
                        component = cwd + "/" + component;

                    if (coDirectory *dir = coDirectory::open(component.c_str()))
                    {
                        for (int j = 0; j < dir->count(); ++j)
                        {
                            if (dir->is_exe(j) && !dir->is_directory(j) && !strcmp(dir->name(j), argv[0]))
                            {
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found)
                    {
                        executable = component + "/" + argv[0];
                        break;
                    }
                }
            }
        }
    }
#endif

#ifdef DEBUG
    std::cerr << "setupEnvironment(): executable=" << executable << std::endl;
#endif

    // guess COVISEDIR
    if (!executable.empty())
    {
        std::string dir = executable;
#ifdef _WIN32
        std::string::size_type idx = dir.find_last_of("\\/");
#else
        std::string::size_type idx = dir.rfind('/');
#endif
        if (idx == std::string::npos)
        {
            dir = coDirectory::current()->path();
        }

        for (;;)
        {
            bool archsuffixfound = false, pythonfound = false;
#ifdef _WIN32
            std::string::size_type idx = dir.find_last_of("\\/");
#else
            std::string::size_type idx = dir.rfind('/');
#endif

            if (idx == std::string::npos)
                break;

            dir = executable.substr(0, idx);
#ifdef DEBUG
            std::cerr << "setupEnvironment(): checking directory " << dir << std::endl;
#endif

            if (coDirectory *cd = coDirectory::open(dir.c_str()))
            {
                for (int i = 0; i < cd->count(); ++i)
                {
                    if (!strcmp(cd->name(i), ARCHSUFFIX))
                        archsuffixfound = true;
                    if (!strcmp(cd->name(i), "Python"))
                        pythonfound = true;
                    if (pythonfound && archsuffixfound)
                        break;
                }
            }
            if (pythonfound && archsuffixfound)
            {
                if (!dir.empty())
                {
                    covisedir = dir;
#ifdef DEBUG
                    std::cerr << "setupEnvironment(): COVISEDIR determined to be " << covisedir << std::endl;
#endif
                }
                break;
            }
        }
    }

#ifdef __APPLE__
    std::string bundlepath = getBundlePath();
    if (!bundlepath.empty())
        covisedir = bundlepath + "/Contents/Resources";
#endif

    if (covisedir.empty())
    {
        std::cerr << "setupEnvironment(): COVISEDIR not set and could not be determined" << std::endl;
        return false;
    }

    setvar("COVISEDIR", covisedir);

    const char *vv_shader_path = getenv("VV_SHADER_PATH");
    std::string vv1 = covisedir + "/share/covise/shader";
    std::string vv2 = covisedir + "/src/3rdparty/deskvox/virvo/shader";
    if (coFile::exists((vv1 + "/vv_texrend.fsh").c_str())) {
        setvar("VV_SHADER_PATH", vv1);
    } else if (coFile::exists((vv2 + "/vv_texrend.fsh").c_str())) {
        setvar("VV_SHADER_PATH", vv2);
    } else if (!vv_shader_path) {
        std::cerr << "VV_SHADER_PATH not set and Virvo shaders not found in " << vv1 << " and " << vv2 << ", using " << vv1 << std::endl;
        setvar("VV_SHADER_PATH", vv1);
    }
    if (vv_shader_path && strcmp(vv_shader_path, getenv("VV_SHADER_PATH")))
        std::cerr << "setupEnvironment: overriding VV_SHADER_PATH from " << vv_shader_path << " to " << getenv("VV_SHADER_PATH") << std::endl;

    addpath("COVISE_PATH", covisedir);
#ifdef _WIN32
    prependpath("PATH", covisedir + "\\" + ARCHSUFFIX + "\\bin");
#endif

    return true;
}
}
