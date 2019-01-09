/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COV_WINCOMPAT_H
#define COV_WINCOMPAT_H

#include <stdarg.h>
#include <string.h>
#include <stdio.h>

#ifndef WINCOMPATINLINE
#ifdef __cplusplus
#define WINCOMPATINLINE inline
#else
#define WINCOMPATINLINE static
#endif
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <float.h>
#if !defined(__MINGW32__)
// don't do this, otherwise std::isnan does not work#define isnan _isnan
namespace std
{
    inline int isnan(double X){return _isnan(X);};
    inline int isinf(double X){return !_finite(X);};
    inline int finite(double X){return _finite(X);};
}
#endif
#include <winsock2.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AMD_MEAN
#include <windows.h>
#undef WIN32_LEAN_AMD_MEAN
#else
#include <windows.h>
#endif
#ifndef _WIN32_WCE
#include <sys/types.h>
#include <sys/timeb.h>
#include <direct.h>
#endif
#ifndef SURFACE

WINCOMPATINLINE char *basename(char *pathptr)
{
    static char fullPath[MAX_PATH];
    static char dummystr[] = ".";
    char *filebn = 0;
    char *retval = dummystr;
    if (GetFullPathName(pathptr, MAX_PATH, fullPath, &filebn) != 0)
    {
        if (*filebn != 0)
            retval = filebn;
    }
    return retval;
}
#endif

#if !defined(__MINGW32__)
WINCOMPATINLINE int strcasecmp(const char *s1, const char *s2)
{
    return stricmp(s1, s2);
}

WINCOMPATINLINE int strncasecmp(const char *s1, const char *s2, size_t n)
{
    return strnicmp(s1, s2, n);
}
#endif

WINCOMPATINLINE void sleep(int time)
{
    Sleep(time * 1000);
}

WINCOMPATINLINE void usleep(int time)
{
    Sleep(time / 1000);
}

#if !defined(__MINGW32__)
WINCOMPATINLINE int strncasecmp(const char *s1, const char *s2, size_t n)
{
    return strnicmp(s1, s2, n);
}

#if defined(_MSC_VER) && (_MSC_VER < 1900)
WINCOMPATINLINE int snprintf(char *str, size_t size, const char *format, ...)
{
    int result;
    va_list ap;
    va_start(ap, format);
    result = _vsnprintf(str, size, format, ap);
    va_end(ap);
    str[size - 1] = '\0';
    return result;
}
#endif
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1020)
WINCOMPATINLINE int vsnprintf(char *str, size_t size, const char *format, va_list ap)
{
    int result = _vsnprintf(str, size, format, ap);
    str[size - 1] = '\0';
    return result;
}
#endif

#if defined(__MINGW32__)
#include <sys/time.h>
#else
#ifdef timezone
#undef timezone
#endif
struct timezone
{
    int tz_minuteswest;
    int tz_dsttime;
};

WINCOMPATINLINE int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    struct __timeb64 currentTime;
    (void)tz;
#if _MSC_VER < 1400
    _ftime64(&currentTime);
#else
    _ftime64_s(&currentTime);
#endif
    tv->tv_sec = (long)currentTime.time;
    tv->tv_usec = (long)currentTime.millitm * 1000;

    return 0;
}
#endif

#else /* unix */
#include <unistd.h>
#include <sys/time.h>
#endif /* _WIN32 or _WIN64 */

#if defined(_WIN32) || defined(__hpux)
WINCOMPATINLINE char *strtok_r(char *s1, const char *s2, char **lasts)
{
    char *ret;

    if (s1 == NULL)
        s1 = *lasts;
    while (*s1 && strchr(s2, *s1))
        ++s1;
    if (*s1 == '\0')
        return NULL;
    ret = s1;
    while (*s1 && !strchr(s2, *s1))
        ++s1;
    if (*s1)
        *s1++ = '\0';
    *lasts = s1;
    return ret;
}
#endif

#if !defined(__linux__) && !defined(__APPLE__)
WINCOMPATINLINE char *strsep(char **s, const char *delim)
{
    if (!s)
        return NULL;

    char *start = *s;
    char *p = *s;

    if (p == NULL)
        return NULL;

    while (*p)
    {
        const char *d = delim;
        while (*d)
        {
            if (*p == *d)
            {
                *p = '\0';
                *s = p + 1;
                return start;
            }
            d++;
        }
        p++;
    }

    *s = NULL;
    return start;
}
#endif

#if defined(__APPLE__)
WINCOMPATINLINE off_t lseek64(int fildes, off_t offset, int whence)
{
    return lseek(fildes, offset, whence);
}
#endif

#endif /* COV_WINCOMPAT_H */
