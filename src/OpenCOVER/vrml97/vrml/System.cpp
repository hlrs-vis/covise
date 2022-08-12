/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  System.cpp
//  A class to contain system-dependent/non-standard utilities
//

#include <stdarg.h>

#include <fcntl.h> // open() modes

#if defined(__MINGW32__)
#include <sys/time.h>
#endif

#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#define closesocket close
#else
#include <util/coTypes.h>
#include <io.h>
#include <stdio.h>
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>
#include <Shellapi.h>

#endif
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include <iostream>
using std::cerr;
using std::endl;

#include "config.h"
#include "System.h"
#include "Byteswap.h"

using namespace vrml;

System *System::the = NULL;

VrmlMessage::VrmlMessage(size_t len)
    : size(len)
    , pos(0)
{
    buf = new char[len];
}

VrmlMessage::~VrmlMessage()
{
    delete[] buf;
}

void VrmlMessage::append(int &i)
{
    memcpy(&buf[pos], &i, sizeof(int));
#ifdef BYTESWAP
    int *ip = (int *)&buf[pos];
    byteSwap(*ip);
#endif
    pos += sizeof(int);
    if (pos >= size)
        cerr << "append(int)" << endl;
}

void VrmlMessage::append(float &f)
{
    memcpy(&buf[pos], &f, sizeof(float));
#ifdef BYTESWAP
    float *fp = (float *)&buf[pos];
    byteSwap(*fp);
#endif
    pos += sizeof(float);
    if (pos >= size)
        cerr << "append(float)" << endl;
}

void VrmlMessage::append(double &d)
{
    memcpy(&buf[pos], &d, sizeof(double));
#ifdef BYTESWAP
    double *dp = (double *)&buf[pos];
    byteSwap(*dp);
#endif
    pos += sizeof(double);
    if (pos >= size)
        cerr << "append(double)" << endl;
}

void VrmlMessage::append(const char *string, size_t len)
{
    memcpy(&buf[pos], string, len);
    pos += len;
    if (pos >= size)
        cerr << "append(char)" << endl;
}

System::System()
    : correctBackFaceCulling(true)
    , correctSpatializedAudio(true)
    , frameCounter(0)
{
}

System::~System()
{
}

// Should make these iostream objects...

void System::error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

void System::warn(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "Warning: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// Write to the browser status line...

void System::inform(const char *fmt, ...)
{
    static char lastbuf[1024] = { 0 };
    char buf[1024];

    va_list ap;
    va_start(ap, fmt);
    vsprintf(buf, fmt, ap);
    va_end(ap);
    if (strcmp(lastbuf, buf))
    {
        fprintf(stderr, "%s", buf);
        putc('\n', stderr);
        strcpy(lastbuf, buf);
    }
}

#ifdef DEBUG
void System::debug(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}
#else
void System::debug(const char *, ...)
{
}
#endif

// This won't work if netscape isn't running...

bool System::loadUrl(const char *url, int np, char **parameters)
{
    if (!url)
        return false;
    char *buf = new char[strlen(url) + 200];
    int result = 1;
#ifndef _WIN32
    if (np)
        sprintf(buf, "/bin/csh -c \"netscape -remote 'openURL(%s, %s)'\" &",
                url, parameters[0]);
    else
        sprintf(buf, "/bin/csh -c \"netscape -remote 'openURL(%s)'\" &", url);
    result = system(buf);
    fprintf(stderr, "%s\n", buf);
#else
    if (np)
    {
        ShellExecute(NULL, "open", url,
                     parameters[0], NULL, SW_SHOWNORMAL);
    }
    else
    {
        ShellExecute(NULL, "open", url,
                     NULL, NULL, SW_SHOWNORMAL);
    }
#endif

    delete[] buf;
    return (result >= 0); // _WIN32
}

// This won't work under windows...
#ifndef _WIN32
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h> // memset
#include <ctype.h>
#else
#include <windows.h>
#include <WinSock2.h>
#endif
#ifndef WIN32
typedef int SOCKET;
#endif

int System::connectSocket(const char *host, int port)
{
    struct sockaddr_in sin;
    struct hostent *he;

    SOCKET sockfd = -1;

    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);

    // Check for dotted number format
    char *s;
    for (s = (char *)host; *s; ++s)
        if (!(isdigit(*s) || *s == '.'))
            break;

    if (*s) // Not dotted number
        he = gethostbyname(host);
    else
    {
        u_long addr;
        addr = inet_addr(host);
        he = gethostbyaddr((char *)&addr, sizeof(addr), AF_INET);
    }

    if (he)
    {
        memcpy((char *)&sin.sin_addr, he->h_addr, he->h_length);
        sockfd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sockfd != -1)
            if (connect(sockfd, (struct sockaddr *)&sin, sizeof(sin)) == -1)
            {
                closesocket(sockfd);
                sockfd = -1;
            }
    }

    return sockfd;
}

const char *System::httpHost(const char *url, int *port)
{
    static char hostname[256];
    const char *s = strstr(url, "//");
    char *p = hostname;

    if (s)
    {
        for (s += 2; *s && *s != '/' && *s != ':'; *p++ = *s++)
            /* empty */;
        if (*s == ':' && isdigit(*(s + 1)))
            *port = atoi(s + 1);
    }
    *p = '\0';
    return hostname;
}

void System::removeFile(const char *fn)
{
    remove(fn);
}

void System::update()
{
    frameCounter++;
    if (frameCounter < 0)
        frameCounter = 0;
}

int System::frame()
{
    return frameCounter;
}

double System::realTime()
{
#if defined(_WIN32) && !defined(__MINGW32__)
    struct __timeb64 currentTime;
#if _MSC_VER < 1400
    _ftime64(&currentTime);
#else
    _ftime64_s(&currentTime);
#endif
    return (currentTime.time + (double)currentTime.millitm / 1000.0);
#else
    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    return (currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0);
#endif
}

void System::saveTimestamp(const char *name)
{
#ifdef _WIN32
    FILE *fp = fopen("vrml-timestamps.txt", "a");
#else
    FILE *fp = fopen("/tmp/vrml-timestamps.txt", "a");
#endif
    if (fp)
    {
        fprintf(fp, "%s: %f\n", name, realTime());
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "saving timestamp failed: %s: %f\n", name, realTime());
    }
}

bool vrml::System::doOptimize()
{
	return true;
}
