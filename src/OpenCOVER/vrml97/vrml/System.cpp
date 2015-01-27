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

#ifdef HAVE_LIBWWW
#include <WWWLib.h>
#include <WWWInit.h>
#endif

#ifdef HAVE_LIBCURL
#include <curl/curl.h>
#include <curl/easy.h>
#endif

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
#ifdef HAVE_LIBWWW
    try
    {
        HTProfile_newPreemptiveClient("COVISE VRML Library", "6.1");
    }
    catch (...)
    {
        fprintf(stderr, "VRML97 HTProfile_newPreemptiveClient failed\n");
    }
#endif

#ifdef HAVE_LIBCURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
}

System::~System()
{
#ifdef HAVE_LIBWWW
    try
    {
        HTProfile_delete();
    }
    catch (...)
    {
    }
#endif

#ifdef HAVE_LIBCURL
    curl_global_cleanup();
#endif
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

int System::connectSocket(const char *host, int port)
{
    struct sockaddr_in sin;
    struct hostent *he;

    int sockfd = -1;

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

#ifdef HAVE_LIBCURL
struct CurlData
{
    FILE *fp;
    char *filename;
};

static ssize_t curl_fwrite(void *buffer, size_t size, size_t nmemb, void *stream)
{
    CurlData *curlData = static_cast<CurlData *>(stream);
    if (!curlData->fp)
    {
        curlData->fp = fopen(curlData->filename, "wb");
        if (!curlData->fp)
            return -1;
    }
    return fwrite(buffer, size, nmemb, curlData->fp);
}
#endif

// This isn't particularly robust or complete...

const char *System::httpFetch(const char *url)
{
#if defined(HAVE_LIBCURL)
    CURL *curl = curl_easy_init();
    if (curl)
    {
        CurlData curlData = { NULL, tempnam(0, "VR") };
        /* Set url to get */
        curl_easy_setopt(curl, CURLOPT_URL, url);
        /* Define our callback to get called when there's data to be written */
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_fwrite);
        /* Set a pointer to our struct to pass to the callback */
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &curlData);

        /* Switch on full protocol/debug output */
        //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

        CURLcode res = curl_easy_perform(curl);
        if (CURLE_OK != res)
        {
            /* we failed */
            System::warn("Failed to fetch %s, curl told us %d\n", url, res);
            free(curlData.filename);
            curlData.filename = NULL;
        }

        if (curlData.fp)
            fclose(curlData.fp);

        /* always cleanup */
        curl_easy_cleanup(curl);

        return curlData.filename;
    }
    return NULL;
#elif defined(HAVE_LIBWWW)
    HTRequest *request = HTRequest_new();
    char *result = tempnam(0, "VR");
    if (result)
    {
        if (!HTLoadToFile(url, request, result))
        {
            free(result);
            result = NULL;
        }
    }
    HTRequest_delete(request);

    return result;
#else
    int port = 80;
    const char *hostname = httpHost(url, &port);

    if (port == 80)
        System::inform("Connecting to %s ...", hostname);
    else
        System::inform("Connecting to %s:%d ...", hostname, port);

    int sockfd;
    if ((sockfd = System::connectSocket(hostname, port)) != -1)
        System::inform("connected.");
    else
        System::warn("Connect failed: %s (errno %d).\n",
                     strerror(errno), errno);

    // Copy to a local temp file
    char *result = 0;
    if (sockfd != -1 && (result = tempnam(0, "VR")))
    {
#ifdef _WIN32
        int fd = open(result, O_RDWR | O_CREAT | O_BINARY, 0777);
#else
        int fd = open(result, O_RDWR | O_CREAT, 0777);
#endif
        if (fd != -1)
        {
            char *abspath = strstr((char *)url, "//");
            if (abspath)
                abspath = strchr(abspath + 2, '/');
            if (!abspath)
                abspath = (char *)url;
            char host[256];
            strncpy(host, strstr(url, "//") + 2, sizeof(host));
            host[255] = '\0';
            strtok(host, "/");

            char request[1024];
#ifdef _WIN32
            sprintf(request, "GET %s HTTP/1.0\nHost: %s\nAccept: */*\n\r\n", abspath, host);
#else
            snprintf(request, sizeof(request), "GET %s HTTP/1.0\nHost: %s\nAccept: */*\n\r\n", abspath, host);
#endif

            int nbytes = (int)strlen(request);
#ifdef _WIN32
            if (::send(sockfd, (char *)request, nbytes, 0) != nbytes)
#else
            if (write(sockfd, request, nbytes) != nbytes)
#endif
                System::warn("http GET failed: %s (errno %d)\n",
                             strerror(errno), errno);
            else
            {
                int gothdr = 0, nread = 0, nwrote = 0, nmore;
                int restheaderlength = 0;
                char *start;
#ifdef _WIN32
                while ((nmore = ::recv(sockfd, (char *)request + restheaderlength, sizeof(request) - (1 + restheaderlength), 0)) > 0)
#else
                while ((nmore = read(sockfd, request + restheaderlength, sizeof(request) - (1 + restheaderlength))) > 0)
#endif
                {
                    request[nmore + restheaderlength] = '\0';
                    nread += nmore;

                    // Skip header (should read return code, content-type...)
                    if (gothdr)
                        start = request;
                    else
                    {
                        start = strstr(request, "\r\n\r\n");
                        if (start)
                            start += 4;
                        else
                        {
                            start = strstr(request, "\n\r\n");
                            if (start)
                                start += 3;
                            else
                            {
                                start = strstr(request, "\n\n");
                                if (start)
                                    start += 2;
                            }
                        }
                        if (!start)
                        {
                            request[0] = request[nmore + restheaderlength - 4]; // copy four bytes of this header to the beginning of the buffer,
                            // so that we make sure, that we don't miss a delimiter
                            request[1] = request[nmore + restheaderlength - 3];
                            request[2] = request[nmore + restheaderlength - 2];
                            request[3] = request[nmore + restheaderlength - 1];
                            restheaderlength = 4;
                            continue;
                        }
                        gothdr = 1;
                    }

                    nmore -= (int)(start - (request + restheaderlength));
                    restheaderlength = 0;
                    if (write(fd, start, nmore) != nmore)
                    {
                        System::warn("http: temp file write error writing %d bytes to %s\n", nmore, result);
                        System::warn("start - request %d\n", start - request);
                        System::warn("start %ld\n", start);
                        System::warn("request %ld\n", request);
                        System::warn(start);
                        System::warn(request);

                        break;
                    }
                    nwrote += nmore;
                }

                System::inform("Read %dk from %s", (nread + 1023) / 1024, url);
                System::inform("Wrote %d bytes to %s\n", nwrote, result);
                //char tmpstr[1024];
                //sprintf(tmpstr,"cp %s /tmp/test.jpg",result);
                //system(tmpstr);
            }

            close(fd);
        }
    }

    if (sockfd != -1)
    {
        closesocket(sockfd);
    }
    return result;
#endif
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
