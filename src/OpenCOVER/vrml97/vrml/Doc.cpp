/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  Doc.cpp
//  A class to contain document references. This is just a shell until
//  a real http protocol library is found...
//

#include "config.h"

#include "Doc.h"
#include "System.h"
#include <util/coErr.h>

#include <iostream>
#include <fstream>
using std::ostream;
using std::ofstream;
using std::ios;
using std::cerr;
using std::endl;

#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>

using namespace vrml;

Doc::Doc(const char *url, const Doc *relative)
    : d_url(0)
    , d_ostream(0)
    , d_fp(0)
    ,
#if HAVE_LIBPNG || HAVE_ZLIB
    d_gz(0)
    ,
#endif
    d_tmpfile(0)
{
    if (url)
        seturl(url, relative);
}

Doc::Doc(Doc *doc)
    : d_url(0)
    , d_ostream(0)
    , d_fp(0)
    ,
#if HAVE_LIBPNG || HAVE_ZLIB
    d_gz(0)
    ,
#endif
    d_tmpfile(0)
{
    if (doc)
        seturl(doc->url());
}

Doc::~Doc()
{
    delete[] d_url;
    delete d_ostream;
    if (d_tmpfile)
    {
        System::the->removeFile(d_tmpfile);
        delete[] d_tmpfile;
        d_tmpfile = 0;
    }
}

void Doc::seturl(const char *url, const Doc *relative)
{
    delete[] d_url;
    d_url = 0;

    if (url)
    {
        const char *path = "";

        if (relative && !isAbsolute(url))
            path = relative->urlPath();

        d_url = new char[strlen(path) + strlen(url) + 1];
        strcpy(d_url, path);
        strcat(d_url, url);
    }
}

const char *Doc::url() const { return d_url; }

const char *Doc::urlBase() const
{
    if (!d_url)
        return "";

    static char path[1024];
    char *p, *s = path;
    strncpy(path, d_url, sizeof(path) - 1);
    path[sizeof(path) - 1] = '\0';
    char *slash = s + strlen(s);
    while (slash > s)
    {
        if (*slash == '\\')
        {
            s = slash + 1;
            break;
        }
        if (*slash == '/')
        {
            s = slash + 1;
            break;
        }
        if (*slash == ':')
        {
            s = slash + 1;
            break;
        }
        slash--;
    }
    if ((p = strrchr(s, '.')) != 0)
        *p = '\0';

    return s;
}

const char *Doc::urlExt() const
{
    if (!d_url)
        return "";

    static char ext[20];
    char *p;

    if ((p = strrchr(d_url, '.')) != 0)
    {
        strncpy(ext, p + 1, sizeof(ext) - 1);
        ext[sizeof(ext) - 1] = '\0';
    }
    else
        ext[0] = '\0';

    return &ext[0];
}

const char *Doc::urlPath() const
{
    if (!d_url)
        return "";

    static char path[1024];

    strcpy(path, d_url);
    char *quest = strchr(path, '?');
    if (quest)
        *quest = '\0';
    char *slash = path + strlen(path);
    while (slash > path)
    {
        if (*slash == '\\')
        {
            *(slash + 1) = '\0';
            break;
        }
        if (*slash == '/')
        {
            *(slash + 1) = '\0';
            break;
        }
        slash--;
    }
    if (slash == path)
        path[0] = '\0';
    return &path[0];
}

const char *Doc::urlProtocol() const
{
    if (d_url)
    {
        static char protocol[12];
        const char *s = d_url;

#ifdef _WIN32
        if (strncmp(s + 1, ":\\", 2) == 0 || strncmp(s + 1, ":/", 2) == 0)
            return "file";
#endif

        for (unsigned int i = 0; i < sizeof(protocol); ++i, ++s)
        {
            if (*s == 0 || !isalpha(*s))
            {
                protocol[i] = '\0';
                break;
            }
            protocol[i] = tolower(*s);
        }
        protocol[sizeof(protocol) - 1] = '\0';
        if (*s == ':')
            return protocol;
    }

    return "file";
}

const char *Doc::urlModifier() const
{
    char *mod = d_url ? strrchr(d_url, '#') : 0;
    return mod ? mod : "";
}

const char *Doc::localName()
{
    static char buf[1024];
    if (filename(buf, sizeof(buf)))
        return &buf[0];
    return 0;
}

const char *Doc::localPath()
{
    static char buf[1024];
    if (filename(buf, sizeof(buf)))
    {

        char *slash = buf + strlen(buf);
        while (slash > buf)
        {
            if (*slash == '\\')
            {
                *(slash + 1) = '\0';
                break;
            }
            if (*slash == '/')
            {
                *(slash + 1) = '\0';
                break;
            }
            slash--;
        }
        if (slash == buf)
            buf[0] = '\0';
        return &buf[0];
    }
    return 0;
}

const char *Doc::stripProtocol(const char *url)
{
    const char *s = url;

#ifdef _WIN32
    if (strncmp(s + 1, ":\\", 2) == 0 || strncmp(s + 1, ":/", 2) == 0 || strncmp(s + 1, "://", 3) == 0)
        return url;
#endif

    // strip off protocol if any
    while (*s && *s > 0 && isalpha(*s))
        ++s;

    if (*s == ':')
        return s + 1;

    return url;
}

// Converts a url into a local filename

bool Doc::filename(char *fn, int nfn)
{
    fn[0] = '\0';

    char *e = 0, *s = (char *)stripProtocol(d_url);

    char *endString = NULL;
    int endLength = 0;
    if ((e = strrchr(s, '#')) != 0)
    {
        endLength = (int)strlen(e);
        endString = new char[endLength + 1];
        strcpy(endString, e);
        *e = '\0';
    }

    const char *protocol = urlProtocol();

    // Get a local copy of http files
    if (strcmp(protocol, "http") == 0 || strcmp(protocol, "https") == 0)
    {
        if (d_tmpfile) // Already fetched it
            s = d_tmpfile;
        else if ((s = (char *)System::the->httpFetch(d_url)))
        {
            d_tmpfile = new char[strlen(s) + 1 + endLength];
            strcpy(d_tmpfile, s);
            free(s); // assumes tempnam or equiv...
            s = d_tmpfile;
        }

        if (s)
        {
            strncpy(fn, s, nfn - 1);
            fn[nfn - 1] = '\0';
        }
    }

    // Unrecognized protocol (need ftp here...)

    else if (strcmp(protocol, "file") != 0)
    {
        // we have a local file now but does this exist?
        // if not, the try to fetch it from a remote site if possible
        cerr << "file " << s << " not local, try to get it from remote " << endl;

        if ((s = (char *)System::the->remoteFetch(d_url)))
        {
            d_tmpfile = new char[strlen(s) + 1 + endLength];
            strcpy(d_tmpfile, s);
            //delete [] s; This causes a memory leak
            s = NULL;
            // XXX: there is a problem here: freeing sth not malloced...
            //free(const_cast<char *>(s));        // assumes tempnam or equiv...
            s = d_tmpfile;
        }

        if (s)
        {
            strncpy(fn, s, nfn - 1);
            fn[nfn - 1] = '\0';
        }
    }
    else if (s)
    {
        // we have a local file now but does this exist?
        // if not, the try to fetch it from a remote site if possible
#ifdef WIN32
        struct _stat64 sbuf;
        if (_stat64(s, &sbuf))
        {
            cerr << "file " << s << " not local, try to get it from remote " << endl;

            if (d_tmpfile) // Already fetched it
            {
                s = d_tmpfile;
            }
            else if ((s = (char *)System::the->remoteFetch(s)))
            {
                d_tmpfile = new char[strlen(s) + 1 + endLength];
                strcpy(d_tmpfile, s);
                // XXX: there is a problem here: freeing sth not malloced...
                //free(const_cast<char *>(s));        // assumes tempnam or equiv...
                s = d_tmpfile;
            }
        }
#else
        struct stat sbuf;
        if (stat(s, &sbuf))
        {
            cerr << "file " << s << " not local, try to get it from remote " << endl;

            if (d_tmpfile) // Already fetched it
            {
                s = d_tmpfile;
            }
            else if ((s = (char *)System::the->remoteFetch(s)))
            {
                d_tmpfile = new char[strlen(s) + 1 + endLength];
                strcpy(d_tmpfile, s);
                // XXX: there is a problem here: freeing sth not malloced...
                //free(const_cast<char *>(s));        // assumes tempnam or equiv...
                s = d_tmpfile;
            }
        }
#endif
        if (s)
        {
            strncpy(fn, s, nfn - 1);
            fn[nfn - 1] = '\0';
        }
    }

    if (s && e)
    {
        strcat(s, endString);
        delete[] endString;
    }

    return s && *s;
}

// Having both fopen and outputStream is dumb...

FILE *Doc::fopen(const char *mode)
{
    if (d_fp)
    {
        System::the->error("Doc::fopen: %s is already open.\n", d_url ? d_url : "");
    }
    char fn[256];
    if (filename(fn, sizeof(fn)))
    {
        if (strcmp(fn, "-") == 0)
        {
            if (*mode == 'r')
                d_fp = stdin;
            else if (*mode == 'w')
                d_fp = stdout;
        }
        else
        {
            d_fp = ::fopen(fn, mode);
        }
    }

    return d_fp;
}

void Doc::fclose()
{
    if (d_fp && (strcmp(d_url, "-") != 0) && (strncmp(d_url, "-#", 2) != 0))
        ::fclose(d_fp);

    d_fp = 0;
    if (d_tmpfile)
    {
        System::the->removeFile(d_tmpfile);
        delete[] d_tmpfile;
        d_tmpfile = 0;
    }
}

#if HAVE_LIBPNG || HAVE_ZLIB

// For (optionally) gzipped files

gzFile Doc::gzopen(const char *mode)
{
    if (d_fp || d_gz)
        System::the->error("Doc::gzopen: %s is already open.\n", d_url ? d_url : "");

    char fn[1024];
    if (filename(fn, sizeof(fn)))
    {
        struct stat statbuf;
        if (stat(fn, &statbuf) == -1)
        {
            cerr << "Doc::gzopen(): failed to stat " << fn << ": " << strerror(errno) << endl;

#ifdef _WIN32
        }
        else if (_S_IFDIR & statbuf.st_mode)
        {
#else
        }
        else if (S_ISDIR(statbuf.st_mode))
        {
#endif
            cerr << "Doc::gzopen(): " << fn << " is a directory" << endl;
        }
        else
        {
            d_gz = ::gzopen(fn, mode);
        }
    }

    return d_gz;
}

void Doc::gzclose()
{
    if (d_gz)
        ::gzclose(d_gz);

    d_gz = 0;
    if (d_tmpfile)
    {
        System::the->removeFile(d_tmpfile);
        delete[] d_tmpfile;
        d_tmpfile = 0;
    }
}
#endif

ostream &Doc::outputStream()
{
    d_ostream = new ofstream(stripProtocol(d_url), ios::out);
    return *d_ostream;
}

bool Doc::isAbsolute(const char *url)
{
    const char *s = stripProtocol(url);
    return (*s == '/' || *s == '\\' || *(s + 1) == ':');
}
