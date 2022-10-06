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
using std::cerr;
using std::endl;
using std::ios;
using std::ofstream;
using std::ostream;

#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>

#include <boost/filesystem/path.hpp>

using namespace vrml;

Doc::Doc(const std::string &url, const Doc *relative)
{
    seturl(url, relative);
}

Doc::Doc(Doc *doc)
{
    if (doc)
        seturl(doc->url());
}

Doc::~Doc()
{
    if (!d_tmpfile.empty() && d_isTmp)
        System::the->removeFile(d_tmpfile.c_str());
}

void Doc::seturl(const std::string &url, const Doc *relative)
{
    d_url = "";
    if (relative && !isAbsolute(url))
        d_url = relative->urlPath();
    d_url += url;
}

const std::string &Doc::url() const { return d_url; }

std::string Doc::urlBase() const
{
    return boost::filesystem::path(d_url).stem().string();
}

std::string Doc::urlExt() const
{
    return boost::filesystem::path(d_url).extension().string();
}

std::string Doc::urlPath() const
{
    return boost::filesystem::path(d_url).parent_path().string() + "/";
}

std::string Doc::urlProtocol() const
{

    std::string protocol;
    const char *s = d_url.c_str();

#ifdef _WIN32
    if (strncmp(s + 1, ":\\", 2) == 0 || strncmp(s + 1, ":/", 2) == 0)
        return "file";
#endif

    for (unsigned int i = 0; i < d_url.size(); ++i, ++s)
    {
        if (*s == 0 || !isalpha(*s))
        {
            break;
        }
        protocol += tolower(*s);
    }
    if (*s == ':')
        return protocol;

    return "file";
}

std::string Doc::urlModifier() const
{
  auto p = d_url.find_last_of('#');
  return p == std::string::npos ? "" : d_url.substr(p);
}

std::string Doc::localName()
{
    return initFilePath();
}

std::string Doc::localPath()
{
    initFilePath();
    return boost::filesystem::path(d_tmpfile).parent_path().string();
}

std::string Doc::stripProtocol(const std::string &url)
{
    const char *s = url.c_str();

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

const std::string &Doc::initFilePath()
{
    if(d_tmpfile.empty())
        d_tmpfile = System::the->remoteFetch(d_url);
    return d_tmpfile;
}
// Having both fopen and outputStream is dumb...

FILE *Doc::fopen(const char *mode)
{
    if(d_url == "-"){
            if (*mode == 'r')
                d_fp = stdin;
            else if (*mode == 'w')
                d_fp = stdout;
    }
    else{
        auto fn = initFilePath();
        d_fp = ::fopen(fn.c_str(), mode);

    }
    return d_fp;
}


void Doc::fclose()
{
    if (d_fp && d_url !=  "-"  && (strncmp(d_url.c_str(), "-#", 2) != 0))
        ::fclose(d_fp);

    d_fp = nullptr;
    d_tmpfile = "";
}

#if HAVE_LIBPNG || HAVE_ZLIB

// For (optionally) gzipped files

gzFile Doc::gzopen(const char *mode)
{
    auto fn = initFilePath();
    struct stat statbuf;
    if (stat(fn.c_str(), &statbuf) == -1)
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
        d_gz = ::gzopen(fn.c_str(), mode);
    }

    return d_gz;
}

void Doc::gzclose()
{
    if (d_gz)
        ::gzclose(d_gz);

    d_gz = nullptr;
    d_tmpfile = "";
}
#endif

ostream &Doc::outputStream()
{
    d_ostream.reset(new ofstream(stripProtocol(d_url), ios::out));
    return *d_ostream;
}

bool Doc::isAbsolute(const std::string &url)
{
    return boost::filesystem::path(url).is_absolute();
}
