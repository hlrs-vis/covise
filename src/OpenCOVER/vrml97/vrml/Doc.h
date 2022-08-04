/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _DOC_
#define _DOC_
//
//  Document class
//

#include "config.h"

#if HAVE_LIBPNG || HAVE_ZLIB
#include <zlib.h>
#endif

#include <iostream>
#include <memory>
#include <stdio.h>

namespace vrml
{

class VRMLEXPORT Doc
{

public:
    Doc(const std::string &url = "", const Doc *relative = 0);
    Doc(Doc *);
    ~Doc();

    void seturl(const std::string &url, const Doc *relative = 0);

    const std::string &url() const; // "http://www.foo.com/dir/file.xyz#Viewpoint"
    std::string urlBase() const; // "file" or ""
    std::string urlExt() const; // "xyz#Viewpoint" or ""
    std::string urlPath() const; // "http://www.foo.com/dir/" or ""
    std::string urlProtocol() const; // "http"
    std::string urlModifier() const; // "#Viewpoint" or ""

    std::string localName(); // "/tmp/file.xyz" or NULL
    std::string localPath(); // "/tmp/" or NULL

    FILE *fopen(const char *mode);
    void fclose();

#if HAVE_LIBPNG || HAVE_ZLIB
    // For (optionally) compressed files
    gzFile gzopen(const char *mode);
    void gzclose();
#endif

    std::ostream &outputStream();

private:
    const std::string &initFilePath();
    static std::string stripProtocol(const std::string &url);
    static bool isAbsolute(const std::string &url);

    std::string d_url;
    std::unique_ptr<std::ostream> d_ostream;
    FILE *d_fp;
#if HAVE_LIBPNG || HAVE_ZLIB
    gzFile d_gz = nullptr;
#endif
    std::string d_tmpfile; // Local copy of http: files
	bool d_isTmp = false; //is local copy a temp file
};
}
#endif // _DOC_
