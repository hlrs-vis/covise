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

#include <stdio.h>

namespace vrml
{

class VRMLEXPORT Doc
{

public:
    Doc(const char *url = 0, const Doc *relative = 0);
    Doc(Doc *);
    ~Doc();

    void seturl(const char *url, const Doc *relative = 0);

    const char *url() const; // "http://www.foo.com/dir/file.xyz#Viewpoint"
    const char *urlBase() const; // "file" or ""
    const char *urlExt() const; // "xyz" or ""
    const char *urlPath() const; // "http://www.foo.com/dir/" or ""
    const char *urlProtocol() const; // "http"
    const char *urlModifier() const; // "#Viewpoint" or ""

    const char *localName(); // "/tmp/file.xyz" or NULL
    const char *localPath(); // "/tmp/" or NULL

    FILE *fopen(const char *mode);
    void fclose();

#if HAVE_LIBPNG || HAVE_ZLIB
    // For (optionally) compressed files
    gzFile gzopen(const char *mode);
    void gzclose();
#endif

    std::ostream &outputStream();

protected:
    static const char *stripProtocol(const char *url);
    static bool isAbsolute(const char *url);
    bool filename(char *fn, int nfn);

    char *d_url;
    std::ostream *d_ostream;
    FILE *d_fp;
#if HAVE_LIBPNG || HAVE_ZLIB
    gzFile d_gz;
#endif
    char *d_tmpfile; // Local copy of http: files
};
}
#endif // _DOC_
