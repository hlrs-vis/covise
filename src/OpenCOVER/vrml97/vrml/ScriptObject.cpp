/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  ScriptObject.cpp
//  An abstract class to encapsulate the interface between the VRML97
//  scene graph and the supported scripting languages.
//

#include "config.h"
#include "ScriptObject.h"
#include "Doc.h"
#include "VrmlMFString.h"
#include "VrmlNodeScript.h"
#include "VrmlScene.h"
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
// JavaScript
#if HAVE_JAVASCRIPT
#include "ScriptJS.h"
#endif

// Java via Sun JDK
#if HAVE_JDK
#include "ScriptJDK.h"
#endif

#include <iostream>
using std::cerr;
using std::endl;
using namespace vrml;

ScriptObject::ScriptObject() {}

ScriptObject::~ScriptObject() {}

ScriptObject *ScriptObject::create(VrmlNodeScript *node,
                                   VrmlMFString &url, VrmlSFString &relativeUrl)
{
    // Try each url until we find one we like
    int i, n = url.size();
    for (i = 0; i < n; ++i)
    {
        if (!url[i])
            continue;

        // Get the protocol & mimetype...

        int slen = (int)strlen(url[i]);
#if HAVE_JAVASCRIPT
        // Need to handle external .js files too... should be done now
        if (strncmp(url[i], "javascript:", 11) == 0 || strncmp(url[i], "vrmlscript:", 11) == 0)
        {
            return new ScriptJS(node, url[i] + 11);
        }
        if (slen > 3 && (strcmp(url[i] + slen - 3, ".js") == 0 || strcmp(url[i] + slen - 3, ".JS") == 0))
        {
            Doc relDoc(relativeUrl.get());
            Doc doc(url[i], &relDoc);
            FILE *fp = doc.fopen("rb");
            if (fp)
            {
                int file, size, rsize;
                file = fileno(fp);
                struct stat statbuf;
                if (fstat(file, &statbuf) < 0)
                {
                    cerr << " could not stat file " << url[i] << endl;
                    continue;
                }
                size = statbuf.st_size;
                char *buf;
                buf = new char[size + 1];
                if ((rsize = read(file, buf, size)) < size)
                {
                    cerr << " could only read " << rsize << " of " << size << " bytes from file " << file << endl;
                }
                cerr << " read " << rsize << " of " << size << " bytes from file " << file << buf << endl;
                buf[rsize] = '\0';
                return new ScriptJS(node, buf);
            }
        }
#endif

#if HAVE_JDK
        if (slen > 6 && (strcmp(url[i] + slen - 6, ".class") == 0 || strcmp(url[i] + slen - 6, ".CLASS") == 0))
        {
            Doc *relative = 0;
            if (node->scene())
                relative = node->scene()->url();
            Doc doc(url[i], relative);
            if (doc.localName())
                return new ScriptJDK(node, doc.urlBase(), doc.localPath());
        }
#endif
    }

    return 0;
}
