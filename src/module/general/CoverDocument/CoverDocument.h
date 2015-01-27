/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Module CoverDocument
//
// This class sends a dummy geometry to COVER and adds a
// attribute:
//
//     DOCUMENT <name> <numFiles> fullpath1 fullpath2 ..
//
// Initial version: 2004-04-29 [dr]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//    we 2004-04-30  Translated to new API

#ifndef _COVER_DOCUMENT_H
#define _COVER_DOCUMENT_H

#include <api/coModule.h>
using namespace covise;
#include <api/coStepFile.h>

class CoverDocument : public coModule
{
private:
    //  member functions
    virtual int compute(const char *port);

    // One output object: Dummy point
    coOutputPort *outPort;

    //parameter: filename
    coFileBrowserParam *filenameParam;

    //parameter: Document name
    coStringParam *docnameParam;

public:
    CoverDocument(int argc, char *argv[]);
};
#endif
