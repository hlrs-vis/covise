/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_MODULE_H
#define _READ_MODULE_H

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadModule
//
// This module reads polygon meshes in all formats supported by assimp
//
#include <util/coviseCompat.h>
#include <api/coModule.h>
using namespace covise;

class ReadModule : public coModule
{
public:

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paraName, bool inMapLoading);

    // ports
    coOutputPort *p_polyOut;
    coOutputPort *p_normOut;
    coOutputPort *p_linesOut;
    coOutputPort *p_colorOut;

    // parameter
    coFileBrowserParam *p_filename;
    coBooleanParam *p_triangulate;
    coBooleanParam *p_joinVertices;


public:
    ReadModule(int argc, char *argv[]);
    virtual ~ReadModule();
};
#endif
