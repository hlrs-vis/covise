/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ReadFconfig_H
#define _ReadFconfig_H

#include <appl/ApplInterface.h>
using namespace covise;
#include <api/coSimpleModule.h>
#include <stdlib.h>
#include <stdio.h>


class ReadFconfig : public coSimpleModule
{

private:
    //////////  member functions
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);

    // Create the mesh
    void createMesh();
	coFileBrowserParam * fileName;
    coIntScalarParam *dimX;
    coIntScalarParam *dimY;
    coIntScalarParam *dimZ;

    coOutputPort *mesh, *data;

public:
    ReadFconfig(int argc, char *argv[]);

    virtual ~ReadFconfig()
    {
    }

};
#endif // _ReadFconfig_H
