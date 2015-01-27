/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POLY_TO_TETRA_USG_H
#define _POLY_TO_TETRA_USG_H

#include <api/coModule.h>
#include <do/coDistributedObject.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>

using namespace covise;

class PolyToTetraUSG : public coModule
{

public:
    PolyToTetraUSG(int argc, char **argv);
    virtual ~PolyToTetraUSG();

private:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);

    coDistributedObject *convert(coDistributedObject *obj,
                                 int child = 0, int level = 0);

    coInputPort *p_mesh;
    coOutputPort *p_meshOut;

    int numPolyhedra;
};

#endif
