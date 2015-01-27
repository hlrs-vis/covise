/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vorticity Transport
// Filip Sadlo 2006 - 2008
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "VorticityTransport.h"

#include "linalg.h"

#include "unstructured.h"
#include "unigeom.h"
#include "unisys.h"

#include "vorticity_transport_impl.cpp" // ### including .cpp

static Unstructured *unst = NULL;

UniSys us = UniSys(NULL);

int main(int argc, char *argv[])
{
    myModule *application = new myModule(argc, argv);
    application->start(argc, argv);
    return 0;
}

void myModule::postInst() {}

void myModule::param(const char *, bool)
{
    // force min/max
    adaptMyParams();
}

int myModule::compute(const char *)
{
    // force min/max
    adaptMyParams();

    // system wrapper
    us = UniSys(this);

    // create unstructured wrapper for input
    if (us.inputChanged("ucd", 0))
    {
        if (unst)
            delete unst;
        std::vector<coDoVec3 *> vvec;
        vvec.push_back((coDoVec3 *)(velocity->getCurrentObject()));
        unst = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                NULL, &vvec);
    }

    // scalar components come first in Covise-Unstructured
    int compVelo = 0;

    // go
    {
        // setup geometry wrapper for input
        UniGeom ugeomCoreLines = UniGeom(coreLines);

        // setup geometry wrapper for output
        UniGeom ugeomTubes = UniGeom(tubes);
        UniGeom ugeomLines = UniGeom(pathLines);

        // compute
        vorticity_transport_impl(&us,
                                 unst, compVelo,
                                 &ugeomCoreLines,
                                 startTime.getValue(),
                                 integrationTime.getValue(),
                                 integStepsMax.getValue(),
                                 &ugeomTubes,
                                 &ugeomLines);

        // output data already assigned to ports
    }

    return SUCCESS;
}
