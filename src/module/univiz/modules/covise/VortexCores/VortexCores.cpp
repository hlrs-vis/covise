/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vortex Core Lines
// Ronald Peikert, Martin Roth <=2005 and Filip Sadlo >=2006
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "VortexCores.h"

#include "linalg.h"

#include "unstructured.h"
#include "unigeom.h"
#include "unisys.h"

#include "computeVortexCores.h"
#include "ucd_vortex_cores_impl.cpp" // ### including .cpp

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

        // setup geometry wrapper for output
        UniGeom ugeom = UniGeom(coreLines);

        // compute
        ucd_vortex_cores_impl(&us,
                              unst, compVelo,
                              method->getValue() + 1,
                              variant->getValue() + 1,
                              minVertices.getValue(),
                              maxExceptions.getValue(),
                              minStrength.getValue(),
                              maxAngle.getValue(),
                              &ugeom);

        // output data already assigned to ports
    }

    return SUCCESS;
}
