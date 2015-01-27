/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vortex Core Lines
// Ronald Peikert, Martin Roth, Dirk Bauer <=2005 and Filip Sadlo >=2006
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
    myModule *application = new myModule;
    application->start(argc, argv);
    return 0;
}

void myModule::postInst() {}

void myModule::param(const char *name) {}

int myModule::compute()
{
    // system wrapper
    us = UniSys(this);

    // create unstructured wrapper for input
    if (us.inputChanged("ucd", 0))
    {
        if (unst)
            delete unst;
        std::vector<DO_Unstructured_V3D_Data *> vvec;
        vvec.push_back((DO_Unstructured_V3D_Data *)(velocity->getObj()));
        unst = new Unstructured((DO_UnstructuredGrid *)(grid->getObj()),
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
                              method->getValue(),
                              variant->getValue(),
                              minVertices->getValue(),
                              maxExceptions->getValue(),
                              minStrength->getValue(),
                              maxAngle->getValue(),
                              &ugeom);

        // output data already assigned to ports
    }

    return SUCCESS;
}
