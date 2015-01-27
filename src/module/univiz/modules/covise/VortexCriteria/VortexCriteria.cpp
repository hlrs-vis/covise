/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vortex Criteria
// Filip Sadlo 2007
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "VortexCriteria.h"

#include "linalg.h"

#include "unstructured.h"
#include "unisys.h"

#include "vortex_criteria_impl.cpp" // ### including .cpp

static Unstructured *unst_in = NULL;

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
        if (unst_in)
            delete unst_in;
        std::vector<coDoVec3 *> vvec;
        vvec.push_back((coDoVec3 *)(velocity->getCurrentObject()));
        unst_in = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                   NULL, &vvec);
    }

    // scalar components come first in Covise-Unstructured
    int compVelo = 0;

    // compute gradient
    if (us.inputChanged("ucd", 0) || us.parameterChanged("smoothingRange"))
    {
        us.moduleStatus("computing gradient", 5);
        unst_in->gradient(compVelo, false, smoothingRange.getValue());
        us.moduleStatus("computing gradient", 50);
    }

    // go
    {

        // wrapper for output
        Unstructured *unst_scalar;
        coDoFloat *scalarData;
        {
            // alloc output field (future: TODO: do it inside Unstructured)
            int nCells, nConn, nNodes;
            //((coDoUnstructuredGrid *) grid->getCurrentObject())->get_grid_size(&nCells, &nConn, &nNodes);
            ((coDoUnstructuredGrid *)grid->getCurrentObject())->getGridSize(&nCells, &nConn, &nNodes);
            scalarData = new coDoFloat(scalar->getObjName(), nNodes);

            // wrap
            std::vector<coDoFloat *> svec;
            svec.push_back((coDoFloat *)scalarData);
            unst_scalar = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                           &svec, NULL);
        }

        // compute
        vortex_criteria_impl(&us, unst_in, compVelo,
                             NULL, 0, // ### TODO
                             unst_scalar,
                             quantity->getValue() + 1,
                             smoothingRange.getValue() + 1,
                             NULL);

        // output data already assigned to ports
        scalar->setCurrentObject(scalarData);

        delete unst_scalar;
    }

    return SUCCESS;
}
