/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Flow Topology
// Ronald Peikert and Filip Sadlo 2005
// CGL ETHZ

#include "stdlib.h"
#include "stdio.h"

#include "FlowTopo.h"

#include "linalg.h"

#include "unstructured.h"
#include "unifield.h"
#include "unisys.h"

static Unstructured *unst = NULL;

UniSys us = UniSys(NULL);

#include "flow_topo_impl.cpp" // ### including .cpp

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

    // create unstructured object for input
    if (us.inputChanged("ucd", 0))
    {
        if (unst)
            delete unst;
        std::vector<coDoFloat *> svec;
        if (wallDistance->getCurrentObject())
            svec.push_back((coDoFloat *)(wallDistance->getCurrentObject()));
        std::vector<coDoVec3 *> vvec;
        vvec.push_back((coDoVec3 *)(velocity->getCurrentObject()));
        unst = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                (wallDistance->getCurrentObject() ? &svec : NULL),
                                &vvec);
    }

    // scalar components come first in Covise-Unstructured
    // ### HACK, should retrieve from Unstructured?
    int compWallDist;
    int compVelo;
    if (wallDistance->getCurrentObject())
    {
        compWallDist = 0;
        compVelo = 1;
    }
    else
    {
        compWallDist = -1;
        compVelo = 0;
    }

#if 0
  // Generate output data
  coDoFloat *odata;
  {
    int numEl, numConn, numCoord;
    ((coDoUnstructuredGrid *) (grid->getObj()))->get_grid_size(&numEl, &numConn, &numCoord);
    odata = new coDoVec3(matDeriv->getObjName(), numCoord);
  }
#endif

    // go
    {
        // setup Unifield for output, without allocating or assigning data
        std::vector<coOutputPort *> outPortVec;
        outPortVec.push_back(criticalPoints);
        outPortVec.push_back(criticalPointsData);
        UniField *unif = new UniField(outPortVec);

        // compute
        flow_topo_impl(&us, unst, compVelo, compWallDist, &unif,
                       divideByWallDist->getValue(),
                       interiorCritPoints->getValue(),
                       boundaryCritPoints->getValue(),
                       generateSeeds->getValue(),
                       seedsPerCircle.getValue(),
                       radius.getValue(),
                       offset.getValue());

        if (unif)
        {

            // assign output data to ports
            coDoStructuredGrid *wgrid;
            coDoFloat *wdat;
            unif->getField(&wgrid, &wdat, (coDoVec3 **)NULL);

            criticalPoints->setCurrentObject(wgrid);
            criticalPointsData->setCurrentObject(wdat);

            // delete field wrapper (but not the field)
            delete unif;
        }
    }

    return SUCCESS;
}
