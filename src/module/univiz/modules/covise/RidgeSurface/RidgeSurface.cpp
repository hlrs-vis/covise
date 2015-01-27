/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Ridge Surface
// Filip Sadlo 2007
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "RidgeSurface.h"

#include "covise_ext.h"
#include "linalg.h"
#include "unstructured.h"
#include "unigeom.h"
#include "unisys.h"

#include "ridge_surface_impl.cpp" // ### including .cpp

static Unstructured *unst = NULL;
static Unstructured *temp = NULL;
static bool *excludeNodes = NULL;

UniSys us = UniSys(NULL);

int main(int argc, char *argv[])
{
    myModule *application = new myModule(argc, argv);
    application->start(argc, argv);
    return 0;
}

void myModule::handleParamWidgets(void)
{
}

void myModule::postInst()
{
}

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

    // create Unstructured wrapper for scalar input
    int compScalar = 0;
    int compClipScalar = -1;
    if (us.inputChanged("ucd", 0))
    {
        if (unst)
            delete unst;
        std::vector<coDoFloat *> svec;
        svec.push_back((coDoFloat *)(scalar->getCurrentObject()));
        if (clipScalar->getCurrentObject())
        {
            svec.push_back((coDoFloat *)(clipScalar->getCurrentObject()));
            compClipScalar = 1;
        }
        unst = new Unstructured((coDoUnstructuredGrid *)(grid->getCurrentObject()),
                                &svec,
                                NULL);
    }

    // geometry wrapper for output
    UniGeom ugeom = UniGeom(ridges,
                            (generateNormals->getValue() ? normals : NULL));

    // compute
    if (!ridge_surface_impl(&us,
                            unst, compScalar, compClipScalar,
                            &temp,
                            //compGradient, compHess, compEigenvals, compEigenvectExtr,
                            &excludeNodes,
                            //*level,
                            0.0,
                            smoothingRange.getValue(),
                            mode->getValue() + 1,
                            extremum->getValue() + 1,
                            1, // useBisection
                            excludeFLT_MAX->getValue(),
                            excludeLonelyNodes->getValue(),
                            HessExtrEigenvalMin.getValue(),
                            PCAsubdomMaxPerc.getValue(),
                            scalarMin.getValue(),
                            scalarMax.getValue(),
                            clipScalarMin.getValue(),
                            clipScalarMax.getValue(),
                            minSize.getValue(),
                            filterByCell->getValue(),
                            combineExceptions->getValue(),
                            maxExceptions.getValue(),
                            -FLT_MAX, // min x
                            FLT_MAX, // max x
                            -FLT_MAX, // min y
                            FLT_MAX, // max y
                            -FLT_MAX, // min z
                            FLT_MAX, // max z
                            //clip_lower_data,
                            0,
                            //clip_higher_data,
                            0,
                            generateNormals->getValue(),
                            &ugeom))
    {
        return FAIL;
    }

    // output data already assigned to ports

    return SUCCESS;
}
