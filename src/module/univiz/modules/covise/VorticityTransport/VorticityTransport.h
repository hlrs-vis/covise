/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vorticity Transport
// Filip Sadlo 2006 - 2008
// Computer Graphics Laboratory, ETH Zurich

#include "api/coSimpleModule.h"

#include "covise_ext.h"
std::vector<myParam *> myParams;

class myModule : public coSimpleModule
{
private:
    // Member functions
    virtual void postInst();
    virtual void param(const char *name, bool);
    virtual int compute(const char *);

    // Inputs
    coInputPort *grid;
    coInputPort *velocity;
    coInputPort *coreLines;

    // Outputs
    coOutputPort *tubes;
    coOutputPort *pathLines;
    coOutputPort *test;

    // Parameters
    coFloatParamExt startTime;
    coFloatParamExt integrationTime;
    coIntScalarParamExt integStepsMax;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Vorticity Transport")
    {
        // Inputs
        grid = addInputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        velocity = addInputPort("velocity", "Vec3", "Vector Data");
        coreLines = addInputPort("coreLines", "Lines", "Vortex Core Lines");

        // Outputs
        tubes = addOutputPort("tubes", "Polygons", "Striped Tubes");
        pathLines = addOutputPort("pathLines", "Lines", "Path Lines");

        // Parameters
        startTime.p = addFloatParam("startTime", "start time for integration");
        integrationTime.p = addFloatParam("integrationTime", "integration time");
        integStepsMax.p = addInt32Param("integStepsMax", "maximum number of integration steps");

        // default values
        startTime.setValue(0.0, -FLT_MAX, FLT_MAX);
        integrationTime.setValue(0.1, 0.0, FLT_MAX);
        integStepsMax.setValue(100, 1, INT_MAX);
    }

    virtual ~myModule()
    {
    }
};
