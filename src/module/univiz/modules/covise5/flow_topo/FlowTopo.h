/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coModule.h"

class myModule : public coModule
{
private:
    // Member functions
    virtual void postInst();
    virtual void param(const char *name);
    virtual int compute();

    // Inputs
    coInPort *grid;
    coInPort *velocity;
    coInPort *wallDistance;

    // Ouputs
    coOutPort *criticalPoints;
    coOutPort *criticalPointsData;

    // Parameters
    coBooleanParam *divideByWallDist;
    coBooleanParam *interiorCritPoints;
    coBooleanParam *boundaryCritPoints;
    coBooleanParam *generateSeeds;
    coIntScalarParam *seedsPerCircle;
    coFloatScalarParam *radius;
    coFloatScalarParam *offset;

public:
    myModule()
        : coModule("Flow Topology")
    {
        // ports
        grid = addInPort("grid", "DO_UnstructuredGrid",
                         "Unstructured grid");
        velocity = addInPort("velocity", "DO_Unstructured_V3D_Data",
                             "Vector data");
        wallDistance = addInPort("wallDistance", "DO_Unstructured_S3D_Data",
                                 "Scalar data");
        wallDistance->setRequired(0);
        criticalPoints = addOutPort("criticalPoints", "DO_StructuredGrid",
                                    "Structured grid");
        criticalPointsData = addOutPort("criticalPointsData", "DO_Structured_S3D_Data",
                                        "Scalar data");

        // params
        divideByWallDist = addBooleanParam("divideByWallDist", "divide by wall distance");
        interiorCritPoints = addBooleanParam("interiorCritPoints", "compute interior critical points");
        boundaryCritPoints = addBooleanParam("boundaryCritPoints", "compute boundary critical points");
        generateSeeds = addBooleanParam("generateSeeds", "generate seeds");
        seedsPerCircle = addIntScalarParam("seedsPerCircle", "seeds per circle");
        radius = addFloatScalarParam("radius", "radius");
        offset = addFloatScalarParam("offset", "offset");

        // param default values
        divideByWallDist->setValue(false);
        interiorCritPoints->setValue(true);
        boundaryCritPoints->setValue(false);
        generateSeeds->setValue(false);
        seedsPerCircle->setValue(8);
        radius->setValue(1.0);
        offset->setValue(1.0);
    }

    virtual ~myModule()
    {
    }
};
