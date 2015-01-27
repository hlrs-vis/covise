/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
    coInputPort *wallDistance;

    // Outputs
    coOutputPort *criticalPoints;
    coOutputPort *criticalPointsData;

    // Parameters
    coBooleanParam *divideByWallDist;
    coBooleanParam *interiorCritPoints;
    coBooleanParam *boundaryCritPoints;
    coBooleanParam *generateSeeds;
    coIntScalarParamExt seedsPerCircle;
    coFloatParamExt radius;
    coFloatParamExt offset;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Flow Topology")
    {
        // ports
        grid = addInputPort("grid", "UnstructuredGrid",
                            "Unstructured Grid");
        velocity = addInputPort("velocity", "Vec3", "Vector Data");
        wallDistance = addInputPort("wallDistance", "Float", "Scalar Data");
        wallDistance->setRequired(0);
        criticalPoints = addOutputPort("criticalPoints", "StructuredGrid",
                                       "Critical Points");
        criticalPointsData = addOutputPort("criticalPointsData", "Float",
                                           "Critical Points Data");

        // params
        divideByWallDist = addBooleanParam("divideByWallDist", "divide by wall distance");
        interiorCritPoints = addBooleanParam("interiorCritPoints", "compute interior critical points");
        boundaryCritPoints = addBooleanParam("boundaryCritPoints", "compute boundary critical points");
        generateSeeds = addBooleanParam("generateSeeds", "generate seeds");
        seedsPerCircle.p = addInt32Param("seedsPerCircle", "seeds per circle");
        radius.p = addFloatParam("radius", "radius");
        offset.p = addFloatParam("offset", "offset");

        // param default values
        divideByWallDist->setValue(false);
        interiorCritPoints->setValue(true);
        boundaryCritPoints->setValue(false);
        generateSeeds->setValue(false);
        seedsPerCircle.setValue(8, 1, INT_MAX);
        radius.setValue(1.0, 0.0, FLT_MAX);
        offset.setValue(1.0, 0.0, FLT_MAX);
    }

    virtual ~myModule()
    {
    }
};
