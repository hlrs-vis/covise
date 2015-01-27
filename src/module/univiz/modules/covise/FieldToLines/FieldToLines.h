/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Field To Lines
// Filip Sadlo 2007
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
    coInputPort *lineFieldGrid;
    coInputPort *lineFieldData;

    // Outputs
    coOutputPort *lines;

    // Parameters
    coIntScalarParam *nodesX;
    coIntScalarParam *nodesY;
    coIntScalarParam *nodesZ;
    coIntScalarParamExt stride;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Field To Lines")
    {
        lineFieldGrid = addInputPort("lineFieldGrid", "StructuredGrid", "Structured Grid");
        lineFieldData = addInputPort("lineFieldData", "Vec2", "Line Data");
        lines = addOutputPort("lines", "Lines", "Lines From Field");

        nodesX = addInt32Param("nodesX", "number of nodes in grid in x-direction");
        nodesY = addInt32Param("nodesY", "number of nodes in grid in y-direction");
        nodesZ = addInt32Param("nodesZ", "number of nodes in grid in z-direction");
        stride.p = addInt32Param("stride", "take each stride\'th line in each dimension");

        // default values
        nodesX->setValue(10);
        nodesY->setValue(10);
        nodesZ->setValue(10);
        stride.setValue(1, 1, INT_MAX);
    }

    virtual ~myModule()
    {
    }
};
