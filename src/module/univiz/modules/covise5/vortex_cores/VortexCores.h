/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vortex Core Lines
// Ronald Peikert, Martin Roth, Dirk Bauer <=2005 and Filip Sadlo >=2006
// Computer Graphics Laboratory, ETH Zurich

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

    // Ouputs
    coOutPort *coreLines;

    // Parameters
    coChoiceParam *method;
    coChoiceParam *variant;
    coIntScalarParam *minVertices;
    coIntScalarParam *maxExceptions;
    coFloatScalarParam *minStrength;
    coFloatScalarParam *maxAngle;

public:
    myModule()
        : coModule("Vortex Corelines")
    {
        grid = addInPort("grid", "DO_UnstructuredGrid",
                         "Unstructured grid");
        velocity = addInPort("velocity", "DO_Unstructured_V3D_Data",
                             "Vector data");
        coreLines = addOutPort("coreLines", "DO_Lines", "Lines");

        method = addChoiceParam("method", "method");
        variant = addChoiceParam("variant", "variant");
        minVertices = addIntScalarParam("minVertices", "minimal number of vertices");
        maxExceptions = addIntScalarParam("maxExceptions", "maximal number of exceptions");
        minStrength = addFloatScalarParam("minStrength", "minimal strength");
        maxAngle = addFloatScalarParam("maxAngle", "maximal angle between v and w field");

        // default values
        char *methodChoice[] = { "Levy", "Sujudi-Haimes" };
        method->updateValue(2, methodChoice, 1);

        char *variantChoice[] = { "triangle", "quad Newton" };
        variant->updateValue(2, variantChoice, 1);

        minVertices->setValue(10);
        maxExceptions->setValue(3);
        minStrength->setValue(1.0);
        maxAngle->setValue(30.0);
    }

    virtual ~myModule()
    {
    }
};
