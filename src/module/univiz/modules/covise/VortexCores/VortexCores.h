/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vortex Core Lines
// Ronald Peikert, Martin Roth <=2005 and Filip Sadlo >=2006
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

    // Outputs
    coOutputPort *coreLines;

    // Parameters
    coChoiceParam *method;
    coChoiceParam *variant;
    coIntScalarParamExt minVertices;
    coIntScalarParamExt maxExceptions;
    coFloatParamExt minStrength;
    coFloatParamExt maxAngle;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Vortex Corelines")
    {
        grid = addInputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        velocity = addInputPort("velocity", "Vec3", "Vector Data");
        coreLines = addOutputPort("coreLines", "Lines", "Vortex Core Lines");

        method = addChoiceParam("method", "method");
        variant = addChoiceParam("variant", "variant");
        minVertices.p = addInt32Param("minVertices", "minimal number of vertices");
        maxExceptions.p = addInt32Param("maxExceptions", "maximal number of exceptions");
        minStrength.p = addFloatParam("minStrength", "minimal strength");
        maxAngle.p = addFloatParam("maxAngle", "maximal angle between v and w field");

        // default values
        const char *methodChoice[] = { "Levy", "Sujudi-Haimes" };
        method->updateValue(2, methodChoice, 0);

        const char *variantChoice[] = { "triangle", "quad Newton" };
        variant->updateValue(2, variantChoice, 0);

        minVertices.setValue(10, 2, INT_MAX);
        maxExceptions.setValue(3, 1, INT_MAX);
        minStrength.setValue(1.0, 0.0, FLT_MAX);
        maxAngle.setValue(30.0, 0.0, 90.0);
    }

    virtual ~myModule()
    {
    }
};
