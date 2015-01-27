/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ReadUnstructured
// Filip Sadlo 2008
// Computer Graphics Laboratory, ETH Zurich

#include "api/coSimpleModule.h"

#include "covise_ext.h"

const char *defaultChoice[] = { "no choice" };

std::vector<myParam *> myParams;

class myModule : public coSimpleModule
{
private:
    // Member functions
    virtual void postInst();
    virtual void param(const char *name, bool);
    virtual int compute(const char *);

    // Inputs

    // Outputs
    coOutputPort *grid;
    coOutputPort *scalar;
    coOutputPort *vector;

    // Parameters
    coFileBrowserParam *fileName;
    coChoiceParam *scalarComponent;
    coChoiceParam *vectorComponent;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "ReadUnstructured")
    {
        grid = addOutputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        scalar = addOutputPort("scalar", "Float", "Scalar Data");
        vector = addOutputPort("vector", "Vec3", "Vector Data");

        fileName = addFileBrowserParam("fileName", "name for unstructured input");
        scalarComponent = addChoiceParam("scalar_component", "scalar component");
        vectorComponent = addChoiceParam("vector_component", "vector component");

        // default values
        scalarComponent->setValue(1, defaultChoice, 0);
        vectorComponent->setValue(1, defaultChoice, 0);
    }

    virtual ~myModule()
    {
    }
};
