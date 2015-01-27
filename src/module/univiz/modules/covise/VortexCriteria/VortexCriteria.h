/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Vortex Criteria
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
    coInputPort *grid;
    coInputPort *velocity;

    // Outputs
    coOutputPort *scalar;

    // Parameters
    coChoiceParam *quantity;
    coIntScalarParamExt smoothingRange;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Vortex Criteria")
    {
        grid = addInputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        velocity = addInputPort("velocity", "Vec3", "Vector Data");
        scalar = addOutputPort("scalar", "Float", "Scalar Criterion");

        quantity = addChoiceParam("quantity", "quantity");
        smoothingRange.p = addInt32Param("smoothingRange", "smoothing range for gradient");

        // default values
        const char *quantityChoice[] = { "helicity",
                                         "velo-norm helicity",
                                         "vorticity mag",
                                         "z vorticity",
                                         "lambda2",
                                         "Q",
                                         "delta",
                                         "div accel",
                                         "divergence" };
        quantity->updateValue(9, quantityChoice, 0);
        smoothingRange.setValue(1, 1, INT_MAX);
    }

    virtual ~myModule()
    {
    }
};
