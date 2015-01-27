/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Statistics
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
    coInputPort *scalar;
    coInputPort *vector;

    // Outputs

    // Parameters

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Statistics")
    {
        grid = addInputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        vector = addInputPort("vector", "Vec3", "Vector Data");
        vector->setRequired(0);
        scalar = addInputPort("scalar", "Float", "Scalar Data");
        scalar->setRequired(0);

        // default values
    }

    virtual ~myModule()
    {
    }
};
