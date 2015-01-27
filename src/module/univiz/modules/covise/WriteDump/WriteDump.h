/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// WriteDump
// Filip Sadlo 2008
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
    //coInputPort		*scalar;
    coInputPort *vector;

    // Outputs

    // Parameters
    coFileBrowserParam *fileName;
    coBooleanParam *sequenceName;
    coFloatParamExt startTime;
    coFloatParamExt timeStep;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "WriteDump")
    {
        grid = addInputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        vector = addInputPort("vector", "Vec3", "Vector Data");
        //vector->setRequired(0);
        //scalar         = addInputPort("scalar", "Float", "Scalar Data");
        //scalar->setRequired(0);

        fileName = addFileBrowserParam("fileName", "name for unstructured output");
        sequenceName = addBooleanParam("sequenceName", "for sequential name");
        startTime.p = addFloatParam("startTime", "start time");
        timeStep.p = addFloatParam("timeStep", "time step");

        // default values
        sequenceName->setValue(false);
        startTime.setValue(0.0, -FLT_MAX, FLT_MAX);
        timeStep.setValue(1.0, -FLT_MAX, FLT_MAX);
    }

    virtual ~myModule()
    {
    }
};
