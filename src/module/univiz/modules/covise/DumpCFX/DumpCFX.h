/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// DumpCFX
// Filip Sadlo 2008
// Computer Graphics Laboratory, ETH Zurich

#include "api/coSimpleModule.h"

#include "covise_ext.h"

#define OUTPUT_ENABLE 0

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
#if OUTPUT_ENABLE
    coOutputPort *grid;
    //coOutputPort *scalar;
    coOutputPort *vector;
#endif

    // Parameters
    coFileBrowserParam *fileName;
    //coChoiceParam       *scalarComponent;
    coChoiceParam *vectorComponent;
    coIntScalarParamExt domain;
    coIntScalarParamExt levelOfInterest;
    coIntScalarParamExt firstTimeStep;
    coIntScalarParamExt timeStepNb;
    coIntScalarParamExt mmapFileSizeMax;
    coBooleanParam *generateMMapFiles;
    coBooleanParam *deleteDumpFiles;
    coFileBrowserParam *outputPath;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "DumpCFX")
    {
#if OUTPUT_ENABLE
        grid = addOutputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        //scalar           = addOutputPort("scalar", "Float", "Scalar Data");
        vector = addOutputPort("vector", "Vec3", "Vector Data");
#endif

        fileName = addFileBrowserParam("fileName", "name for unstructured input");
        //scalarComponent  = addChoiceParam( "scalar_component",   "scalar component");
        vectorComponent = addChoiceParam("vector_component", "vector component");
        domain.p = addInt32Param("domain", "domain to read, 0 for all domains.");
        levelOfInterest.p = addInt32Param("levelOfInterest", "level of interest, 3 for all variables");
        firstTimeStep.p = addInt32Param("firstTimeStep", "first time step to read");
        timeStepNb.p = addInt32Param("timeStepNb", "number of time steps to read, 0 for all time steps");
        mmapFileSizeMax.p = addInt32Param("mmapFileSizeMax", "Maximum size of mmap file (in bytes). Set to 0 if address space is large enough (64-bit systems), then only a single file is generated. Otherwise set to e.g. 300 MB, so multiple files are generated.");
        generateMMapFiles = addBooleanParam("generateMMapFiles", "generate mmap file(s), used e.g. by the FLE module for path line integration");
        deleteDumpFiles = addBooleanParam("deleteDumpFiles", "delete dump files (after generation of mmap files)");
        outputPath = addFileBrowserParam("outputPath", "path for the dump file output");

        // default values
        //scalarComponent->setValue(1, defaultChoice, 0);
        vectorComponent->setValue(1, defaultChoice, 0);
        domain.setValue(0, 0, INT_MAX);
        levelOfInterest.setValue(1, 1, 3);
        firstTimeStep.setValue(1, 1, INT_MAX);
        timeStepNb.setValue(1, 0, INT_MAX);
        mmapFileSizeMax.setValue(0, 0, INT_MAX);
        generateMMapFiles->setValue(true);
        deleteDumpFiles->setValue(true);
    }

    virtual ~myModule()
    {
    }
};
