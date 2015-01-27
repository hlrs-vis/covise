/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Finite Lyapunov Exponents
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
    void handleParamWidgets(void);

    // Inputs
    coInputPort *velocityGrid;
    coInputPort *velocity;
    coInputPort *samplingGrid;

    // Outputs
    coOutputPort *outputGrid;
    coOutputPort *FLE;
    coOutputPort *eigenvalMax;
    coOutputPort *eigenvalMed;
    coOutputPort *eigenvalMin;
    coOutputPort *integrationSize;
    coOutputPort *map;
    coOutputPort *trajectoriesGrid;
    coOutputPort *trajectoriesData;

    // Parameters
    coFloatVectorParam *origin;
    coIntVectorParam *cells;
    coFloatParamExt cellSize;
    coBooleanParam *unsteady;
    coFileBrowserParam *velocityFile;
    coFloatParamExt startTime;
    coChoiceParam *mode;
    coBooleanParam *ln;
    coBooleanParam *divT;
    coFloatParamExt integrationTime;
    coFloatParamExt integrationLength;
    coIntScalarParamExt timeIntervals;
    coFloatParamExt sepFactorMin;
    coIntScalarParamExt integStepsMax;
    coBooleanParam *forward;
    coIntScalarParamExt smoothingRange;
    coBooleanParam *omitBoundaryCells;
    coBooleanParam *gradNeighDisabled;
    coBooleanParam *execute;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Finite Lyapunov Exponents")
    {
        // Inputs
        velocityGrid = addInputPort("velocityGrid", "UnstructuredGrid", "Vector Grid");
        velocity = addInputPort("velocity", "Vec3", "Vector Data");
        samplingGrid = addInputPort("samplingGrid", "UnstructuredGrid", "Sampling Grid");
        samplingGrid->setRequired(0);

        // Outputs
        outputGrid = addOutputPort("outputGrid", "UnstructuredGrid", "Output Grid");
        FLE = addOutputPort("FLE", "Float", "Finite Lyapunov Exponent");
        eigenvalMax = addOutputPort("eigenvalMax", "Float", "Maximum Eigenvalue");
        eigenvalMed = addOutputPort("eigenvalMed", "Float", "Medium Eigenvalue");
        eigenvalMin = addOutputPort("eigenvalMin", "Float", "Minimum Eigenvalue");
        integrationSize = addOutputPort("integrationSize", "Float", "Integration Time/Length");
        map = addOutputPort("map", "Vec3", "Flow Map");

        trajectoriesGrid = addOutputPort("trajectoriesGrid", "StructuredGrid", "Trajectories Grid");
        trajectoriesData = addOutputPort("trajectoriesData", "Vec2", "Trajectories Data");

        // Parameters
        origin = addFloatVectorParam("origin", "origin of sampling grid");
        cells = addInt32VectorParam("cells", "number of cells per dimension of sampling grid");
        cellSize.p = addFloatParam("cellSize", "side length of a cell in sampling grid");
        unsteady = addBooleanParam("unsteady", "unsteady vector field input");
        velocityFile = addFileBrowserParam("velocityFile", "velocity file for unsteady input");
        startTime.p = addFloatParam("startTime", "start time for integration");
        mode = addChoiceParam("mode", "Finite Lyapunov exponent variant. FLLE and FTLEA are usually of little use. See the cited paper for a description (of FTLEM).");
        ln = addBooleanParam("ln", "compute logarithm");
        divT = addBooleanParam("divT", "divide by T");
        integrationTime.p = addFloatParam("integrationTime", "integration time");
        integrationLength.p = addFloatParam("integrationLength", "integration length");
        timeIntervals.p = addInt32Param("timeIntervals", "Number of time intervals. Used e.g. for FSLE, larger values produce better quantization but use more memory.");
        sepFactorMin.p = addFloatParam("sepFactorMin", "minimum separation factor");
        integStepsMax.p = addInt32Param("integStepsMax", "maximum number of integration steps");
        forward = addBooleanParam("forward", "integration in forward direction");
        smoothingRange.p = addInt32Param("smoothingRange", "smoothing range for gradient computation");
        omitBoundaryCells = addBooleanParam("omitBoundaryCells", "omit boundary cells");
        gradNeighDisabled = addBooleanParam("gradNeighDisabled", "do not compute and mark gradient at undefined flow map nodes");
        execute = addBooleanParam("execute", "allow execution");

        // Default values
        cells->setValue(0, 10);
        cells->setValue(1, 10);
        cells->setValue(2, 10);
        cellSize.setValue(0.01, 0.0, FLT_MAX);
        unsteady->setValue(false);
        //velocityFile->setValue("velocityFile","data/visit/tube.dat *.dat*");
        velocityFile->disable();
        startTime.setValue(0.0, -FLT_MAX, FLT_MAX);
        const char *modeChoice[] = { "FTLE",
                                     "FLLE",
                                     "FSLE",
                                     "FTLEM",
                                     "FTLEA" };
        mode->updateValue(5, modeChoice, 0);
        ln->setValue(true);
        divT->setValue(true);
        integrationTime.setValue(0.1, 0.0, FLT_MAX);
        integrationLength.setValue(0.1, 0.0, FLT_MAX);
        integrationLength.disable();
        timeIntervals.setValue(50, 1, INT_MAX);
        timeIntervals.disable();
        sepFactorMin.setValue(1.1, 1.0, FLT_MAX);
        sepFactorMin.disable();
        integStepsMax.setValue(100, 1, INT_MAX);
        forward->setValue(true);
        smoothingRange.setValue(1, 1, INT_MAX);
        omitBoundaryCells->setValue(false);
        gradNeighDisabled->setValue(true);
        execute->setValue(true);
    }

    virtual ~myModule()
    {
    }
};
