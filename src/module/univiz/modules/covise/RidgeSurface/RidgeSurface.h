/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Ridge Surface
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
    coInputPort *grid;
    coInputPort *scalar;
    coInputPort *clipScalar;

    // Outputs
    coOutputPort *ridges;
    coOutputPort *normals;

    // Parameters
    coIntScalarParamExt smoothingRange;
    coChoiceParam *mode;
    coChoiceParam *extremum;
    coBooleanParam *excludeFLT_MAX;
    coBooleanParam *excludeLonelyNodes;
    coFloatParamExt HessExtrEigenvalMin;
    coFloatParamExt PCAsubdomMaxPerc;
    coFloatParamExt scalarMin;
    coFloatParamExt scalarMax;
    coFloatParamExt clipScalarMin;
    coFloatParamExt clipScalarMax;
    coIntScalarParamExt minSize;
    coBooleanParam *filterByCell;
    coBooleanParam *combineExceptions;
    coIntScalarParamExt maxExceptions;
    coBooleanParam *generateNormals;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "Ridge Surface")
    {
        // Inputs
        grid = addInputPort("grid", "UnstructuredGrid", "Scalar Grid");
        scalar = addInputPort("scalar", "Float", "Scalar Data");
        clipScalar = addInputPort("clipScalar", "Float", "Scalar Clip Data");
        clipScalar->setRequired(0);

        // Outputs
        ridges = addOutputPort("ridges", "Polygons", "Ridge Surfaces");
        normals = addOutputPort("normals", "Vec3", "Surface Normals");

        // Parameters
        smoothingRange.p = addInt32Param("smoothingRange", "smoothing range for gradient computation");
        mode = addChoiceParam("mode", "method for consistent eigenvector orientation");
        extremum = addChoiceParam("extremum", "ridges (maximum) or valleys (minimum)");
        excludeFLT_MAX = addBooleanParam("excludeFLT_MAX", "exclude nodes with FLT_MAX (produced e.g. by FLE module for marking nodes with invalid data)");
        excludeLonelyNodes = addBooleanParam("excludeLonelyNodes", "exclude lonely nodes (nodes that do not have enough neighbors (due to excludeFLT_MAX)");
        HessExtrEigenvalMin.p = addFloatParam("HessExtrEigenvalMin", "minimum absolute value of of second derivative across ridge (used for suppressing flat ridges)");
        PCAsubdomMaxPerc.p = addFloatParam("PCAsubdomMaxPerc", "the second largest absolute eigenvalue must not be larger than this percentage of the largest absolute eigenvalue");
        scalarMin.p = addFloatParam("scalarMin", "minimum value of the scalar field for ridge extraction");
        scalarMax.p = addFloatParam("scalarMax", "maximum value of the scalar field for ridge extraction");
        clipScalarMin.p = addFloatParam("clipScalarMin", "minimum value of the scalar clipping field for clipped ridge extraction");
        clipScalarMax.p = addFloatParam("clipScalarMax", "maximum value of the scalar clipping field for clipped ridge extraction");
        minSize.p = addInt32Param("minSize", "ridges with less than this number of triangles are suppressed");
        filterByCell = addBooleanParam("filterByCell", "ridge filtering is based on cells. Otherwise it is based on cell edges (recommended).");
        combineExceptions = addBooleanParam("combineExceptions", "instead of rejecting a triangle if a condition is violated at any corner, the violations are summed up and the the triangle is rejected if the count reaches or exceeds maxExceptions");
        maxExceptions.p = addInt32Param("maxExceptions", "a triangle is rejected if it exhibits this count of exceptions");
        generateNormals = addBooleanParam("generateNormals", "generate normals. However it is recommended to use the GenNormals module instead.");

        // Default values
        smoothingRange.setValue(1, 1, INT_MAX);

        const char *modeChoice[] = { "cell nodes PCA", "edge nodes PCA" };
        mode->updateValue(2, modeChoice, 0);

        const char *extremumChoice[] = { "maximum ridges", "minimum ridges" };
        extremum->updateValue(2, extremumChoice, 0);

        excludeFLT_MAX->setValue(true);
        excludeLonelyNodes->setValue(true);
        HessExtrEigenvalMin.setValue(0.0, 0.0, FLT_MAX);
        PCAsubdomMaxPerc.setValue(1.0, 0.0, 1.0);
        scalarMin.setValue(0, -FLT_MAX, FLT_MAX);
        scalarMax.setValue(1e20, -FLT_MAX, FLT_MAX);
        clipScalarMin.setValue(0, -FLT_MAX, FLT_MAX);
        clipScalarMax.setValue(1e20, -FLT_MAX, FLT_MAX);
        minSize.setValue(1, 1, INT_MAX);
        filterByCell->setValue(false);
        combineExceptions->setValue(false);
        maxExceptions.setValue(1, 1, 3);
        generateNormals->setValue(false);
    }

    virtual ~myModule()
    {
    }
};
