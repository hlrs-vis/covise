/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DerivOperators.h"
#include <do/covise_gridmethods.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>

enum
{
    Gradient,
    Divergence,
    Curl,
    GradientMagnitude
};
const char *DerivOperators::operators[] = { "gradient", "divergence", "curl", "gradient magnitude" };

DerivOperators::DerivOperators(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Derivative-operator tool box")
{
    p_grid_ = addInputPort("GridIn0", "UnstructuredGrid", "input grid");
    p_inData_ = addInputPort("DataIn0", "Float|Vec3", "input data");
    p_outData_ = addOutputPort("DataOut0", "Float|Vec3", "output data");
    p_whatToDo_ = addChoiceParam("whatToDo", "Choose operator");
    p_whatToDo_->setValue(4, operators, 0);
    //   p_perCell_= addBooleanParam("perCell","Data per cell");
    //   p_perCell_->setValue(1);
}

DerivOperators::~DerivOperators()
{
}

void
DerivOperators::outputDummy()
{
    coDistributedObject *out = NULL;
    switch (p_whatToDo_->getValue())
    {
    case Gradient:
    case Curl:
        out = new coDoVec3(p_outData_->getObjName(), 0);
        break;
    case Divergence:
    case GradientMagnitude:
        out = new coDoFloat(p_outData_->getObjName(), 0);
        break;
    default:
        break;
    }
    p_outData_->setCurrentObject(out);
}

void
DerivOperators::copyAttributesToOutObj(coInputPort **input_ports,
                                       coOutputPort **output_ports,
                                       int i)
{
    if (i == 0 && output_ports[0]->getCurrentObject() && input_ports[1]->getCurrentObject())
    {
        copyAttributes(output_ports[0]->getCurrentObject(), input_ports[1]->getCurrentObject());
    }
}

int
DerivOperators::compute(const char *)
{
    // open object
    const coDistributedObject *in_grid = p_grid_->getCurrentObject();
    const coDistributedObject *in_data = p_inData_->getCurrentObject();
    const coDoUnstructuredGrid *inGrid = dynamic_cast<const coDoUnstructuredGrid *>(in_grid);
    if (!inGrid)
    {
        outputDummy();
        return SUCCESS;
    }
    int no_el;
    int no_vl;
    int no_points;
    int *el = NULL, *vl = NULL, *tl = NULL;
    float *xc = NULL, *yc = NULL, *zc = NULL;
    inGrid->getGridSize(&no_el, &no_vl, &no_points);
    inGrid->getAddresses(&el, &vl, &xc, &yc, &zc);
    inGrid->getTypeList(&tl);

    float *sdata = NULL;
    float *vdata[3] = { NULL, NULL, NULL };
    int option = p_whatToDo_->getValue();
    if (const coDoFloat *in_sdata = dynamic_cast<const coDoFloat *>(in_data))
    {
        int no_dpoints = in_sdata->getNumPoints();
        if (no_dpoints == 0)
        {
            outputDummy();
            return SUCCESS;
        }
        else if (no_dpoints != no_points)
        {
            sendError("Only input data per vertex is supported");
            return FAIL;
        }

        if (option != Gradient && option != GradientMagnitude)
        {
            sendWarning("Trying this operator on a scalar field results in a dummy object");
            outputDummy(); // this option is not valid for scalar fields
            return SUCCESS;
        }

        in_sdata->getAddress(&sdata);
    }
    else if (const coDoVec3 *in_vdata = dynamic_cast<const coDoVec3 *>(in_data))
    {
        int no_dpoints = in_vdata->getNumPoints();
        if (no_dpoints == 0)
        {
            outputDummy();
            return SUCCESS;
        }
        else if (no_dpoints != no_points)
        {
            sendError("Only input data per vertex is supported");
            return FAIL;
        }

        if (option == Gradient || option == GradientMagnitude)
        {
            sendWarning("Trying this operator on a vector field results in a dummy object");
            outputDummy(); // this option is not valid for scalar fields
            return SUCCESS;
        }

        in_vdata->getAddresses(&vdata[0], &vdata[1], &vdata[2]);
    }
    else
    {
        outputDummy();
        return SUCCESS;
    }

    switch (option)
    {
    case Gradient: // gradient
        if (gradient(no_el, no_vl, no_points, el, vl, tl, xc, yc, zc, sdata) != 0)
        {
            return FAIL;
        }
        break;
    case GradientMagnitude: // gradient magnitude
        if (gradientMagnitude(no_el, no_vl, no_points, el, vl, tl, xc, yc, zc, sdata) != 0)
        {
            return FAIL;
        }
        break;
    case Divergence: // divergence
        if (divergence(no_el, no_vl, no_points, el, vl, tl, xc, yc, zc, vdata) != 0)
        {
            return FAIL;
        }
        break;
    case Curl: // curl
        if (curl(no_el, no_vl, no_points, el, vl, tl, xc, yc, zc, vdata) != 0)
        {
            return FAIL;
        }
        break;
    default:
        sendError("Unsupported parameter option");
        return FAIL;
    }
    return SUCCESS;
}

int
DerivOperators::gradient(int no_el, int no_vl, int no_points,
                         const int *el, const int *vl, const int *tl,
                         const float *xc, const float *yc, const float *zc,
                         float *sdata)
{
    float **gradient[3];
    float *scalx, *scaly, *scalz;
    gradient[0] = &scalx;
    gradient[1] = &scaly;
    gradient[2] = &scalz;
    scalx = new float[no_el];
    scaly = new float[no_el];
    scalz = new float[no_el];
    if (grid_methods::derivativesAtCenter(gradient, no_points, 1, &sdata,
                                          no_el, no_vl, tl, el, vl, xc, yc, zc) != 0)
    {
        delete[] gradient[0][0];
        delete[] gradient[1][0];
        delete[] gradient[2][0];
        sendWarning("Found degenerate element");
        return -1;
    }
    // construct output object
    coDoVec3 *out = new coDoVec3(p_outData_->getObjName(), no_el,
                                 gradient[0][0], gradient[1][0], gradient[2][0]);
    p_outData_->setCurrentObject(out);
    delete[] gradient[0][0];
    delete[] gradient[1][0];
    delete[] gradient[2][0];
    return 0;
}

int
DerivOperators::gradientMagnitude(int no_el, int no_vl, int no_points,
                                  const int *el, const int *vl, const int *tl,
                                  const float *xc, const float *yc, const float *zc,
                                  float *sdata)
{
    float *arrays[] = { new float[no_el], new float[no_el], new float[no_el] };
    float **gradient[3] = { &arrays[0], &arrays[1], &arrays[2] };
    if (grid_methods::derivativesAtCenter(gradient, no_points, 1, &sdata,
                                          no_el, no_vl, tl, el, vl, xc, yc, zc) != 0)
    {
        delete[] gradient[0][0];
        delete[] gradient[1][0];
        delete[] gradient[2][0];
        sendWarning("Found degenerate element");
        return -1;
    }
    // construct output object
    coDoFloat *out = new coDoFloat(p_outData_->getObjName(), no_el);
    float *mag = out->getAddress();
    for (int i = 0; i < no_el; ++i)
        mag[i] = sqrt(gradient[0][0][i] * gradient[0][0][i]
                      + gradient[1][0][i] * gradient[1][0][i]
                      + gradient[2][0][i] * gradient[2][0][i]);
    delete[] gradient[0][0];
    delete[] gradient[1][0];
    delete[] gradient[2][0];
    p_outData_->setCurrentObject(out);
    return 0;
}

int
DerivOperators::divergence(int no_el, int no_vl, int no_points,
                           const int *el, const int *vl, const int *tl,
                           const float *xc, const float *yc, const float *zc,
                           float *vdata[3])
{
    float **gradient[3];
    float *gradX[3];
    float *gradY[3];
    float *gradZ[3];
    gradX[0] = new float[no_el];
    gradX[1] = new float[no_el];
    gradX[2] = new float[no_el];
    gradY[0] = new float[no_el];
    gradY[1] = new float[no_el];
    gradY[2] = new float[no_el];
    gradZ[0] = new float[no_el];
    gradZ[1] = new float[no_el];
    gradZ[2] = new float[no_el];
    gradient[0] = gradX;
    gradient[1] = gradY;
    gradient[2] = gradZ;
    if (grid_methods::derivativesAtCenter(gradient, no_points, 3, vdata,
                                          no_el, no_vl, tl, el, vl, xc, yc, zc) != 0)
    {
        delete[] gradient[0][0];
        delete[] gradient[1][0];
        delete[] gradient[2][0];
        delete[] gradient[0][1];
        delete[] gradient[1][1];
        delete[] gradient[2][1];
        delete[] gradient[0][2];
        delete[] gradient[1][2];
        delete[] gradient[2][2];
        sendWarning("Found degenerate element");
        return -1;
    }
    // construct output object
    int elem;
    for (elem = 0; elem < no_el; ++elem)
    {
        gradient[0][0][elem] += gradient[1][1][elem];
        gradient[0][0][elem] += gradient[2][2][elem];
    }
    coDoFloat *out = new coDoFloat(p_outData_->getObjName(), no_el,
                                   gradient[0][0]);
    p_outData_->setCurrentObject(out);
    delete[] gradient[0][0];
    delete[] gradient[1][0];
    delete[] gradient[2][0];
    delete[] gradient[0][1];
    delete[] gradient[1][1];
    delete[] gradient[2][1];
    delete[] gradient[0][2];
    delete[] gradient[1][2];
    delete[] gradient[2][2];
    return 0;
}

int
DerivOperators::curl(int no_el, int no_vl, int no_points,
                     const int *el, const int *vl, const int *tl,
                     const float *xc, const float *yc, const float *zc,
                     float *vdata[3])
{
    float **gradient[3];
    float *gradX[3];
    float *gradY[3];
    float *gradZ[3];
    gradX[0] = new float[no_el];
    gradX[1] = new float[no_el];
    gradX[2] = new float[no_el];
    gradY[0] = new float[no_el];
    gradY[1] = new float[no_el];
    gradY[2] = new float[no_el];
    gradZ[0] = new float[no_el];
    gradZ[1] = new float[no_el];
    gradZ[2] = new float[no_el];
    gradient[0] = gradX;
    gradient[1] = gradY;
    gradient[2] = gradZ;
    if (grid_methods::derivativesAtCenter(gradient, no_points, 3, vdata,
                                          no_el, no_vl, tl, el, vl, xc, yc, zc) != 0)
    {
        delete[] gradient[0][0];
        delete[] gradient[1][0];
        delete[] gradient[2][0];
        delete[] gradient[0][1];
        delete[] gradient[1][1];
        delete[] gradient[2][1];
        delete[] gradient[0][2];
        delete[] gradient[1][2];
        delete[] gradient[2][2];
        sendWarning("Found degenerate element");
        return -1;
    }
    // construct output object
    int elem;
    float *curl[3];
    curl[0] = new float[no_el];
    curl[1] = new float[no_el];
    curl[2] = new float[no_el];
    for (elem = 0; elem < no_el; ++elem)
    {
        curl[0][elem] = gradient[1][2][elem] - gradient[2][1][elem];
        curl[1][elem] = gradient[2][0][elem] - gradient[0][2][elem];
        curl[2][elem] = gradient[0][1][elem] - gradient[1][0][elem];
    }
    coDoVec3 *out = new coDoVec3(p_outData_->getObjName(), no_el,
                                 curl[0], curl[1], curl[2]);
    p_outData_->setCurrentObject(out);
    delete[] curl[0];
    delete[] curl[1];
    delete[] curl[2];
    delete[] gradient[0][0];
    delete[] gradient[1][0];
    delete[] gradient[2][0];
    delete[] gradient[0][1];
    delete[] gradient[1][1];
    delete[] gradient[2][1];
    delete[] gradient[0][2];
    delete[] gradient[1][2];
    delete[] gradient[2][2];
    return 0;
}

MODULE_MAIN(Tools, DerivOperators)
