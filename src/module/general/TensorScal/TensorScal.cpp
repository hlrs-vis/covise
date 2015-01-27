/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                     (C)2001 Vircinity  **
 **                                                                        **
 ** Description:  COVISE TensScal  application module                      **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  S. Leseduarte                                                 **
 **                                                                        **
 **                                                                        **
 ** Date:  01.12.97  V1.0                                                  **
 ** Date   08.11.00                                                        **
\**************************************************************************/

#include "TensorScal.h"
#include <do/coDoData.h>
#include <do/coDoData.h>

TensScal::TensScal(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Calculate invariants from Tensors")
{
    const char *ChoiseVal[] = { "Spur", "Stress effective value" };

    //parameters

    p_option = addChoiceParam("option", "Options");
    p_option->setValue(2, ChoiseVal, 1);

    //ports
    p_inPort = addInputPort("vdataIn", "Tensor", "input tensor data");
    p_outPort = addOutputPort("sdataOut", "Float", "output scalar data");
}

int TensScal::compute(const char *)
{

    coDoFloat *u_scalar_data = NULL;

    //int option = p_option->getValue();

    const coDistributedObject *obj = p_inPort->getCurrentObject();

    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort->getName());
        return FAIL;
    }
    else if (obj->isType("USTTDT"))
    {
        // unstructured tensor data
        const coDoTensor *tensor = (const coDoTensor *)(obj);
        coDoTensor::TensorType type = tensor->getTensorType();
        int option = p_option->getValue();
        float *scalar = NULL;
        int nopoints = tensor->getNumPoints();
        float *t_addr = NULL;
        tensor->getAddress(&t_addr);

        // it would be better to introduce new methods in the
        // tensor class, but now we may not change the libraries...
        switch (type)
        {
        case coDoTensor::S2D:
            switch (option)
            {
            case 0:
                scalar = S2D_Spur(nopoints, t_addr);
                break;
            case 1:
                scalar = S2D_Stress(nopoints, t_addr);
                break;
            default:
                break;
            }
            break;
        case coDoTensor::F2D:
            switch (option)
            {
            case 0:
                scalar = F2D_Spur(nopoints, t_addr);
                break;
            case 1:
                sendWarning("Stress effective value only defined for symmetric tensors");
            default:
                break;
            }
            break;
        case coDoTensor::S3D:
            switch (option)
            {
            case 0:
                scalar = S3D_Spur(nopoints, t_addr);
                break;
            case 1:
                scalar = S3D_Stress(nopoints, t_addr);
                break;
            default:
                break;
            }
            break;
        case coDoTensor::F3D:
            switch (option)
            {
            case 0:
                scalar = S3D_Spur(nopoints, t_addr);
                break;
            case 1:
                sendWarning("Stress effective value only defined for symmetric tensors");
                break;
            default:
                break;
            }
            break;
        case coDoTensor::UNKNOWN:
        default:
            break;
        }

        if (scalar == NULL)
            return FAIL;

        // unstructured scalar data output
        u_scalar_data = new coDoFloat(p_outPort->getObjName(), nopoints, scalar);
        delete[] scalar;
        if (!u_scalar_data->objectOk())
        {
            sendError("Failed to create the object '%s' for the port '%s'", p_outPort->getObjName(), p_outPort->getName());
            return FAIL;
        }
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort->getName());
        return FAIL;
    }
    p_outPort->setCurrentObject(u_scalar_data);

    return SUCCESS;
}

float *
TensScal::S2D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 3];
        ret[point] += t_addr[point * 3 + 1];
        // ret[point] *= 0.5;
    }
    return ret;
}

float *
TensScal::S2D_Stress(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = sqrt((t_addr[point * 3] - t_addr[point * 3 + 1]) * (t_addr[point * 3] - t_addr[point * 3 + 1]) + 4.0f * t_addr[point * 3 + 2] * t_addr[point * 3 + 2]);
    }
    return ret;
}

float *
TensScal::S3D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 6];
        ret[point] += t_addr[point * 6 + 1];
        ret[point] += t_addr[point * 6 + 2];
        // ret[point] *= 0.33333333333333333;
    }
    return ret;
}

float *
TensScal::S3D_Stress(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        float spur = t_addr[point * 6] + t_addr[point * 6 + 1] + t_addr[point * 6 + 2];
        spur *= 0.3333333333333f;
        float contrib = t_addr[point * 6] - spur;
        ret[point] = contrib * contrib;
        contrib = t_addr[point * 6 + 1] - spur;
        ret[point] += contrib * contrib;
        contrib = t_addr[point * 6 + 2] - spur;
        ret[point] += contrib * contrib;
        contrib = t_addr[point * 6 + 3];
        ret[point] += 2.0f * contrib * contrib;
        contrib = t_addr[point * 6 + 4];
        ret[point] += 2.0f * contrib * contrib;
        contrib = t_addr[point * 6 + 5];
        ret[point] += 2.0f * contrib * contrib;

        ret[point] *= 1.5;

        ret[point] = sqrt(ret[point]);
    }
    return ret;
}

float *
TensScal::F2D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 4];
        ret[point] += t_addr[point * 4 + 3];
        // ret[point] *= 0.5;
    }
    return ret;
}

float *
TensScal::F3D_Spur(int nopoints, const float *t_addr)
{
    int point;
    float *ret = new float[nopoints];
    for (point = 0; point < nopoints; ++point)
    {
        ret[point] = t_addr[point * 9];
        ret[point] += t_addr[point * 9 + 4];
        ret[point] += t_addr[point * 9 + 8];
        // ret[point] *= 0.33333333333333333;
    }
    return ret;
}

MODULE_MAIN(Tools, TensScal)
