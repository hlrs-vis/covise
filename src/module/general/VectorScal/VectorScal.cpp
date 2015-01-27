/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE VectScalNew  application module                   **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  R. Beller, Sasha Cioringa                                     **
 **                                                                        **
 **                                                                        **
 ** Date:  01.12.97  V1.0                                                  **
 ** Date   08.11.00                                                        **
\**************************************************************************/

#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include "VectorScal.h"

using namespace covise;

VectScal::VectScal(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Convert Vectors to Scalars")
{
    const char *ChoiseVal[] = { "Length", "X-Component", "Y-Component", "Z-Component", "Alpha", "X-Y", "X-Z", "Y-Z" };

    //parameters

    p_option = addChoiceParam("option", "Options");
    p_option->setValue(8, ChoiseVal, 0);

    //ports
    p_inPort = addInputPort("vdataIn", "TriangleStrips|Polygons|UnstructuredGrid|Vec3|RGBA", "input vector data");
    p_outPort = addOutputPort("sdataOut", "Float|Vec3", "output scalar or vector data");
}

int VectScal::compute(const char *)
{

    int i;
    int num_values;

    float *u = NULL, *v = NULL, *w = NULL, *a = NULL;
    bool deleteData = false;
    float *s_out;
    float *v1_out;
    float *v2_out;
    float *v3_out;

    coDoFloat *u_scalar_data = NULL;
    coDoVec3 *u_vector_data = NULL;

    int option = p_option->getValue();

    const coDistributedObject *obj = p_inPort->getCurrentObject();

    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort->getName());
        return FAIL;
    }
    if (obj->isType("USTVDT"))
    {
        // unstructured vector data
        const coDoVec3 *vector_data = (const coDoVec3 *)obj;
        num_values = vector_data->getNumPoints();
        vector_data->getAddresses(&u, &v, &w);
    }
    // ------- Unstructured grid
    else if (obj->isType("UNSGRD"))
    {
        // unstructured vector data
        const coDoUnstructuredGrid *unsgrd = (const coDoUnstructuredGrid *)obj;
        int numConn, numElem;
        int *elemList, *connList;
        unsgrd->getAddresses(&elemList, &connList, &u, &v, &w);
        unsgrd->getGridSize(&numElem, &numConn, &num_values);
    }
    // ------- Polygons
    else if (obj->isType("POLYGN"))
    {
        // unstructured vector data
        const coDoPolygons *polygons = (const coDoPolygons *)obj;
        int *elemList, *connList;
        num_values = polygons->getNumPoints();
        polygons->getAddresses(&u, &v, &w, &elemList, &connList);
    }
    // ------- coDoTriangleStrips
    else if (obj->isType("TRIANG"))
    {
        // unstructured vector data
        const coDoTriangleStrips *tristrips = (const coDoTriangleStrips *)obj;
        int *elemList, *connList;
        num_values = tristrips->getNumPoints();
        tristrips->getAddresses(&u, &v, &w, &elemList, &connList);
    }
    // ------- coDoTriangleStrips
    else if (dynamic_cast<const coDoRGBA *>(obj))
    {
        // unstructured vector data
        const coDoRGBA *colors = (const coDoRGBA *)obj;
        num_values = colors->getNumPoints();
        deleteData = true;
        u = new float[num_values];
        v = new float[num_values];
        w = new float[num_values];
        a = new float[num_values];
        for (int i = 0; i < num_values; i++)
        {
            colors->getFloatRGBA(i, u + i, v + i, w + i, a + i);
        }
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort->getName());
        return FAIL;
    }

    if (option > 4)
    {
        // unstructured vector data output
        u_vector_data = new coDoVec3(p_outPort->getObjName(), num_values);
        if (!u_vector_data->objectOk())
        {
            const char *name = NULL;
            if (p_outPort->getCurrentObject())
                name = p_outPort->getCurrentObject()->getName();
            sendError("Failed to create the object '%s' for the port '%s'", name, p_outPort->getName());
            return FAIL;
        }
        u_vector_data->getAddresses(&v1_out, &v2_out, &v3_out);
    }
    else
    {
        // unstructured scalar data output
        u_scalar_data = new coDoFloat(p_outPort->getObjName(), num_values);
        if (!u_scalar_data->objectOk())
        {
            const char *name = NULL;
            if (p_outPort->getCurrentObject())
                name = p_outPort->getCurrentObject()->getName();
            sendError("Failed to create the object '%s' for the port '%s'", name, p_outPort->getName());
            return FAIL;
        }
        u_scalar_data->getAddress(&s_out);
    }
    switch (option)
    {

    case 0: // length of vectors of the discreet vector field
        if (a)
        {
            for (i = 0; i < num_values; i++)
            {
                *s_out = sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i] + a[i] * a[i]);
                s_out++;
            }
        }
        else
        {
            for (i = 0; i < num_values; i++)
            {
                *s_out = sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
                s_out++;
            }
        }
        break;

    case 1: // x-coordinate of the discreet vector field
        if (num_values)
            memcpy(s_out, u, (size_t)(num_values * sizeof(float)));
        break;

    case 2: // y-coordinate of the discreet vector field
        if (num_values)
            memcpy(s_out, v, (size_t)(num_values * sizeof(float)));
        break;

    case 3: // z-coordinate of the discreet vector field
        if (num_values)
            memcpy(s_out, w, (size_t)(num_values * sizeof(float)));
        break;
    case 4: // Alpha values
        if (num_values && a)
            memcpy(s_out, a, (size_t)(num_values * sizeof(float)));
        break;
    case 5: // Alpha values
        if (num_values && u && v)
        {
            memcpy(v1_out, u, (size_t)(num_values * sizeof(float)));
            memcpy(v2_out, v, (size_t)(num_values * sizeof(float)));
            for (int i = 0; i < num_values; i++)
            {
                v3_out[i] = 0.0;
            }
        }
        break;
    case 6: // Alpha values
        if (num_values && u && w)
        {
            memcpy(v1_out, u, (size_t)(num_values * sizeof(float)));
            memcpy(v3_out, w, (size_t)(num_values * sizeof(float)));
            for (int i = 0; i < num_values; i++)
            {
                v2_out[i] = 0.0;
            }
        }
        break;
    case 7: // Alpha values
        if (num_values && v && w)
        {
            memcpy(v2_out, v, (size_t)(num_values * sizeof(float)));
            memcpy(v3_out, w, (size_t)(num_values * sizeof(float)));
            for (int i = 0; i < num_values; i++)
            {
                v1_out[i] = 0.0;
            }
        }
        break;
    }
    if (option > 4)
        p_outPort->setCurrentObject(u_vector_data);
    else
        p_outPort->setCurrentObject(u_scalar_data);

    if (deleteData)
    {
        delete[] u;
        delete[] v;
        delete[] w;
        delete[] a;
    }

    return SUCCESS;
}

MODULE_MAIN(Tools, VectScal)
