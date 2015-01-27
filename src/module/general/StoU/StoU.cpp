/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE StoU application module                           **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner, Sasha Cioringa                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 ** Date:  05.09.98                                                        **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include "StoU.h"

StoU::StoU(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Convert structured to unstructured grids")
{
    const char *ChoiseVal[] = { "tetrahedrons", "hexahedrons", "pyramids", "prisms" };
    char buffer[10];
    int i;
    //parameters

    p_option = addChoiceParam("option", "Conversion Options");
    p_option->setValue(4, ChoiseVal, 1);

    p_inPort = new coInputPort *[MAX_DATA_PORTS + 1];
    p_outPort = new coOutputPort *[MAX_DATA_PORTS + 1];

    //ports
    p_inPort[0] = addInputPort("meshIn", "StructuredGrid|RectilinearGrid|UniformGrid", "Grid");

    for (i = 1; i < MAX_DATA_PORTS + 1; i++)
    {
        sprintf(buffer, "dataIn_%d", i);
        p_inPort[i] = addInputPort(buffer, "Float|Vec3|IntArr", "Data");
        p_inPort[i]->setRequired(0);
    }

    p_outPort[0] = addOutputPort("meshOut", "UnstructuredGrid", "unstructured Grid");
    for (i = 1; i < MAX_DATA_PORTS + 1; i++)
    {
        sprintf(buffer, "dataOut_%d", i);
        p_outPort[i] = addOutputPort(buffer, "Float|Vec3|IntArr", "unstructured data");
        p_outPort[i]->setDependencyPort(p_inPort[i]);
    }
}

StoU::~StoU()
{
    delete[] p_inPort;
    delete[] p_outPort;
}

int StoU::compute(const char *)
{
    int option_param, factor = 1;

    option_param = p_option->getValue();
    Buffer *buffer;

    coDoUnstructuredGrid *u_grid_out;
    const coDistributedObject *mesh_obj = p_inPort[0]->getCurrentObject();

    if (!mesh_obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort[0]->getName());
        return FAIL;
    }
    if (mesh_obj->isType("STRGRD"))
        buffer = new Buffer(this, (coDoStructuredGrid *)mesh_obj);
    else if (mesh_obj->isType("RCTGRD"))
        buffer = new Buffer(this, (coDoRectilinearGrid *)mesh_obj);
    else if (mesh_obj->isType("UNIGRD"))
        buffer = new Buffer(this, (coDoUniformGrid *)mesh_obj);
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort[0]->getName());
        return FAIL;
    }

    if (buffer->i_dim == 0 && buffer->j_dim == 0 && buffer->k_dim == 0)
        sendWarning("WARNING: Data object at port '%s' is empty", p_inPort[0]->getName());

    switch (option_param)
    {

    case 0:
        u_grid_out = buffer->create_tetrahedrons(p_outPort[0]->getObjName());
        factor = 6;
        break;
    case 1:
        u_grid_out = buffer->create_hexahedrons(p_outPort[0]->getObjName());
        factor = 1;
        break;
    case 2:
        u_grid_out = buffer->create_pyramids(p_outPort[0]->getObjName());
        factor = 3;
        break;
    case 3:
        u_grid_out = buffer->create_prisms(p_outPort[0]->getObjName());
        factor = 2;
        break;
    default:
        delete buffer;
        return FAIL;
        break;
    }

    p_outPort[0]->setCurrentObject(u_grid_out);

    //data

    for (int i = 1; i < MAX_DATA_PORTS + 1; i++)
    {

        const coDistributedObject *data_obj = p_inPort[i]->getCurrentObject();
        if (data_obj != NULL)
        {
            if (data_obj->isType("USTSDT"))
            {
                float *scalar_data;
                coDoFloat *in_sdata = ((coDoFloat *)data_obj);
                int nelem = in_sdata->getNumPoints();
                if (buffer->i_dim * buffer->j_dim * buffer->k_dim != nelem)
                {
                    delete buffer;
                    sendError("ERROR: The grid size does not match with the data size at port '%s'", p_inPort[i]->getName());
                    return FAIL;
                }
                else
                {
                    in_sdata->getAddress(&scalar_data);
                    coDoFloat *out_sdata = new coDoFloat(p_outPort[i]->getObjName(), nelem, scalar_data);
                    if (!out_sdata)
                    {
                        delete buffer;
                        sendError("Failed to  create object '%s' for port '%s' ", p_outPort[i]->getObjName(), p_outPort[i]->getName());
                        return FAIL;
                    }
                    p_outPort[i]->setCurrentObject(out_sdata);
                }
            }
            else if (data_obj->isType("USTVDT"))
            {
                float *u_data, *v_data, *w_data;
                coDoVec3 *in_vdata = ((coDoVec3 *)data_obj);
                int nelem = in_vdata->getNumPoints();
                if (buffer->i_dim * buffer->j_dim * buffer->k_dim != nelem)
                {
                    delete buffer;
                    sendError("ERROR: The grid size does not match with the data size at port '%s'", p_inPort[i]->getName());
                    return FAIL;
                }
                else
                {
                    in_vdata->getAddresses(&u_data, &v_data, &w_data);
                    coDoVec3 *out_vdata = new coDoVec3(p_outPort[i]->getObjName(), nelem, u_data, v_data, w_data);
                    if (!out_vdata)
                    {
                        delete buffer;
                        sendError("Failed to  create object '%s' for port '%s' ", p_outPort[i]->getObjName(), p_outPort[i]->getName());
                        return FAIL;
                    }
                    p_outPort[i]->setCurrentObject(out_vdata);
                }
            }
            else if (data_obj->isType("INTARR"))
            {
                coDoIntArr *in_arr = (coDoIntArr *)data_obj;
                if (option_param == 1)
                {
                    int numDim = in_arr->getNumDimensions();
                    int *dimArray = new int[numDim];
                    int *in_array, r;
                    in_arr->getAddress(&in_array);
                    for (r = 0; r < numDim; r++)
                        dimArray[r] = in_arr->getDimension(r);
                    p_outPort[i]->setCurrentObject(new coDoIntArr(p_outPort[i]->getObjName(), numDim, dimArray, in_array));
                    delete[] dimArray;
                }
                else
                {
                    int r, in_index = 0, out_index = 0;
                    int *in_array, *out_array;
                    int numDim = in_arr->getNumDimensions();
                    int *dimArray = new int[numDim];
                    coDoIntArr *out_arr = NULL;

                    in_arr->getAddress(&in_array);

                    for (r = 0; r < numDim; r++)
                        dimArray[r] = in_arr->getDimension(r) * factor;
                    out_arr = new coDoIntArr(p_outPort[i]->getObjName(), numDim, dimArray);
                    out_arr->getAddress(&out_array);

                    for (r = 0; r < numDim; r++)
                        for (int s = 0; s < in_arr->getDimension(r); s++)
                        {
                            for (int t = 0; t < factor; t++)
                                out_array[out_index++] = in_array[in_index];
                            in_index++;
                        }

                    p_outPort[i]->setCurrentObject(out_arr);
                    delete[] dimArray;
                }
            }
            else
            {
                delete buffer;
                sendError("Received illegal type at port '%s'", p_inPort[i]->getName());
                return FAIL;
            }
        }
    }

    delete buffer;
    return SUCCESS;
}

Buffer::Buffer(const coModule *mod, coDoStructuredGrid *s_grid_in)
    : module(mod)
{
    s_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
    s_grid_in->getAddresses(&x_in, &y_in, &z_in);
    x_S = x_in;
    y_S = y_in;
    z_S = z_in;
    alloc = 0;
}

Buffer::Buffer(const coModule *mod, coDoRectilinearGrid *r_grid_in)
    : module(mod)
{
    float *x_tmp, *y_tmp, *z_tmp;

    r_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
    r_grid_in->getAddresses(&x_in, &y_in, &z_in);
    x_S = new float[i_dim * j_dim * k_dim];
    y_S = new float[i_dim * j_dim * k_dim];
    z_S = new float[i_dim * j_dim * k_dim];

    alloc = 1;
    x_tmp = x_S;
    y_tmp = y_S;
    z_tmp = z_S;
    for (int i = 0; i < i_dim; i++)
        for (int j = 0; j < j_dim; j++)
            for (int k = 0; k < k_dim; k++)
            {
                *x_tmp = x_in[i];
                *y_tmp = y_in[j];
                *z_tmp = z_in[k];
                x_tmp++;
                y_tmp++;
                z_tmp++;
            }
}

Buffer::Buffer(const coModule *mod, coDoUniformGrid *u_grid_in)
    : module(mod)
{
    float *x_tmp, *y_tmp, *z_tmp;

    u_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);

    x_S = new float[i_dim * j_dim * k_dim];
    y_S = new float[i_dim * j_dim * k_dim];
    z_S = new float[i_dim * j_dim * k_dim];
    alloc = 1;
    x_tmp = x_S;
    y_tmp = y_S;
    z_tmp = z_S;
    for (int i = 0; i < i_dim; i++)
        for (int j = 0; j < j_dim; j++)
            for (int k = 0; k < k_dim; k++)
            {
                u_grid_in->getPointCoordinates(i, x_tmp, j, y_tmp, k, z_tmp);
                x_tmp++;
                y_tmp++;
                z_tmp++;
            }
}

Buffer::~Buffer()
{
    if (alloc)
    {
        delete[] x_S;
        delete[] y_S;
        delete[] z_S;
    }
}

//======================================================================
// create the unstructured grids
//======================================================================

coDoUnstructuredGrid *Buffer::create_hexahedrons(const char *objectname)
{
    const char *COLOR = "COLOR";
    const char *color = "pink";
    float *x_out, *y_out, *z_out;
    int *el, *cl, *tl, v, i, j, k, n;
    coDoUnstructuredGrid *grid_out;
    grid_out = new coDoUnstructuredGrid(objectname, (i_dim - 1) * (j_dim - 1) * (k_dim - 1), (i_dim - 1) * (j_dim - 1) * (k_dim - 1) * 8, i_dim * j_dim * k_dim, 1);
    if (grid_out->objectOk())
    {
        grid_out->getAddresses(&el, &cl, &x_out, &y_out, &z_out);
        grid_out->getTypeList(&tl);
        grid_out->addAttribute(COLOR, color);
        v = 0;
        n = 0;
        for (i = 1; i < i_dim; i++)
            for (j = 1; j < j_dim; j++)
                for (k = 1; k < k_dim; k++)
                {
                    cl[n] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 1] = i * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 2] = i * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 3] = (i - 1) * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 4] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 5] = i * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 6] = i * j_dim * k_dim + j * k_dim + k;
                    cl[n + 7] = (i - 1) * j_dim * k_dim + j * k_dim + k;
                    el[v] = n;
                    tl[v] = TYPE_HEXAGON;
                    v++;
                    n += 8;
                }
    }
    else
    {
        module->sendError("Failed to create object '%s'", objectname);
        return NULL;
    }
    memcpy(x_out, x_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(y_out, y_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(z_out, z_S, i_dim * j_dim * k_dim * sizeof(float));
    return grid_out;
}

coDoUnstructuredGrid *Buffer::create_tetrahedrons(const char *objectname)
{
    const char *COLOR = "COLOR";
    const char *color = "pink";
    float *x_out, *y_out, *z_out;
    int *el, *cl, *tl, m, v, i, j, k, n;
    coDoUnstructuredGrid *grid_out;
    grid_out = new coDoUnstructuredGrid(objectname, (i_dim - 1) * (j_dim - 1) * (k_dim - 1) * 6, (i_dim - 1) * (j_dim - 1) * (k_dim - 1) * 6 * 4, i_dim * j_dim * k_dim, 1);
    if (grid_out->objectOk())
    {
        grid_out->getAddresses(&el, &cl, &x_out, &y_out, &z_out);
        grid_out->getTypeList(&tl);
        grid_out->addAttribute(COLOR, color);
        v = 0;
        n = 0;
        for (i = 1; i < i_dim; i++)
            for (j = 1; j < j_dim; j++)
                for (k = 1; k < k_dim; k++)
                {
                    cl[n] = cl[n + 4] = cl[n + 17] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 1] = cl[n + 8] = cl[n + 12] = cl[n + 16] = cl[n + 20] = i * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 2] = cl[n + 5] = cl[n + 14] = i * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 6] = (i - 1) * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 18] = cl[n + 21] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 9] = cl[n + 22] = i * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 10] = cl[n + 13] = i * j_dim * k_dim + j * k_dim + k;
                    cl[n + 3] = cl[n + 7] = cl[n + 11] = cl[n + 15] = cl[n + 19] = cl[n + 23] = (i - 1) * j_dim * k_dim + j * k_dim + k;
                    for (m = 0; m < 6; m++)
                    {
                        el[v] = v * 4;
                        tl[v] = TYPE_TETRAHEDER;
                        v++;
                    }
                    n += 24;
                }
    }
    else
    {
        module->sendError("Failed to create object '%s'", objectname);
        return NULL;
    }
    memcpy(x_out, x_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(y_out, y_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(z_out, z_S, i_dim * j_dim * k_dim * sizeof(float));
    return grid_out;
}

coDoUnstructuredGrid *Buffer::create_pyramids(const char *objectname)
{
    const char *COLOR = "COLOR";
    const char *color = "pink";
    float *x_out, *y_out, *z_out;
    int *el, *cl, *tl, m, v, i, j, k, n;
    coDoUnstructuredGrid *grid_out;

    grid_out = new coDoUnstructuredGrid(objectname, (i_dim - 1) * (j_dim - 1) * (k_dim - 1) * 3, (i_dim - 1) * (j_dim - 1) * (k_dim - 1) * 3 * 5, i_dim * j_dim * k_dim, 1);
    if (grid_out->objectOk())
    {
        grid_out->getAddresses(&el, &cl, &x_out, &y_out, &z_out);
        grid_out->getTypeList(&tl);
        grid_out->addAttribute(COLOR, color);
        v = 0;
        n = 0;
        for (i = 1; i < i_dim; i++)
            for (j = 1; j < j_dim; j++)
                for (k = 1; k < k_dim; k++)
                {
                    cl[n] = cl[n + 10] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 1] = cl[n + 5] = cl[n + 11] = i * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 2] = cl[n + 8] = i * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 3] = (i - 1) * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 13] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 6] = cl[n + 12] = i * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 7] = i * j_dim * k_dim + j * k_dim + k;
                    cl[n + 4] = cl[n + 9] = cl[n + 14] = (i - 1) * j_dim * k_dim + j * k_dim + k;
                    for (m = 0; m < 3; m++)
                    {
                        el[v] = v * 5;
                        tl[v] = TYPE_PYRAMID;
                        v++;
                    }
                    n += 15;
                }
    }
    else
    {
        module->sendError("Failed to create object '%s'", objectname);
        return NULL;
    }
    memcpy(x_out, x_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(y_out, y_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(z_out, z_S, i_dim * j_dim * k_dim * sizeof(float));

    return grid_out;
}

coDoUnstructuredGrid *Buffer::create_prisms(const char *objectname)
{
    const char *COLOR = "COLOR";
    const char *color = "pink";
    float *x_out, *y_out, *z_out;
    int *el, *cl, *tl, v, i, j, k, n;
    coDoUnstructuredGrid *grid_out;
    grid_out = new coDoUnstructuredGrid(objectname, (i_dim - 1) * (j_dim - 1) * (k_dim - 1) * 2, (i_dim - 1) * (j_dim - 1) * (k_dim - 1) * 2 * 6, i_dim * j_dim * k_dim, 1);
    if (grid_out->objectOk())
    {
        grid_out->getAddresses(&el, &cl, &x_out, &y_out, &z_out);
        grid_out->getTypeList(&tl);
        grid_out->addAttribute(COLOR, color);
        v = 0;
        n = 0;
        for (i = 1; i < i_dim; i++)
            for (j = 1; j < j_dim; j++)
                for (k = 1; k < k_dim; k++)
                {
                    cl[n] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 3] = i * j_dim * k_dim + (j - 1) * k_dim + k - 1;
                    cl[n + 4] = cl[n + 11] = i * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 1] = cl[n + 8] = (i - 1) * j_dim * k_dim + j * k_dim + k - 1;
                    cl[n + 2] = cl[n + 7] = (i - 1) * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 5] = cl[n + 10] = i * j_dim * k_dim + (j - 1) * k_dim + k;
                    cl[n + 9] = i * j_dim * k_dim + j * k_dim + k;
                    cl[n + 6] = (i - 1) * j_dim * k_dim + j * k_dim + k;
                    el[v] = v * 6;
                    tl[v] = TYPE_PRISM;
                    v++;
                    el[v] = v * 6;
                    tl[v] = TYPE_PRISM;
                    v++;
                    n += 12;
                }
    }
    else
    {
        module->sendError("Failed to create object '%s'", objectname);
        return NULL;
    }
    memcpy(x_out, x_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(y_out, y_S, i_dim * j_dim * k_dim * sizeof(float));
    memcpy(z_out, z_S, i_dim * j_dim * k_dim * sizeof(float));
    return grid_out;
}

MODULE_MAIN(Converter, StoU)
