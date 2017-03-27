/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE FilterCrop application module                     **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "FilterCrop.h"

#define TRUE 1
#define FALSE 0

coDistributedObject **r;
//
// static stub callback functions calling the real class
// member functions
//

FilterCrop::FilterCrop(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Reduce or crop a data set")
{
    pMeshIn = addInputPort("GridIn0", "StructuredGrid|RectilinearGrid|UniformGrid", "input mesh");
    pDataIn = addInputPort("DataIn0", "Float|Vec3", "input data");
    pDataIn->setRequired(0);

    pMeshOut = addOutputPort("GridOut0", "StructuredGrid|RectilinearGrid|UniformGrid", "Cuttingplane");
    pDataOut = addOutputPort("DataOut0", "Float|Vec3", "interpolated data");
    pDataOut->setDependencyPort(pDataIn);

    pIMin = addIntSliderParam("i_min", "min i-index");
    pIMin->setValue(1, 100, 1);
    pIMax = addIntSliderParam("i_max", "max i-index");
    pIMax->setValue(1, 100, 100);
    pJMin = addIntSliderParam("j_min", "min j-index");
    pJMin->setValue(1, 100, 1);
    pJMax = addIntSliderParam("j_max", "max j-index");
    pJMax->setValue(1, 100, 100);
    pKMin = addIntSliderParam("k_min", "min k-index");
    pKMin->setValue(1, 100, 1);
    pKMax = addIntSliderParam("k_max", "max k-index");
    pKMax->setValue(1, 100, 100);
    pSample = addInt32Param("sample", "sampling factor");
    pSample->setValue(2);
}

FilterCrop::~FilterCrop()
{
}

//
//
//..........................................................................
//
//

// Description file :
//MODULE C_FilterCrop
//CATEGORY Filter
//DESCRIPTION "Reduce or cropp a data set"
//INPUT meshIn "coDoStructuredGrid|coDoRectilinearGrid|coDoUniformGrid" "input mesh" req
//INPUT dataIn "coDoFloat|coDoVec3" "input data" req
//OUTPUT meshOut "coDoStructuredGrid|coDoRectilinearGrid|coDoUniformGrid" "reduced mesh" default
//OUTPUT dataOut "coDoFloat|coDoVec3" "reduced data" default
//PARIN i_min Slider "min i-index" "1 100 1"
//PARIN i_max Slider "max i-index" "1 100 100"
//PARIN j_min Slider "min j-index" "1 100 1"
//PARIN j_max Slider "max j-index" "1 100 100"
//PARIN k_min Slider "min k-index" "1 100 1"
//PARIN k_max Slider "max k-index" "1 100 100"
//PARIN sample Scalar "sampling factor" "2"

//
//
//..........................................................................
//
//

int FilterCrop::compute(const char *)
{
    i_min = pIMin->getValue();
    i_max = pIMax->getValue();
    j_min = pJMin->getValue();
    j_max = pJMax->getValue();
    k_min = pKMin->getValue();
    k_max = pKMax->getValue();
    sample = pSample->getValue();

    int no_of_objects = pDataIn->getCurrentObject() ? 2 : 1;

    int mesh, data;
    int nelem = 0;

    r = new coDistributedObject *[3];
    r[0] = NULL;
    r[1] = NULL;

    //	get output data object	names
    GridOut = pMeshOut->getObjName();
    //fprintf(stderr, "creating obj with name %s\n", GridOut);
    DataOut = pDataOut->getObjName();

    //	retrieve grid object from shared memeory
    const coDistributedObject *data_obj = pMeshIn->getCurrentObject();
    if (data_obj != NULL)
    {
        gtype = data_obj->getType();
        if (strcmp(gtype, "STRGRD") == 0)
        {
            s_grid_in = (coDoStructuredGrid *)data_obj;
            s_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
            s_grid_in->getAddresses(&x_in, &y_in, &z_in);
        }

        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            r_grid_in = (coDoRectilinearGrid *)data_obj;
            r_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
            r_grid_in->getAddresses(&x_in, &y_in, &z_in);
        }

        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            u_grid_in = (coDoUniformGrid *)data_obj;
            u_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
        }

        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return STOP_PIPELINE;
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'meshIn' can not be retrieved from SHM");
        return STOP_PIPELINE;
    }
    mesh = TRUE;
    if (i_dim == 0 && j_dim == 0 && k_dim == 0)
    {
        Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    //	retrieve data object from shared memeory
    if (no_of_objects > 1)
    {
        data_obj = pDataIn->getCurrentObject();
        if (data_obj != NULL)
        {
            dtype = data_obj->getType();
            if (strcmp(dtype, "USTSDT") == 0)
            {
                s_data_in = (coDoFloat *)data_obj;
                nelem = s_data_in->getNumPoints();
                s_data_in->getAddress(&s_in);
            }

            else if (strcmp(dtype, "USTVDT") == 0)
            {
                v_data_in = (coDoVec3 *)data_obj;
                nelem = v_data_in->getNumPoints();
                v_data_in->getAddresses(&u_in, &v_in, &w_in);
            }

            else
            {
                Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
                return STOP_PIPELINE;
            }
            data = TRUE;
        }
        else
        {
            data = FALSE;
        }
    }
    else
    {
        data_obj = NULL;
        data = FALSE;
    }
    if (nelem == 0)
    {
        Covise::sendWarning("WARNING: Data object 'dataIn' is empty");
    }

    // check dimensions
    if (mesh && data)
    {
        if (nelem != i_dim * j_dim * k_dim)
        {
            Covise::sendError("ERROR: Objects have different dimensions");
            return STOP_PIPELINE;
        }
    }
    if (!mesh)
    {
        Covise::sendError("ERROR: No mesh object");
        return STOP_PIPELINE;
    }

    if (isPartOfMultiblock())
    {
        i_min = 1;
        i_max = i_dim;
        j_min = 1;
        j_max = j_dim;
        k_min = 1;
        k_max = k_dim;
        Covise::sendWarning("Detected block-structured grid: I only use the feature sample.");
    }
    else
    {
        // check slider parameters

        if (i_min > i_dim)
        {
            i_min = 1;
            Covise::sendWarning("WARNING: i-min out of range, setting new one");
        }
        if (i_max > i_dim)
        {
            i_max = i_dim;
            Covise::sendWarning("WARNING: i-max out of range, setting new one");
        }
        if (i_min > i_max)
        {
            int i = i_min;
            i_min = i_max;
            i_max = i;
            Covise::sendWarning("WARNING: i-min > i_max, switching boundaries");
        }
        pIMin->setValue(1, i_dim, i_min);
        pIMax->setValue(1, i_dim, i_max);

        if (j_min > j_dim)
        {
            j_min = 1;
            Covise::sendWarning("WARNING: j-min out of range, setting new one");
        }
        if (j_max > j_dim)
        {
            j_max = j_dim;
            Covise::sendWarning("WARNING: j-max out of range, setting new one");
        }
        if (i_min > i_max)
        {
            int j = j_min;
            j_min = j_max;
            j_max = j;
            Covise::sendWarning("WARNING: j-min > j_max, switching boundaries");
        }
        pJMin->setValue(1, j_dim, j_min);
        pJMax->setValue(1, j_dim, j_max);

        if (k_min > k_dim)
        {
            k_min = 1;
            Covise::sendWarning("WARNING: k-min out of range, setting new one");
        }
        if (k_max > k_dim)
        {
            k_max = k_dim;
            Covise::sendWarning("WARNING: k-max out of range, setting new one");
        }
        if (k_min > k_max)
        {
            int k = k_min;
            k_min = k_max;
            k_max = k;
            Covise::sendWarning("WARNING: k-min > k_max, switching boundaries");
        }
        pKMin->setValue(1, k_dim, k_min);
        pKMax->setValue(1, k_dim, k_max);
    }

    l_dim = 1 + (i_max - i_min) / sample;
    m_dim = 1 + (j_max - j_min) / sample;
    n_dim = 1 + (k_max - k_min) / sample;

    // create output objects
    if (pMeshIn->getCurrentObject())
    {
        if (GridOut == NULL)
        {
            Covise::sendError("ERROR: Object name not correct for 'meshOut'");
            return STOP_PIPELINE;
        }

        coDistributedObject *grid = NULL;

        if (strcmp(gtype, "STRGRD") == 0)
        {
            grid = create_strgrid_plane();
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            grid = create_rectgrid_plane();
        }
        else
        {
            grid = create_unigrid_plane();
        }

        if (grid)
        {
            grid->copyAllAttributes(pMeshIn->getCurrentObject());
        }
        pMeshOut->setCurrentObject(grid);
    }

    if (no_of_objects > 1)
    {
        if (DataOut != NULL)
        {
            coDistributedObject *data = NULL;
            if (strcmp(dtype, "USTSDT") == 0)
            {
                data = create_scalar_plane();
            }
            else
            {
                data = create_vector_plane();
            }

            if (data)
            {
                data->copyAllAttributes(pDataIn->getCurrentObject());
            }
            pDataOut->setCurrentObject(data);
        }
    }
    r[2] = NULL;

    return CONTINUE_PIPELINE;
}

//======================================================================
// create the cutting planes
//======================================================================
coDistributedObject *FilterCrop::create_strgrid_plane()
{
    int i_sample, j_sample, k_sample;

    s_grid_out = new coDoStructuredGrid(GridOut, l_dim, m_dim, n_dim);
    if (!s_grid_out->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'meshOut' failed");
        return NULL;
    }
    s_grid_out->getAddresses(&x_out, &y_out, &z_out);

    int idims[] = { i_dim, j_dim, k_dim };
    int odims[] = { l_dim, m_dim, n_dim };
    //     cerr << "min_max_sample: (" << i_min << ", " << i_max << ", " << sample << ") (";
    //     cerr << j_min << ", " << j_max << ") (" << k_min << ", " << k_max << ")\n";
    for (int i = 0; i < l_dim; i++)
        for (int j = 0; j < m_dim; j++)
            for (int k = 0; k < n_dim; k++)
            {
                // make sure that there is no gap at the end
                if (i != l_dim - 1)
                    i_sample = i_min - 1 + sample * i;
                else
                    i_sample = i_max - 1;
                if (j != m_dim - 1)
                    j_sample = j_min - 1 + sample * j;
                else
                    j_sample = j_max - 1;
                if (k != n_dim - 1)
                    k_sample = k_min - 1 + sample * k;
                else
                    k_sample = k_max - 1;

                int iOut = coIndex(i, j, k, odims);
                int iIn = coIndex(i_sample, j_sample, k_sample, idims);

                x_out[iOut] = x_in[iIn];
                y_out[iOut] = y_in[iIn];
                z_out[iOut] = z_in[iIn];
            }
    r[0] = s_grid_out;
    return s_grid_out;
}

//======================================================================
// create the cutting planes
//======================================================================
coDistributedObject *FilterCrop::create_rectgrid_plane()
{
    int ip;

    r_grid_out = new coDoRectilinearGrid(GridOut, l_dim, m_dim, n_dim);
    if (!r_grid_out->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'meshOut' failed");
        return NULL;
    }
    r_grid_out->getAddresses(&x_out, &y_out, &z_out);

    ip = i_min - 1;
    for (int i = 0; i < l_dim; i++)
    {
        x_out[i] = x_in[ip];
        ip = ip + sample;
    }
    ip = j_min - 1;
    for (int j = 0; j < m_dim; j++)
    {
        y_out[j] = y_in[ip];
        ip = ip + sample;
    }
    ip = k_min - 1;
    for (int k = 0; k < n_dim; k++)
    {
        z_out[k] = z_in[ip];
        ip = ip + sample;
    }
    r[0] = r_grid_out;
    return r_grid_out;
}

//======================================================================
// create the cutting planes
//======================================================================
coDistributedObject *FilterCrop::create_unigrid_plane()
{

    u_grid_in->getPointCoordinates(i_min - 1, &x_min,
                                   j_min - 1, &y_min, k_min - 1, &z_min);
    u_grid_in->getPointCoordinates(i_max - 1, &x_max,
                                   j_max - 1, &y_max, k_max - 1, &z_max);
    u_grid_out = new coDoUniformGrid(GridOut, l_dim, m_dim, n_dim,
                                     x_min, x_max, y_min, y_max, z_min, z_max);
    if (!u_grid_out->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'meshOut' failed");
        return NULL;
    }
    r[0] = u_grid_out;
    return  u_grid_out;
}

//======================================================================
// create the cutting planes
//======================================================================
coDistributedObject *FilterCrop::create_scalar_plane()
{
    int i_sample, j_sample, k_sample;

    s_data_out = new coDoFloat(DataOut, l_dim * m_dim * n_dim);
    if (!s_data_out->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'dataOut' failed");
        return NULL;
    }

    s_data_out->getAddress(&s_out);

    int idims[] = { i_dim, j_dim, k_dim };
    int odims[] = { l_dim, m_dim, n_dim };
    for (int i = 0; i < l_dim; i++)
        for (int j = 0; j < m_dim; j++)
            for (int k = 0; k < n_dim; k++)
            {
                if (i != l_dim - 1)
                    i_sample = i_min - 1 + sample * i;
                else
                    i_sample = i_max - 1;
                if (j != m_dim - 1)
                    j_sample = j_min - 1 + sample * j;
                else
                    j_sample = j_max - 1;
                if (k != n_dim - 1)
                    k_sample = k_min - 1 + sample * k;
                else
                    k_sample = k_max - 1;

                int iOut = coIndex(i, j, k, odims);
                int iIn = coIndex(i_sample, j_sample, k_sample, idims);

                s_out[iOut] = s_in[iIn];
            }
    r[1] = s_data_out;
    return s_data_out;
}

//======================================================================
// create the cutting planes
//======================================================================
coDistributedObject *FilterCrop::create_vector_plane()
{
    int i_sample, j_sample, k_sample;

    v_data_out = new coDoVec3(DataOut, l_dim * m_dim * n_dim);
    if (!v_data_out->objectOk())
    {
        Covise::sendError("ERROR: creation of data object 'dataOut' failed");
        return NULL;
    }

    v_data_out->getAddresses(&u_out, &v_out, &w_out);

    int idims[] = { i_dim, j_dim, k_dim };
    int odims[] = { l_dim, m_dim, n_dim };
    for (int i = 0; i < l_dim; i++)
        for (int j = 0; j < m_dim; j++)
            for (int k = 0; k < n_dim; k++)
            {
                if (i != l_dim - 1)
                    i_sample = i_min - 1 + sample * i;
                else
                    i_sample = i_max - 1;
                if (j != m_dim - 1)
                    j_sample = j_min - 1 + sample * j;
                else
                    j_sample = j_max - 1;
                if (k != n_dim - 1)
                    k_sample = k_min - 1 + sample * k;
                else
                    k_sample = k_max - 1;

                int iOut = coIndex(i, j, k, odims);
                int iIn = coIndex(i_sample, j_sample, k_sample, idims);

                u_out[iOut] = u_in[iIn];
                v_out[iOut] = v_in[iIn];
                w_out[iOut] = w_in[iIn];
            }
    r[1] = v_data_out;
    return v_data_out;
}

MODULE_MAIN(Filter, FilterCrop)
