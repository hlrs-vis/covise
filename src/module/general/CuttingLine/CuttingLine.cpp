/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  Extract a line from a structured data set                **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  11.11.94  V1.0      						  **
 **									  **
 ** changed to new API:   22. 02. 2001					  **
 **  	 Sven Kufer							  **
 **	 (C) VirCinity IT-Consulting GmbH				  **
 **       Nobelstrasse 15						  **
 **       D- 70569 Stuttgart                                           	  **
\**************************************************************************/

#include "CuttingLine.h"

CuttingLine::CuttingLine(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Extract a line from a data set, create an unstructured 2D object")
{
    p_meshIn = addInputPort("GridIn0", "StructuredGrid|RectilinearGrid|UniformGrid", "input mesh");
    p_dataIn = addInputPort("DataIn0", "Float|Vec3", "input data");

    p_dataOut = addOutputPort("DataOut0", "Vec2", "2D Data");

    param_plane = addChoiceParam("cutting_direction", "cutting direction");
    char *choices[] = { (char *)"cut along i", (char *)"cut along j", (char *)"cut along k" };
    param_plane->setValue(3, choices, 0);

    param_ind_i = addIntSliderParam("i_index", "value of i-index");
    param_ind_i->setValue(1, 100, 2);

    param_ind_j = addIntSliderParam("j_index", "value of j-index");
    param_ind_j->setValue(1, 100, 2);

    param_ind_k = addIntSliderParam("k_index", "value of k-index");
    param_ind_k->setValue(1, 100, 2);

    strcpy(COLOR, "COLOR");
    strcpy(color, "pink");
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
int CuttingLine::compute(const char *)
{
    const coDistributedObject *data_obj, *grid_obj;
    int nelem;

    //	get input data object names
    grid_obj = p_meshIn->getCurrentObject();
    data_obj = p_dataIn->getCurrentObject();

    //	get output data object	names
    DataOut = p_dataOut->getObjName();

    //	get parameter
    param_ind_i->getValue(i_fro_min, i_fro_max, i_index);
    param_ind_j->getValue(j_fro_min, j_fro_max, j_index);
    param_ind_k->getValue(k_fro_min, k_fro_max, k_index);

    direction = param_plane->getValue();

    //	retrieve grid object from shared memeory

    if (grid_obj != NULL)
    {
        s_grid_in = dynamic_cast<const coDoStructuredGrid *>(grid_obj);
        r_grid_in = dynamic_cast<const coDoRectilinearGrid *>(grid_obj);
        u_grid_in = dynamic_cast<const coDoUniformGrid *>(grid_obj);

        const coDoAbstractStructuredGrid *grid = dynamic_cast<const coDoAbstractStructuredGrid *>(grid_obj);
        if (grid)
        {
            grid->getGridSize(&i_dim, &j_dim, &k_dim);
        }

        if (s_grid_in)
        {
            s_grid_in->getAddresses(&x_in, &y_in, &z_in);
        }
        else if (r_grid_in)
        {
            r_grid_in->getAddresses(&x_in, &y_in, &z_in);
        }
        else if (!u_grid_in)
        {
            sendError("ERROR: Data object 'meshIn' has wrong data type");
            return STOP_PIPELINE;
        }
    }
    else
    {
        sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return STOP_PIPELINE;
    }
    if (i_dim == 0 && j_dim == 0 && k_dim == 0)
    {
        sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    //	retrieve data object from shared memeory

    if (data_obj != NULL)
    {
        s_data_in = dynamic_cast<const coDoFloat *>(data_obj);

        if (s_data_in)
        {
            nelem = s_data_in->getNumPoints();
            s_data_in->getAddress(&s_in);
        }

        else if (dynamic_cast<const coDoVec3 *>(data_obj))
        {
            sendError("ERROR: Vector Data not yet implemented");
            return STOP_PIPELINE;
            /*v_data_in = (coDoVec3 *)data_obj;
         v_data_in->getGridSize(&isize, &jsize, &ksize);
         v_data_in->getAddresses(&u_in, &v_in, &w_in); */
        }

        else
        {
            sendError("ERROR: Data object 'dataIn' has wrong data type");
            return STOP_PIPELINE;
        }
    }
    else
    {
        sendError("ERROR: Data object 'dataIn' can't be accessed in shared memory");
        return STOP_PIPELINE;
    }
    if (nelem == 0)
    {
        sendWarning("WARNING: Data object 'dataIn' is empty");
    }

    // check dimensions
    if (nelem != i_dim * j_dim * k_dim)
    {
        sendError("ERROR: Objects have different dimensions");
        return STOP_PIPELINE;
    }

    // check slider parameter
    if (i_index > i_dim)
    {
        i_index = i_dim / 2;
        sendWarning("WARNING: i-index out of range,  set new one");
        param_ind_i->setValue(0, i_fro_max, i_index);
    }
    if (i_fro_max > i_dim)
    {
        i_fro_max = i_dim;
        sendWarning("WARNING: upper value for i-slider out of range,  set new one");
        param_ind_i->setValue(0, i_fro_max, i_index);
    }

    if (j_index > j_dim)
    {
        j_index = j_dim / 2;
        sendWarning("WARNING: j-index out of range,  set new one");
        param_ind_j->setValue(0, j_fro_max, j_index);
    }
    if (j_fro_max > j_dim)
    {
        j_fro_max = j_dim;
        sendWarning("WARNING: upper value for j-slider out of range,  set new one");
        param_ind_j->setValue(0, j_fro_max, j_index);
    }

    if (k_index > k_dim)
    {
        k_index = k_dim / 2;
        sendWarning("WARNING: k-index out of range,  set new one");
        param_ind_k->setValue(0, k_fro_max, k_index);
    }
    if (k_fro_max > k_dim)
    {
        k_fro_max = k_dim;
        sendWarning("WARNING: upper value for k-slider out of range,  set new one");
        param_ind_k->setValue(0, k_fro_max, k_index);
    }

    // create output objects

    if (s_grid_in)
    {
        CuttingLine::create_strgrid_data();
    }

    else if (r_grid_in)
        CuttingLine::create_rectgrid_data();

    else
        CuttingLine::create_unigrid_data();

    p_dataOut->setCurrentObject(data_out);
    return CONTINUE_PIPELINE;
}

//======================================================================
// create the cutting planes
//======================================================================
void CuttingLine::create_strgrid_data()
{
    int i;

    if (direction == 0)
    {
        data_out = new coDoVec2(DataOut, i_dim - i_index);
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < i_dim - i_index; i++)
            {
                *(x_out + i) = *(x_in + (i + i_index) * j_dim * k_dim + j_index * k_dim + k_index);
                *(y_out + i) = *(s_in + (i + i_index) * j_dim * k_dim + j_index * k_dim + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
    else if (direction == 1)
    {
        data_out = new coDoVec2(DataOut, j_dim - j_index);
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < j_dim - j_index; i++)
            {
                *(x_out + i) = *(y_in + i_index * j_dim * k_dim + (i + j_index) * k_dim + k_index);
                *(y_out + i) = *(s_in + i_index * j_dim * k_dim + (i + j_index) * k_dim + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
    else if (direction == 2)
    {
        data_out = new coDoVec2(DataOut, k_dim - k_index);
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < k_dim - k_index; i++)
            {
                *(x_out + i) = *(z_in + i_index * j_dim * k_dim + j_index * k_dim + i + k_index);
                *(y_out + i) = *(s_in + i_index * j_dim * k_dim + j_index * k_dim + i + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void CuttingLine::create_rectgrid_data()
{
    int i;

    if (direction == 1)
    {
        data_out = new coDoVec2(DataOut, i_dim - i_index);
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < i_dim - i_index; i++)
            {
                *(x_out + i) = *(x_in + (i + i_index));
                *(y_out + i) = *(s_in + (i + i_index) * j_dim * k_dim + j_index * k_dim + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
    else if (direction == 2)
    {
        data_out = new coDoVec2(DataOut, j_dim - j_index);
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < j_dim - j_index; i++)
            {
                *(x_out + i) = *(y_in + (i + j_index));
                *(y_out + i) = *(s_in + i_index * j_dim * k_dim + (i + j_index) * k_dim + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
    else if (direction == 3)
    {
        data_out = new coDoVec2(DataOut, k_dim - k_index);
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < k_dim - k_index; i++)
            {
                *(x_out + i) = *(z_in + i + k_index);
                *(y_out + i) = *(s_in + i_index * j_dim * k_dim + j_index * k_dim + i + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void CuttingLine::create_unigrid_data()
{
    int i;
    float n;

    if (direction == 0)
    {
        data_out = new coDoVec2(DataOut, i_dim - i_index);
        //");
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < i_dim - i_index; i++)
            {
                u_grid_in->getPointCoordinates(i + i_index, (x_out + i), 0, &n, 0, &n);
                *(y_out + i) = *(s_in + (i + i_index) * j_dim * k_dim + j_index * k_dim + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
    else if (direction == 1)
    {
        data_out = new coDoVec2(DataOut, j_dim - j_index);
        //");
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < j_dim - j_index; i++)
            {
                u_grid_in->getPointCoordinates(0, &n, i + j_index, (x_out + i), 0, &n);
                *(y_out + i) = *(s_in + i_index * j_dim * k_dim + (i + j_index) * k_dim + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
    else if (direction == 2)
    {
        data_out = new coDoVec2(DataOut, k_dim - k_index);
        data_out->addAttribute("COMMANDS", "AUTOSCALE");
        if (data_out->objectOk())
        {
            data_out->getAddresses(&x_out, &y_out);
            data_out->addAttribute(COLOR, color);

            for (i = 0; i < k_dim - k_index; i++)
            {
                u_grid_in->getPointCoordinates(0, &n, 0, &n, i + k_index, (x_out + i));
                *(y_out + i) = *(s_in + i_index * j_dim * k_dim + j_index * k_dim + i + k_index);
            }
        }
        else
        {
            sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
}

MODULE_MAIN(Filter, CuttingLine)
