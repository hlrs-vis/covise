/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                                          **
 **                                                                          **
 ** Description: Assemble blocks of unstructured grids( per timestep)        **
 **                                                                          **
 ** Name:        AssembleUsg                                                 **
 ** Category:    Tools                                                       **
 **                                                                          **
 ** Author: Sven Kufer		                                            **
 **         (C)  VirCinity IT- Consulting GmbH                               **
 **         Nobelstrasse 15                               		    **
 **         D- 70569 Stuttgart    			       		    **
 **                                                                          **
 **  19.02.2001                                                              **
\****************************************************************************/

#include "AssembleUsg.h"
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

AssembleUsg::AssembleUsg(int argc, char *argv[])
    : coModule(argc, argv, "Assemble blocks of unstructured grids (per timestep)")
{

    p_gridIn = addInputPort("GridIn0", "UnstructuredGrid|Polygons", "set of grids");
    p_dataIn = addInputPort("DataIn0", "Vec3|Float", "set of data");
    p_dataIn->setRequired(0);

    p_gridOut = addOutputPort("GridOut0", "UnstructuredGrid|Polygons", "assembled grid");
    p_dataOut = addOutputPort("DataOut0", "Vec3|Float", "assembled data");

    p_dataOut->setDependencyPort(p_dataIn);

    p_removeUnnescessaryTimesteps = addBooleanParam("removeUnnescessaryTimesteps", "Remove a timestep-set if it contains only one element");
    p_removeUnnescessaryTimesteps->setValue(0);
}

int AssembleUsg::compute(const char *)
{
    const coDistributedObject *obj_in = p_gridIn->getCurrentObject();
    const coDistributedObject *dobj_in = p_dataIn->getCurrentObject();
    const coDistributedObject *const *gelem_in;
    const coDistributedObject *const *delem_in;

    coDistributedObject **gelem_out, **delem_out, *data_out, *grid_out;
    coDoSet *set_out, *dset_out;

    int i, num_sets, dnum_sets;

    char buffer[100];

    //
    // do some checks first
    //

    if (obj_in == NULL)
    {
        sendError("No input grid available.");
        return STOP_PIPELINE;
    }

    const coDoSet *set_in = dynamic_cast<const coDoSet *>(obj_in);
    if (!set_in)
    {
        sendError("Only for sets implemented.");
        return STOP_PIPELINE;
    }

    gridType = GT_NONE;
    if (!checkGrid(obj_in))
    {
        // error message was already printed by checkGrid
        return STOP_PIPELINE;
    }
    if (gridType == GT_NONE)
    {
        sendError("No valid grid found.");
        return STOP_PIPELINE;
    }

    const coDoSet *dset_in = dynamic_cast<const coDoSet *>(dobj_in);
    if (dobj_in)
    {
        if (!dset_in)
        {
            sendError("Only for sets implemented.");
            return STOP_PIPELINE;
        }

        dataType = DT_NONE;
        if (!checkData(dobj_in))
        {
            // error message was already printed by checkGrid
            return STOP_PIPELINE;
        }
        if (dataType == DT_NONE)
        {
            sendError("No valid data found.");
            return STOP_PIPELINE;
        }
    }

    //
    // handle grid_port
    //

    if ((obj_in->getAttribute("TIMESTEP") != NULL) && (!p_removeUnnescessaryTimesteps->getValue() || (set_in->getNumElements() > 1)))
    {
        gelem_in = set_in->getAllElements(&num_sets);
        if (!dynamic_cast<const coDoSet *>(gelem_in[0]))
        {
            sendError("Found no block structure. Please bypass AssembleUsg.");
            return STOP_PIPELINE;
        }
        gelem_out = new coDistributedObject *[num_sets + 1];
        gelem_out[num_sets] = NULL;

        for (i = 0; i < num_sets; i++)
        {
            sprintf(buffer, "%s_%d", p_gridOut->getObjName(), i);
            // --------------------------------------------------------
            gelem_out[i] = unpack_grid(gelem_in[i], buffer);
            if (gelem_out[i] == NULL)
            {
                Covise::sendError("Error constructing new grid.");
                return STOP_PIPELINE;
            }
            copyAttributes(gelem_out[i], gelem_in[i]);
            // --------------------------------------------------------
        }

        set_out = new coDoSet(p_gridOut->getObjName(), gelem_out);
        copyAttributes(set_out, set_in);
        p_gridOut->setCurrentObject(set_out);
    }
    else
    {
        // --------------------------------------------------------
        grid_out = unpack_grid(set_in, p_gridOut->getObjName());
        if (grid_out == NULL)
        {
            Covise::sendError("Error constructing new grid.");
            return STOP_PIPELINE;
        }
        copyAttributes(grid_out, set_in);
        // --------------------------------------------------------
        p_gridOut->setCurrentObject(grid_out);
    }

    //
    // handle data port
    //

    if (dset_in)
    {
        if ((dobj_in->getAttribute("TIMESTEP") != NULL) && (!p_removeUnnescessaryTimesteps->getValue() || (set_in->getNumElements() > 1)))
        {
            delem_in = dset_in->getAllElements(&dnum_sets);
            delem_out = new coDistributedObject *[dnum_sets + 1];
            delem_out[dnum_sets] = NULL;

            for (i = 0; i < dnum_sets; i++)
            {
                sprintf(buffer, "%s_%d", p_dataOut->getObjName(), i);
                // --------------------------------------------------------
                delem_out[i] = unpack_data(delem_in[i], buffer);
                if (delem_out[i] == NULL)
                {
                    Covise::sendError("Error constructing new data.");
                    return STOP_PIPELINE;
                }
                copyAttributes(delem_out[i], delem_in[i]);
                // --------------------------------------------------------
            }

            if (delem_out[0] == NULL)
            {
                return STOP_PIPELINE;
            }
            dset_out = new coDoSet(p_dataOut->getObjName(), delem_out);
            copyAttributes(dset_out, dset_in);
            p_dataOut->setCurrentObject(dset_out);
        }
        else
        {
            // --------------------------------------------------------
            data_out = unpack_data(dobj_in, p_dataOut->getObjName());
            if (data_out == NULL)
            {
                Covise::sendError("Error constructing new data.");
                return STOP_PIPELINE;
            }
            copyAttributes(data_out, dobj_in);
            // --------------------------------------------------------
            p_dataOut->setCurrentObject(data_out);
        }
    }

    return CONTINUE_PIPELINE;
}

void AssembleUsg::copyAttributes(coDistributedObject *tgt, const coDistributedObject *src)
{
    const char **attr_names, **attr_values;
    int num_attrib;
    num_attrib = src->getAllAttributes(&attr_names, &attr_values);
    tgt->addAttributes(num_attrib, attr_names, attr_values);
    return;
}

bool AssembleUsg::checkGrid(const coDistributedObject *obj_in)
{
    const coDoSet *set_in;
    if ((set_in = dynamic_cast<const coDoSet *>(obj_in)))
    {
        int num;
        const coDistributedObject *const *elems = set_in->getAllElements(&num);
        for (int i = 0; i < num; ++i)
        {
            if (!checkGrid(elems[i]))
            {
                return false;
            }
        }
    }
    else
    {
        GRID_TYPE currType = GT_NONE;
        if (dynamic_cast<const coDoUnstructuredGrid *>(obj_in))
        {
            currType = GT_UNSGRD;
            lastGridElement = obj_in;
        }
        else if (dynamic_cast<const coDoPolygons *>(obj_in))
        {
            currType = GT_POLYGN;
            lastGridElement = obj_in;
        }
        else
        {
            sendError("Found element other than UNSGRD and Polygons in set.");
            return false;
        }
        if ((gridType != GT_NONE) && (currType != gridType))
        {
            sendError("Found different grid elements in set.");
            return false;
        }
        gridType = currType;
    }
    return true;
}

bool AssembleUsg::checkData(const coDistributedObject *obj_in)
{
    const coDoSet *set_in;
    if ((set_in = dynamic_cast<const coDoSet *>(obj_in)))
    {
        int num;
        const coDistributedObject *const *elems = set_in->getAllElements(&num);
        for (int i = 0; i < num; ++i)
        {
            if (!checkData(elems[i]))
            {
                return false;
            }
        }
    }
    else
    {
        DATA_TYPE currType = DT_NONE;
        if (dynamic_cast<const coDoFloat *>(obj_in))
        {
            currType = DT_FLOAT;
            lastDataElement = obj_in;
        }
        else if (dynamic_cast<const coDoVec3 *>(obj_in))
        {
            currType = DT_VEC3;
            lastDataElement = obj_in;
        }
        else
        {
            sendError("Found data other than float and vec3 in set.");
            return false;
        }
        if ((dataType != DT_NONE) && (currType != dataType))
        {
            sendError("Found different data element in set.");
            return false;
        }
        dataType = currType;
    }
    return true;
}

void AssembleUsg::findGrid(const coDoSet *set_in)
{
    int num_elems;
    const coDistributedObject *const *set_elem_in = set_in->getAllElements(&num_elems);

    const coDoUnstructuredGrid *grid_elem;
    const coDoPolygons *poly_elem;
    const coDoSet *set_elem;

    int *elem, *conn, *tl, num_elem, num_conn, num_coord;
    float *x_coord, *y_coord, *z_coord;

    for (int i = 0; i < num_elems; ++i)
    {
        if ((set_elem = dynamic_cast<const coDoSet *>(set_elem_in[i])))
        {
            findGrid(set_elem);
        }
        else
        {
            if ((grid_elem = dynamic_cast<const coDoUnstructuredGrid *>(set_elem_in[i])))
            {
                grid_elem->getAddresses(&elem, &conn, &x_coord, &y_coord, &z_coord);
                grid_elem->getGridSize(&num_elem, &num_conn, &num_coord);
                grid_elem->getTypeList(&tl);
            }
            else if ((poly_elem = dynamic_cast<const coDoPolygons *>(set_elem_in[i])))
            {
                poly_elem->getAddresses(&x_coord, &y_coord, &z_coord, &conn, &elem);
                num_elem = poly_elem->getNumPolygons();
                num_conn = poly_elem->getNumVertices();
                num_coord = poly_elem->getNumPoints();
                tl = NULL;
            }
            else
            {
                continue;
            }
            elem_in.push_back(elem);
            conn_in.push_back(conn);
            tl_in.push_back(tl);
            x_coord_in.push_back(x_coord);
            y_coord_in.push_back(y_coord);
            z_coord_in.push_back(z_coord);
            num_elem_in.push_back(num_elem);
            num_conn_in.push_back(num_conn);
            num_coord_in.push_back(num_coord);
        }
    }
}

void AssembleUsg::findData(const coDoSet *set_in)
{
    int num_elems;
    const coDistributedObject *const *set_elem_in = set_in->getAllElements(&num_elems);

    const coDoFloat *float_elem;
    const coDoVec3 *vec3_elem;
    const coDoSet *set_elem;

    float *x_data, *y_data, *z_data;

    for (int i = 0; i < num_elems; ++i)
    {
        if ((set_elem = dynamic_cast<const coDoSet *>(set_elem_in[i])))
        {
            findData(set_elem);
        }
        else
        {
            if ((float_elem = dynamic_cast<const coDoFloat *>(set_elem_in[i])))
            {
                float_elem->getAddress(&x_data);
                x_data_in.push_back(x_data);
                num_points_in.push_back(float_elem->getNumPoints());
            }
            else if ((vec3_elem = dynamic_cast<const coDoVec3 *>(set_elem_in[i])))
            {
                vec3_elem->getAddresses(&x_data, &y_data, &z_data);
                x_data_in.push_back(x_data);
                y_data_in.push_back(y_data);
                z_data_in.push_back(z_data);
                num_points_in.push_back(float_elem->getNumPoints());
            }
        }
    }
}

coDistributedObject *AssembleUsg::unpack_grid(const coDistributedObject *obj_in, const char *obj_name)
{
    coDoUnstructuredGrid *grid_out;
    coDoPolygons *poly_out;
    coDistributedObject *obj_out;

    float *x_coord_out, *y_coord_out, *z_coord_out;
    int *elem_out, *conn_out;
    int num_elem_out = 0, num_conn_out = 0, num_coord_out = 0;
    int *tl_out = NULL;
    int i, j;

    // clear global vectors
    x_coord_in.clear();
    y_coord_in.clear();
    z_coord_in.clear();
    elem_in.clear();
    conn_in.clear();
    tl_in.clear();
    num_elem_in.clear();
    num_conn_in.clear();
    num_coord_in.clear();

    findGrid((coDoSet *)obj_in);
    int num_elems = elem_in.size();

    for (i = 0; i < num_elems; ++i)
    {
        num_elem_out += num_elem_in[i];
        num_conn_out += num_conn_in[i];
        num_coord_out += num_coord_in[i];
    }

    if (gridType == GT_UNSGRD)
    {
        grid_out = new coDoUnstructuredGrid(obj_name, num_elem_out, num_conn_out, num_coord_out, ((coDoUnstructuredGrid *)lastGridElement)->hasTypeList());
        grid_out->getAddresses(&elem_out, &conn_out, &x_coord_out, &y_coord_out, &z_coord_out);
        grid_out->getTypeList(&tl_out);
        obj_out = grid_out;
    }
    else
    {
        poly_out = new coDoPolygons(obj_name, num_coord_out, num_conn_out, num_elem_out);
        poly_out->getAddresses(&x_coord_out, &y_coord_out, &z_coord_out, &conn_out, &elem_out);
        obj_out = poly_out;
    }

    int base_coords = 0;
    int base_conn = 0;
    int base_elem = 0;

    for (i = 0; i < num_elems; i++)
    {
        for (j = 0; j < num_coord_in[i]; j++)
        {
            x_coord_out[j + base_coords] = x_coord_in[i][j];
            y_coord_out[j + base_coords] = y_coord_in[i][j];
            z_coord_out[j + base_coords] = z_coord_in[i][j];
        }

        for (j = 0; j < num_conn_in[i]; j++)
            conn_out[j + base_conn] = base_coords + conn_in[i][j];

        for (j = 0; j < num_elem_in[i]; j++)
            elem_out[j + base_elem] = base_conn + elem_in[i][j];

        if ((gridType == GT_UNSGRD) && ((coDoUnstructuredGrid *)lastGridElement)->hasTypeList())
        {
            for (j = 0; j < num_elem_in[i]; j++)
                tl_out[j + base_elem] = tl_in[i][j];
        }

        base_coords += num_coord_in[i];
        base_elem += num_elem_in[i];
        base_conn += num_conn_in[i];
    }

    copyAttributes(obj_out, lastGridElement);
    return obj_out;
}

coDistributedObject *AssembleUsg::unpack_data(const coDistributedObject *obj_in, const char *obj_name)
{
    coDistributedObject *obj_out = NULL;
    coDoVec3 *data_out;
    coDoFloat *sdata_out;

    float *x_data_out = NULL, *y_data_out = NULL, *z_data_out = NULL;
    int num_points_out = 0;

    // clear global values
    x_data_in.clear();
    y_data_in.clear();
    z_data_in.clear();
    num_points_in.clear();

    findData((coDoSet *)obj_in);
    int num_elems = x_data_in.size();

    for (int i = 0; i < num_elems; ++i)
    {
        num_points_out += num_points_in[i];
    }

    if (dataType == DT_FLOAT)
    {
        sdata_out = new coDoFloat(obj_name, num_points_out);
        sdata_out->getAddress(&x_data_out);
        obj_out = sdata_out;

        int base_points = 0;
        for (int i = 0; i < num_elems; i++)
        {
            for (int j = 0; j < num_points_in[i]; j++)
            {
                x_data_out[j + base_points] = x_data_in[i][j];
            }
            base_points += num_points_in[i];
        }
    }
    else
    {
        data_out = new coDoVec3(obj_name, num_points_out);
        data_out->getAddresses(&x_data_out, &y_data_out, &z_data_out);
        obj_out = data_out;

        int base_points = 0;
        for (int i = 0; i < num_elems; i++)
        {
            for (int j = 0; j < num_points_in[i]; j++)
            {
                x_data_out[j + base_points] = x_data_in[i][j];
                y_data_out[j + base_points] = y_data_in[i][j];
                z_data_out[j + base_points] = z_data_in[i][j];
            }
            base_points += num_points_in[i];
        }
    }

    copyAttributes(obj_out, lastDataElement);
    return obj_out;
}

MODULE_MAIN(Converter, AssembleUsg)
