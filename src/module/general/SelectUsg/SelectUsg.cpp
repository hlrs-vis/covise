/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Select      application module                    **
 **                                                                        **
 **                                                                        **
 **                       (C) Vircinity 2000, 2001                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Andreas Werner, Sasha Cioringa                                **
 **                                                                        **
 **                                                                        **
 ** Date:  05.01.97                                                        **
 ** Date:  30.05.01                                                        **
\**************************************************************************/

#include "SelectUsg.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoUnstructuredGrid.h>
#include <util/coRestraint.h>

SelectUsg::SelectUsg(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Select parts from unstructured grids")
{
    const char *ChoiceVal[] = { "Element", "Property", "Component" };

    //parameters

    p_type = addChoiceParam("Type", "Type of selection");
    p_type->setValue(3, ChoiceVal, 0);

    p_selection = addStringParam("Selection", "Number selection");
    p_selection->setValue("0-9999999");

    //ports
    p_inPort1 = addInputPort("mesh", "UnstructuredGrid", "Mesh Input");
    p_inPort2 = addInputPort("index", "IntArr", "Index Input");
    p_inPort2->setRequired(0);
    p_inPort3 = addInputPort("dataIn", "Float|Vec3", "Data Input");
    p_inPort3->setRequired(0);

    p_outPort1 = addOutputPort("meshOut", "UnstructuredGrid", "Mesh Output");
    p_outPort2 = addOutputPort("dataOut", "Float|Vec3", "Data Output");
    p_outPort2->setDependencyPort(p_inPort3);

    run_no = 0;
}

int SelectUsg::compute(const char *)
{
    coRestraint sel;
    run_no++;

    ////////// Get type of selection

    int type = p_type->getValue(); // type of selection

    ////////// Get the selection

    const char *selection = p_selection->getValue();
    char subselection[256];
    int i = 0, num_sub = 0;
    // selection is seperated by blanks
    while (selection[i])
    {
        if (selection[i] == ' ')
        {
            subselection[num_sub] = '\0';
            sel.add(subselection);
            num_sub = 0;
        }
        else
        {
            subselection[num_sub++] = selection[i];
        }
        i++;
    }
    subselection[num_sub] = '\0';
    sel.add(subselection);

    ////////////////////////////////////////////////////////////////
    /////// Get the mesh

    float *vert_x, *vert_y, *vert_z;
    int *elem_list, *conn_list, *type_list;
    int num_elem, num_conn, num_coord;
    coDoUnstructuredGrid *mesh;

    const coDistributedObject *mesh_obj = p_inPort1->getCurrentObject();
    if (!mesh_obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort1->getName());
        return FAIL;
    }

    if (mesh_obj->isType("UNSGRD"))
    {
        mesh = ((coDoUnstructuredGrid *)mesh_obj);
        mesh->getGridSize(&num_elem, &num_conn, &num_coord);
        mesh->getAddresses(&elem_list, &conn_list, &vert_x, &vert_y, &vert_z);
        mesh->getTypeList(&type_list);
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort1->getName());
        return FAIL;
    }
    ////////////////////////////////////////////////////////////////
    ////////// Get the data

    const coDistributedObject *data_obj = p_inPort3->getCurrentObject();

    ////////////////////////////////////////////////////////////////
    ////////// Get the index

    int num_dim, num_indices, num_types;
    int *idx;
    coDoIntArr *index;

    const coDistributedObject *index_obj = p_inPort2->getCurrentObject();
    if (!index_obj)
    {
        // if no external index array is found we create one and fill it
        // with cell numbers
        int *elemNumbs = new int[num_elem];
        int i;
        for (i = 0; i < num_elem; ++i)
        {
            elemNumbs[i] = i;
        }

        char objName[256];
        sprintf(objName, "%s-%s", p_inPort2->getName(), p_outPort1->getObjName());
        index_obj = new coDoIntArr(objName, 1, &num_elem, elemNumbs);
        delete[] elemNumbs;

        sendInfo("Did not receive object at port %s set element numbers instead", p_inPort2->getName());
    }

    if (index_obj->isType("INTARR"))
    {
        index = ((coDoIntArr *)index_obj);
        num_dim = index->getNumDimensions();
        if (num_dim == 1)
        { // 1D index field
            num_indices = index->getDimension(0);
            index->getAddress(&idx);
        }
        else if (num_dim == 2)
        { // 2D index field
            num_indices = index->getDimension(0);
            num_types = index->getDimension(1);

            if (num_types != 3)
            {
                sendError("ERROR: Index array must have 3 columns");
                return FAIL;
            }
            index->getAddress(&idx);
        }
        else
        {
            sendError("ERROR: Indexing data must be 1D or 2D integer field");
            return FAIL;
        }

        if (num_indices == 0)
        {
            // dummy output
            if (data_obj)
            {
                if (data_obj->isType("USTVDT"))
                    p_outPort2->setCurrentObject(new coDoVec3(p_outPort2->getObjName(), 0));
                else if (data_obj->isType("USTSDT"))
                    p_outPort2->setCurrentObject(new coDoFloat(p_outPort2->getObjName(), 0));
                else
                {
                    sendError("Data may only be unstructured scalar or vector");
                    return FAIL;
                }
            }
            p_outPort1->setCurrentObject(new coDoUnstructuredGrid(p_outPort1->getObjName(), 0, 0, 0, 1));
            return SUCCESS;
        }

        if (num_indices != num_elem)
        {
            sendError("ERROR: Wrong number of indices: mesh=%d index=%d", num_elem, num_indices);
            return FAIL;
        }
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort2->getName());
        return FAIL;
    }

    ////////////////////////////////////////////////////////////////
    /////////////// find selected Cells

    int j;
    int numSelCells = 0;
    int *selCell = new int[num_indices];
    for (i = 0; i < num_elem; i++)
    {
        if (num_dim == 1) // 1D index field
        {
            if ((selCell[i] = sel(idx[i])) > 0)
                numSelCells++;
        }
        else if (num_dim == 2) // 2D index field (3 "columns")
        {
            switch (type)
            {
            case 0: // Element ID
                if ((selCell[i] = sel(idx[i])) > 0)
                {
                    numSelCells++;
                }
                break;
            case 1: // Property ID
                if ((selCell[i] = sel(idx[i + num_elem])) > 0)
                {
                    numSelCells++;
                }
                break;
            case 2: // Component ID
                if ((selCell[i] = sel(idx[i + 2 * num_elem])) > 0)
                {
                    numSelCells++;
                }
                break;
            };
        }
    }

    if (numSelCells == 0)
    {
        // Here we should create an empty object?!
        // sl: Yes!!!
        // output dummy
        if (data_obj)
        {
            if (data_obj->isType("USTVDT"))
                p_outPort2->setCurrentObject(new coDoVec3(p_outPort2->getObjName(), 0));
            else if (data_obj->isType("USTSDT"))
                p_outPort2->setCurrentObject(new coDoFloat(p_outPort2->getObjName(), 0));
            else
            {
                sendError("Data may only be unstructured scalar or vector");
                return FAIL;
            }
        }
        p_outPort1->setCurrentObject(new coDoUnstructuredGrid(p_outPort1->getObjName(), 0, 0, 0, 1));
        return SUCCESS;
        /*******************************************************
             sendError("No cells with selected types");
             return FAIL;
      ********************************************************/
    }

    ////////////////////////////////////////////////////////////////
    ///////  find selected Vertices, create translation

    int *vertexTrans = new int[num_coord];
    for (i = 0; i < num_coord; i++)
        vertexTrans[i] = -2;

    int numSelConn = 0;
    for (i = 0; i < num_elem; i++)
    {
        if (selCell[i])
        {
            int start = elem_list[i];
            int endP1 = (i == num_elem - 1) ? num_conn : elem_list[i + 1];
            numSelConn += endP1 - start;
            for (j = start; j < endP1; j++)
                vertexTrans[conn_list[j]] = -1;
        }
    }

    int numSelVert = 0;
    for (i = 0; i < num_coord; i++)
    {
        if (vertexTrans[i] == -1)
        {
            vertexTrans[i] = numSelVert;
            numSelVert++;
        }
    }

    sendInfo("%d Cells, %d Conn's, %d Vertices",
             numSelCells, numSelConn, numSelVert);

    ////////////////////////////////////////////////////////////////
    ///////  create mesh output

    coDoUnstructuredGrid *out_mesh
        = new coDoUnstructuredGrid(p_outPort1->getObjName(), numSelCells, numSelConn,
                                   numSelVert, 1);
    if (!out_mesh)
    {
        sendError("Failed to create object '%s' for port '%s' ", p_outPort1->getObjName(), p_outPort1->getName());
        return FAIL;
    }

    int *o_elem_list, *o_conn_list, *o_type_list;
    float *o_vert_x, *o_vert_y, *o_vert_z;

    out_mesh->getAddresses(&o_elem_list, &o_conn_list, &o_vert_x, &o_vert_y, &o_vert_z);
    out_mesh->getTypeList(&o_type_list);

    int o_conn = 0, o_elem = 0, elem;
    for (elem = 0; elem < num_elem; elem++)
    {
        if (selCell[elem])
        {
            o_type_list[o_elem] = type_list[elem];
            o_elem_list[o_elem] = o_conn;
            int start = elem_list[elem];
            int endP1 = (elem == num_elem - 1) ? num_conn : elem_list[elem + 1];
            for (i = start; i < endP1; i++)
                o_conn_list[o_conn++] = vertexTrans[conn_list[i]];
            o_elem++;
        }
    }

    int o_vert = 0, vert;
    for (vert = 0; vert < num_coord; vert++)
        if (vertexTrans[vert] >= 0)
        {
            o_vert_x[o_vert] = vert_x[vert];
            o_vert_y[o_vert] = vert_y[vert];
            o_vert_z[o_vert] = vert_z[vert];
            o_vert++;
        }

    ////////////////////////////////////////////////////////////////
    ////////// Get the data and create data output

    //   coDistributedObject *data_obj = p_inPort3->getCurrentObject();

    if (data_obj != NULL)
    {
        if (data_obj->isType("USTSDT"))
        {
            float *in_data, *out_data;
            int num_data;
            coDoFloat *s_in_data = (coDoFloat *)data_obj;
            coDoFloat *s_out_data = NULL;
            s_in_data->getAddress(&in_data);
            num_data = s_in_data->getNumPoints();

            if (num_data == num_elem)
                s_out_data = new coDoFloat(p_outPort2->getObjName(), numSelCells);
            else if (num_data == num_coord)
                s_out_data = new coDoFloat(p_outPort2->getObjName(), numSelVert);
            else
            {
                if (num_data == 0)
                {
                    p_outPort1->setCurrentObject(out_mesh);
                    p_outPort2->setCurrentObject(new coDoFloat(p_outPort2->getObjName(), 0));
                    return SUCCESS;
                }
                sendError("ERROR: Data does not match the grid!");
                delete out_mesh;
                return FAIL;
            }
            if (!s_out_data)
            {
                sendError("Failed to  create object '%s' for port '%s' ", p_outPort2->getObjName(), p_outPort2->getName());
                delete out_mesh;
                return FAIL;
            }

            s_out_data->getAddress(&out_data);
            if (num_data == num_elem)
                ////// element-based data
                for (i = 0; i < num_elem; i++)
                    if (selCell[i])
                        *(out_data)++ = *(in_data)++;
                    else
                        (in_data)++;
            else ////// vertex-based data
                for (i = 0; i < num_coord; i++)
                    if (vertexTrans[i] >= 0)
                        *(out_data)++ = *(in_data)++;
                    else
                        (in_data)++;

            p_outPort2->setCurrentObject(s_out_data);
        }
        else if (data_obj->isType("USTVDT"))
        {
            float *in_u_data, *in_v_data, *in_w_data;
            float *out_u_data, *out_v_data, *out_w_data;
            int num_data;
            coDoVec3 *v_in_data = (coDoVec3 *)data_obj;
            coDoVec3 *v_out_data = NULL;
            v_in_data->getAddresses(&in_u_data, &in_v_data, &in_w_data);
            num_data = v_in_data->getNumPoints();

            if (num_data == num_elem)
                v_out_data = new coDoVec3(p_outPort2->getObjName(), numSelCells);
            else if (num_data == num_coord)
                v_out_data = new coDoVec3(p_outPort2->getObjName(), numSelVert);
            else
            {
                if (num_data == 0)
                {
                    p_outPort1->setCurrentObject(out_mesh);
                    p_outPort2->setCurrentObject(new coDoVec3(p_outPort2->getObjName(), 0));
                    return SUCCESS;
                }
                sendError("ERROR: Data does not match the grid!");
                delete out_mesh;
                return FAIL;
            }
            if (!v_out_data)
            {
                sendError("Failed to  create object '%s' for port '%s' ", p_outPort2->getObjName(), p_outPort2->getName());
                delete out_mesh;
                return FAIL;
            }

            v_out_data->getAddresses(&out_u_data, &out_v_data, &out_w_data);
            if (num_data == num_elem)
                ////// element-based data
                for (i = 0; i < num_elem; i++)
                    if (selCell[i])
                    {
                        *(out_u_data)++ = *(in_u_data)++;
                        *(out_v_data)++ = *(in_v_data)++;
                        *(out_w_data)++ = *(in_w_data)++;
                    }
                    else
                    {
                        (in_u_data)++;
                        (in_v_data)++;
                        (in_w_data)++;
                    }
            else ////// vertex-based data
                for (i = 0; i < num_coord; i++)
                    if (vertexTrans[i] >= 0)
                    {
                        *(out_u_data)++ = *(in_u_data)++;
                        *(out_v_data)++ = *(in_v_data)++;
                        *(out_w_data)++ = *(in_w_data)++;
                    }
                    else
                    {
                        (in_u_data)++;
                        (in_v_data)++;
                        (in_w_data)++;
                    }

            p_outPort2->setCurrentObject(v_out_data);
        }
        else
        {
            sendError("Received illegal type at port '%s'", p_inPort3->getName());
            delete out_mesh;
            return FAIL;
        }
    }

    p_outPort1->setCurrentObject(out_mesh);

    return SUCCESS;
}

void SelectUsg::copyAttributesToOutObj(coInputPort **input_ports,
                                       coOutputPort **output_ports, int i)
{
    switch (i)
    {
    case 0: // output grid attributes
        if (input_ports[0] && output_ports[i])
            copyAttributes(output_ports[i]->getCurrentObject(), input_ports[0]->getCurrentObject());
        break;
    case 1: // output data attributes
        if (input_ports[2] && output_ports[i])
            copyAttributes(output_ports[i]->getCurrentObject(), input_ports[2]->getCurrentObject());
        break;
    default:
        break;
    }
}

MODULE_MAIN(Filter, SelectUsg)
