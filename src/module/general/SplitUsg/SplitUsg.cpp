/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
 **                                                     (C)2001 Vircinity  **
 **                                                                        **
 ** Description: Splits grid into subgrids with elements of the same       **
 **              dimensionality                                            **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Sergio Leseduarte                           **
 **                            Vircinity GmbH                              **
 **                            Nobelstr. 15                                **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  29.1.2001  (coding begins)                                      **
\**************************************************************************/

#include "SplitUsg.h"

SplitUSG::SplitUSG(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Split a grid according to dimensionality")
{
    // Input ports
    p_grid = addInputPort("Grid", "UnstructuredGrid",
                          "grid with elements of sundry dimensionalities");
    p_data_scal = addInputPort("S_Data", "Float",
                               "scalar data of input grid");
    p_data_scal->setRequired(0);
    p_data_vect = addInputPort("V_Data", "Vec3",
                               "vector data of input grid");
    p_data_vect->setRequired(0);

    // Output ports
    p_grid_3D = addOutputPort("Grid3D", "UnstructuredGrid",
                              "3D grid");
    p_data_3D_scal = addOutputPort("S_Grid3D_Data", "Float",
                                   "scalar data of 3D grid");
    p_data_3D_scal->setDependencyPort(p_data_scal);
    p_data_3D_vect = addOutputPort("V_Grid3D_Data", "Vec3",
                                   "vector data of 3D grid");
    p_data_3D_vect->setDependencyPort(p_data_vect);

    p_grid_2D = addOutputPort("Grid2D", "Polygons",
                              "2D grid");
    p_data_2D_scal = addOutputPort("S_Grid2D_Data", "Float",
                                   "scalar data of 2D grid");
    p_data_2D_scal->setDependencyPort(p_data_scal);
    p_data_2D_vect = addOutputPort("V_Grid2D_Data", "Vec3",
                                   "vector data of 2D grid");
    p_data_2D_vect->setDependencyPort(p_data_vect);

    p_grid_1D = addOutputPort("Grid1D", "Lines",
                              "1D grid");
    p_data_1D_scal = addOutputPort("S_Grid1D_Data", "Float",
                                   "scalar data of 1D grid");
    p_data_1D_scal->setDependencyPort(p_data_scal);
    p_data_1D_vect = addOutputPort("V_Grid1D_Data", "Vec3",
                                   "vector data of 1D grid");
    p_data_1D_vect->setDependencyPort(p_data_vect);

    p_grid_0D = addOutputPort("Grid0D", "Points",
                              "0D grid");
    p_data_0D_scal = addOutputPort("S_Grid0D_Data", "Float",
                                   "scalar data of 0D grid");
    p_data_0D_scal->setDependencyPort(p_data_scal);
    p_data_0D_vect = addOutputPort("V_Grid0D_Data", "Vec3",
                                   "vector data of 0D grid");
    p_data_0D_vect->setDependencyPort(p_data_vect);
}

void
SplitUSG::copyAttributesToOutObj(coInputPort **input_ports,
                                 coOutputPort **output_ports,
                                 int i)
{
    int j = i % 3;
    if (input_ports[j] && output_ports[i])
        copyAttributes(output_ports[i]->getCurrentObject(), input_ports[j]->getCurrentObject());
}

int SplitUSG::compute(const char *)
{
    in_data_scal = 0;
    in_data_vect = 0;
    scal_num_of_points = 0;
    vect_num_of_points = 0;

#ifdef _DEBUG_SPLIT_
    nou_grid_3D = 0;
    nou_data_3D_scal = 0;
    nou_data_3D_vect = 0;
    nou_grid_2D = 0;
    nou_data_2D_scal = 0;
    nou_data_2D_vect = 0;
    nou_grid_1D = 0;
    nou_data_1D_scal = 0;
    nou_data_1D_vect = 0;
    nou_grid_0D = 0;
    nou_data_0D_scal = 0;
    nou_data_0D_vect = 0;
#endif

    // The grid is compulsory
    const coDistributedObject *in_obj_grid = p_grid->getCurrentObject();
    if (!in_obj_grid)
    {
        sendError("Did not receive grid at port '%s'", p_grid->getName());
        return FAIL;
    }
    if (!in_obj_grid->isType("UNSGRD"))
    {
        sendError("Type of grid is not UNSGRD at port '%s'", p_grid->getName());
        return FAIL;
    }
    in_grid = (coDoUnstructuredGrid *)in_obj_grid;

    // Have we got scalar data?
    const coDistributedObject *in_data = p_data_scal->getCurrentObject();
    if (!in_data)
    {
        in_data_scal = 0;
    }
    else if (!in_data->isType("USTSDT"))
    {
        sendError("Type of data is not USTSDT at port '%s'", p_data_scal->getName());
        return FAIL;
    }
    else
    {
        in_data_scal = (coDoFloat *)in_data;
    }

    // Have we got vector data?
    in_data = p_data_vect->getCurrentObject();
    if (!in_data)
    {
        in_data_vect = 0;
    }
    else if (!in_data->isType("USTVDT"))
    {
        sendError("Type of data is not USTVDT at port '%s'", p_data_vect->getName());
        return FAIL;
    }
    else
    {
        in_data_vect = (coDoVec3 *)in_data;
    }

    // Get grid and data size
    in_grid->getGridSize(&numEl, &numConn, &numCoord);
    if (in_data_scal)
    {
        scal_num_of_points = in_data_scal->getNumPoints();
        in_data_scal->getAddress(&scal_data);
    }
    if (in_data_vect)
    {
        vect_num_of_points = in_data_vect->getNumPoints();
        in_data_vect->getAddresses(&vect_data_x, &vect_data_y, &vect_data_z);
    }

    // Get grid addresses and type list (if available)
    in_grid->getAddresses(&elem_list, &conn_list, &x_in, &y_in, &z_in);
    type_list = 0;
    if (in_grid->hasTypeList())
        in_grid->getTypeList(&type_list);

    // Assign state to flag scal_cell_or_vertex
    if (in_data_scal)
    {
        if (numEl == scal_num_of_points)
            scal_cell_or_vertex = S_PER_CELL;
        else if (numCoord == scal_num_of_points)
            scal_cell_or_vertex = S_PER_VERTEX;
        else if (scal_num_of_points == 0)
            scal_cell_or_vertex = S_NULL;
        else
        {
            sendError("The number of scalar data cannot be unambiguously\
                     interpreted as cell-centred nor coordinate-centred at port '%s'",
                      p_data_scal->getName());
            return FAIL;
        }
    }

    // Assign state to flag vect_cell_or_vertex
    if (in_data_vect)
    {
        if (numEl == vect_num_of_points)
            vect_cell_or_vertex = V_PER_CELL;
        else if (numCoord == vect_num_of_points)
            vect_cell_or_vertex = V_PER_VERTEX;
        else if (vect_num_of_points == 0)
            vect_cell_or_vertex = V_NULL;
        else
        {
            sendError("The number of vector data cannot be unambiguously\
                     interpreted as cell-centred nor coordinate-centred at port '%s'",
                      p_data_vect->getName());
            return FAIL;
        }
    }

    out_grid_3D = 0;
    out_grid_2D = 0;
    out_grid_1D = 0;
    out_grid_0D = 0;
    out_data_3D_scal = out_data_2D_scal = out_data_1D_scal = out_data_0D_scal = 0;
    out_data_3D_vect = out_data_2D_vect = out_data_1D_vect = out_data_0D_vect = 0;

    coord_map = new int[numCoord];

    ///////////////////////////////////////////////////////////////////7
    build_out_3D_grid();
    if (!out_grid_3D)
    {
        out_grid_3D = new coDoUnstructuredGrid(p_grid_3D->getObjName(),
                                               0, 0, 0, type_list != 0);
#ifdef _DEBUG_SPLIT_
        nou_grid_3D++;
#endif
    }
    if (in_data_scal && !out_data_3D_scal)
    {
        out_data_3D_scal = new coDoFloat(p_data_3D_scal->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_3D_scal++;
#endif
    }
    if (in_data_vect && !out_data_3D_vect)
    {
        out_data_3D_vect = new coDoVec3(p_data_3D_vect->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_3D_vect++;
#endif
    }

    ///////////////////////////////////////////////////////////////////7
    build_out_2D_grid();
    if (!out_grid_2D)
    {
        out_grid_2D = new coDoPolygons(p_grid_2D->getObjName(),
                                       0, 0, 0);
#ifdef _DEBUG_SPLIT_
        nou_grid_2D++;
#endif
        if (!out_grid_2D)
        {
            sendError("Could not create output 2D grid with 0 elements\n");
            return FAIL;
        }
    }
    if (in_data_scal && !out_data_2D_scal)
    {
        out_data_2D_scal = new coDoFloat(p_data_2D_scal->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_2D_scal++;
#endif
    }
    if (in_data_vect && !out_data_2D_vect)
    {
        out_data_2D_vect = new coDoVec3(p_data_2D_vect->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_2D_vect++;
#endif
    }

    ///////////////////////////////////////////////////////////////////7
    build_out_1D_grid();
    if (!out_grid_1D)
    {
        out_grid_1D = new coDoLines(p_grid_1D->getObjName(),
                                    0, 0, 0);
#ifdef _DEBUG_SPLIT_
        nou_grid_1D++;
#endif
    }
    if (in_data_scal && !out_data_1D_scal)
    {
        out_data_1D_scal = new coDoFloat(p_data_1D_scal->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_1D_scal++;
#endif
    }
    if (in_data_vect && !out_data_1D_vect)
    {
        out_data_1D_vect = new coDoVec3(p_data_1D_vect->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_1D_vect++;
#endif
    }

    ///////////////////////////////////////////////////////////////////7
    build_out_0D_grid();
    if (!out_grid_0D)
    {
        out_grid_0D = new coDoPoints(p_grid_0D->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_grid_0D++;
#endif
    }
    if (in_data_scal && !out_data_0D_scal)
    {
        out_data_0D_scal = new coDoFloat(p_data_0D_scal->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_0D_scal++;
#endif
    }
    if (in_data_vect && !out_data_0D_vect)
    {
        out_data_0D_vect = new coDoVec3(p_data_0D_vect->getObjName(), 0);
#ifdef _DEBUG_SPLIT_
        nou_data_0D_vect++;
#endif
    }

    ///////////////////////////////////////////////////////////////////7

    delete[] coord_map;

    copyAttributes(out_grid_3D, in_grid);
    copyAttributes(out_grid_2D, in_grid);
    out_grid_2D->addAttribute("vertexOrder", "2");
    copyAttributes(out_grid_1D, in_grid);
    copyAttributes(out_grid_0D, in_grid);

    p_grid_3D->setCurrentObject(out_grid_3D);
    p_grid_2D->setCurrentObject(out_grid_2D);
    p_grid_1D->setCurrentObject(out_grid_1D);
    p_grid_0D->setCurrentObject(out_grid_0D);

#ifdef _DEBUG_SPLIT_
    nou_grid_3D--;
    nou_grid_2D--;
    nou_grid_1D--;
    nou_grid_0D--;
#endif

    if (out_data_3D_scal)
    {
        copyAttributes(out_data_3D_scal, in_data_scal);
        p_data_3D_scal->setCurrentObject(out_data_3D_scal);
#ifdef _DEBUG_SPLIT_
        nou_data_3D_scal--;
#endif
    }
    if (out_data_2D_scal)
    {
        copyAttributes(out_data_2D_scal, in_data_scal);
        p_data_2D_scal->setCurrentObject(out_data_2D_scal);
#ifdef _DEBUG_SPLIT_
        nou_data_2D_scal--;
#endif
    }
    if (out_data_1D_scal)
    {
        copyAttributes(out_data_1D_scal, in_data_scal);
        p_data_1D_scal->setCurrentObject(out_data_1D_scal);
#ifdef _DEBUG_SPLIT_
        nou_data_1D_scal--;
#endif
    }
    if (out_data_0D_scal)
    {
        copyAttributes(out_data_0D_scal, in_data_scal);
        p_data_0D_scal->setCurrentObject(out_data_0D_scal);
#ifdef _DEBUG_SPLIT_
        nou_data_0D_scal--;
#endif
    }

    if (out_data_3D_vect)
    {
        copyAttributes(out_data_3D_vect, in_data_vect);
        p_data_3D_vect->setCurrentObject(out_data_3D_vect);
#ifdef _DEBUG_SPLIT_
        nou_data_3D_vect--;
#endif
    }
    if (out_data_2D_vect)
    {
        copyAttributes(out_data_2D_vect, in_data_vect);
        p_data_2D_vect->setCurrentObject(out_data_2D_vect);
#ifdef _DEBUG_SPLIT_
        nou_data_2D_vect--;
#endif
    }
    if (out_data_1D_vect)
    {
        copyAttributes(out_data_1D_vect, in_data_vect);
        p_data_1D_vect->setCurrentObject(out_data_1D_vect);
#ifdef _DEBUG_SPLIT_
        nou_data_1D_vect--;
#endif
    }
    if (out_data_0D_vect)
    {
        copyAttributes(out_data_0D_vect, in_data_vect);
        p_data_0D_vect->setCurrentObject(out_data_0D_vect);
#ifdef _DEBUG_SPLIT_
        nou_data_0D_vect--;
#endif
    }

#ifdef _DEBUG_SPLIT_
    if (nou_grid_3D)
        sendWarning("nou_grid_3D");
    if (nou_grid_2D)
        sendWarning("nou_grid_2D");
    if (nou_grid_1D)
        sendWarning("nou_grid_1D");
    if (nou_grid_0D)
        sendWarning("nou_grid_0D");
    if (nou_data_3D_scal)
        sendWarning("nou_data_3D_scal");
    if (nou_data_2D_scal)
        sendWarning("nou_data_2D_scal");
    if (nou_data_1D_scal)
        sendWarning("nou_data_1D_scal");
    if (nou_data_0D_scal)
        sendWarning("nou_data_0D_scal");
    if (nou_data_3D_vect)
        sendWarning("nou_data_3D_vect");
    if (nou_data_2D_vect)
        sendWarning("nou_data_2D_vect");
    if (nou_data_1D_vect)
        sendWarning("nou_data_1D_vect");
    if (nou_data_0D_vect)
        sendWarning("nou_data_0D_vect");
#endif
    /*
      if(in_data_scal){
        copyAttributes(out_data_3D_scal,in_data_scal);
        copyAttributes(out_data_2D_scal,in_data_scal);
        copyAttributes(out_data_1D_scal,in_data_scal);
        copyAttributes(out_data_0D_scal,in_data_scal);
        p_data_3D_scal->setCurrentObject(out_data_3D_scal);
        p_data_2D_scal->setCurrentObject(out_data_2D_scal);
        p_data_1D_scal->setCurrentObject(out_data_1D_scal);
        p_data_0D_scal->setCurrentObject(out_data_0D_scal);
      }

   if(in_data_vect){
   copyAttributes(out_data_3D_vect,in_data_vect);
   copyAttributes(out_data_2D_vect,in_data_vect);
   copyAttributes(out_data_1D_vect,in_data_vect);
   copyAttributes(out_data_0D_vect,in_data_vect);
   p_data_3D_vect->setCurrentObject(out_data_3D_vect);
   p_data_2D_vect->setCurrentObject(out_data_2D_vect);
   p_data_1D_vect->setCurrentObject(out_data_1D_vect);
   p_data_0D_vect->setCurrentObject(out_data_0D_vect);
   }
   */

    return SUCCESS;
}

// build_out_3D_grid counts the dimensions
// of the 3D output grid and sets up a map
// for the coordinates. The map is an array (coord_map)
// with as many ints as coordinates in in_grid.
// At the beginning all items in the map are 0. The elements
// of the grid are sequentially read and if one is a 3D one,
// the coordinates of the corresponding items in the
// connectivity list are checked with a positive number (mark)
// in the map. This number is increased by one whenever
// assigned to an unmarked item of the coordinate list.
// See the header file for more information.

// build_out_3D_grid also creates the output objects
// (both grid) and data, and it fills the lists (it calls fill_data_objects).

void SplitUSG::build_out_3D_grid()
{
    unsigned int element;
    unsigned int numElem3D = 0, numConn3D = 0;

    unsigned int numConnNextElement;
    unsigned int numConnThisElement = 0;

    memset(coord_map, 0, numCoord * sizeof(int));
    mark = 0;
    for (element = 0; element < (unsigned int)numEl; ++element)
    {
        if (type_list)
        {
            switch (type_list[element])
            {
            case TYPE_PRISM:
                ++numElem3D;
                numConn3D += 6;
                put_marks_in_map(element, 6);
                break;
            case TYPE_TETRAHEDER:
                ++numElem3D;
                numConn3D += 4;
                put_marks_in_map(element, 4);
                break;
            case TYPE_PYRAMID:
                ++numElem3D;
                numConn3D += 5;
                put_marks_in_map(element, 5);
                break;
            case TYPE_HEXAEDER:
                ++numElem3D;
                numConn3D += 8;
                put_marks_in_map(element, 8);
                break;
            default:
                break;
                // Not a 3D type
            }
        }
        else
        {
            if ((int)element < numEl - 1)
                numConnNextElement = elem_list[element + 1];
            else
                numConnNextElement = numConn;
            switch (numConnNextElement - numConnThisElement)
            {
            case 4: // TETRAHEDER
                // I assume this is a tetrahedron and not a quad!!!
                ++numElem3D;
                numConn3D += 4;
                put_marks_in_map(element, 4);
                break;
            case 6: // PRISM
                ++numElem3D;
                numConn3D += 6;
                put_marks_in_map(element, 6);
                break;
            case 5: // PYRAMID
                ++numElem3D;
                numConn3D += 5;
                put_marks_in_map(element, 5);
                break;
            case 8: // HEXAHERDRON
                ++numElem3D;
                numConn3D += 8;
                put_marks_in_map(element, 8);
                break;
            default:
                break;
                // Not a 3D type
            }
            numConnThisElement = numConnNextElement;
        }
    }

    // Now create output grid (3D)...
    if (numElem3D)
    {
        out_grid_3D = new coDoUnstructuredGrid(p_grid_3D->getObjName(),
                                               numElem3D,
                                               numConn3D, mark, type_list != 0);
#ifdef _DEBUG_SPLIT_
        nou_grid_3D++;
#endif
        if (!out_grid_3D->objectOk())
        {
            sendError("Failed to create object '%s' for port '%s'",
                      p_grid_3D->getObjName(), p_grid_3D->getName());
        }
        // ... and scalar data if necessary ...
        if (in_data_scal)
        {
            switch (scal_cell_or_vertex)
            {
            case S_PER_CELL:
                out_data_3D_scal = new coDoFloat(p_data_3D_scal->getObjName(),
                                                 numElem3D);
#ifdef _DEBUG_SPLIT_
                nou_data_3D_scal++;
#endif
                break;
            case S_PER_VERTEX:
                out_data_3D_scal = new coDoFloat(p_data_3D_scal->getObjName(),
                                                 mark);
#ifdef _DEBUG_SPLIT_
                nou_data_3D_scal++;
#endif
                break;
            case S_NULL:
                out_data_3D_scal = new coDoFloat(p_data_3D_scal->getObjName(),
                                                 0);
                break;
            }
            if (!out_data_3D_scal->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_3D_scal->getObjName(),
                          p_data_3D_scal->getName());
            }
        }
        // ... and vector data if necessary ...
        if (in_data_vect)
        {
            switch (vect_cell_or_vertex)
            {
            case V_PER_CELL:
                out_data_3D_vect = new coDoVec3(p_data_3D_vect->getObjName(),
                                                numElem3D);
#ifdef _DEBUG_SPLIT_
                nou_data_3D_vect++;
#endif
                break;
            case V_PER_VERTEX:
                out_data_3D_vect = new coDoVec3(p_data_3D_vect->getObjName(),
                                                mark);
#ifdef _DEBUG_SPLIT_
                nou_data_3D_vect++;
#endif
                break;
            case V_NULL:
                out_data_3D_vect = new coDoVec3(p_data_3D_vect->getObjName(),
                                                0);
                break;
            }
            if (!out_data_3D_vect->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_3D_vect->getObjName(),
                          p_data_3D_vect->getName());
            }
        }
        fill_data_objects(3); // 3 -> three dimensions
    }
}

// write_conn_list fills the connectivity list of an output grid.
// conn_out_list: pointer to the connectivity list
// numConnOut: indicates the position of the connectivity
//             list where the information for this element begins
// num_of_nodes: number of nodes of this element
// element: number of element at issue
void SplitUSG::write_conn_list(int *conn_out_list, unsigned int numConnOut,
                               int num_of_nodes, unsigned int element)
{
    for (int i = 0; i < num_of_nodes; ++i)
    {
        conn_out_list[i + numConnOut] = coord_map[conn_list[i + elem_list[element]]] - 1;
    }
}

// The lists of the output data created in count_out_?D_grid
// are filled by fill_data_objects
void SplitUSG::fill_data_objects(unsigned int dimension)
{
    unsigned int element;
    unsigned int numConnNextElement;
    unsigned int numConnThisElement = 0;

    switch (dimension)
    {
    case 3:
    {
        int *elem_3D_list, *conn_3D_list, *type_3D_list = NULL;
        float *x_3D, *y_3D, *z_3D;
        unsigned int numElem3D = 0, numConn3D = 0;
        if (out_grid_3D)
        {
            out_grid_3D->getAddresses(&elem_3D_list, &conn_3D_list,
                                      &x_3D, &y_3D, &z_3D);
            if (type_list)
                out_grid_3D->getTypeList(&type_3D_list);
            float *scal3D_data = NULL; // scalar data list
            // vector data lists
            float *vect3D_data_x = NULL, *vect3D_data_y = NULL, *vect3D_data_z = NULL;

            // ... get addresses to fill scalar or vector data
            if (in_data_scal)
                out_data_3D_scal->getAddress(&scal3D_data);
            if (in_data_vect)
                out_data_3D_vect->getAddresses(&vect3D_data_x,
                                               &vect3D_data_y,
                                               &vect3D_data_z);

            for (element = 0; element < (unsigned int)numEl; ++element)
            {
                int it_is_not_3D = 0;
                int advance_numConn3D;
                // in this IF I fill the element and connectivity
                // and type lists of the grid
                if (type_list)
                {
                    switch (type_list[element])
                    {
                    case TYPE_PRISM:
                        elem_3D_list[numElem3D] = numConn3D;
                        type_3D_list[numElem3D] = TYPE_PRISM;
                        write_conn_list(conn_3D_list, numConn3D, 6, element);
                        advance_numConn3D = 6;
                        break;
                    case TYPE_TETRAHEDER:
                        elem_3D_list[numElem3D] = numConn3D;
                        type_3D_list[numElem3D] = TYPE_TETRAHEDER;
                        write_conn_list(conn_3D_list, numConn3D, 4, element);
                        advance_numConn3D = 4;
                        break;
                    case TYPE_PYRAMID:
                        elem_3D_list[numElem3D] = numConn3D;
                        type_3D_list[numElem3D] = TYPE_PYRAMID;
                        write_conn_list(conn_3D_list, numConn3D, 5, element);
                        advance_numConn3D = 5;
                        break;
                    case TYPE_HEXAEDER:
                        elem_3D_list[numElem3D] = numConn3D;
                        type_3D_list[numElem3D] = TYPE_HEXAEDER;
                        write_conn_list(conn_3D_list, numConn3D, 8, element);
                        advance_numConn3D = 8;
                        break;
                    default:
                        it_is_not_3D = 1;
                        break;
                        // Not a 3D type
                    }
                }
                else
                {
                    sendWarning("Grids without element type info are deprecated.");
                    if ((int)element < numEl - 1)
                        numConnNextElement = elem_list[element + 1];
                    else
                        numConnNextElement = numConn;
                    switch (numConnNextElement - numConnThisElement)
                    {
                    case 4: // TETRAHEDER
                        elem_3D_list[numElem3D] = numConn3D;
                        // I assume this is a tetrahedron and not a quad!!!
                        write_conn_list(conn_3D_list, numConn3D, 4, element);
                        advance_numConn3D = 4;
                        break;
                    case 6: // PRISM
                        elem_3D_list[numElem3D] = numConn3D;
                        write_conn_list(conn_3D_list, numConn3D, 6, element);
                        advance_numConn3D = 6;
                        break;
                    case 5: // PYRAMID
                        elem_3D_list[numElem3D] = numConn3D;
                        write_conn_list(conn_3D_list, numConn3D, 5, element);
                        advance_numConn3D = 5;
                        break;
                    case 8: // HEXAHERDRON
                        elem_3D_list[numElem3D] = numConn3D;
                        write_conn_list(conn_3D_list, numConn3D, 8, element);
                        advance_numConn3D = 8;
                        break;
                    default:
                        it_is_not_3D = 1;
                        break;
                        // Not a 3D type
                    }
                } // end of else typ_list

                if (!it_is_not_3D)
                {
                    // if data is element-based, we fill the lists
                    if (in_data_scal && scal_cell_or_vertex == S_PER_CELL)
                    {
                        scal3D_data[numElem3D] = scal_data[element];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_CELL)
                    {
                        vect3D_data_x[numElem3D] = vect_data_x[element];
                        vect3D_data_y[numElem3D] = vect_data_y[element];
                        vect3D_data_z[numElem3D] = vect_data_z[element];
                    }
                    ++numElem3D; // advance element counter for 3D list
                    numConn3D += advance_numConn3D;
                }
            } // End of element loop

            // Now it remains filling the coordinates of the
            // 3D grid and eventually the scalar and vector data
            // if node-based
            for (int globalNode = 0; globalNode < numCoord; ++globalNode)
            {
                if (coord_map[globalNode])
                {
                    int globalNode3D = coord_map[globalNode] - 1;
                    x_3D[globalNode3D] = x_in[globalNode];
                    y_3D[globalNode3D] = y_in[globalNode];
                    z_3D[globalNode3D] = z_in[globalNode];
                    if (in_data_scal && scal_cell_or_vertex == S_PER_VERTEX)
                    {
                        scal3D_data[globalNode3D] = scal_data[globalNode];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_VERTEX)
                    {
                        vect3D_data_x[globalNode3D] = vect_data_x[globalNode];
                        vect3D_data_y[globalNode3D] = vect_data_y[globalNode];
                        vect3D_data_z[globalNode3D] = vect_data_z[globalNode];
                    }
                }
            }
        }
        break;
    }
    case 2:
    {
        int *elem_2D_list, *conn_2D_list;
        float *x_2D, *y_2D, *z_2D;
        unsigned int numElem2D = 0, numConn2D = 0;
        if (out_grid_2D)
        {
            out_grid_2D->getAddresses(&x_2D, &y_2D, &z_2D,
                                      &conn_2D_list, &elem_2D_list);
            float *scal2D_data = NULL; // scalar data list
            // vector data lists
            float *vect2D_data_x = NULL, *vect2D_data_y = NULL, *vect2D_data_z = NULL;

            // ... get addresses to fill scalar or vector data
            if (in_data_scal)
                out_data_2D_scal->getAddress(&scal2D_data);
            if (in_data_vect)
                out_data_2D_vect->getAddresses(&vect2D_data_x,
                                               &vect2D_data_y,
                                               &vect2D_data_z);

            for (element = 0; element < (unsigned int)numEl; ++element)
            {
                int it_is_not_2D = 0;
                int advance_numConn2D;

                // in this IF I fill the element and connectivity
                // lists of the grid
                if (type_list)
                {
                    switch (type_list[element])
                    {
                    case TYPE_QUAD:
                        elem_2D_list[numElem2D] = numConn2D;
                        write_conn_list(conn_2D_list, numConn2D, 4, element);
                        advance_numConn2D = 4;
                        break;
                    case TYPE_TRIANGLE:
                        elem_2D_list[numElem2D] = numConn2D;
                        write_conn_list(conn_2D_list, numConn2D, 3, element);
                        advance_numConn2D = 3;
                        break;
                    default:
                        it_is_not_2D = 1;
                        break;
                        // Not a 2D type
                    }
                }
                else
                {
                    sendWarning("Grids without element type info are deprecated.");
                    if ((int)element < numEl - 1)
                        numConnNextElement = elem_list[element + 1];
                    else
                        numConnNextElement = numConn;
                    switch (numConnNextElement - numConnThisElement)
                    {
                    case 3: // TRIANGLE
                        elem_2D_list[numElem2D] = numConn2D;
                        write_conn_list(conn_2D_list, numConn2D, 3, element);
                        advance_numConn2D = 3;
                        break;
                    default:
                        it_is_not_2D = 1;
                        break;
                        // Not a 2D type
                    }
                } // end of else typ_list
                if (!it_is_not_2D)
                {
                    // if data is element-based, we fill the lists
                    if (in_data_scal && scal_cell_or_vertex == S_PER_CELL)
                    {
                        scal2D_data[numElem2D] = scal_data[element];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_CELL)
                    {
                        vect2D_data_x[numElem2D] = vect_data_x[element];
                        vect2D_data_y[numElem2D] = vect_data_y[element];
                        vect2D_data_z[numElem2D] = vect_data_z[element];
                    }
                    ++numElem2D; // advance element counter for 2D list
                    numConn2D += advance_numConn2D;
                }
            } // End of element loop

            // Now it remains filling the coordinates of the
            // 2D grid and eventually the scalar and vector data
            // if node-based
            for (int globalNode = 0; globalNode < numCoord; ++globalNode)
            {
                if (coord_map[globalNode])
                {
                    int globalNode2D = coord_map[globalNode] - 1;
                    x_2D[globalNode2D] = x_in[globalNode];
                    y_2D[globalNode2D] = y_in[globalNode];
                    z_2D[globalNode2D] = z_in[globalNode];
                    if (in_data_scal && scal_cell_or_vertex == S_PER_VERTEX)
                    {
                        scal2D_data[globalNode2D] = scal_data[globalNode];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_VERTEX)
                    {
                        vect2D_data_x[globalNode2D] = vect_data_x[globalNode];
                        vect2D_data_y[globalNode2D] = vect_data_y[globalNode];
                        vect2D_data_z[globalNode2D] = vect_data_z[globalNode];
                    }
                }
            }
        }
        break;
    }
    case 1:
    {
        int *elem_1D_list, *conn_1D_list;
        float *x_1D, *y_1D, *z_1D;
        unsigned int numElem1D = 0, numConn1D = 0;
        if (out_grid_1D)
        {
            out_grid_1D->getAddresses(&x_1D, &y_1D, &z_1D,
                                      &conn_1D_list, &elem_1D_list);
            float *scal1D_data = NULL; // scalar data list
            // vector data lists
            float *vect1D_data_x = NULL, *vect1D_data_y = NULL, *vect1D_data_z = NULL;

            // ... get addresses to fill scalar or vector data
            if (in_data_scal)
                out_data_1D_scal->getAddress(&scal1D_data);
            if (in_data_vect)
                out_data_1D_vect->getAddresses(&vect1D_data_x,
                                               &vect1D_data_y,
                                               &vect1D_data_z);

            for (element = 0; element < (unsigned int)numEl; ++element)
            {
                int it_is_not_1D = 0;
                int advance_numConn1D;

                // in this IF I fill the element and connectivity
                // lists of the grid
                if (type_list)
                {
                    switch (type_list[element])
                    {
                    case TYPE_BAR:
                        elem_1D_list[numElem1D] = numConn1D;
                        write_conn_list(conn_1D_list, numConn1D, 2, element);
                        advance_numConn1D = 2;
                        break;
                    default:
                        it_is_not_1D = 1;
                        break;
                        // Not a 1D type
                    }
                }
                else
                {
                    sendWarning("Grids without element type info are deprecated.");
                    if ((int)element < numEl - 1)
                        numConnNextElement = elem_list[element + 1];
                    else
                        numConnNextElement = numConn;
                    switch (numConnNextElement - numConnThisElement)
                    {
                    case 2: // BAR
                        elem_1D_list[numElem1D] = numConn1D;
                        write_conn_list(conn_1D_list, numConn1D, 2, element);
                        advance_numConn1D = 2;
                        break;
                    default:
                        it_is_not_1D = 1;
                        break;
                        // Not a 1D type
                    }
                } // end of else typ_list
                if (!it_is_not_1D)
                {
                    // if data is element-based, we fill the lists
                    if (in_data_scal && scal_cell_or_vertex == S_PER_CELL)
                    {
                        scal1D_data[numElem1D] = scal_data[element];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_CELL)
                    {
                        vect1D_data_x[numElem1D] = vect_data_x[element];
                        vect1D_data_y[numElem1D] = vect_data_y[element];
                        vect1D_data_z[numElem1D] = vect_data_z[element];
                    }
                    ++numElem1D; // advance element counter for 1D list
                    numConn1D += advance_numConn1D;
                }
            } // End of element loop

            // Now it remains filling the coordinates of the
            // 1D grid and eventually the scalar and vector data
            // if node-based
            for (int globalNode = 0; globalNode < numCoord; ++globalNode)
            {
                if (coord_map[globalNode])
                {
                    int globalNode1D = coord_map[globalNode] - 1;
                    x_1D[globalNode1D] = x_in[globalNode];
                    y_1D[globalNode1D] = y_in[globalNode];
                    z_1D[globalNode1D] = z_in[globalNode];
                    if (in_data_scal && scal_cell_or_vertex == S_PER_VERTEX)
                    {
                        scal1D_data[globalNode1D] = scal_data[globalNode];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_VERTEX)
                    {
                        vect1D_data_x[globalNode1D] = vect_data_x[globalNode];
                        vect1D_data_y[globalNode1D] = vect_data_y[globalNode];
                        vect1D_data_z[globalNode1D] = vect_data_z[globalNode];
                    }
                }
            }
        }
        break;
    }
    case 0:
    {
        float *x_0D, *y_0D, *z_0D;
        unsigned int numElem0D = 0, numConn0D = 0;
        if (out_grid_0D)
        {
            out_grid_0D->getAddresses(&x_0D, &y_0D, &z_0D);
            float *scal0D_data = NULL; // scalar data list
            // vector data lists
            float *vect0D_data_x = NULL, *vect0D_data_y = NULL, *vect0D_data_z = NULL;

            // ... get addresses to fill scalar or vector data
            if (in_data_scal)
                out_data_0D_scal->getAddress(&scal0D_data);
            if (in_data_vect)
                out_data_0D_vect->getAddresses(&vect0D_data_x,
                                               &vect0D_data_y,
                                               &vect0D_data_z);

            for (element = 0; element < (unsigned int)numEl; ++element)
            {
                int it_is_not_0D = 0;
                int advance_numConn0D;
                // if data is element-based, we fill the lists
                if (in_data_scal && scal_cell_or_vertex == S_PER_CELL)
                {
                    scal0D_data[numElem0D] = scal_data[element];
                }
                if (in_data_vect && vect_cell_or_vertex == V_PER_CELL)
                {
                    vect0D_data_x[numElem0D] = vect_data_x[element];
                    vect0D_data_y[numElem0D] = vect_data_y[element];
                    vect0D_data_z[numElem0D] = vect_data_z[element];
                }

                // in this IF I fill the element and connectivity
                // lists of the grid
                if (type_list)
                {
                    switch (type_list[element])
                    {
                    case TYPE_POINT:
                        advance_numConn0D = 1;
                        break;
                    default:
                        it_is_not_0D = 1;
                        break;
                        // Not a 0D type
                    }
                }
                else
                {
                    sendWarning("Grids without element type info are deprecated.");
                    if ((int)element < numEl - 1)
                        numConnNextElement = elem_list[element + 1];
                    else
                        numConnNextElement = numConn;
                    switch (numConnNextElement - numConnThisElement)
                    {
                    case 1: // POINT
                        advance_numConn0D = 1;
                        break;
                    default:
                        it_is_not_0D = 1;
                        break;
                        // Not a 0D type
                    }
                } // end of else typ_list
                if (!it_is_not_0D)
                {
                    // if data is element-based, we fill the lists
                    if (in_data_scal && scal_cell_or_vertex == S_PER_CELL)
                    {
                        scal0D_data[numElem0D] = scal_data[element];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_CELL)
                    {
                        vect0D_data_x[numElem0D] = vect_data_x[element];
                        vect0D_data_y[numElem0D] = vect_data_y[element];
                        vect0D_data_z[numElem0D] = vect_data_z[element];
                    }
                    ++numElem0D; // advance element counter for 0D list
                    numConn0D += advance_numConn0D;
                }
            } // End of element loop

            // Now it remains filling the coordinates of the
            // 0D grid and eventually the scalar and vector data
            // if node-based
            for (int globalNode = 0; globalNode < numCoord; ++globalNode)
            {
                if (coord_map[globalNode])
                {
                    int globalNode0D = coord_map[globalNode] - 1;
                    x_0D[globalNode0D] = x_in[globalNode];
                    y_0D[globalNode0D] = y_in[globalNode];
                    z_0D[globalNode0D] = z_in[globalNode];
                    if (in_data_scal && scal_cell_or_vertex == S_PER_VERTEX)
                    {
                        scal0D_data[globalNode0D] = scal_data[globalNode];
                    }
                    if (in_data_vect && vect_cell_or_vertex == V_PER_VERTEX)
                    {
                        vect0D_data_x[globalNode0D] = vect_data_x[globalNode];
                        vect0D_data_y[globalNode0D] = vect_data_y[globalNode];
                        vect0D_data_z[globalNode0D] = vect_data_z[globalNode];
                    }
                }
            }
        }
        break;
    }
    }
}

// build_out_2D_grid counts the dimensions
// of the 2D output grid and sets up a map
// for the coordinates. The map is an array (coord_map)
// with as many ints as coordinates in in_grid.
// At the beginning all items in the map are 0. The elements
// of the grid are sequentially read and if one is a 2D one,
// the coordinates of the corresponding items in the
// connectivity list are checked with a positive number (mark)
// in the map. This number is increased by one whenever
// assigned to a new coordinate item.

// build_out_2D_grid also creates the output objects
// (both grid) and data, and it fills the lists (it calls fill_data_objects).

void SplitUSG::build_out_2D_grid()
{
    unsigned int element;
    unsigned int numElem2D = 0, numConn2D = 0;

    unsigned int numConnNextElement;
    unsigned int numConnThisElement = 0;

    memset(coord_map, 0, numCoord * sizeof(int));
    mark = 0;
    for (element = 0; element < (unsigned int)numEl; ++element)
    {
        if (type_list)
        {
            switch (type_list[element])
            {
            case TYPE_QUAD:
                ++numElem2D;
                numConn2D += 4;
                put_marks_in_map(element, 4);
                break;
            case TYPE_TRIANGLE:
                ++numElem2D;
                numConn2D += 3;
                put_marks_in_map(element, 3);
                break;
            default:
                break;
                // Not a 2D type
            }
        }
        else
        {
            if ((int)element < numEl - 1)
                numConnNextElement = elem_list[element + 1];
            else
                numConnNextElement = numConn;
            switch (numConnNextElement - numConnThisElement)
            {
            case 3: // TRIANGLE
                ++numElem2D;
                numConn2D += 3;
                put_marks_in_map(element, 3);
                break;
            default:
                break;
                // Not a 2D type
            }
            numConnThisElement = numConnNextElement;
        }
    }

    // Now create output grid (2D)...
    if (numElem2D)
    {
        out_grid_2D = new coDoPolygons(p_grid_2D->getObjName(), mark, numConn2D,
                                       numElem2D);
#ifdef _DEBUG_SPLIT_
        nou_grid_2D++;
#endif
        if (!out_grid_2D->objectOk())
        {
            sendError("Failed to create object '%s' for port '%s'",
                      p_grid_2D->getObjName(), p_grid_2D->getName());
        }
        // ... and scalar data if necessary ...
        if (in_data_scal)
        {
            switch (scal_cell_or_vertex)
            {
            case S_PER_CELL:
                out_data_2D_scal = new coDoFloat(p_data_2D_scal->getObjName(),
                                                 numElem2D);
#ifdef _DEBUG_SPLIT_
                nou_data_2D_scal++;
#endif
                break;
            case S_PER_VERTEX:
                out_data_2D_scal = new coDoFloat(p_data_2D_scal->getObjName(),
                                                 mark);
#ifdef _DEBUG_SPLIT_
                nou_data_2D_scal++;
#endif
                break;
            case S_NULL:
                out_data_2D_scal = new coDoFloat(p_data_2D_scal->getObjName(),
                                                 0);
                break;
            }
            if (!out_data_2D_scal->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_2D_scal->getObjName(),
                          p_data_2D_scal->getName());
            }
        }
        // ... and vector data if necessary ...
        if (in_data_vect)
        {
            switch (vect_cell_or_vertex)
            {
            case V_PER_CELL:
                out_data_2D_vect = new coDoVec3(p_data_2D_vect->getObjName(),
                                                numElem2D);
#ifdef _DEBUG_SPLIT_
                nou_data_2D_vect++;
#endif
                break;
            case V_PER_VERTEX:
                out_data_2D_vect = new coDoVec3(p_data_2D_vect->getObjName(),
                                                mark);
#ifdef _DEBUG_SPLIT_
                nou_data_2D_vect++;
#endif
                break;
            case V_NULL:
                out_data_2D_vect = new coDoVec3(p_data_2D_vect->getObjName(),
                                                0);
                break;
            }
            if (!out_data_2D_vect->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_2D_vect->getObjName(),
                          p_data_2D_vect->getName());
            }
        }
        fill_data_objects(2); // 2 -> two dimensions
    }
}

// build_out_1D_grid counts the dimensions
// of the 1D output grid and sets up a map
// for the coordinates. The map is an array (coord_map)
// with as many ints as coordinates in in_grid.
// At the beginning all items in the map are 0. The elements
// of the grid are sequentially read and if one is a 1D one,
// the coordinates of the corresponding items in the
// connectivity list are checked with a positive number (mark)
// in the map. This number is increased by one whenever
// assigned to a new coordinate item.

// build_out_1D_grid also creates the output objects
// (both grid) and data, and it fills the lists (it calls fill_data_objects).

void SplitUSG::build_out_1D_grid()
{
    unsigned int element;
    unsigned int numElem1D = 0, numConn1D = 0;

    unsigned int numConnNextElement;
    unsigned int numConnThisElement = 0;

    memset(coord_map, 0, numCoord * sizeof(int));
    mark = 0;
    for (element = 0; element < (unsigned int)numEl; ++element)
    {
        if (type_list)
        {
            switch (type_list[element])
            {
            case TYPE_BAR:
                ++numElem1D;
                numConn1D += 2;
                put_marks_in_map(element, 2);
                break;
            default:
                break;
                // Not a 1D type
            }
        }
        else
        {
            if ((int)element < numEl - 1)
                numConnNextElement = elem_list[element + 1];
            else
                numConnNextElement = numConn;
            switch (numConnNextElement - numConnThisElement)
            {
            case 2: // BAR
                ++numElem1D;
                numConn1D += 2;
                put_marks_in_map(element, 2);
                break;
            default:
                break;
                // Not a 1D type
            }
            numConnThisElement = numConnNextElement;
        }
    }

    // Now create output grid (1D)...
    if (numElem1D)
    {
        out_grid_1D = new coDoLines(p_grid_1D->getObjName(), mark, numConn1D, numElem1D);
#ifdef _DEBUG_SPLIT_
        nou_grid_1D++;
#endif
        if (!out_grid_1D->objectOk())
        {
            sendError("Failed to create object '%s' for port '%s'",
                      p_grid_1D->getObjName(), p_grid_1D->getName());
        }
        // ... and scalar data if necessary ...
        if (in_data_scal)
        {
            switch (scal_cell_or_vertex)
            {
            case S_PER_CELL:
                out_data_1D_scal = new coDoFloat(p_data_1D_scal->getObjName(),
                                                 numElem1D);
#ifdef _DEBUG_SPLIT_
                nou_data_1D_scal++;
#endif
                break;
            case S_PER_VERTEX:
                out_data_1D_scal = new coDoFloat(p_data_1D_scal->getObjName(),
                                                 mark);
#ifdef _DEBUG_SPLIT_
                nou_data_1D_scal++;
#endif
                break;
            case S_NULL:
                out_data_1D_scal = new coDoFloat(p_data_1D_scal->getObjName(),
                                                 0);
                break;
            }
            if (!out_data_1D_scal->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_1D_scal->getObjName(),
                          p_data_1D_scal->getName());
            }
        }
        // ... and vector data if necessary ...
        if (in_data_vect)
        {
            switch (vect_cell_or_vertex)
            {
            case V_PER_CELL:
                out_data_1D_vect = new coDoVec3(p_data_1D_vect->getObjName(),
                                                numElem1D);
#ifdef _DEBUG_SPLIT_
                nou_data_1D_vect++;
#endif
                break;
            case V_PER_VERTEX:
                out_data_1D_vect = new coDoVec3(p_data_1D_vect->getObjName(),
                                                mark);
#ifdef _DEBUG_SPLIT_
                nou_data_1D_vect++;
#endif
                break;
            case V_NULL:
                out_data_1D_vect = new coDoVec3(p_data_1D_vect->getObjName(),
                                                0);
                break;
            }
            if (!out_data_1D_vect->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_1D_vect->getObjName(),
                          p_data_1D_vect->getName());
            }
        }
        fill_data_objects(1); // 1 -> one dimensions
    }
}

// build_out_0D_grid counts the dimensions
// of the 0D output grid and sets up a map
// for the coordinates. The map is an array (coord_map)
// with as many ints as coordinates in in_grid.
// At the beginning all items in the map are 0. The elements
// of the grid are sequentially read and if one is a 0D one,
// the coordinates of the corresponding items in the
// connectivity list are checked with a positive number (mark)
// in the map. This number is increased by one whenever
// assigned to a new coordinate item.

// build_out_0D_grid also creates the output objects
// (both grid) and data, and it fills the lists (it calls fill_data_objects).

void SplitUSG::build_out_0D_grid()
{
    unsigned int element;
    unsigned int numElem0D = 0, numConn0D = 0;

    unsigned int numConnNextElement;
    unsigned int numConnThisElement = 0;

    memset(coord_map, 0, numCoord * sizeof(int));
    mark = 0;
    for (element = 0; element < (unsigned int)numEl; ++element)
    {
        if (type_list)
        {
            switch (type_list[element])
            {
            case TYPE_POINT:
                ++numElem0D;
                numConn0D += 1;
                put_marks_in_map(element, 1);
                break;
            default:
                break;
                // Not a 0D type
            }
        }
        else
        {
            if ((int)element < numEl - 1)
                numConnNextElement = elem_list[element + 1];
            else
                numConnNextElement = numConn;
            switch (numConnNextElement - numConnThisElement)
            {
            case 1: // POINT
                ++numElem0D;
                numConn0D += 1;
                put_marks_in_map(element, 1);
                break;
            default:
                break;
                // Not a 1D type
            }
            numConnThisElement = numConnNextElement;
        }
    }

    // Now create output grid (0D)...
    if (numElem0D)
    {
        out_grid_0D = new coDoPoints(p_grid_0D->getObjName(), mark);
#ifdef _DEBUG_SPLIT_
        nou_grid_0D++;
#endif
        if (!out_grid_0D->objectOk())
        {
            sendError("Failed to create object '%s' for port '%s'",
                      p_grid_0D->getObjName(), p_grid_0D->getName());
        }
        // ... and scalar data if necessary ...
        if (in_data_scal)
        {
            switch (scal_cell_or_vertex)
            {
            case S_PER_CELL:
                out_data_0D_scal = new coDoFloat(p_data_0D_scal->getObjName(),
                                                 numElem0D);
#ifdef _DEBUG_SPLIT_
                nou_data_0D_scal++;
#endif
                break;
            case S_PER_VERTEX:
                out_data_0D_scal = new coDoFloat(p_data_0D_scal->getObjName(),
                                                 mark);
#ifdef _DEBUG_SPLIT_
                nou_data_0D_scal++;
#endif
                break;
            case S_NULL:
                out_data_0D_scal = new coDoFloat(p_data_0D_scal->getObjName(),
                                                 0);
                break;
            }
            if (!out_data_0D_scal->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_0D_scal->getObjName(),
                          p_data_0D_scal->getName());
            }
        }
        // ... and vector data if necessary ...
        if (in_data_vect)
        {
            switch (vect_cell_or_vertex)
            {
            case V_PER_CELL:
                out_data_0D_vect = new coDoVec3(p_data_0D_vect->getObjName(),
                                                numElem0D);
#ifdef _DEBUG_SPLIT_
                nou_data_0D_vect++;
#endif
                break;
            case V_PER_VERTEX:
                out_data_0D_vect = new coDoVec3(p_data_0D_vect->getObjName(),
                                                mark);
#ifdef _DEBUG_SPLIT_
                nou_data_0D_vect++;
#endif
                break;
            case V_NULL:
                out_data_0D_vect = new coDoVec3(p_data_0D_vect->getObjName(),
                                                0);
                break;
            }
            if (!out_data_0D_vect->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'",
                          p_data_0D_vect->getObjName(),
                          p_data_0D_vect->getName());
            }
        }
        fill_data_objects(0); // 0 -> zero dimensions
    }
}

// Given an element put_marks_in_map puts the marks in coord_map
// to be able to reconstruct the order of the coordinates
// of the points for the output grids.
// int element: element for whose nodes we want to put a mark
//              (if a mark has not been set by a previous element
//               for the node at issue)
// int numLocalNodes: number of nodes of this element
void SplitUSG::put_marks_in_map(int element, int numLocalNodes)
{
    for (int localNode = 0; localNode < numLocalNodes; ++localNode)
    {
        if (!coord_map[conn_list[elem_list[element] + localNode]])
        {
            ++mark;
            coord_map[conn_list[elem_list[element] + localNode]] = mark;
        }
    }
}

MODULE_MAIN(Filter, SplitUSG)
