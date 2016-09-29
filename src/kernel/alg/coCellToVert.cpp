/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                     (C)2005 Visenso ++
// ++ Description: Interpolation from Cell Data to Vertex Data            ++
// ++                 ( CellToVert module functionality  )                ++
// ++                                                                     ++
// ++ Author: Sven Kufer( sk@visenso.de)                                  ++
// ++                                                                     ++
// ++**********************************************************************/

#include "coCellToVert.h"
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>

namespace covise
{
inline double sqr(float x)
{
    return double(x) * double(x);
}
}

using namespace covise;

#define NODES_IN_ELEM(i) (((i) == num_elem - 1) ? num_conn - elem_list[(i)] : elem_list[(i) + 1] - elem_list[(i)])

////// workin' routines
bool
coCellToVert::interpolate(bool unstructured, int num_elem, int num_conn, int num_point,
                          const int *elem_list, const int *conn_list, const int *type_list, const int *neighbour_cells, const int *neighbour_idx,
                          const float *xcoord, const float *ycoord, const float *zcoord,
                          int numComp, int &dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                          float *out_data_0, float *out_data_1, float *out_data_2, Algorithm algo_option)
{
    // check for errors
    if (!elem_list || !conn_list || !xcoord || !ycoord || !zcoord || !in_data_0)
    {
        return (false);
    }

    if (numComp != 1 && numComp != 3)
    {
        //Covise::sendError("incorrect input type in data_in");
        return (false);
    }

    // copy original data if already vertex based
    if (dataSize == num_point)
    {
        int i;
        for (i = 0; i < num_point; i++)
        {
            out_data_0[i] = in_data_0[i];
            if (numComp == 3)
            {
                out_data_1[i] = in_data_1[i];
                out_data_2[i] = in_data_2[i];
            }
        }
        return true;
    }

    if (unstructured)
    {
        switch (algo_option)
        {
        case SIMPLE:
            return simpleAlgo(num_elem, num_conn, num_point,
                              elem_list, conn_list,
                              numComp, dataSize, in_data_0, in_data_1, in_data_2,
                              out_data_0, out_data_1, out_data_2);

        case SQR_WEIGHT:
            return weightedAlgo(num_elem, num_conn, num_point,
                                elem_list, conn_list, type_list, neighbour_cells, neighbour_idx,
                                xcoord, ycoord, zcoord,
                                numComp, dataSize, in_data_0, in_data_1, in_data_2,
                                out_data_0, out_data_1, out_data_2);
        }
    }
    else
    {
        return simpleAlgo(num_elem, num_conn, num_point,
                          elem_list, conn_list,
                          numComp, dataSize, in_data_0, in_data_1, in_data_2,
                          out_data_0, out_data_1, out_data_2);
    }

    return true;
}

bool
coCellToVert::simpleAlgo(int num_elem, int num_conn, int num_point,
                         const int *elem_list, const int *conn_list,
                         int numComp, int dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                         float *out_data_0, float *out_data_1, float *out_data_2)
{
    int i, j, n, vertex;
    enum
    {
        SCALAR = 1,
        VECTOR = 3
    };

    float *weight_num;
    weight_num = new float[num_point];

    if (!weight_num)
        return false;

    // reset everything to 0, != 0 to prevent div/0 errors
    for (vertex = 0; vertex < num_point; vertex++)
        weight_num[vertex] = 1.0e-30f;

    if (numComp == SCALAR)
        for (vertex = 0; vertex < num_point; vertex++)
            out_data_0[vertex] = 0.0;
    else
    {
        for (vertex = 0; vertex < num_point; vertex++)
            out_data_0[vertex] = 0.0;
        for (vertex = 0; vertex < num_point; vertex++)
            out_data_1[vertex] = 0.0;
        for (vertex = 0; vertex < num_point; vertex++)
            out_data_2[vertex] = 0.0;
    }

    if (numComp == SCALAR)
        for (i = 0; i < num_elem; i++)
        {
            n = NODES_IN_ELEM(i);
            for (j = 0; j < n; j++)
            {
                vertex = conn_list[elem_list[i] + j];
                weight_num[vertex] += 1.0;
                if (i < dataSize)
                    out_data_0[vertex] += in_data_0[i];
            }
        }
    else
        for (i = 0; i < num_elem; i++)
        {
            n = NODES_IN_ELEM(i);
            for (j = 0; j < n; j++)
            {
                vertex = conn_list[elem_list[i] + j];
                weight_num[vertex] += 1.0;
                if (i < dataSize)
                {
                    out_data_0[vertex] += in_data_0[i];
                    out_data_1[vertex] += in_data_1[i];
                    out_data_2[vertex] += in_data_2[i];
                }
            }
        }

    // divide value sum by 'weight' (# adjacent cells)

    if (numComp == SCALAR)
    {
        for (vertex = 0; vertex < num_point; vertex++)
            if (weight_num[vertex] >= 1.0)
                out_data_0[vertex] /= weight_num[vertex];
    }
    else
    {
        for (vertex = 0; vertex < num_point; vertex++)
        {
            if (weight_num[vertex] >= 1.0)
            {
                out_data_0[vertex] /= weight_num[vertex];
                out_data_1[vertex] /= weight_num[vertex];
                out_data_2[vertex] /= weight_num[vertex];
            }
        }
    }

    // clean up
    delete[] weight_num;

    return true;
}

bool
coCellToVert::weightedAlgo(int num_elem, int num_conn, int num_point,
                           const int *elem_list, const int *conn_list, const int *type_list, const int *neighbour_cells, const int *neighbour_idx,
                           const float *xcoord, const float *ycoord, const float *zcoord,
                           int numComp, int dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                           float *out_data_0, float *out_data_1, float *out_data_2)
{
    if (!neighbour_cells || !neighbour_idx)
        return false;

    // now go through all elements and calculate their center

    int *ePtr = (int *)elem_list;
    int *tPtr = (int *)type_list;

    int el_type;
    int elem, num_vert_elem;
    int vert, vertex, *vertex_id;

    float *xc, *yc, *zc;

    float *cell_center_0 = new float[num_elem];
    float *cell_center_1 = new float[num_elem];
    float *cell_center_2 = new float[num_elem];

    //static const int num_vertices_per_element[] = {0,2,3,4,4,5,6,8};

    for (elem = 0; elem < num_elem; elem++)
    {
        el_type = *tPtr++; // get this elements type (then go on to the next one)
        //num_vert_elem = num_vertices_per_element[el_type];
        if (elem == num_elem - 1)
        {
            num_vert_elem = num_conn - elem_list[elem];
        }
        else
        {
            num_vert_elem = elem_list[elem + 1] - elem_list[elem];
        }

        // # of vertices in current element
        vertex_id = (int *)conn_list + (*ePtr++); // get ptr to the first vertex-id of current element
        // then go on to the next one

        // place where to store the calculated center-coordinates in
        xc = &cell_center_0[elem];
        yc = &cell_center_1[elem];
        zc = &cell_center_2[elem];

        // reset
        (*xc) = (*yc) = (*zc) = 0.0;

        //FIXME doesn't make sense for Polyhedrons
        // the center can be calculated now
        if (el_type == TYPE_POLYHEDRON)
        {
            int num_averaged = 0;
            int facestart = conn_list[elem_list[elem]];
            bool face_done = true;
            for (vert = 0; vert < num_vert_elem; vert++)
            {
                int cur_vert = conn_list[elem_list[elem] + vert];
                if (face_done)
                {
                    facestart = cur_vert;
                    face_done = false;
                }
                else if (facestart == cur_vert)
                {
                    face_done = true;
                    continue;
                }
                (*xc) += xcoord[cur_vert];
                (*yc) += ycoord[cur_vert];
                (*zc) += zcoord[cur_vert];
                ++num_averaged;
            }
            (*xc) /= num_averaged;
            (*yc) /= num_averaged;
            (*zc) /= num_averaged;
        }
        else
        {
            for (vert = 0; vert < num_vert_elem; vert++)
            {
                (*xc) += xcoord[*vertex_id];
                (*yc) += ycoord[*vertex_id];
                (*zc) += zcoord[*vertex_id];
                vertex_id++;
            }
            (*xc) /= num_vert_elem;
            (*yc) /= num_vert_elem;
            (*zc) /= num_vert_elem;
        }
    }

    int actIndex = 0;
    int ni, cp;

    int *niPtr = (int *)neighbour_idx + 1;
    int *cellPtr = (int *)neighbour_cells;

    double weight_sum, weight;
    double value_sum_0, value_sum_1, value_sum_2;
    float vx, vy, vz, ccx, ccy, ccz;

    for (vertex = 0; vertex < num_point; vertex++)
    {
        weight_sum = 0.0;
        value_sum_0 = 0.0;
        value_sum_1 = 0.0;
        value_sum_2 = 0.0;

        vx = xcoord[vertex];
        vy = ycoord[vertex];
        vz = zcoord[vertex];

        ni = *niPtr;

        while (actIndex < ni) // loop over neighbour cells
        {
            cp = *cellPtr;
            ccx = cell_center_0[cp];
            ccy = cell_center_1[cp];
            ccz = cell_center_2[cp];

            // cells with 0 volume are not weigthed
            //XXX: was soll das?
            //weight = (weight==0.0) ? 0 : (1.0/weight);
            weight = sqr(vx - ccx) + sqr(vy - ccy) + sqr(vz - ccz);
            weight_sum += weight;

            if (cp < dataSize)
            {
                if (numComp == 1)
                    value_sum_0 += weight * in_data_0[cp];
                else
                {
                    value_sum_0 += weight * in_data_0[cp];
                    value_sum_1 += weight * in_data_1[cp];
                    value_sum_2 += weight * in_data_2[cp];
                }
            }

            actIndex++;
            cellPtr++;
        }

        niPtr++;
        if (weight_sum == 0)
            weight_sum = 1.0;

        if (numComp == 1)
            out_data_0[vertex] = (float)(value_sum_0 / weight_sum);
        else
        {
            out_data_0[vertex] = (float)(value_sum_0 / weight_sum);
            out_data_1[vertex] = (float)(value_sum_1 / weight_sum);
            out_data_2[vertex] = (float)(value_sum_2 / weight_sum);
        }
    }

    return true;
}

coDistributedObject *
coCellToVert::interpolate(bool unstructured, int num_elem, int num_conn, int num_point,
                          const int *elem_list, const int *conn_list, const int *type_list,
                          const int *neighbour_cells, const int *neighbour_idx,
                          const float *xcoord, const float *ycoord, const float *zcoord,
                          int numComp, int &dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                          const char *objName, Algorithm algo_option)
{
    coDistributedObject *data_return = NULL;

    float *out_data_0 = NULL;
    float *out_data_1 = NULL;
    float *out_data_2 = NULL;

    if (numComp == 1)
    {
        coDoFloat *sdata = new coDoFloat(objName, num_point);
        sdata->getAddress(&out_data_0);
        data_return = (coDistributedObject *)sdata;
    }
    else if (numComp == 3)
    {
        coDoVec3 *vdata = new coDoVec3(objName, num_point);
        vdata->getAddresses(&out_data_0, &out_data_1, &out_data_2);
        data_return = (coDistributedObject *)vdata;
    }

    if (!interpolate(unstructured, num_elem, num_conn, num_point,
                     elem_list, conn_list, type_list, neighbour_cells, neighbour_idx, xcoord, ycoord, zcoord,
                     numComp, dataSize, in_data_0, in_data_1, in_data_2, out_data_0, out_data_1, out_data_2, algo_option))
    {
        data_return = NULL;
    }

    return data_return;
}

coDistributedObject *
coCellToVert::interpolate(const coDistributedObject *geo_in,
                          int numComp, int &dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                          const char *objName, Algorithm algo_option)
{
    int num_elem, num_conn, num_point;
    int *elem_list, *conn_list, *type_list = NULL;
    float *xcoord, *ycoord, *zcoord;

    if (!geo_in)
    {
        return NULL;
    }

    int *neighbour_cells = NULL;
    int *neighbour_idx = NULL;

    if (const coDoPolygons *pgrid_in = dynamic_cast<const coDoPolygons *>(geo_in))
    {
        num_elem = pgrid_in->getNumPolygons();
        num_point = pgrid_in->getNumPoints();
        num_conn = pgrid_in->getNumVertices();
        pgrid_in->getAddresses(&xcoord, &ycoord, &zcoord, &conn_list, &elem_list);
    }
    else if (const coDoLines *lgrid_in = dynamic_cast<const coDoLines *>(geo_in))
    {
        num_elem = lgrid_in->getNumLines();
        num_point = lgrid_in->getNumPoints();
        num_conn = lgrid_in->getNumVertices();
        lgrid_in->getAddresses(&xcoord, &ycoord, &zcoord, &conn_list, &elem_list);
    }
    else if (const coDoUnstructuredGrid *ugrid_in = dynamic_cast<const coDoUnstructuredGrid *>(geo_in))
    {
        ugrid_in->getGridSize(&num_elem, &num_conn, &num_point);
        ugrid_in->getAddresses(&elem_list, &conn_list, &xcoord, &ycoord, &zcoord);
        ugrid_in->getTypeList(&type_list);
        if (algo_option == SQR_WEIGHT)
        {
            int vertex;
            ugrid_in->getNeighborList(&vertex, &neighbour_cells, &neighbour_idx);
        }
    }
    else
        return NULL;

    bool unstructured = (dynamic_cast<const coDoUnstructuredGrid *>(geo_in) != NULL);
    return interpolate(unstructured, num_elem, num_conn, num_point,
                       elem_list, conn_list, type_list, neighbour_cells, neighbour_idx, xcoord, ycoord, zcoord,
                       numComp, dataSize, in_data_0, in_data_1, in_data_2, objName, algo_option);
}

coDistributedObject *
coCellToVert::interpolate(const coDistributedObject *geo_in, const coDistributedObject *data_in, const char *objName,
                          Algorithm algo_option)
{
    if (!geo_in || !data_in)
    {
        return NULL;
    }

    float *in_data_0=NULL, *in_data_1=NULL, *in_data_2=NULL;
    int dataSize=0;

    int numComp = 0;
    if (const coDoFloat *s_data_in = dynamic_cast<const coDoFloat *>(data_in))
    {
        s_data_in->getAddress(&in_data_0);
        dataSize = s_data_in->getNumPoints();
        numComp = 1;
    }
    else if (const coDoInt *s_data_in = dynamic_cast<const coDoInt *>(data_in))
    {
    }
    else if (const coDoByte *s_data_in = dynamic_cast<const coDoByte *>(data_in))
    {
    }
    else if (const coDoVec3 *v_data_in = dynamic_cast<const coDoVec3 *>(data_in))
    {
        v_data_in->getAddresses(&in_data_0, &in_data_1, &in_data_2);
        dataSize = v_data_in->getNumPoints();
        numComp = 3;
    }

    return interpolate(geo_in, numComp, dataSize, in_data_0, in_data_1, in_data_2, objName, algo_option);

}
