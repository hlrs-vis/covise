/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISOSURFACEGPMUTIL_H_
#define _ISOSURFACEGPMUTIL_H_

#include <math.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <do/Triangulate.h>

#if defined(__GNUC__) && (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 2)
#pragma GCC diagnostic warning "-Wuninitialized"
#endif

/************************************************************/
/* Structs and methods used in the IsoSurfaceComp module */
/* for supporting Generalized Polyhedral Meshes                    */
/************************************************************/
namespace covise
{

typedef struct
{
    float v[3];
    int dimension;
} S_V_DATA;

typedef struct
{
    float x;
    float y;
    float z;
} EDGE_VECTOR;

typedef struct
{
    int vertex1;
    int vertex2;
    int vertex3;

    EDGE_VECTOR normal;
} TRIANGLE;

typedef struct
{
    bool intersection_at_vertex1;
    bool intersection_at_vertex2;

    int int_flag;
    int vertex1;
    int vertex2;

    float data_vertex1;
    float data_vertex2;

    S_V_DATA data_vertex_int;
    EDGE_VECTOR intersection;
} ISOSURFACE_EDGE_INTERSECTION;

typedef struct
{
    std::vector<int> ring;
    std::vector<int> ring_index;
    std::vector<int> polyhedron_faces;
} CONTOUR;

typedef std::vector<ISOSURFACE_EDGE_INTERSECTION> ISOSURFACE_EDGE_INTERSECTION_VECTOR;

typedef std::vector<TRIANGLE> TESSELATION;

typedef std::vector<int> CONNECTIVITY_VECTOR;
typedef std::vector<int> POLYGON;
typedef std::vector<int> PROCESSED_ELEMENTS;

typedef std::vector<int>::iterator POLYGON_ITERATOR;
typedef std::vector<int>::iterator ITERATOR;

float dot_product(EDGE_VECTOR vector1, EDGE_VECTOR vector2)
{
    float scalar;
    scalar = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;

    return scalar;
}

EDGE_VECTOR cross_product(EDGE_VECTOR vector1, EDGE_VECTOR vector2)
{
    EDGE_VECTOR normal;

    normal.x = (vector1.y * vector2.z) - (vector2.y * vector1.z);
    normal.y = (vector2.x * vector1.z) - (vector1.x * vector2.z);
    normal.z = (vector1.x * vector2.y) - (vector2.x * vector1.y);

    return normal;
}

double vec_length(EDGE_VECTOR &vector)
{
    return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

ISOSURFACE_EDGE_INTERSECTION VertexInterpolate(float x1, float y1, float z1, float x2, float y2, float z2, float isovalue, float data1, float data2, int v1, int v2)
{
    ISOSURFACE_EDGE_INTERSECTION edge;

    /************************************************/
    /* Case 1:  Isosurface passes through Vertex 1  */
    /************************************************/

    if (isovalue == data1)
    {
        edge.int_flag = 1;
        edge.vertex1 = v1;
        edge.vertex2 = v2;
        edge.data_vertex1 = data1;
        edge.data_vertex2 = data2;
        edge.intersection_at_vertex1 = true;
        edge.intersection_at_vertex2 = false;

        if (x1 < x2)
            edge.intersection.x = x1 + 0.01f * fabs(x2 - x1);
        else if (x1 > x2)
            edge.intersection.x = x1 - 0.01f * fabs(-x2 + x1);
        else if (x1 == x2)
            edge.intersection.x = x1;

        if (y1 < y2)
            edge.intersection.y = y1 + 0.01f * fabs(y2 - y1);
        else if (y1 > y2)
            edge.intersection.y = y1 - 0.01f * fabs(-y2 + y1);
        else if (y1 == y2)
            edge.intersection.y = y1;

        if (z1 < z2)
            edge.intersection.z = z1 + 0.01f * fabs(z2 - z1);
        else if (z1 > z2)
            edge.intersection.z = z1 - 0.01f * fabs(-z2 + z1);
        else if (z1 == z2)
            edge.intersection.z = z1;

        return (edge);
    }

    /************************************************/
    /* Case 2:  Isosurface passes through Vertex 2  */
    /************************************************/

    if (isovalue == data2)
    {
        edge.int_flag = 1;
        edge.vertex1 = v1;
        edge.vertex2 = v2;
        edge.data_vertex1 = data1;
        edge.data_vertex2 = data2;
        edge.intersection_at_vertex1 = false;
        edge.intersection_at_vertex2 = true;

        if (x1 < x2)
            edge.intersection.x = x2 - 0.01f * fabs(x2 - x1);
        else if (x1 > x2)
            edge.intersection.x = x2 + 0.01f * fabs(-x2 + x1);
        else if (x1 == x2)
            edge.intersection.x = x2;

        if (y1 < y2)
            edge.intersection.y = y2 - 0.01f * fabs(y2 - y1);
        else if (y1 > y2)
            edge.intersection.y = y2 + 0.01f * fabs(-y2 + y1);
        else if (y1 == y2)
            edge.intersection.y = y2;

        if (z1 < z2)
            edge.intersection.z = z2 - 0.01f * fabs(z2 - z1);
        else if (z1 > z2)
            edge.intersection.z = z2 + 0.01f * fabs(-z2 + z1);
        else if (z1 == z2)
            edge.intersection.z = z2;

        return (edge);
    }

    /****************************************************/
    /* Case 3:  Isosurface passes between both vertices */
    /****************************************************/

    else
    {
        edge.int_flag = 1;
        edge.vertex1 = v1;
        edge.vertex2 = v2;
        edge.data_vertex1 = data1;
        edge.data_vertex2 = data2;
        edge.intersection_at_vertex1 = false;
        edge.intersection_at_vertex2 = false;
        edge.intersection.x = x1 + (x2 - x1) * ((isovalue - data1) / (data2 - data1));
        edge.intersection.y = y1 + (y2 - y1) * ((isovalue - data1) / (data2 - data1));
        edge.intersection.z = z1 + (z2 - z1) * ((isovalue - data1) / (data2 - data1));

        return (edge);
    }
}

bool test_intersection(ISOSURFACE_EDGE_INTERSECTION_VECTOR &intsec_vector, ISOSURFACE_EDGE_INTERSECTION &intsec, float *x_coord_in, float *y_coord_in, float *z_coord_in, bool &improper_topology)
{
    /***********************************************************************************************************************/
    /* Test if intersection has already been calculated                                                                                                                  */
    /*                                                                                                                                                                                           */
    /* According to O'Rourke a valid polyhedral surface must satisfy the following three conditions:                                          */
    /*                                                                                                                                                                                           */
    /* 	a) components intersect "properly"                                                                                                                               */
    /* 	b) local topology is "proper"                                                                                                                                           */
    /* 	c) global topology is "proper"                                                                                                                                         */
    /*                                                                                                                                                                                           */
    /* It has been noted that some datasets contain cells that violate the second and consequently also third conditions,         */
    /* therefore for an adequate analysis it has to be determined if the cell is a "proper" polyhedron or not.  Examples of         */
    /* these cells are transition cells found in multi-resolution grids.  For more information refer to "Computational Geometry   */
    /* in C" by Joseph O'Rourke.                                                                                                                                                  */
    /***********************************************************************************************************************/

    for (std::vector<ISOSURFACE_EDGE_INTERSECTION>::iterator existent_intsec = intsec_vector.begin(); existent_intsec < intsec_vector.end(); ++existent_intsec)
    {
        /***********************************************/
        /* Case 1:  Local topology of the cell is "proper" */
        /***********************************************/

        /* Edge intersections share both vertices */
        if (existent_intsec->vertex1 == intsec.vertex1 && existent_intsec->vertex2 == intsec.vertex2)
        {
            return true;
        }

        /* Edge intersections share both vertices (swapped) */
        if (existent_intsec->vertex1 == intsec.vertex2 && existent_intsec->vertex2 == intsec.vertex1)
        {
            return true;
        }

        /*************************************************/
        /* Case 2:  Local topology of the cell is "improper" */
        /*************************************************/

        /* Check for T-vertices */
        if (existent_intsec->vertex1 == intsec.vertex1 || existent_intsec->vertex2 == intsec.vertex2 || existent_intsec->vertex1 == intsec.vertex2 || existent_intsec->vertex2 == intsec.vertex1)
        {
            int vtx1;
            int vtx2;

            if (intsec.vertex1 == existent_intsec->vertex1 || intsec.vertex2 == existent_intsec->vertex2)
            {
                vtx1 = existent_intsec->vertex1;
                vtx2 = existent_intsec->vertex2;
            }

            if (intsec.vertex1 == existent_intsec->vertex2 || intsec.vertex2 == existent_intsec->vertex1)
            {
                vtx1 = existent_intsec->vertex2;
                vtx2 = existent_intsec->vertex1;
            }

            /* Check direction of edges */
            EDGE_VECTOR d1;
            d1.x = x_coord_in[vtx1] - x_coord_in[vtx2];
            d1.y = y_coord_in[vtx1] - y_coord_in[vtx2];
            d1.z = z_coord_in[vtx1] - z_coord_in[vtx2];

            EDGE_VECTOR d2;
            d2.x = x_coord_in[intsec.vertex1] - x_coord_in[intsec.vertex2];
            d2.y = y_coord_in[intsec.vertex1] - y_coord_in[intsec.vertex2];
            d2.z = z_coord_in[intsec.vertex1] - z_coord_in[intsec.vertex2];

            double length1 = vec_length(d1);
            double length2 = vec_length(d2);

            if ((length1 > 0) && (length2 > 0))
            {
                double cosangle = dot_product(d1, d2) / (length1 * length2);

                /******************************************************************************************************/
                /* Cell is topologically "improper"  --> Two edges have the same direction (and share a common vertex)  */
                /*                                                                                                                                                                */
                /* The chosen tolerance has brought good results however the possibility always remains that a certain  */
                /* dataset contains extremely small triangles (polygons) which are practically degenerate.  Under these   */
                /* circumstances a "proper" cell could be treated as "improper".                                                                */
                /******************************************************************************************************/

                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                {
                    improper_topology = true;
                    return true;
                }
            }
        }
    }

    return false;
}

float map_to_isosurface(float coord_x1, float coord_x2, float coord_y1, float coord_y2, float coord_z1, float coord_z2, float coord_isox, float coord_isoy, float coord_isoz, float data_1, float data_2, bool int_vertex1, bool int_vertex2)
{
    float mapped_value;
    float dist_x1x2;
    float dist_x1xiso;

    if (int_vertex1 == true)
    {
        mapped_value = data_1;
    }

    else if (int_vertex2 == true)
    {
        mapped_value = data_2;
    }

    else
    {
        dist_x1x2 = sqrt(pow(coord_x1 - coord_x2, 2) + pow(coord_y1 - coord_y2, 2) + pow(coord_z1 - coord_z2, 2));

        // Avoid division by zero
        if (dist_x1x2 == 0)
        {
            mapped_value = data_1;
        }

        else
        {
            dist_x1xiso = sqrt(pow(coord_x1 - coord_isox, 2) + pow(coord_y1 - coord_isoy, 2) + pow(coord_z1 - coord_isoz, 2));
            mapped_value = data_1 + ((data_2 - data_1) / dist_x1x2) * dist_x1xiso;
        }
    }

    return mapped_value;
}

ISOSURFACE_EDGE_INTERSECTION_VECTOR calculate_intersections(int num_elem_in, int *elem_in, int num_conn_in, int *conn_in, float *x_coord_in, float *y_coord_in, float *z_coord_in, float *isodata_in, float isovalue, float *sdata_in, float *udata_in, float *vdata_in, float *wdata_in, bool isomap, bool &improper_topology)
{
    int i;
    int j;

    float data_vertex1;
    float data_vertex2;

    ISOSURFACE_EDGE_INTERSECTION intersec;
    ISOSURFACE_EDGE_INTERSECTION_VECTOR intsec_vector;

    /* Avoid Unnecessary Reallocation */
    intsec_vector.reserve(15);

    for (i = 0; i < num_elem_in; i++)
    {
        if (i < num_elem_in - 1)
        {
            for (j = elem_in[i]; j <= elem_in[i + 1] - 1; j++)
            {
                if (j < elem_in[i + 1] - 1)
                {
                    if (((isodata_in[conn_in[j]] <= isovalue) && (isovalue < isodata_in[conn_in[j + 1]])) || (isodata_in[conn_in[j]] > isovalue && isovalue >= isodata_in[conn_in[j + 1]]) || ((isodata_in[conn_in[j]] < isovalue) && (isovalue <= isodata_in[conn_in[j + 1]])) || (isodata_in[conn_in[j]] >= isovalue && isovalue > isodata_in[conn_in[j + 1]]))
                    {
                        data_vertex1 = isodata_in[conn_in[j]];
                        data_vertex2 = isodata_in[conn_in[j + 1]];

                        intersec = VertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j + 1]], isovalue, data_vertex1, data_vertex2, conn_in[j], conn_in[j + 1]);

                        if (intersec.int_flag == 1)
                        {
                            /* Check for repeated intersections  */
                            if (!test_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in, improper_topology))
                            {
                                if (sdata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 1;

                                    /* Map scalar data only if required */
                                    if (isomap == true)
                                    {
                                        intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, sdata_in[conn_in[j]], sdata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    }
                                }

                                else if (udata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 3;
                                    intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, udata_in[conn_in[j]], udata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[1] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, vdata_in[conn_in[j]], vdata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[2] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, wdata_in[conn_in[j]], wdata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                }

                                intsec_vector.push_back(intersec);
                            }
                        }
                    }
                }

                else if (j == elem_in[i + 1] - 1)
                {
                    if ((isodata_in[conn_in[j]] <= isovalue && isovalue < isodata_in[conn_in[elem_in[i]]]) || (isodata_in[conn_in[j]] > isovalue && isovalue >= isodata_in[conn_in[elem_in[i]]]) || (isodata_in[conn_in[j]] < isovalue && isovalue <= isodata_in[conn_in[elem_in[i]]]) || (isodata_in[conn_in[j]] >= isovalue && isovalue > isodata_in[conn_in[elem_in[i]]]))
                    {
                        data_vertex1 = isodata_in[conn_in[j]];
                        data_vertex2 = isodata_in[conn_in[elem_in[i]]];

                        intersec = VertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[elem_in[i]]], isovalue, data_vertex1, data_vertex2, conn_in[j], conn_in[elem_in[i]]);

                        if (intersec.int_flag == 1)
                        {
                            /* Check for repeated intersections  */
                            if (!test_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in, improper_topology))
                            {
                                if (sdata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 1;

                                    /* Map scalar data only if required */
                                    if (isomap == true)
                                    {
                                        intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, sdata_in[conn_in[j]], sdata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    }
                                }

                                else if (udata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 3;
                                    intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, udata_in[conn_in[j]], udata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[1] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, vdata_in[conn_in[j]], vdata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[2] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, wdata_in[conn_in[j]], wdata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                }

                                intsec_vector.push_back(intersec);
                            }
                        }
                    }
                }
            }
        }

        else
        {
            for (j = elem_in[i]; j <= (num_conn_in - 1); j++)
            {
                if (j < (num_conn_in - 1))
                {
                    if ((isodata_in[conn_in[j]] <= isovalue && isovalue < isodata_in[conn_in[j + 1]]) || (isodata_in[conn_in[j]] > isovalue && isovalue >= isodata_in[conn_in[j + 1]]) || (isodata_in[conn_in[j]] < isovalue && isovalue <= isodata_in[conn_in[j + 1]]) || (isodata_in[conn_in[j]] >= isovalue && isovalue > isodata_in[conn_in[j + 1]]))
                    {
                        data_vertex1 = isodata_in[conn_in[j]];
                        data_vertex2 = isodata_in[conn_in[j + 1]];

                        intersec = VertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j + 1]], isovalue, data_vertex1, data_vertex2, conn_in[j], conn_in[j + 1]);

                        if (intersec.int_flag == 1)
                        {
                            /* Check for repeated intersections  */
                            if (!test_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in, improper_topology))
                            {
                                if (sdata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 1;

                                    /* Map scalar data only if required */
                                    if (isomap == true)
                                    {
                                        intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, sdata_in[conn_in[j]], sdata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    }
                                }

                                else if (udata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 3;
                                    intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, udata_in[conn_in[j]], udata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[1] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, vdata_in[conn_in[j]], vdata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[2] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[j + 1]], y_coord_in[conn_in[j]], y_coord_in[conn_in[j + 1]], z_coord_in[conn_in[j]], z_coord_in[conn_in[j + 1]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, wdata_in[conn_in[j]], wdata_in[conn_in[j + 1]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                }

                                intsec_vector.push_back(intersec);
                            }
                        }
                    }
                }

                else if (j == (num_conn_in - 1))
                {
                    if ((isodata_in[conn_in[j]] <= isovalue && isovalue < isodata_in[conn_in[elem_in[i]]]) || (isodata_in[conn_in[j]] > isovalue && isovalue >= isodata_in[conn_in[elem_in[i]]]) || (isodata_in[conn_in[j]] < isovalue && isovalue <= isodata_in[conn_in[elem_in[i]]]) || (isodata_in[conn_in[j]] >= isovalue && isovalue > isodata_in[conn_in[elem_in[i]]]))
                    {
                        data_vertex1 = isodata_in[conn_in[j]];
                        data_vertex2 = isodata_in[conn_in[elem_in[i]]];

                        intersec = VertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[elem_in[i]]], isovalue, data_vertex1, data_vertex2, conn_in[j], conn_in[elem_in[i]]);

                        if (intersec.int_flag == 1)
                        {
                            /* Check for repeated intersections  */
                            if (!test_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in, improper_topology))
                            {
                                if (sdata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 1;

                                    /* Map scalar data only if required */
                                    if (isomap == true)
                                    {
                                        intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, sdata_in[conn_in[j]], sdata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    }
                                }

                                else if (udata_in != NULL)
                                {
                                    intersec.data_vertex_int.dimension = 3;
                                    intersec.data_vertex_int.v[0] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, udata_in[conn_in[j]], udata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[1] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, vdata_in[conn_in[j]], vdata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                    intersec.data_vertex_int.v[2] = map_to_isosurface(x_coord_in[conn_in[j]], x_coord_in[conn_in[elem_in[i]]], y_coord_in[conn_in[j]], y_coord_in[conn_in[elem_in[i]]], z_coord_in[conn_in[j]], z_coord_in[conn_in[elem_in[i]]], intersec.intersection.x, intersec.intersection.y, intersec.intersection.z, wdata_in[conn_in[j]], wdata_in[conn_in[elem_in[i]]], intersec.intersection_at_vertex1, intersec.intersection_at_vertex2);
                                }

                                intsec_vector.push_back(intersec);
                            }
                        }
                    }
                }
            }
        }
    }

    return intsec_vector;
}

int assign_int_index(ISOSURFACE_EDGE_INTERSECTION_VECTOR intsec_vector, int edge_vertex1, int edge_vertex2)
{
    size_t i;
    int index;

    index = 0;

    for (i = 0; i < intsec_vector.size(); i++)
    {
        if ((edge_vertex1 == intsec_vector[i].vertex1) && (edge_vertex2 == intsec_vector[i].vertex2))
        {
            index = (int)i;
            break;
        }

        if ((edge_vertex2 == intsec_vector[i].vertex1) && (edge_vertex1 == intsec_vector[i].vertex2))
        {
            index = (int)i;
            break;
        }
    }

    return index;
}

bool find_intersection(ISOSURFACE_EDGE_INTERSECTION_VECTOR intsec_vector, int &edge_vertex1, int &edge_vertex2, float *x_coord_in, float *y_coord_in, float *z_coord_in, bool improper_topology, int &int_index)
{
    int i;
    bool int_found;

    int_found = false;

    for (i = 0; i < ssize_t(intsec_vector.size()); i++)
    {
        /***********************************************/
        /* Case 1:  Local topology of the cell is "proper" */
        /***********************************************/

        /* Edge intersections share both vertices */
        if ((edge_vertex1 == intsec_vector[i].vertex1) && (edge_vertex2 == intsec_vector[i].vertex2))
        {
            int_found = true;
            int_index = i;
            break;
        }

        /* Edge intersections share both vertices (swapped) */
        else if ((edge_vertex2 == intsec_vector[i].vertex1) && (edge_vertex1 == intsec_vector[i].vertex2))
        {
            int_found = true;
            int_index = i;
            break;
        }

        /*************************************************/
        /* Case 2:  Local topology of the cell is "improper" */
        /*************************************************/

        // Check for T-vertices
        else if (improper_topology)
        {
            if (edge_vertex1 == intsec_vector[i].vertex1 || edge_vertex2 == intsec_vector[i].vertex2 || edge_vertex2 == intsec_vector[i].vertex1 || edge_vertex1 == intsec_vector[i].vertex2)
            {
                int vtx1;
                int vtx2;

                if (edge_vertex1 == intsec_vector[i].vertex1 || edge_vertex2 == intsec_vector[i].vertex2)
                {
                    vtx1 = intsec_vector[i].vertex1;
                    vtx2 = intsec_vector[i].vertex2;
                }

                if (edge_vertex1 == intsec_vector[i].vertex2 || edge_vertex2 == intsec_vector[i].vertex1)
                {
                    vtx1 = intsec_vector[i].vertex2;
                    vtx2 = intsec_vector[i].vertex1;
                }

                // Check direction of edges
                EDGE_VECTOR d1;
                d1.x = x_coord_in[vtx1] - x_coord_in[vtx2];
                d1.y = y_coord_in[vtx1] - y_coord_in[vtx2];
                d1.z = z_coord_in[vtx1] - z_coord_in[vtx2];

                EDGE_VECTOR d2;
                d2.x = x_coord_in[edge_vertex1] - x_coord_in[edge_vertex2];
                d2.y = y_coord_in[edge_vertex1] - y_coord_in[edge_vertex2];
                d2.z = z_coord_in[edge_vertex1] - z_coord_in[edge_vertex2];

                double length1 = vec_length(d1);
                double length2 = vec_length(d2);

                if ((length1 > 0) && (length2 > 0))
                {
                    double cosangle = dot_product(d1, d2) / (length1 * length2);

                    // Cell is topologically "improper"
                    // Two edges have the same direction (and share a common vertex)
                    if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                    {
                        if (edge_vertex1 == intsec_vector[i].vertex1)
                            edge_vertex2 = intsec_vector[i].vertex2;
                        else if (edge_vertex2 == intsec_vector[i].vertex2)
                            edge_vertex1 = intsec_vector[i].vertex1;
                        else if (edge_vertex1 == intsec_vector[i].vertex2)
                            edge_vertex2 = intsec_vector[i].vertex1;
                        else if (edge_vertex2 == intsec_vector[i].vertex1)
                            edge_vertex1 = intsec_vector[i].vertex2;

                        int_found = true;
                        int_index = i;
                        break;
                    }
                }
            }
        }
    }

    return int_found;
}

void find_current_face(CONTOUR &contour, ISOSURFACE_EDGE_INTERSECTION_VECTOR intsec_vector, int &edge_vertex1, int &edge_vertex2, float &data_vertex1, float &data_vertex2, float *isodata_in, int *elem_in, int *conn_in, int *index_list, int *polygon_list, int num_coord_in, int num_conn_in, int num_elem_in, int &ring_counter, int &current_face, float *x_coord_in, float *y_coord_in, float *z_coord_in, bool improper_topology, bool &abort_tracing_isocontour)
{
    bool adjacent_vertices;
    bool T_vertices;

    int i;
    int j;
    int k;
    int face_flag;
    int copy_current_face;
    int neighbour_face1;
    int neighbour_face2;
    int next_vertex;
    int previous_vertex;
    //int vertex_index;
    int new_edge_vertex;

    ITERATOR it;

    face_flag = 0;
    copy_current_face = current_face;

    /******************************************************************************/
    /* Find  the next face of the polyhedron to continue tracing the convex contour */
    /******************************************************************************/

    if ((edge_vertex1 < num_coord_in - 1) && (edge_vertex2 < num_coord_in - 1))
    {
        for (i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
        {
            adjacent_vertices = false;
            neighbour_face1 = polygon_list[i];
            for (j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
            {
                neighbour_face2 = polygon_list[j];
                if (face_flag == 0)
                {
                    /*********************************************************/
                    /* Search among the elements that contain both vertices  */
                    /*********************************************************/

                    if (neighbour_face1 == neighbour_face2)
                    {
                        if (neighbour_face1 < num_elem_in - 1)
                        {
                            for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                            {
                                if (conn_in[k] == edge_vertex1)
                                {
                                    //vertex_index = k;
                                    if (k == elem_in[neighbour_face1])
                                    {
                                        previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k < elem_in[neighbour_face1 + 1] - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k == elem_in[neighbour_face1 + 1] - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[elem_in[neighbour_face1]];
                                    }

                                    if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                    {
                                        adjacent_vertices = true;
                                        break;
                                    }
                                }
                            }
                        }

                        else
                        {
                            for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                            {
                                if (conn_in[k] == edge_vertex1)
                                {
                                    //vertex_index = k;
                                    if (k == elem_in[neighbour_face1])
                                    {
                                        previous_vertex = conn_in[num_conn_in - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k < num_conn_in - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k == num_conn_in - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[elem_in[neighbour_face1]];
                                    }

                                    if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                    {
                                        adjacent_vertices = true;
                                        break;
                                    }
                                }
                            }
                        }

                        /**************************************************/
                        /* Test if the element has already been processed  */
                        /**************************************************/

                        if (adjacent_vertices == true)
                        {
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face1);
                                current_face = neighbour_face1;
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }
            }

            if (face_flag == 1)
            {
                break;
            }
        }

        /**********************************/
        /* No element has been found yet */
        /**********************************/

        if (face_flag == 0)
        {
            /**********************************************/
            /* Case 1:  The cell has an "improper" topology */
            /**********************************************/

            if (improper_topology)
            {
                T_vertices = false;

                /**************************************************/
                /* Search among the elements that share vertex1  */
                /**************************************************/

                if (!T_vertices)
                {
                    for (i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
                    {
                        neighbour_face1 = polygon_list[i];
                        if (neighbour_face1 != current_face)
                        {
                            if (neighbour_face1 < num_elem_in - 1)
                            {
                                for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                                {
                                    if (conn_in[k] == edge_vertex1)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face1])
                                        {
                                            previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < elem_in[neighbour_face1 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == elem_in[neighbour_face1 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face1]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex2 && next_vertex != edge_vertex2)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex1] - x_coord_in[edge_vertex2];
                                            d1.y = y_coord_in[edge_vertex1] - y_coord_in[edge_vertex2];
                                            d1.z = z_coord_in[edge_vertex1] - z_coord_in[edge_vertex2];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex1] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex1] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex1] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex1] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex1] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex1] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            else
                            {
                                for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                                {
                                    if (conn_in[k] == edge_vertex1)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face1])
                                        {
                                            previous_vertex = conn_in[num_conn_in - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face1]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex2 && next_vertex != edge_vertex2)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex1] - x_coord_in[edge_vertex2];
                                            d1.y = y_coord_in[edge_vertex1] - y_coord_in[edge_vertex2];
                                            d1.z = z_coord_in[edge_vertex1] - z_coord_in[edge_vertex2];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex1] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex1] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex1] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex1] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex1] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex1] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /**************************************************/
                        /* Test if the element has already been processed  */
                        /**************************************************/

                        if (T_vertices == true)
                        {
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face1);
                                current_face = neighbour_face1;

                                /* Update edge vertices */
                                edge_vertex2 = new_edge_vertex;
                                data_vertex2 = isodata_in[edge_vertex2];
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }

                /**************************************************/
                /* Search among the elements that share vertex2  */
                /**************************************************/

                if (!T_vertices)
                {
                    for (j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
                    {
                        neighbour_face2 = polygon_list[j];
                        if (neighbour_face2 != current_face)
                        {
                            if (neighbour_face2 < num_elem_in - 1)
                            {
                                for (k = elem_in[neighbour_face2]; k < elem_in[neighbour_face2 + 1]; k++)
                                {
                                    if (conn_in[k] == edge_vertex2)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face2])
                                        {
                                            previous_vertex = conn_in[elem_in[neighbour_face2 + 1] - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < elem_in[neighbour_face2 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == elem_in[neighbour_face2 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face2]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex1 && next_vertex != edge_vertex1)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex2] - x_coord_in[edge_vertex1];
                                            d1.y = y_coord_in[edge_vertex2] - y_coord_in[edge_vertex1];
                                            d1.z = z_coord_in[edge_vertex2] - z_coord_in[edge_vertex1];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex2] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex2] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex2] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex2] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex2] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex2] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            else
                            {
                                for (k = elem_in[neighbour_face2]; k < num_conn_in; k++)
                                {
                                    if (conn_in[k] == edge_vertex2)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face2])
                                        {
                                            previous_vertex = conn_in[num_conn_in - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face2]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex1 && next_vertex != edge_vertex1)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex2] - x_coord_in[edge_vertex1];
                                            d1.y = y_coord_in[edge_vertex2] - y_coord_in[edge_vertex1];
                                            d1.z = z_coord_in[edge_vertex2] - z_coord_in[edge_vertex1];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex2] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex2] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex2] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex2] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex2] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex2] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /**************************************************/
                        /* Test if the element has already been processed  */
                        /**************************************************/

                        if (T_vertices == true)
                        {
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face2);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face2);
                                current_face = neighbour_face2;

                                /* Update edge vertices */
                                edge_vertex1 = new_edge_vertex;
                                data_vertex1 = isodata_in[edge_vertex1];
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }
            }

            /*******************************************************************/
            /* Case 2:  All intersection faces have been processed at least once  */
            /*******************************************************************/

            else
            {
                for (i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
                {
                    adjacent_vertices = false;
                    neighbour_face1 = polygon_list[i];
                    for (j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
                    {
                        neighbour_face2 = polygon_list[j];
                        if (face_flag == 0)
                        {
                            /********************************************************/
                            /* Search among the elements that contain both vertices */
                            /********************************************************/

                            if (neighbour_face1 == neighbour_face2)
                            {
                                if (neighbour_face1 < num_elem_in - 1)
                                {
                                    for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                                    {
                                        if (conn_in[k] == edge_vertex1)
                                        {
                                            //vertex_index = k;
                                            if (k == elem_in[neighbour_face1])
                                            {
                                                previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k < elem_in[neighbour_face1 + 1] - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k == elem_in[neighbour_face1 + 1] - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[elem_in[neighbour_face1]];
                                            }

                                            if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                            {
                                                adjacent_vertices = true;
                                                break;
                                            }
                                        }
                                    }
                                }

                                else
                                {
                                    for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                                    {
                                        if (conn_in[k] == edge_vertex1)
                                        {
                                            //vertex_index = k;
                                            if (k == elem_in[neighbour_face1])
                                            {
                                                previous_vertex = conn_in[num_conn_in - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k < num_conn_in - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k == num_conn_in - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[elem_in[neighbour_face1]];
                                            }

                                            if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                            {
                                                adjacent_vertices = true;
                                                break;
                                            }
                                        }
                                    }
                                }

                                if (adjacent_vertices == true)
                                {
                                    if (neighbour_face1 != copy_current_face)
                                    {
                                        contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                        contour.polyhedron_faces.push_back(neighbour_face1);
                                        current_face = neighbour_face1;
                                        ring_counter++;
                                        face_flag = 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if (face_flag == 1)
                    {
                        break;
                    }
                }
            }

            /* A new element to continue tracing the isocontour could not be found */
            if (current_face == copy_current_face)
            {
                abort_tracing_isocontour = true;
            }
        }
    }

    else if ((edge_vertex1 < num_coord_in - 1) && (edge_vertex2 == num_coord_in - 1))
    {
        for (i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
        {
            adjacent_vertices = false;
            neighbour_face1 = polygon_list[i];
            for (j = index_list[edge_vertex2]; j < num_conn_in; j++)
            {
                neighbour_face2 = polygon_list[j];
                if (face_flag == 0)
                {
                    /********************************************************/
                    /* Search among the elements that contain both vertices */
                    /********************************************************/

                    if (neighbour_face1 == neighbour_face2)
                    {
                        if (neighbour_face1 < num_elem_in - 1)
                        {
                            for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                            {
                                if (conn_in[k] == edge_vertex1)
                                {
                                    //vertex_index = k;
                                    if (k == elem_in[neighbour_face1])
                                    {
                                        previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k < elem_in[neighbour_face1 + 1] - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k == elem_in[neighbour_face1 + 1] - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[elem_in[neighbour_face1]];
                                    }

                                    if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                    {
                                        adjacent_vertices = true;
                                        break;
                                    }
                                }
                            }
                        }

                        else
                        {
                            for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                            {
                                if (conn_in[k] == edge_vertex1)
                                {
                                    //vertex_index = k;
                                    if (k == elem_in[neighbour_face1])
                                    {
                                        previous_vertex = conn_in[num_conn_in - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k < num_conn_in - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k == num_conn_in - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[elem_in[neighbour_face1]];
                                    }

                                    if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                    {
                                        adjacent_vertices = true;
                                        break;
                                    }
                                }
                            }
                        }

                        /*************************************************/
                        /* Test if the element has already been processed */
                        /*************************************************/

                        if (adjacent_vertices == true)
                        {
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face1);
                                current_face = neighbour_face1;
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }
            }

            if (face_flag == 1)
            {
                break;
            }
        }

        /**********************************/
        /* No element has been found yet */
        /**********************************/

        if (face_flag == 0)
        {
            /**********************************************/
            /* Case 1:  The cell has an "improper" topology */
            /**********************************************/

            if (improper_topology)
            {
                T_vertices = false;

                /*************************************************/
                /* Search among the elements that share vertex1 */
                /*************************************************/

                if (!T_vertices)
                {
                    for (i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
                    {
                        neighbour_face1 = polygon_list[i];
                        if (neighbour_face1 != current_face)
                        {
                            if (neighbour_face1 < num_elem_in - 1)
                            {
                                for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                                {
                                    if (conn_in[k] == edge_vertex1)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face1])
                                        {
                                            previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < elem_in[neighbour_face1 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == elem_in[neighbour_face1 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face1]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex2 && next_vertex != edge_vertex2)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex1] - x_coord_in[edge_vertex2];
                                            d1.y = y_coord_in[edge_vertex1] - y_coord_in[edge_vertex2];
                                            d1.z = z_coord_in[edge_vertex1] - z_coord_in[edge_vertex2];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex1] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex1] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex1] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex1] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex1] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex1] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            else
                            {
                                for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                                {
                                    if (conn_in[k] == edge_vertex1)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face1])
                                        {
                                            previous_vertex = conn_in[num_conn_in - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face1]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex2 && next_vertex != edge_vertex2)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex1] - x_coord_in[edge_vertex2];
                                            d1.y = y_coord_in[edge_vertex1] - y_coord_in[edge_vertex2];
                                            d1.z = z_coord_in[edge_vertex1] - z_coord_in[edge_vertex2];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex1] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex1] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex1] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex1] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex1] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex1] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (T_vertices == true)
                        {
                            /* Test if the element has already been processed */
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face1);
                                current_face = neighbour_face1;

                                /* Update edge vertices */
                                edge_vertex2 = new_edge_vertex;
                                data_vertex2 = isodata_in[edge_vertex2];
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }

                /*************************************************/
                /* Search among the elements that share vertex2 */
                /*************************************************/

                if (!T_vertices)
                {
                    for (j = index_list[edge_vertex2]; j < num_conn_in; j++)
                    {
                        neighbour_face2 = polygon_list[j];
                        if (neighbour_face2 != current_face)
                        {
                            if (neighbour_face2 < num_elem_in - 1)
                            {
                                for (k = elem_in[neighbour_face2]; k < elem_in[neighbour_face2 + 1]; k++)
                                {
                                    if (conn_in[k] == edge_vertex2)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face2])
                                        {
                                            previous_vertex = conn_in[elem_in[neighbour_face2 + 1] - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < elem_in[neighbour_face2 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == elem_in[neighbour_face2 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face2]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex1 && next_vertex != edge_vertex1)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex2] - x_coord_in[edge_vertex1];
                                            d1.y = y_coord_in[edge_vertex2] - y_coord_in[edge_vertex1];
                                            d1.z = z_coord_in[edge_vertex2] - z_coord_in[edge_vertex1];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex2] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex2] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex2] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex2] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex2] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex2] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            else
                            {
                                for (k = elem_in[neighbour_face2]; k < num_conn_in; k++)
                                {
                                    if (conn_in[k] == edge_vertex2)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face2])
                                        {
                                            previous_vertex = conn_in[num_conn_in - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face2]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex1 && next_vertex != edge_vertex1)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex2] - x_coord_in[edge_vertex1];
                                            d1.y = y_coord_in[edge_vertex2] - y_coord_in[edge_vertex1];
                                            d1.z = z_coord_in[edge_vertex2] - z_coord_in[edge_vertex1];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex2] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex2] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex2] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex2] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex2] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex2] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /*************************************************/
                        /* Test if the element has already been processed */
                        /*************************************************/

                        if (T_vertices == true)
                        {
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face2);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face2);
                                current_face = neighbour_face2;

                                /* Update edge vertices */
                                edge_vertex1 = new_edge_vertex;
                                data_vertex1 = isodata_in[edge_vertex1];
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }
            }

            /******************************************************************/
            /* Case 2:  All intersection faces have been processed at least once */
            /******************************************************************/

            else
            {
                for (i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
                {
                    adjacent_vertices = false;
                    neighbour_face1 = polygon_list[i];
                    for (j = index_list[edge_vertex2]; j < num_conn_in; j++)
                    {
                        neighbour_face2 = polygon_list[j];
                        if (face_flag == 0)
                        {
                            /* Search among the elements that contain both vertices */
                            if (neighbour_face1 == neighbour_face2)
                            {
                                if (neighbour_face1 < num_elem_in - 1)
                                {
                                    for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                                    {
                                        if (conn_in[k] == edge_vertex1)
                                        {
                                            //vertex_index = k;
                                            if (k == elem_in[neighbour_face1])
                                            {
                                                previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k < elem_in[neighbour_face1 + 1] - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k == elem_in[neighbour_face1 + 1] - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[elem_in[neighbour_face1]];
                                            }

                                            if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                            {
                                                adjacent_vertices = true;
                                                break;
                                            }
                                        }
                                    }
                                }

                                else
                                {
                                    for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                                    {
                                        if (conn_in[k] == edge_vertex1)
                                        {
                                            //vertex_index = k;
                                            if (k == elem_in[neighbour_face1])
                                            {
                                                previous_vertex = conn_in[num_conn_in - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k < num_conn_in - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k == num_conn_in - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[elem_in[neighbour_face1]];
                                            }

                                            if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                            {
                                                adjacent_vertices = true;
                                                break;
                                            }
                                        }
                                    }
                                }

                                if (adjacent_vertices == true)
                                {
                                    if (neighbour_face1 != copy_current_face)
                                    {
                                        contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                        contour.polyhedron_faces.push_back(neighbour_face1);
                                        current_face = neighbour_face1;
                                        ring_counter++;
                                        face_flag = 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if (face_flag == 1)
                    {
                        break;
                    }
                }
            }

            /*A new element to continue tracing the isocontour could not be found */
            if (current_face == copy_current_face)
            {
                abort_tracing_isocontour = true;
            }
        }
    }

    else if ((edge_vertex1 == num_coord_in - 1) && (edge_vertex2 < num_coord_in - 1))
    {
        for (i = index_list[edge_vertex1]; i < num_conn_in; i++)
        {
            adjacent_vertices = false;
            neighbour_face1 = polygon_list[i];
            for (j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
            {
                neighbour_face2 = polygon_list[j];
                if (face_flag == 0)
                {
                    /* Search among the elements that contain both vertices */
                    if (neighbour_face1 == neighbour_face2)
                    {
                        if (neighbour_face1 < num_elem_in - 1)
                        {
                            for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                            {
                                if (conn_in[k] == edge_vertex1)
                                {
                                    //vertex_index = k;
                                    if (k == elem_in[neighbour_face1])
                                    {
                                        previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k < elem_in[neighbour_face1 + 1] - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k == elem_in[neighbour_face1 + 1] - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[elem_in[neighbour_face1]];
                                    }

                                    if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                    {
                                        adjacent_vertices = true;
                                        break;
                                    }
                                }
                            }
                        }

                        else
                        {
                            for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                            {
                                if (conn_in[k] == edge_vertex1)
                                {
                                    //vertex_index = k;
                                    if (k == elem_in[neighbour_face1])
                                    {
                                        previous_vertex = conn_in[num_conn_in - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k < num_conn_in - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[k + 1];
                                    }

                                    else if (k == num_conn_in - 1)
                                    {
                                        previous_vertex = conn_in[k - 1];
                                        next_vertex = conn_in[elem_in[neighbour_face1]];
                                    }

                                    if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                    {
                                        adjacent_vertices = true;
                                        break;
                                    }
                                }
                            }
                        }

                        if (adjacent_vertices == true)
                        {
                            /* Test if the element has already been processed */
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face1);
                                current_face = neighbour_face1;
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }
            }

            if (face_flag == 1)
            {
                break;
            }
        }

        /* All intersection faces have been processed at least once or the cell has an "improper" topology */
        if (face_flag == 0)
        {
            if (improper_topology)
            {
                T_vertices = false;
                if (!T_vertices)
                {
                    /* Search among elements that share vertex1 */
                    for (i = index_list[edge_vertex1]; i < num_conn_in; i++)
                    {
                        neighbour_face1 = polygon_list[i];
                        if (neighbour_face1 != current_face)
                        {
                            if (neighbour_face1 < num_elem_in - 1)
                            {
                                for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                                {
                                    if (conn_in[k] == edge_vertex1)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face1])
                                        {
                                            previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < elem_in[neighbour_face1 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == elem_in[neighbour_face1 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face1]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex2 && next_vertex != edge_vertex2)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex1] - x_coord_in[edge_vertex2];
                                            d1.y = y_coord_in[edge_vertex1] - y_coord_in[edge_vertex2];
                                            d1.z = z_coord_in[edge_vertex1] - z_coord_in[edge_vertex2];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex1] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex1] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex1] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex1] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex1] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex1] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            else
                            {
                                for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                                {
                                    if (conn_in[k] == edge_vertex1)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face1])
                                        {
                                            previous_vertex = conn_in[num_conn_in - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face1]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex2 && next_vertex != edge_vertex2)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex1] - x_coord_in[edge_vertex2];
                                            d1.y = y_coord_in[edge_vertex1] - y_coord_in[edge_vertex2];
                                            d1.z = z_coord_in[edge_vertex1] - z_coord_in[edge_vertex2];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex1] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex1] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex1] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex1] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex1] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex1] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (T_vertices == true)
                        {
                            /* Test if the element has already been processed */
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face1);
                                current_face = neighbour_face1;

                                /* Update edge vertices */
                                edge_vertex2 = new_edge_vertex;
                                data_vertex2 = isodata_in[edge_vertex2];
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }

                if (!T_vertices)
                {
                    /* Search among elements that share vertex2 */
                    for (j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
                    {
                        neighbour_face2 = polygon_list[j];
                        if (neighbour_face2 != current_face)
                        {
                            if (neighbour_face2 < num_elem_in - 1)
                            {
                                for (k = elem_in[neighbour_face2]; k < elem_in[neighbour_face2 + 1]; k++)
                                {
                                    if (conn_in[k] == edge_vertex2)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face2])
                                        {
                                            previous_vertex = conn_in[elem_in[neighbour_face2 + 1] - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < elem_in[neighbour_face2 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == elem_in[neighbour_face2 + 1] - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face2]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex1 && next_vertex != edge_vertex1)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex2] - x_coord_in[edge_vertex1];
                                            d1.y = y_coord_in[edge_vertex2] - y_coord_in[edge_vertex1];
                                            d1.z = z_coord_in[edge_vertex2] - z_coord_in[edge_vertex1];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex2] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex2] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex2] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex2] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex2] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex2] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            else
                            {
                                for (k = elem_in[neighbour_face2]; k < num_conn_in; k++)
                                {
                                    if (conn_in[k] == edge_vertex2)
                                    {
                                        //vertex_index = k;
                                        if (k == elem_in[neighbour_face2])
                                        {
                                            previous_vertex = conn_in[num_conn_in - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k < num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[k + 1];
                                        }

                                        else if (k == num_conn_in - 1)
                                        {
                                            previous_vertex = conn_in[k - 1];
                                            next_vertex = conn_in[elem_in[neighbour_face2]];
                                        }

                                        /* Check for edges that have the same direction and share a common vertex */
                                        if (previous_vertex != edge_vertex1 && next_vertex != edge_vertex1)
                                        {
                                            // Check direction of edges
                                            EDGE_VECTOR d1;
                                            d1.x = x_coord_in[edge_vertex2] - x_coord_in[edge_vertex1];
                                            d1.y = y_coord_in[edge_vertex2] - y_coord_in[edge_vertex1];
                                            d1.z = z_coord_in[edge_vertex2] - z_coord_in[edge_vertex1];

                                            EDGE_VECTOR d2;
                                            d2.x = x_coord_in[edge_vertex2] - x_coord_in[previous_vertex];
                                            d2.y = y_coord_in[edge_vertex2] - y_coord_in[previous_vertex];
                                            d2.z = z_coord_in[edge_vertex2] - z_coord_in[previous_vertex];

                                            EDGE_VECTOR d3;
                                            d3.x = x_coord_in[edge_vertex2] - x_coord_in[next_vertex];
                                            d3.y = y_coord_in[edge_vertex2] - y_coord_in[next_vertex];
                                            d3.z = z_coord_in[edge_vertex2] - z_coord_in[next_vertex];

                                            double length1 = vec_length(d1);
                                            double length2 = vec_length(d2);
                                            double length3 = vec_length(d3);

                                            if ((length1 > 0) && (length2 > 0))
                                            {
                                                double cosangle = dot_product(d1, d2) / (length1 * length2);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = previous_vertex;
                                                    break;
                                                }
                                            }

                                            if ((length1 > 0) && (length3 > 0))
                                            {
                                                double cosangle = dot_product(d1, d3) / (length1 * length3);

                                                // Cell is topologically "improper"
                                                // Two edges have the same direction (and share a common vertex)
                                                if (fabs(cosangle - 1.0) < 0.00001 && cosangle > 0)
                                                {
                                                    T_vertices = true;
                                                    new_edge_vertex = next_vertex;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (T_vertices == true)
                        {
                            /* Test if the element has already been processed */
                            it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face2);

                            /* The current face that contains the vertices has not been processed */
                            if (it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
                            {
                                contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                contour.polyhedron_faces.push_back(neighbour_face2);
                                current_face = neighbour_face2;

                                /* Update edge vertices */
                                edge_vertex1 = new_edge_vertex;
                                data_vertex1 = isodata_in[edge_vertex1];
                                ring_counter++;
                                face_flag = 1;
                                break;
                            }
                        }
                    }
                }
            }

            else
            {
                for (i = index_list[edge_vertex1]; i < num_conn_in; i++)
                {
                    adjacent_vertices = false;
                    neighbour_face1 = polygon_list[i];
                    for (j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
                    {
                        neighbour_face2 = polygon_list[j];
                        if (face_flag == 0)
                        {
                            /* Search among the elements that contain both vertices */
                            if (neighbour_face1 == neighbour_face2)
                            {
                                if (neighbour_face1 < num_elem_in - 1)
                                {
                                    for (k = elem_in[neighbour_face1]; k < elem_in[neighbour_face1 + 1]; k++)
                                    {
                                        if (conn_in[k] == edge_vertex1)
                                        {
                                            //vertex_index = k;
                                            if (k == elem_in[neighbour_face1])
                                            {
                                                previous_vertex = conn_in[elem_in[neighbour_face1 + 1] - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k < elem_in[neighbour_face1 + 1] - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k == elem_in[neighbour_face1 + 1] - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[elem_in[neighbour_face1]];
                                            }

                                            if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                            {
                                                adjacent_vertices = true;
                                                break;
                                            }
                                        }
                                    }
                                }

                                else
                                {
                                    for (k = elem_in[neighbour_face1]; k < num_conn_in; k++)
                                    {
                                        if (conn_in[k] == edge_vertex1)
                                        {
                                            //vertex_index = k;
                                            if (k == elem_in[neighbour_face1])
                                            {
                                                previous_vertex = conn_in[num_conn_in - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k < num_conn_in - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[k + 1];
                                            }

                                            else if (k == num_conn_in - 1)
                                            {
                                                previous_vertex = conn_in[k - 1];
                                                next_vertex = conn_in[elem_in[neighbour_face1]];
                                            }

                                            if (previous_vertex == edge_vertex2 || next_vertex == edge_vertex2)
                                            {
                                                adjacent_vertices = true;
                                                break;
                                            }
                                        }
                                    }
                                }

                                if (adjacent_vertices == true)
                                {
                                    if (neighbour_face1 != copy_current_face)
                                    {
                                        contour.ring.push_back(assign_int_index(intsec_vector, edge_vertex1, edge_vertex2));
                                        contour.polyhedron_faces.push_back(neighbour_face1);
                                        current_face = neighbour_face1;
                                        ring_counter++;
                                        face_flag = 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if (face_flag == 1)
                    {
                        break;
                    }
                }
            }

            /* A new element to continue tracing the isocontour could not be found */
            if (current_face == copy_current_face)
            {
                abort_tracing_isocontour = true;
            }
        }
    }
}

void generate_isocontour(ISOSURFACE_EDGE_INTERSECTION_VECTOR intsec_vector, float data_vertex1, float data_vertex2, int edge_vertex1, int edge_vertex2, int &new_edge_vertex1, int &new_edge_vertex2, int *elem_in, int *conn_in, float isovalue, int num_elem_in, int num_conn_in, int current_face, float *x_coord_in, float *y_coord_in, float *z_coord_in, bool improper_topology, bool &abort_tracing_isocontour, CONTOUR contour, int num_of_rings, int ring_end)
{
    //bool new_int_found;

    int i;
    int int_index;
    int index_v1;
    int index_v2;
    int index_flag_v1;
    int index_flag_v2;
    int counter1;
    int counter2;

    ITERATOR it;
    ITERATOR it2;

    /************************************/
    /* Test configuration of the vertices  */
    /************************************/

    //new_int_found = false;
    index_flag_v1 = 0;
    index_flag_v2 = 0;
    index_v1 = elem_in[current_face];
    index_v2 = elem_in[current_face];

    /***********************************************************************/
    /* Locate index values of the edge vertices in the array of connectivities */
    /***********************************************************************/

    if (current_face < num_elem_in - 1)
    {
        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
        {
            if (index_flag_v1 == 0)
            {
                if (edge_vertex1 != conn_in[index_v1])
                {
                    index_v1++;
                }

                else
                {
                    index_flag_v1 = 1;
                }
            }

            if (index_flag_v2 == 0)
            {
                if (edge_vertex2 != conn_in[index_v2])
                {
                    index_v2++;
                }

                else
                {
                    index_flag_v2 = 1;
                }
            }
        }

        /* This should not happen */
        if (index_v1 == elem_in[current_face + 1] || index_v2 == elem_in[current_face + 1])
        {
            abort_tracing_isocontour = true;
        }
    }

    else
    {
        for (i = elem_in[current_face]; i < num_conn_in; i++)
        {
            if (index_flag_v1 == 0)
            {
                if (edge_vertex1 != conn_in[index_v1])
                {
                    index_v1++;
                }

                else
                {
                    index_flag_v1 = 1;
                }
            }

            if (index_flag_v2 == 0)
            {
                if (edge_vertex2 != conn_in[index_v2])
                {
                    index_v2++;
                }

                else
                {
                    index_flag_v2 = 1;
                }
            }
        }

        /* This should not happen */
        if (index_v1 == num_conn_in || index_v2 == num_conn_in)
        {
            abort_tracing_isocontour = true;
        }
    }

    counter1 = index_v1;
    counter2 = index_v2;

    if (!abort_tracing_isocontour)
    {
        /*******************************/
        /* Determine Tracing Direction */
        /*******************************/

        /*********************************************************/
        /* Configuration 1:  Data vertex 1 (+) --- Data vertex 2 (-) */
        /*********************************************************/

        if (((data_vertex1 > isovalue) && (isovalue >= data_vertex2)) || ((data_vertex1 >= isovalue) && (isovalue > data_vertex2)))
        {
            if (current_face < num_elem_in - 1)
            {
                /***************************/
                /* Case 1:  Clockwise (CW) */
                /***************************/

                if (index_v1 > index_v2)
                {
                    if ((index_v1 != elem_in[current_face + 1] - 1) || (index_v2 != elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == elem_in[current_face + 1])
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == elem_in[current_face + 1])
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == elem_in[current_face + 1] - 1) && (index_v2 == elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = elem_in[current_face + 1] - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = elem_in[current_face + 1] - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                /************************************/
                /* Case 2: Counterclockwise (CCW)  */
                /************************************/

                else if (index_v1 < index_v2)
                {
                    if ((index_v1 != elem_in[current_face]) || (index_v2 != elem_in[current_face + 1] - 1))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = elem_in[current_face + 1] - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = elem_in[current_face + 1] - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == elem_in[current_face]) && (index_v2 == elem_in[current_face + 1] - 1))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == elem_in[current_face + 1])
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == elem_in[current_face + 1])
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            else
            {
                /***************************/
                /* Case 1:  Clockwise (CW) */
                /***************************/

                if (index_v1 > index_v2)
                {
                    if ((index_v1 != num_conn_in - 1) || (index_v2 != elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == num_conn_in)
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == num_conn_in)
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == num_conn_in - 1) && (index_v2 == elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = num_conn_in - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = num_conn_in - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                /************************************/
                /* Case 2: Counterclockwise (CCW)  */
                /************************************/

                else if (index_v1 < index_v2)
                {
                    if ((index_v1 != elem_in[current_face]) || (index_v2 != num_conn_in - 1))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = num_conn_in - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = num_conn_in - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == elem_in[current_face]) && (index_v2 == num_conn_in - 1))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == num_conn_in)
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == num_conn_in)
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ((edge_vertex1 == new_edge_vertex1 && edge_vertex2 == new_edge_vertex2) || (edge_vertex1 == new_edge_vertex2 && edge_vertex2 == new_edge_vertex1))
            {
                abort_tracing_isocontour = true;
            }
        }

        /*********************************************************/
        /* Configuration 2:  Data vertex 1 (-) --- Data vertex 2 (+) */
        /*********************************************************/

        else if (((data_vertex2 > isovalue) && (isovalue >= data_vertex1)) || ((data_vertex2 >= isovalue) && (isovalue > data_vertex1)))
        {
            if (current_face < num_elem_in - 1)
            {
                /***************************/
                /* Case 1:  Clockwise (CW) */
                /***************************/

                if (index_v1 < index_v2)
                {
                    if ((index_v1 != elem_in[current_face]) || (index_v2 != elem_in[current_face + 1] - 1))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == elem_in[current_face + 1])
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == elem_in[current_face + 1])
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == elem_in[current_face]) && (index_v2 == elem_in[current_face + 1] - 1))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = elem_in[current_face + 1] - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = elem_in[current_face + 1] - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                /************************************/
                /* Case 2: Counterclockwise (CCW)  */
                /************************************/

                else if (index_v1 > index_v2)
                {
                    if ((index_v1 != elem_in[current_face + 1] - 1) || (index_v2 != elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = elem_in[current_face + 1] - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = elem_in[current_face + 1] - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == elem_in[current_face + 1] - 1) && (index_v2 == elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == elem_in[current_face + 1])
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == elem_in[current_face + 1])
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            else
            {
                /***************************/
                /* Case 1:  Clockwise (CW) */
                /***************************/

                if (index_v1 < index_v2)
                {
                    if ((index_v1 != elem_in[current_face]) || (index_v2 != num_conn_in - 1))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == num_conn_in)
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == num_conn_in)
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == elem_in[current_face]) && (index_v2 == num_conn_in - 1))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = num_conn_in - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = num_conn_in - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                /************************************/
                /* Case 2: Counterclockwise (CCW)  */
                /************************************/

                else if (index_v1 > index_v2)
                {
                    if ((index_v1 != num_conn_in - 1) || (index_v2 != elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1--;
                            counter2--;

                            if (counter1 == elem_in[current_face] - 1)
                            {
                                counter1 = num_conn_in - 1;
                            }

                            if (counter2 == elem_in[current_face] - 1)
                            {
                                counter2 = num_conn_in - 1;
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    else if ((index_v1 == num_conn_in - 1) && (index_v2 == elem_in[current_face]))
                    {
                        for (i = elem_in[current_face]; i < num_conn_in; i++)
                        {
                            counter1++;
                            counter2++;

                            if (counter1 == num_conn_in)
                            {
                                counter1 = elem_in[current_face];
                            }

                            if (counter2 == num_conn_in)
                            {
                                counter2 = elem_in[current_face];
                            }

                            /* Test if the a new intersection has been encountered */
                            new_edge_vertex1 = conn_in[counter1];
                            new_edge_vertex2 = conn_in[counter2];

                            if (find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2, x_coord_in, y_coord_in, z_coord_in, improper_topology, int_index))
                            {
                                if (num_of_rings == 0)
                                {
                                    it = find(contour.ring.begin(), contour.ring.end(), int_index);
                                    if (it == contour.ring.end() || it == contour.ring.begin())
                                    {
                                        break;
                                    }
                                }

                                else
                                {
                                    it = find(contour.ring.begin(), contour.ring.begin() + ring_end, int_index);
                                    it2 = find(contour.ring.begin() + ring_end, contour.ring.end(), int_index);
                                    if (it == contour.ring.begin() + ring_end && (it2 == contour.ring.end() || it2 == contour.ring.begin() + ring_end))
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ((edge_vertex1 == new_edge_vertex1 && edge_vertex2 == new_edge_vertex2) || (edge_vertex1 == new_edge_vertex2 && edge_vertex2 == new_edge_vertex1))
            {
                abort_tracing_isocontour = true;
            }
        }

        /***********************************************************************************************************************/
        /* Configuration 3:  Degenerate case --> Data vertex 1 (+) --- Data vertex 2 (+) or Data vertex 1 (-) --- Data vertex 2 (-)   */
        /*                                                                                                                                                                                           */
        /* This condition occurs only in topologically "improper" polyhedral cells.  In this case it is not possible to continue              */
        /* tracing the convex contour within the cell.  This situation appears also in transition cells of multi-resolution grids where  */
        /* the isopatch degenerates into a plane.                                                                                                                               */
        /***********************************************************************************************************************/

        else if (((data_vertex1 < isovalue) && (isovalue > data_vertex2)) || ((data_vertex1 > isovalue) && (isovalue < data_vertex2)))
        {
            abort_tracing_isocontour = true;
        }
    }
}

void generate_tesselation(TESSELATION &triangulation, CONTOUR contour, ISOSURFACE_EDGE_INTERSECTION_VECTOR intsec_vector)
{
    bool end_of_triangulation;

    int i;
    int j;
    int polygon_index;
    int temp_polygon_index;
    int predecessor;
    int successor;
    int triangulation_counter;
    int normal_vector_counter;

    TRIANGLE tesselation_triangle;
    TRIANGLE temp_triangle;

    EDGE_VECTOR vector1;
    EDGE_VECTOR vector2;
    EDGE_VECTOR normal;

    POLYGON polygon;

    POLYGON_ITERATOR start;
    POLYGON_ITERATOR end;
    POLYGON_ITERATOR new_end;

    /* Avoid Unnecessary Reallocations */
    polygon.reserve(15);

    /*******************************************/
    /* Triangulate the convex contour(s) found */
    /*******************************************/

    for (i = 0; i < ssize_t(contour.ring_index.size()); i++)
    {
        /* Make sure polygon and vertex containers are empty */
        polygon.clear();

        /* Analyzed contour is not the last one */
        if (i < ssize_t(contour.ring_index.size()) - 1)
        {
            /**************************************************/
            /* Case A - --> The analyzed contour is a triangle   */
            /**************************************************/

            if ((contour.ring_index[i + 1] - contour.ring_index[i]) == 3)
            {
                /* Generate the triangle */
                tesselation_triangle.vertex1 = contour.ring[contour.ring_index[i]];
                tesselation_triangle.vertex2 = contour.ring[contour.ring_index[i] + 1];
                tesselation_triangle.vertex3 = contour.ring[contour.ring_index[i] + 2];

                /* Define two edge vectors of the triangle */
                vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                /* Calculate normal to the triangle */
                normal = cross_product(vector1, vector2);
                tesselation_triangle.normal.x = normal.x;
                tesselation_triangle.normal.y = normal.y;
                tesselation_triangle.normal.z = normal.z;

                /* Store the triangle in the tesselation vector */
                triangulation.push_back(tesselation_triangle);
            }

            /***********************************************************/
            /* Case B ---> The analyzed contour is an arbitrary polygon  */
            /***********************************************************/

            else if ((contour.ring_index[i + 1] - contour.ring_index[i]) > 3)
            {
                for (j = contour.ring_index[i]; j < contour.ring_index[i + 1]; j++)
                {
                    /* Define a polygon data structure with the current ring */
                    polygon.push_back(contour.ring[j]);
                }

                /************************/
                /* Initiate Graham Scan  */
                /************************/

                polygon_index = 1;
                end_of_triangulation = false;
                triangulation_counter = 0;

                do
                {
                    if (polygon.size() > 3)
                    {
                        /***********************************************/
                        /* Update node predecessors and successors  */
                        /***********************************************/

                        if (polygon_index == 1)
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        else if (polygon_index == 0)
                        {
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else if (polygon_index == ssize_t(polygon.size() - 1))
                        {
                            predecessor = polygon_index - 1;
                            successor = 0;
                        }

                        else if (polygon_index == ssize_t(polygon.size()))
                        {
                            polygon_index = 0;
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        /* Define a triangle */
                        tesselation_triangle.vertex1 = polygon[predecessor];
                        tesselation_triangle.vertex2 = polygon[polygon_index];
                        tesselation_triangle.vertex3 = polygon[successor];

                        /* Define two edge vectors of the triangle */
                        vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        /* Store initial triangle */
                        temp_polygon_index = polygon_index;
                        temp_triangle.vertex1 = tesselation_triangle.vertex1;
                        temp_triangle.vertex2 = tesselation_triangle.vertex2;
                        temp_triangle.vertex3 = tesselation_triangle.vertex3;

                        /* Calculate normal to the triangle */
                        normal = cross_product(vector1, vector2);
                        tesselation_triangle.normal.x = normal.x;
                        tesselation_triangle.normal.y = normal.y;
                        tesselation_triangle.normal.z = normal.z;

                        temp_triangle.normal.x = normal.x;
                        temp_triangle.normal.y = normal.y;
                        temp_triangle.normal.z = normal.z;

                        normal_vector_counter = 0;

                        /*******************************************************************************************************/
                        /* Special Case ---> Normal is equal to the null vector                                                                                  */
                        /*                                                                                                                                                                 */
                        /* This result can only be obtained when two edges of the triangle are collinear (degenerate triangle) .       */
                        /* This implies that the cell is too small or that at least two adjacent intersections are almost coincident or */
                        /* actually overlap.                                                                                                                                       */
                        /*******************************************************************************************************/

                        if (normal.x == 0 && normal.y == 0 && normal.z == 0)
                        {
                            do
                            {
                                normal_vector_counter++;

                                /* Update Scan */
                                if (polygon_index + 1 <= ssize_t(polygon.size()))
                                {
                                    polygon_index++;
                                }

                                /***********************************************/
                                /* Update node predecessors and successors  */
                                /***********************************************/

                                if (polygon_index == 0)
                                {
                                    predecessor = (int)polygon.size() - 1;
                                    successor = polygon_index + 1;
                                }

                                else if (polygon_index == ssize_t(polygon.size() - 1))
                                {
                                    predecessor = polygon_index - 1;
                                    successor = 0;
                                }

                                else if (polygon_index == ssize_t(polygon.size()))
                                {
                                    polygon_index = 0;
                                    predecessor = (int)polygon.size() - 1;
                                    successor = polygon_index + 1;
                                }

                                else if (polygon_index != 0 && polygon_index != ssize_t(polygon.size()) - 1 && polygon_index != ssize_t(polygon.size()))
                                {
                                    predecessor = polygon_index - 1;
                                    successor = polygon_index + 1;
                                }

                                /* Define a new triangle */
                                tesselation_triangle.vertex1 = polygon[predecessor];
                                tesselation_triangle.vertex2 = polygon[polygon_index];
                                tesselation_triangle.vertex3 = polygon[successor];

                                /* Define two edge vectors of the triangle */
                                vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                                vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                                vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                                vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                                vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                                vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                                /* Recalculate normal to the triangle */
                                normal = cross_product(vector1, vector2);
                                tesselation_triangle.normal.x = normal.x;
                                tesselation_triangle.normal.y = normal.y;
                                tesselation_triangle.normal.z = normal.z;
                            } while ((normal.x == 0 && normal.y == 0 && normal.z == 0) && normal_vector_counter < ssize_t(polygon.size()));
                        }

                        /* If normal still equals the null vector implies that the cell is too small */
                        /* Proceed to store the last triangle in the tesselation vector */
                        triangulation_counter = (int)polygon.size() + 1;

                        /* Avoid infinite loops with the triangulation counter */
                        if (triangulation_counter > (int)polygon.size())
                        {
                            /* Store the triangle in the triangulation vector */
                            triangulation.push_back(temp_triangle);

                            /* Cut ear from the polygon vector */
                            start = polygon.begin();
                            end = polygon.end();
                            new_end = remove(start, end, polygon[temp_polygon_index]);
                            polygon.erase(new_end, end);

                            /* Update scan */
                            polygon_index = 1;
                        }
                    }

                    /* All ears have been cut; only one triangle is left */
                    if (polygon.size() == 3)
                    {
                        /* Update node predecessors and successors */
                        if (polygon_index == 1)
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        else if (polygon_index == 0)
                        {
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else if (polygon_index == (int)polygon.size() - 1)
                        {
                            predecessor = polygon_index - 1;
                            successor = 0;
                        }

                        else if (polygon_index == (int)polygon.size())
                        {
                            polygon_index = 0;
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        /* Store the last triangle in the tesselation vector */
                        tesselation_triangle.vertex1 = polygon[predecessor];
                        tesselation_triangle.vertex2 = polygon[polygon_index];
                        tesselation_triangle.vertex3 = polygon[successor];

                        /* Define two edge vectors of the triangle */
                        vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        /* Calculate normal to the triangle */
                        normal = cross_product(vector1, vector2);
                        tesselation_triangle.normal.x = normal.x;
                        tesselation_triangle.normal.y = normal.y;
                        tesselation_triangle.normal.z = normal.z;

                        /* Store the triangle in the tesselation vector */
                        triangulation.push_back(tesselation_triangle);
                        end_of_triangulation = true;
                    }
                } while (end_of_triangulation != true);
            }
        }

        /* Analyzed contour is the last one */
        else
        {
            /**************************************************/
            /* Case A - --> The analyzed contour is a triangle   */
            /**************************************************/

            if ((contour.ring.size() - contour.ring_index[i]) == 3)
            {
                /* Generate the triangle */
                tesselation_triangle.vertex1 = contour.ring[contour.ring_index[i]];
                tesselation_triangle.vertex2 = contour.ring[contour.ring_index[i] + 1];
                tesselation_triangle.vertex3 = contour.ring[contour.ring_index[i] + 2];

                /* Define two edge vectors of the triangle */
                vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                /* Calculate normal to the triangle */
                normal = cross_product(vector1, vector2);
                tesselation_triangle.normal.x = normal.x;
                tesselation_triangle.normal.y = normal.y;
                tesselation_triangle.normal.z = normal.z;

                /* Store the triangle in the tesselation vector */
                triangulation.push_back(tesselation_triangle);
            }

            /***********************************************************/
            /* Case B ---> The analyzed contour is an arbitrary polygon  */
            /***********************************************************/

            else if ((contour.ring.size() - contour.ring_index[i]) > 3)
            {
                for (j = contour.ring_index[i]; j < ssize_t(contour.ring.size()); j++)
                {
                    /* Define a polygon data structure with the current convex contour */
                    polygon.push_back(contour.ring[j]);
                }

                /************************/
                /* Initiate Graham Scan  */
                /************************/

                polygon_index = 1;
                end_of_triangulation = false;
                triangulation_counter = 0;

                do
                {
                    if (polygon.size() > 3)
                    {
                        /***********************************************/
                        /* Update node predecessors and successors  */
                        /***********************************************/

                        if (polygon_index == 1)
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        else if (polygon_index == 0)
                        {
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else if (polygon_index == (int)polygon.size() - 1)
                        {
                            predecessor = polygon_index - 1;
                            successor = 0;
                        }

                        else if (polygon_index == (int)polygon.size())
                        {
                            polygon_index = 0;
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        /* Define a triangle */
                        tesselation_triangle.vertex1 = polygon[predecessor];
                        tesselation_triangle.vertex2 = polygon[polygon_index];
                        tesselation_triangle.vertex3 = polygon[successor];

                        /* Define two edge vectors of the triangle */
                        vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        /* Store initial triangle */
                        temp_polygon_index = polygon_index;
                        temp_triangle.vertex1 = tesselation_triangle.vertex1;
                        temp_triangle.vertex2 = tesselation_triangle.vertex2;
                        temp_triangle.vertex3 = tesselation_triangle.vertex3;

                        /* Calculate normal to the triangle */
                        normal = cross_product(vector1, vector2);
                        tesselation_triangle.normal.x = normal.x;
                        tesselation_triangle.normal.y = normal.y;
                        tesselation_triangle.normal.z = normal.z;

                        temp_triangle.normal.x = normal.x;
                        temp_triangle.normal.y = normal.y;
                        temp_triangle.normal.z = normal.z;

                        normal_vector_counter = 0;

                        /*******************************************************************************************************/
                        /* Special Case ---> Normal is equal to the null vector                                                                                  */
                        /*                                                                                                                                                                 */
                        /* This result can only be obtained when two edges of the triangle are collinear (degenerate triangle).        */
                        /* This implies that the cell is too small or that at least two adjacent intersections are almost coincident or */
                        /* actually overlap.                                                                                                                                       */
                        /*******************************************************************************************************/

                        if (normal.x == 0 && normal.y == 0 && normal.z == 0)
                        {
                            do
                            {
                                normal_vector_counter++;

                                /* Update Scan */
                                if (polygon_index + 1 <= ssize_t(polygon.size()))
                                {
                                    polygon_index++;
                                }

                                /***********************************************/
                                /* Update node predecessors and successors  */
                                /***********************************************/

                                if (polygon_index == 1)
                                {
                                    predecessor = polygon_index - 1;
                                    successor = polygon_index + 1;
                                }

                                else if (polygon_index == 0)
                                {
                                    predecessor = (int)polygon.size() - 1;
                                    successor = polygon_index + 1;
                                }

                                else if (polygon_index == (int)polygon.size() - 1)
                                {
                                    predecessor = polygon_index - 1;
                                    successor = 0;
                                }

                                else if (polygon_index == (int)polygon.size())
                                {
                                    polygon_index = 0;
                                    predecessor = (int)polygon.size() - 1;
                                    successor = polygon_index + 1;
                                }

                                else
                                {
                                    predecessor = polygon_index - 1;
                                    successor = polygon_index + 1;
                                }

                                /* Define a new triangle */
                                tesselation_triangle.vertex1 = polygon[predecessor];
                                tesselation_triangle.vertex2 = polygon[polygon_index];
                                tesselation_triangle.vertex3 = polygon[successor];

                                /* Define two edge vectors of the triangle */
                                vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                                vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                                vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                                vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                                vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                                vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                                /* Recalculate normal to the triangle */
                                normal = cross_product(vector1, vector2);

                                tesselation_triangle.normal.x = normal.x;
                                tesselation_triangle.normal.y = normal.y;
                                tesselation_triangle.normal.z = normal.z;
                            } while ((normal.x == 0 && normal.y == 0 && normal.z == 0) && normal_vector_counter < ssize_t(polygon.size()));
                        }

                        /* If normal still equals the null vector implies that the cell is too small */
                        /* Proceed to store the last triangle in the tesselation vector */
                        triangulation_counter = (int)polygon.size() + 1;

                        /* Avoid infinite loops with the triangulation counter */
                        if (triangulation_counter > (int)polygon.size())
                        {
                            /* Store the triangle in the triangulation vector */
                            triangulation.push_back(temp_triangle);

                            /* Cut ear from the polygon vector */
                            start = polygon.begin();
                            end = polygon.end();
                            new_end = remove(start, end, polygon[temp_polygon_index]);
                            polygon.erase(new_end, end);

                            /* Update scan */
                            polygon_index = 1;
                        }
                    }

                    /* All ears have been cut; only one triangle is left */
                    if (polygon.size() == 3)
                    {
                        /***********************************************/
                        /* Update node predecessors and successors  */
                        /***********************************************/

                        if (polygon_index == 1)
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        if (polygon_index == 0)
                        {
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else if (polygon_index == ssize_t(polygon.size() - 1))
                        {
                            predecessor = polygon_index - 1;
                            successor = 0;
                        }

                        else if (polygon_index == ssize_t(polygon.size()))
                        {
                            polygon_index = 0;
                            predecessor = (int)polygon.size() - 1;
                            successor = polygon_index + 1;
                        }

                        else
                        {
                            predecessor = polygon_index - 1;
                            successor = polygon_index + 1;
                        }

                        /* Store the last triangle in the tesselation vector */
                        tesselation_triangle.vertex1 = polygon[predecessor];
                        tesselation_triangle.vertex2 = polygon[polygon_index];
                        tesselation_triangle.vertex3 = polygon[successor];

                        /* Define two edge vectors of the triangle */
                        vector1.x = intsec_vector[tesselation_triangle.vertex2].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector1.y = intsec_vector[tesselation_triangle.vertex2].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector1.z = intsec_vector[tesselation_triangle.vertex2].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        vector2.x = intsec_vector[tesselation_triangle.vertex3].intersection.x - intsec_vector[tesselation_triangle.vertex1].intersection.x;
                        vector2.y = intsec_vector[tesselation_triangle.vertex3].intersection.y - intsec_vector[tesselation_triangle.vertex1].intersection.y;
                        vector2.z = intsec_vector[tesselation_triangle.vertex3].intersection.z - intsec_vector[tesselation_triangle.vertex1].intersection.z;

                        /* Calculate normal to the triangle */
                        normal = cross_product(vector1, vector2);
                        tesselation_triangle.normal.x = normal.x;
                        tesselation_triangle.normal.y = normal.y;
                        tesselation_triangle.normal.z = normal.z;

                        /* Store the triangle in the tesselation vector */
                        triangulation.push_back(tesselation_triangle);
                        end_of_triangulation = true;
                    }
                } while (end_of_triangulation != true);
            }
        }
    }
}
}
#endif
