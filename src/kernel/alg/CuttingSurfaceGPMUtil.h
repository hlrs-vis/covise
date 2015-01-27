/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/Triangulate.h>

namespace covise
{

typedef struct
{
    float v[3];
    int dimension;
} S_V_DATA;

typedef struct
{
    double x;
    double y;
    double z;
} EDGE_VECTOR;
typedef struct
{
    bool intersection_at_vertex1;
    bool intersection_at_vertex2;

    int int_flag;
    int vertex1;
    int vertex2;

    S_V_DATA data_vertex_int;
    EDGE_VECTOR intersection;
} PLANE_EDGE_INTERSECTION;
typedef struct
{
    vector<int> ring;
    vector<int> ring_index;
    vector<int> polyhedron_faces;
} CONTOUR;

// typedef struct
// {
//    vector<int> element_list;
//    vector<int> connectivity_list;
//    vector<float> plane_data;
// }PLANE_CELL_INTERSECTION ;

typedef std::vector<PLANE_EDGE_INTERSECTION> PLANE_EDGE_INTERSECTION_VECTOR;
//typedef std::vector<PLANE_CELL_INTERSECTION> PLANE_CELL_INTERSECTION_VECTOR;

double dot_product(EDGE_VECTOR &vector1, EDGE_VECTOR &vector2)
{
    return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;
}

EDGE_VECTOR cross_product(EDGE_VECTOR &vector1, EDGE_VECTOR &vector2)
{
    EDGE_VECTOR normal;

    normal.x = (vector1.y * vector2.z) - (vector2.y * vector1.z);
    normal.y = (vector2.x * vector1.z) - (vector1.x * vector2.z);
    normal.z = (vector1.x * vector2.y) - (vector2.x * vector1.y);

    return normal;
}

double length(EDGE_VECTOR &vector)
{
    return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

PLANE_EDGE_INTERSECTION PlaneEdgeVertexInterpolate(float x1, float y1, float z1, float x2, float y2, float z2, S_V_DATA data1, S_V_DATA data2, int v1, int v2, double p, EDGE_VECTOR unit_normal_vector, double D1, double D2)
{
    double t;
    double D;

    PLANE_EDGE_INTERSECTION intersection;

    /******************/
    /* Regular Cases  */
    /******************/

    /* Case 1:  Vertex 1 --> Behind; Vertex 2 --> In Front */
    if (D1 < 0 && D2 > 0)
    {
        /* Intersection Point in Slice Plane */
        D = 0;

        /* Calculate Parameter "t" for Line Equation */
        t = (D - x1 * unit_normal_vector.x - y1 * unit_normal_vector.y - z1 * unit_normal_vector.z - p) / (((x2 - x1) * unit_normal_vector.x) + ((y2 - y1) * unit_normal_vector.y) + ((z2 - z1) * unit_normal_vector.z));

        intersection.intersection_at_vertex1 = false;
        intersection.intersection_at_vertex2 = false;
        intersection.int_flag = 1;
        intersection.vertex1 = v1;
        intersection.vertex2 = v2;

        /* Calculate Intersection Coordinates */
        intersection.intersection.x = x1 + t * (x2 - x1);
        intersection.intersection.y = y1 + t * (y2 - y1);
        intersection.intersection.z = z1 + t * (z2 - z1);

        /* Interpolate Intersection Data Value */
        intersection.data_vertex_int.dimension = data1.dimension;
        for (int i = 0; i < intersection.data_vertex_int.dimension; ++i)
            intersection.data_vertex_int.v[i] = (float)(t * (data2.v[i] - data1.v[i]) + data1.v[i]);
    }

    /* Case 2:  Vertex 2 --> Behind; Vertex 1 --> In front */
    if (D1 > 0 && D2 < 0)
    {
        /* Intersection Point in Slice Plane */
        D = 0;

        /* Calculate Parameter "t" for Line Equation */
        t = (D - x1 * unit_normal_vector.x - y1 * unit_normal_vector.y - z1 * unit_normal_vector.z - p) / (((x2 - x1) * unit_normal_vector.x) + ((y2 - y1) * unit_normal_vector.y) + ((z2 - z1) * unit_normal_vector.z));

        intersection.intersection_at_vertex1 = false;
        intersection.intersection_at_vertex2 = false;
        intersection.int_flag = 1;
        intersection.vertex1 = v1;
        intersection.vertex2 = v2;

        /* Calculate Intersection Coordinates */
        intersection.intersection.x = x1 + t * (x2 - x1);
        intersection.intersection.y = y1 + t * (y2 - y1);
        intersection.intersection.z = z1 + t * (z2 - z1);

        /* Interpolate Intersection Data Value */
        intersection.data_vertex_int.dimension = data1.dimension;
        for (int i = 0; i < intersection.data_vertex_int.dimension; ++i)
            intersection.data_vertex_int.v[i] = (float)(t * (data2.v[i] - data1.v[i]) + data1.v[i]);
    }

    /***********************/
    /* Degenerate Cases   */
    /***********************/

    /* Case 3:  Vertex 1 --> Behind; Vertex 2 --> In Slice Plane */
    if (D1 < 0 && D2 == 0)
    {
        /* Calculate Edge-Plane Intersection */
        intersection.intersection_at_vertex1 = false;
        intersection.intersection_at_vertex2 = true;
        intersection.int_flag = 2;
        intersection.vertex1 = v1;
        intersection.vertex2 = v2;

        /* Calculate Intersection Coordinates */
        intersection.intersection.x = x2;
        intersection.intersection.y = y2;
        intersection.intersection.z = z2;

        /* Interpolate Intersection Data Value */
        intersection.data_vertex_int = data2;
    }

    /* Case 4:  Vertex 2  --> Behind; Vertex 1 --> In Slice Plane */
    if (D1 == 0 && D2 < 0)
    {
        /* Calculate Edge-Plane Intersection */
        intersection.intersection_at_vertex1 = true;
        intersection.intersection_at_vertex2 = false;
        intersection.int_flag = 2;
        intersection.vertex1 = v1;
        intersection.vertex2 = v2;

        /* Calculate Intersection Coordinates */
        intersection.intersection.x = x1;
        intersection.intersection.y = y1;
        intersection.intersection.z = z1;

        /* Interpolate Intersection Data Value */
        intersection.data_vertex_int = data1;
    }

    /* Case 5:  Vertex 1 --> In Front;  Vertex 2 --> In Slice Plane */
    if (D1 > 0 && D2 == 0)
    {
        /* Calculate Edge-Plane Intersection */
        intersection.intersection_at_vertex1 = false;
        intersection.intersection_at_vertex2 = true;
        intersection.int_flag = 2;
        intersection.vertex1 = v1;
        intersection.vertex2 = v2;

        /* Calculate Intersection Coordinates */
        intersection.intersection.x = x2;
        intersection.intersection.y = y2;
        intersection.intersection.z = z2;

        /* Interpolate Intersection Data Value */
        intersection.data_vertex_int = data2;
    }

    /* Case 6:  Vertex 2 --> In Front;  Vertex 1 --> In Slice Plane */
    if (D1 == 0 && D2 > 0)
    {
        /* Calculate Edge-Plane Intersection */
        intersection.intersection_at_vertex1 = true;
        intersection.intersection_at_vertex2 = false;
        intersection.int_flag = 2;
        intersection.vertex1 = v1;
        intersection.vertex2 = v2;

        /* Calculate Intersection Coordinates */
        intersection.intersection.x = x1;
        intersection.intersection.y = y1;
        intersection.intersection.z = z1;

        /* Interpolate Intersection Data Value */
        intersection.data_vertex_int = data1;
    }

    return (intersection);
}

bool test_plane_edge_intersection(PLANE_EDGE_INTERSECTION_VECTOR &intsec_vector, PLANE_EDGE_INTERSECTION &intsec, float *x_coord_in, float *y_coord_in, float *z_coord_in)
{
    /**************************************************/
    /* Test if intersection has already been processed */
    /**************************************************/

    for (vector<PLANE_EDGE_INTERSECTION>::iterator existent_intsec = intsec_vector.begin(); existent_intsec < intsec_vector.end(); ++existent_intsec)
    {
        if (existent_intsec->intersection_at_vertex1) // Intersection located at vertex 1
        {
            if (existent_intsec->vertex1 == intsec.vertex1 || existent_intsec->vertex1 == intsec.vertex2)
            {
                return true;
            }
        }
        else if (existent_intsec->intersection_at_vertex2) // Intersection located at vertex 2
        {
            if (existent_intsec->vertex1 == intsec.vertex1 || existent_intsec->vertex1 == intsec.vertex2)
            {
                return true;
            }
        }
        else // Intersection located between two vertices
        {

            // 			// Check intersection coordinates
            //  			if(intsec_vector[i].x == intsec.x && intsec_vector[i].y == intsec.y && intsec_vector[i].z == intsec.z)
            //  			{
            //  				return true;
            //  			}
            //
            // 			// Determine a tolerance to avoid machine precision errors
            // 			if(fabs(intsec_vector[i].x - intsec.x) < 0.000005 && fabs(intsec_vector[i].y - intsec.y) < 0.000005 && fabs(intsec_vector[i].z - intsec.z) < 0.000005)
            //  			{
            //  				return true;
            //  			}

            // Check edge vertices
            if (existent_intsec->vertex1 == intsec.vertex1 && existent_intsec->vertex2 == intsec.vertex2)
            {
                return true;
            }

            // Check edge vertices (swapped)
            if (existent_intsec->vertex1 == intsec.vertex2 && existent_intsec->vertex2 == intsec.vertex1)
            {
                return true;
            }

            // Check T-vertices
            if (existent_intsec->vertex1 == intsec.vertex1 || existent_intsec->vertex2 == intsec.vertex2 || existent_intsec->vertex1 == intsec.vertex2 || existent_intsec->vertex2 == intsec.vertex1)
            {
                // check direction of edges
                EDGE_VECTOR d1;
                d1.x = x_coord_in[existent_intsec->vertex1] - x_coord_in[existent_intsec->vertex2];
                d1.y = y_coord_in[existent_intsec->vertex1] - y_coord_in[existent_intsec->vertex2];
                d1.z = z_coord_in[existent_intsec->vertex1] - z_coord_in[existent_intsec->vertex2];
                EDGE_VECTOR d2;
                d2.x = x_coord_in[intsec.vertex1] - x_coord_in[intsec.vertex2];
                d2.y = y_coord_in[intsec.vertex1] - y_coord_in[intsec.vertex2];
                d2.z = z_coord_in[intsec.vertex1] - z_coord_in[intsec.vertex2];
                double length1 = length(d1);
                double length2 = length(d2);
                if ((length1 > 0) && (length2 > 0))
                {
                    double cosangle = dot_product(d1, d2) / (length1 * length2);
                    if (fabs(cosangle - 1.0) < 0.0001 || fabs(cosangle + 1.0) < 0.0001)
                    {
                        // ignore if the two edges have the same direction (and share a common vertex)
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

PLANE_EDGE_INTERSECTION_VECTOR calculate_intersections(float *udata_in, float *vdata_in, float *wdata_in, int num_elem_in, int *elem_in, int num_conn_in, int *conn_in, float *x_coord_in, float *y_coord_in, float *z_coord_in, float p, EDGE_VECTOR unit_normal_vector)
{
    int i;
    int j;

    S_V_DATA data_vertex1;
    S_V_DATA data_vertex2;
    data_vertex1.dimension = (wdata_in == NULL) ? 1 : 3;
    data_vertex2.dimension = (wdata_in == NULL) ? 1 : 3;

    double D1;
    double D2;

    int end;
    int index;

    EDGE_VECTOR cell_vertex_1;
    EDGE_VECTOR cell_vertex_2;

    PLANE_EDGE_INTERSECTION intersec;
    PLANE_EDGE_INTERSECTION_VECTOR intsec_vector;

    /* Analyze Polyhedron Faces */
    for (i = 0; i < num_elem_in; i++)
    {
        if (i < num_elem_in - 1)
        {
            end = elem_in[i + 1] - 1;
        }
        else
        {
            end = num_conn_in - 1;
        }

        for (j = elem_in[i]; j <= end; j++)
        {
            if (j < end)
            {
                index = j + 1;
            }
            else
            {
                index = elem_in[i];
            }

            /* Vector to First Cell Vertex */
            cell_vertex_1.x = x_coord_in[conn_in[j]];
            cell_vertex_1.y = y_coord_in[conn_in[j]];
            cell_vertex_1.z = z_coord_in[conn_in[j]];

            /* Vector to Second Cell Vertex */
            cell_vertex_2.x = x_coord_in[conn_in[index]];
            cell_vertex_2.y = y_coord_in[conn_in[index]];
            cell_vertex_2.z = z_coord_in[conn_in[index]];

            /* Vertex Data Values */
            //sdata_in->get_point_value(conn_in[j], &data_vertex1);
            //sdata_in->get_point_value(conn_in[index], &data_vertex2);
            data_vertex1.v[0] = udata_in[conn_in[j]];
            data_vertex2.v[0] = udata_in[conn_in[index]];
            if (wdata_in != NULL)
            {
                data_vertex1.v[1] = vdata_in[conn_in[j]];
                data_vertex2.v[1] = vdata_in[conn_in[index]];
                data_vertex1.v[2] = wdata_in[conn_in[j]];
                data_vertex2.v[2] = wdata_in[conn_in[index]];
            }

            /* Calculate Point-Plane Distance (Hessian Normal Form) */
            D1 = dot_product(unit_normal_vector, cell_vertex_1) + p;
            D2 = dot_product(unit_normal_vector, cell_vertex_2) + p;

            /******************/
            /* Regular Cases  */
            /******************/

            /* Case 1:  Vertex 1 --> Behind; Vertex 2 --> In Front */
            if (D1 < 0 && D2 > 0)
            {
                /* Calculate Edge-Plane Intersection */
                intersec = PlaneEdgeVertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[index]], y_coord_in[conn_in[index]], z_coord_in[conn_in[index]], data_vertex1, data_vertex2, conn_in[j], conn_in[index], p, unit_normal_vector, D1, D2);

                /* Discard Repeated Intersections */
                if (!test_plane_edge_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in))
                {
                    intsec_vector.push_back(intersec);
                }
            }

            /* Case 2:  Vertex 2 --> Behind; Vertex 1 --> In front */
            if (D1 > 0 && D2 < 0)
            {
                /* Calculate Edge-Plane Intersection */
                intersec = PlaneEdgeVertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[index]], y_coord_in[conn_in[index]], z_coord_in[conn_in[index]], data_vertex1, data_vertex2, conn_in[j], conn_in[index], p, unit_normal_vector, D1, D2);

                /* Discard Repeated Intersections */
                if (!test_plane_edge_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in))
                {
                    intsec_vector.push_back(intersec);
                }
            }

            /**********************/
            /* Degenerate Cases */
            /**********************/

            /* Case 3:  Vertex 1 --> Behind; Vertex 2 --> In Slice Plane */
            if (D1 < 0 && D2 == 0)
            {
                /* Calculate Edge-Plane Intersection */
                intersec = PlaneEdgeVertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[index]], y_coord_in[conn_in[index]], z_coord_in[conn_in[index]], data_vertex1, data_vertex2, conn_in[j], conn_in[index], p, unit_normal_vector, D1, D2);

                /* Discard Repeated Intersections */
                if (!test_plane_edge_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in))
                {
                    intsec_vector.push_back(intersec);
                }
            }

            /* Case 4:  Vertex 2  --> Behind; Vertex 1 --> In Slice Plane */
            if (D1 == 0 && D2 < 0)
            {
                /* Calculate Edge-Plane Intersection */
                intersec = PlaneEdgeVertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[index]], y_coord_in[conn_in[index]], z_coord_in[conn_in[index]], data_vertex1, data_vertex2, conn_in[j], conn_in[index], p, unit_normal_vector, D1, D2);

                /* Discard Repeated Intersections */
                if (!test_plane_edge_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in))
                {
                    intsec_vector.push_back(intersec);
                }
            }

            /* Case 5:  Vertex 1 --> In Front;  Vertex 2 --> In Slice Plane */
            if (D1 > 0 && D2 == 0)
            {
                /* Calculate Edge-Plane Intersection */
                intersec = PlaneEdgeVertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[index]], y_coord_in[conn_in[index]], z_coord_in[conn_in[index]], data_vertex1, data_vertex2, conn_in[j], conn_in[index], p, unit_normal_vector, D1, D2);

                /* Discard Repeated Intersections */
                if (!test_plane_edge_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in))
                {
                    intsec_vector.push_back(intersec);
                }
            }

            /* Case 6:  Vertex 2 --> In Front;  Vertex 1 --> In Slice Plane */
            if (D1 == 0 && D2 > 0)
            {
                /* Calculate Edge-Plane Intersection */
                intersec = PlaneEdgeVertexInterpolate(x_coord_in[conn_in[j]], y_coord_in[conn_in[j]], z_coord_in[conn_in[j]], x_coord_in[conn_in[index]], y_coord_in[conn_in[index]], z_coord_in[conn_in[index]], data_vertex1, data_vertex2, conn_in[j], conn_in[index], p, unit_normal_vector, D1, D2);

                /* Discard Repeated Intersections */
                if (!test_plane_edge_intersection(intsec_vector, intersec, x_coord_in, y_coord_in, z_coord_in))
                {
                    intsec_vector.push_back(intersec);
                }
            }

            /* Case 7:  Vertex 1 --> In Slice Plane; Vertex 2 --> In Slice Plane */
            if (D1 == 0 && D2 == 0)
            {
                /* Discard intersections when both of them are located within the slice plane */
            }
        }
    }

    return intsec_vector;
}

int assign_int_index(PLANE_EDGE_INTERSECTION_VECTOR &intsec_vector, int edge_vertex1, int edge_vertex2)
{
    int i;
    int index;

    index = 0;

    for (i = 0; i < intsec_vector.size(); i++)
    {
        PLANE_EDGE_INTERSECTION intsec = intsec_vector[i];
        if (intsec.intersection_at_vertex1 == false && intsec.intersection_at_vertex2 == false)
        {
            if ((edge_vertex1 == intsec.vertex1) && (edge_vertex2 == intsec.vertex2))
            {
                index = i;
            }

            if ((edge_vertex2 == intsec.vertex1) && (edge_vertex1 == intsec.vertex2))
            {
                index = i;
            }
        }

        if (intsec_vector[i].intersection_at_vertex1 == true)
        {
            if (intsec.vertex1 == edge_vertex1 || intsec.vertex1 == edge_vertex2)
            {
                index = i;
            }
        }

        if (intsec.intersection_at_vertex2 == true)
        {
            if (intsec.vertex2 == edge_vertex1 || intsec.vertex2 == edge_vertex2)
            {
                index = i;
            }
        }
    }

    return index;
}

bool find_intersection(PLANE_EDGE_INTERSECTION_VECTOR &intsec_vector, int edge_vertex1, int edge_vertex2)
{
    bool int_found;

    int int_found_flag;

    int_found_flag = 0;
    int_found = false;

    for (vector<PLANE_EDGE_INTERSECTION>::iterator intsec = intsec_vector.begin(); intsec < intsec_vector.end(); ++intsec)
    {
        if (int_found_flag == 0)
        {
            if (intsec->intersection_at_vertex1 == false && intsec->intersection_at_vertex2 == false)
            {
                if ((edge_vertex1 == intsec->vertex1) && (edge_vertex2 == intsec->vertex2))
                {
                    int_found = true;
                    int_found_flag = 1;
                }

                if ((edge_vertex2 == intsec->vertex1) && (edge_vertex1 == intsec->vertex2))
                {
                    int_found = true;
                    int_found_flag = 1;
                }
            }

            if (intsec->intersection_at_vertex1 == true)
            {
                if (intsec->vertex1 == edge_vertex1 || intsec->vertex1 == edge_vertex2)
                {
                    int_found = true;
                    int_found_flag = 1;
                }
            }

            if (intsec->intersection_at_vertex2 == true)
            {
                if (intsec->vertex2 == edge_vertex1 || intsec->vertex2 == edge_vertex2)
                {
                    int_found = true;
                    int_found_flag = 1;
                }
            }
        }
    }

    return int_found;
}

// void find_current_face(CONTOUR &contour, int edge_vertex1, int edge_vertex2, int *index_list, int *polygon_list, int num_coord_in, int num_conn_in, int &current_face)
// {
// 	int i;
// 	int j;
// 	int face_flag;
// 	int copy_current_face;
// 	int neighbour_face1;
// 	int neighbour_face2;
//
// 	ITERATOR it;
//
// 	face_flag = 0;
// 	copy_current_face = current_face;
//
// 	/******************************************************************************/
// 	/* Find  the next face of the polyhedron to continue tracing the convex contour */
// 	/******************************************************************************/
//
// 	if((edge_vertex1 < num_coord_in - 1) && (edge_vertex2 < num_coord_in - 1))
// 	{
// 		for(i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
// 		{
// 			neighbour_face1 = polygon_list[i];
// 			for(j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
// 			{
// 				neighbour_face2 = polygon_list[j];
// 				if(face_flag == 0)
// 				{
// 					/* Search among the elements that contain both vertices */
//  					if(neighbour_face1 == neighbour_face2)
//  					{
// 						/* Test if the element has already been processed */
// 						it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);
//
// 						/* The current face that contains the vertices has not been processed */
// 						if(it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
// 						{
// 							contour.polyhedron_faces.push_back(neighbour_face1);
// 							current_face = neighbour_face1;
// 							face_flag = 1;
// 						}
//  					}
// 				}
// 			}
// 		}
//
// 		/* All intersection faces have been processed at least once */
// 		if(face_flag == 0)
// 		{
// 			for(i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
// 			{
// 				neighbour_face1 = polygon_list[i];
// 				for(j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
// 				{
// 					neighbour_face2 = polygon_list[j];
// 					if(face_flag == 0)
// 					{
// 						/* Search among the elements that contain both vertices */
//  						if(neighbour_face1 == neighbour_face2)
//  						{
// 							if(neighbour_face1 != copy_current_face)
// 							{
// 								contour.polyhedron_faces.push_back(neighbour_face1);
// 								current_face = neighbour_face1;
// 								face_flag = 1;
// 							}
//  						}
// 					}
// 				}
// 			}
// 		}
// 	}
//
// 	else
// 	{
// 		if((edge_vertex1 < num_coord_in - 1) && (edge_vertex2 == num_coord_in - 1))
// 		{
// 			for(i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
// 			{
// 				neighbour_face1 = polygon_list[i];
// 				for(j = index_list[edge_vertex2]; j < num_conn_in; j++)
// 				{
// 					neighbour_face2 = polygon_list[j];
// 					if(face_flag == 0)
// 					{
// 						/* Search among the elements that contain both vertices */
//  						if(neighbour_face1 == neighbour_face2)
//  						{
// 							/* Test if the element has already been processed */
// 							it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);
//
// 							/* The current face that contains the vertices has not been processed */
// 							if(it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
// 							{
// 								contour.polyhedron_faces.push_back(neighbour_face1);
// 								current_face = neighbour_face1;
// 								face_flag = 1;
// 							}
//  						}
// 					}
// 				}
// 			}
//
// 			/* All intersection faces have been processed at least once */
// 			if(face_flag == 0)
// 			{
// 				for(i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
// 				{
// 					neighbour_face1 = polygon_list[i];
// 					for(j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
// 					{
// 						neighbour_face2 = polygon_list[j];
// 						if(face_flag == 0)
// 						{
// 							/* Search among the elements that contain both vertices */
//  							if(neighbour_face1 == neighbour_face2)
//  							{
// 								if(neighbour_face1 != copy_current_face)
// 								{
// 									contour.polyhedron_faces.push_back(neighbour_face1);
// 									current_face = neighbour_face1;
// 									face_flag = 1;
// 								}
//  							}
// 						}
// 					}
// 				}
// 			}
// 		}
//
// 		if((edge_vertex1 == num_coord_in - 1) && (edge_vertex2 < num_coord_in - 1))
// 		{
// 			for(i = index_list[edge_vertex1]; i < num_conn_in; i++)
// 			{
// 				neighbour_face1 = polygon_list[i];
// 				for(j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
// 				{
// 					neighbour_face2 = polygon_list[j];
// 					if(face_flag == 0)
// 					{
// 						/* Search among the elements that contain both vertices */
// 						if(neighbour_face1 == neighbour_face2)
//  						{
// 							/* Test if the element has already been processed */
// 							it = find(contour.polyhedron_faces.begin(), contour.polyhedron_faces.end(), neighbour_face1);
//
// 							/* The current face that contains the vertices has not been processed */
// 							if(it == contour.polyhedron_faces.end() || contour.polyhedron_faces.size() == 0)
// 							{
// 								contour.polyhedron_faces.push_back(neighbour_face1);
// 								current_face = neighbour_face1;
// 								face_flag = 1;
// 							}
// 						}
// 					}
// 				}
// 			}
//
// 			/* All intersection faces have been processed at least once */
// 			if(face_flag == 0)
// 			{
// 				for(i = index_list[edge_vertex1]; i < index_list[edge_vertex1 + 1]; i++)
// 				{
// 					neighbour_face1 = polygon_list[i];
// 					for(j = index_list[edge_vertex2]; j < index_list[edge_vertex2 + 1]; j++)
// 					{
// 						neighbour_face2 = polygon_list[j];
// 						if(face_flag == 0)
// 						{
// 							/* Search among the elements that contain both vertices */
//  							if(neighbour_face1 == neighbour_face2)
//  							{
// 								if(neighbour_face1 != copy_current_face)
// 								{
// 									contour.polyhedron_faces.push_back(neighbour_face1);
// 									current_face = neighbour_face1;
// 									face_flag = 1;
// 								}
//  							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// bool test_angle(float component1a, float component2a, float component1b, float component2b)
// {
// 	float cross_prod;
// 	float distance1;
// 	float distance2;
//
// 	cross_prod = component1a*component2b - component1b*component2a;
//
// 	if(cross_prod == 0)
// 	{
// 		/* Calculate Manhattan-Distances */
// 		distance1 = fabs(component1a) + fabs(component2a);
// 		distance2 = fabs(component1b) + fabs(component2b);
// 	}
//
// 	return cross_prod > 0 || cross_prod == 0 && distance1 > distance2;
// }
//
//
// void intersection_quicksortYZ(PLANE_EDGE_INTERSECTION_VECTOR &new_intsec_vector, vector<int> &vertex_polynome, int start, int end)
// {
// 	int i;
// 	int j;
//
// 	PLANE_EDGE_INTERSECTION quicksort_pivot;
//
// 	/* Start Quicksort */
// 	i = start;
// 	j = end;
//
// 	quicksort_pivot = new_intsec_vector[(start + end)/2];
//
// 	while(i <= j)
// 	{
//
// 		while(test_angle(new_intsec_vector[i].y, new_intsec_vector[i].z, quicksort_pivot.y, quicksort_pivot.z))
// 		{
// 			i++;
// 		}
//
// 		while(test_angle(quicksort_pivot.y, quicksort_pivot.z, new_intsec_vector[j].y, new_intsec_vector[j].z))
// 		{
// 			j--;
// 		}
//
// 		if(i <= j)
// 		{
// 			std::swap(vertex_polynome[i++], vertex_polynome[j--]);
// 			std::swap(new_intsec_vector[i++], new_intsec_vector[j--]);
// 		}
// 	}
//
// 	 if (start < j)
// 	{
// 		intersection_quicksortYZ(new_intsec_vector, vertex_polynome, start, j);
// 	}
//
//         if (i < end)
// 	{
// 		intersection_quicksortYZ(new_intsec_vector, vertex_polynome, i, end);
// 	}
// }
//
//
//
// void generate_capping_contour(CONTOUR &capping_contour, PLANE_EDGE_INTERSECTION_VECTOR intsec_vector, EDGE_VECTOR unit_normal_vector, vector<float> &data_vector)
// {
// 	int i;
//
// 	int pivot_index;
//
//
// 	vector<int> vertex_polynome;
//
//
//
//
//
// 	PLANE_EDGE_INTERSECTION_VECTOR new_intsec_vector;
//
//
//
//
//
// 	/*******************************************************/
// 	/* Case 1 --> Intersections form a point, line, or triangle */
// 	/*******************************************************/
//
// 	if(intsec_vector.size() <= 3)
// 	{
// 		for(i = 0; i < intsec_vector.size(); i++)
// 		{
// 			capping_contour.ring.push_back(i);
// 		}
//
// 		capping_contour.ring_index.push_back(0);
// 	}
//
// 	/*****************************************************/
// 	/* Case 2 --> Intersections form an arbitrary polygon  */
// 	/*****************************************************/
//
// 	if(intsec_vector.size() > 3)
// 	{
// 		pivot_index = 0;
// 		vertex_polynome.reserve(15);
// 		new_intsec_vector.reserve(15);
//
// 		for(i = 0; i < intsec_vector.size(); i++)
// 		{
// 			vertex_polynome.push_back(i);
// 			new_intsec_vector.push_back(intsec_vector[i]);
//
// 		}
//
// 		/*********************************/
// 		/* Find Edge of Intersection Hull   */
// 		/*********************************/
//
// 		/* General Case */
// 		if(unit_normal_vector.z != 0)
// 		{
// 			/* G*/
// 		}
//
// 		/* Special Cases */
// 		if(unit_normal_vector.z == 0)
// 		{
// 			/* Special Case 1 --> Sampling Plane Parallel to YZ-Axis */
// 			if(unit_normal_vector.x != 0 && unit_normal_vector.y == 0)
// 			{
// 				for(i = 1; i < intsec_vector.size(); i++)
// 				{
// 					/* Search minimum Z-Coordinate */
// 					if(intsec_vector[pivot_index].z > intsec_vector[i].z || intsec_vector[pivot_index].z == intsec_vector[i].z && intsec_vector[pivot_index].y > intsec_vector[i].y)
// 					{
// 						pivot_index = i;
// 					}
// 				}
// 			}
//
// 			/* Substract Pivot */
// 			for(i = 0; i < new_intsec_vector.size(); i++)
// 			{
// 				new_intsec_vector[i].x = new_intsec_vector[i].x - new_intsec_vector[pivot_index].x;
// 				new_intsec_vector[i].y = new_intsec_vector[i].y - new_intsec_vector[pivot_index].y;
// 				new_intsec_vector[i].z = new_intsec_vector[i].z - new_intsec_vector[pivot_index].z;
// 			}
//
// 			std::swap(vertex_polynome[0], vertex_polynome[pivot_index]);
// 			std::swap(new_intsec_vector[0], new_intsec_vector[pivot_index]);
//
// 			intersection_quicksortYZ(new_intsec_vector, vertex_polynome, 1, new_intsec_vector.size() - 1);
//
//
// //   			capping_contour.polyhedron_faces.erase(capping_contour.polyhedron_faces.begin(), capping_contour.polyhedron_faces.end());
//
// 			for(i = 0; i < vertex_polynome.size(); i++)
// 			{
// 				capping_contour.ring.push_back(vertex_polynome[i]);
// 			}
//
// 			capping_contour.ring_index.push_back(0);
//
// // 			/* Special Case 2 --> Sampling Plane Parallel to ZX-Plane */
// // 			else if(unit_normal_vector.x == 0 && unit_normal_vector.y != 0)
// // 			{
// // 			}
// //
// // 			/* Special Case 3 --> Sampling Plane rotates along the Z-Axis */
// // 			else if(unit_normal_vector.x != 0 && unit_normal_vector.y != 0)
// // 			{
// // 			}
//
// 		}
// 	}
//
// 	/* Finish Data Output Vector */
// 	/* Note:  data vector must not be ordered according to the Graham Scan */
// 	for(i = 0; i < intsec_vector.size(); i++)
// 	{
// 		data_vector.push_back(intsec_vector[i].data_vertex_int);
// 	}
// }

// void generate_capping_contour(CONTOUR &capping_contour, PLANE_EDGE_INTERSECTION_VECTOR &intsec_vector, int *index_list, int *polygon_list, int num_coord_in, int num_conn_in, int num_elem_in, int *elem_in, int *conn_in, vector<float> &data_vector)
// {
// 	bool extreme_edge_found;
//
// 	//char info[256];
//
// 	int i;
// 	int intersection_pointer1;
// 	int intersection_pointer2;
// 	int current_face;
// 	int previous_face;
// 	int edge_vertex1;
// 	int edge_vertex2;
// 	int new_edge_vertex1;
// 	int new_edge_vertex2;
// 	int index_v1;
// 	int index_v2;
// 	int index_flag_v1;
// 	int index_flag_v2;
// 	int counter1;
// 	int counter2;
// 	int counter3;
// 	int new_int_found;
// 	int loop_counter;
//
// 	double extreme_edge_magnitude;
// 	double base_edge_magnitude;
// 	double cosine;
// 	double lastCosine;
//
// 	EDGE_VECTOR extreme_edge;
// 	EDGE_VECTOR base_edge;
// 	EDGE_VECTOR test_vector1;
// 	EDGE_VECTOR test_vector2;
//
// 	vector<int> int_index_vector;
// 	vector<double> cosine_vector;
// 	vector<double> difference_vector;
//
// 	int smaller_position;
// 	bool angleChanged;
// 	double distance1;
// 	double distance2;
//
//
// 	/*******************************************************/
// 	/* Case 1 --> Intersections form a point, line, or triangle */
// 	/*******************************************************/
//
// 	if(intsec_vector.size() <= 3)
// 	{
// 		for(i = 0; i < intsec_vector.size(); i++)
// 		{
// 			capping_contour.ring.push_back(i);
// 		}
//
// 		capping_contour.ring_index.push_back(0);
// 	}
//
// 	/*****************************************************/
// 	/* Case 2 --> Intersections form an arbitrary polygon  */
// 	/*****************************************************/
//
// 	if(intsec_vector.size() > 3)
// 	{
// 		/**********************/
// 		/* Find Extreme Edge  */
// 		/**********************/
//
// 		/* First Intersection as Pivot */
// 		intersection_pointer1 = 0;
// 		edge_vertex1 = intsec_vector[0].vertex1;
// 		edge_vertex2 = intsec_vector[0].vertex2;
// 		loop_counter = 0;
// 		counter3 = 0;
// 		previous_face = -1;
// 		extreme_edge_found = false;
//
// 		do
// 		{
// 			do
// 			{
// 				if(loop_counter == 1)
// 				{
// 					previous_face = current_face;
// 				}
//
// 				loop_counter++;
//
// 				/* Find Current Face */
// 				find_current_face(capping_contour, edge_vertex1, edge_vertex2, index_list, polygon_list, num_coord_in, num_conn_in, current_face);
//
// 				/*****************************/
// 				/* Test Vertex Configuration  */
// 				/*****************************/
//
// 				index_flag_v1 = 0;
// 				index_flag_v2 = 0;
// 				new_int_found = 0;
// 				index_v1 = elem_in[current_face];
// 				index_v2 = elem_in[current_face];
//
// 				/***********************************************************************/
// 				/* Locate index values of the edge vertices in the array of connectivities */
// 				/***********************************************************************/
//
// 				if(current_face < num_elem_in - 1)
// 				{
// 					for(i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
// 					{
// 						if(index_flag_v1 == 0)
// 						{
// 							if(edge_vertex1 != conn_in[index_v1])
// 							{
// 								index_v1++;
// 							}
//
// 							else
// 							{
// 								index_flag_v1 = 1;
// 							}
// 						}
//
// 						if(index_flag_v2 == 0)
// 						{
// 							 if(edge_vertex2 != conn_in[index_v2])
// 							{
// 								index_v2++;
// 							}
//
// 							else
// 							{
// 								index_flag_v2 = 1;
// 							}
// 						}
// 					}
// 				}
//
// 				else
// 				{
// 					for(i = elem_in[current_face]; i < num_conn_in; i++)
// 					{
// 						if(index_flag_v1 == 0)
// 						{
// 							if(edge_vertex1 != conn_in[index_v1])
// 							{
// 								index_v1++;
// 							}
//
// 							else
// 							{
// 								index_flag_v1 = 1;
// 							}
// 						}
//
// 						if(index_flag_v2 == 0)
// 						{
// 							if(edge_vertex2 != conn_in[index_v2])
// 							{
// 								index_v2++;
// 							}
//
// 							else
// 							{
// 								index_flag_v2 = 1;
// 							}
// 						}
// 					}
// 				}
//
// 				counter1 = index_v1;
// 				counter2 = index_v2;
//
// 				/***************************/
// 				/* Locate Next Intersection */
// 				/***************************/
//
// 				if(current_face < num_elem_in - 1)
// 				{
// 					/**************************/
// 					/* Search Clockwise (CW) */
// 					/**************************/
//
// 					if(index_v1 > index_v2)
// 					{
// 						if((index_v1 != elem_in[current_face + 1] - 1) || (index_v2 != elem_in[current_face]))
// 						{
// 							for(i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
// 							{
// 								counter1++;
// 								counter2++;
//
// 								if(counter1 == elem_in[current_face + 1])
// 								{
// 									counter1 = elem_in[current_face];
// 								}
//
// 								if(counter2 == elem_in[current_face + 1])
// 								{
// 									counter2 = elem_in[current_face];
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
//
// 						if((index_v1 == elem_in[current_face + 1] - 1) && (index_v2 == elem_in[current_face]))
// 						{
// 							for(i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
// 							{
// 								counter1--;
// 								counter2--;
//
// 								if(counter1 == elem_in[current_face] - 1)
// 								{
// 									counter1 = elem_in[current_face + 1] - 1;
// 								}
//
// 								if(counter2 == elem_in[current_face] - 1)
// 								{
// 									counter2 = elem_in[current_face + 1] - 1;
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
// 					}
//
// 					/************************************/
// 					/* Search Counterclockwise (CCW)   */
// 					/************************************/
//
// 					if(index_v1 < index_v2)
// 					{
// 						if((index_v1 != elem_in[current_face]) || (index_v2 != elem_in[current_face +1] - 1))
// 						{
// 							for(i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
// 							{
// 								counter1--;
// 								counter2--;
//
// 								if(counter1 == elem_in[current_face] - 1)
// 								{
// 									counter1 = elem_in[current_face + 1] - 1;
// 								}
//
// 								if(counter2 == elem_in[current_face] - 1)
// 								{
// 									counter2 = elem_in[current_face + 1] - 1;
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
//
// 						if((index_v1 == elem_in[current_face]) && (index_v2 == elem_in[current_face +1] - 1))
// 						{
// 							for(i = elem_in[current_face]; i < elem_in[current_face + 1]; i++)
// 							{
// 								counter1++;
// 								counter2++;
//
// 								if(counter1 == elem_in[current_face + 1])
// 								{
// 									counter1 = elem_in[current_face];
// 								}
//
// 								if(counter2 == elem_in[current_face + 1])
// 								{
// 									counter2 = elem_in[current_face];
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
// 					}
// 				}
//
// 				else
// 				{
// 					/**************************/
// 					/* Search Clockwise (CW) */
// 					/**************************/
//
// 					if(index_v1 > index_v2)
// 					{
// 						if((index_v1 != num_conn_in - 1) || (index_v2 != elem_in[current_face]))
// 						{
// 							for(i = elem_in[current_face]; i < num_conn_in; i++)
// 							{
// 								counter1++;
// 								counter2++;
//
// 								if(counter1 == num_conn_in)
// 								{
// 									counter1 = elem_in[current_face];
// 								}
//
// 								if(counter2 == num_conn_in)
// 								{
// 									counter2 = elem_in[current_face];
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
//
// 						if((index_v1 == num_conn_in - 1) && (index_v2 == elem_in[current_face]))
// 						{
// 							for(i = elem_in[current_face]; i < num_conn_in; i++)
// 							{
// 								counter1--;
// 								counter2--;
//
// 								if(counter1 == elem_in[current_face] - 1)
// 								{
// 									counter1 = num_conn_in - 1;
// 								}
//
// 								if(counter2 == elem_in[current_face] - 1)
// 								{
// 									counter2 = num_conn_in - 1;
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
// 					}
//
// 					/************************************/
// 					/* Search Counterclockwise (CCW)   */
// 					/************************************/
//
// 					if(index_v1 < index_v2)
// 					{
// 						if((index_v1 != elem_in[current_face]) || (index_v2 != num_conn_in - 1))
// 						{
// 							for(i = elem_in[current_face]; i < num_conn_in; i++)
// 							{
// 								counter1--;
// 								counter2--;
//
// 								if(counter1 == elem_in[current_face] - 1)
// 								{
// 									counter1 = num_conn_in - 1;
// 								}
//
// 								if(counter2 == elem_in[current_face] - 1)
// 								{
// 									counter2 = num_conn_in - 1;
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
//
// 						if((index_v1 == elem_in[current_face]) && (index_v2 == num_conn_in - 1))
// 						{
// 							for(i = elem_in[current_face]; i < num_conn_in; i++)
// 							{
// 								counter1++;
// 								counter2++;
//
// 								if(counter1 == num_conn_in)
// 								{
// 									counter1 = elem_in[current_face];
// 								}
//
// 								if(counter2 == num_conn_in)
// 								{
// 									counter2 = elem_in[current_face];
// 								}
//
// 								if(new_int_found == 0)
// 								{
// 									/* Test if the a new intersection has been encountered */
// 									new_edge_vertex1 = conn_in[counter1];
// 									new_edge_vertex2 = conn_in[counter2];
//
// 									if(find_intersection(intsec_vector, new_edge_vertex1, new_edge_vertex2))
// 									{
// 										new_int_found = 1;
// 									}
// 								}
// 							}
// 						}
// 					}
// 				}
//
// 				intersection_pointer2 = assign_int_index(intsec_vector, new_edge_vertex1, new_edge_vertex2);
// 			}
// 			while(intersection_pointer1 == intersection_pointer2 && previous_face != current_face);
//
// 			/* Extreme Edge Found */
// 			if(intersection_pointer1 != intersection_pointer2)
// 			{
// 				extreme_edge_found = true;
// 			}
//
// 			/* Extreme Edge Not Found --> Select Different Pivot */
// 			if(!extreme_edge_found)
// 			{
// 				counter3++;
// 				intersection_pointer1 = counter3;
// 				edge_vertex1 = intsec_vector[counter3].vertex1;
// 				edge_vertex2 = intsec_vector[counter3].vertex2;
// 				capping_contour.polyhedron_faces.erase(capping_contour.polyhedron_faces.begin(), capping_contour.polyhedron_faces.end());
// 				previous_face = -1;
// 				loop_counter = 0;
// 			}
// 		}
// 		while(!extreme_edge_found && counter3 < intsec_vector.size());
//
// 		/* Define Extreme Edge Vector */
// 		extreme_edge.x = intsec_vector[intersection_pointer2].x - intsec_vector[intersection_pointer1].x;
// 		extreme_edge.y = intsec_vector[intersection_pointer2].y - intsec_vector[intersection_pointer1].y;
// 		extreme_edge.z = intsec_vector[intersection_pointer2].z - intsec_vector[intersection_pointer1].z;
// 		/* Define Extreme Edge Magnitude */
// 		extreme_edge_magnitude = length(extreme_edge);
//
// 		/*****************/
// 		/* Graham Scan  */
// 		/*****************/
//
// 		/* Store first two intersections of the convex hull */
// 		capping_contour.ring.push_back(intersection_pointer1);
// 		capping_contour.ring.push_back(intersection_pointer2);
// 		capping_contour.ring_index.push_back(0);
//
// 		/* Initiate Scan */
// 		for(i = 0; i < intsec_vector.size(); i++)
// 		{
// 			if(i != intersection_pointer1 && i != intersection_pointer2)
// 			{
// 				/* Scan Intersection Points */
// 				base_edge.x = intsec_vector[i].x - intsec_vector[intersection_pointer1].x;
// 				base_edge.y = intsec_vector[i].y - intsec_vector[intersection_pointer1].y;
// 				base_edge.z = intsec_vector[i].z - intsec_vector[intersection_pointer1].z;
// 				/* Define Vector Magnitude */
// 				base_edge_magnitude = length(base_edge);
//
// 				if (extreme_edge_magnitude > 0 && base_edge_magnitude > 0)
// 				{
// 					/* Define Angle between Vectors */
// 					cosine = dot_product(base_edge, extreme_edge)/(extreme_edge_magnitude*base_edge_magnitude);
// 				}
// 				else
// 				{
// 					cosine = 0.0;
// 				}
//
// 				cosine_vector.push_back(cosine);
//
// 				/* Store Intersection Indices */
// 				int_index_vector.push_back(i);
// 			}
// 		}
//
// 		/*******************/
// 		/* Debugging Info  */
// 		/*******************/
//
// 		//sprintf(info, "Angle Vector Size = %d", cosine_vector.size());
// 		//Covise::send_info(info);
// 		//sprintf(info, "Intersection Index Vector Size = %d", int_index_vector.size());
// 		//Covise::send_info(info);
//
// 		// calculate angle of the first two vertices in the ring
// 		base_edge.x = intsec_vector[intersection_pointer2].x - intsec_vector[intersection_pointer1].x;
// 		base_edge.y = intsec_vector[intersection_pointer2].y - intsec_vector[intersection_pointer1].y;
// 		base_edge.z = intsec_vector[intersection_pointer2].z - intsec_vector[intersection_pointer1].z;
// 		base_edge_magnitude = length(base_edge);
// 		if (extreme_edge_magnitude > 0 && base_edge_magnitude > 0)
// 		{
// 			lastCosine = dot_product(base_edge, extreme_edge)/(extreme_edge_magnitude*base_edge_magnitude);
// 		}
// 		else
// 		{
// 			lastCosine = 0.0;
// 		}
//
// 		angleChanged = false;
// 		const double ANGLE_THRESHHOLD = 0.0001; // higher values may display slightly ?konkave? cells correctly
// 		do
// 		{
//
// 			smaller_position = 0;
// 			for(i = 1; i < cosine_vector.size(); i++)
// 			{
// 				if(fabs(cosine_vector[smaller_position]-cosine_vector[i]) < ANGLE_THRESHHOLD)
// 				{
// 					test_vector1.x = intsec_vector[int_index_vector[smaller_position]].x - intsec_vector[intersection_pointer1].x;
// 					test_vector1.y = intsec_vector[int_index_vector[smaller_position]].y - intsec_vector[intersection_pointer1].y;
// 					test_vector1.z = intsec_vector[int_index_vector[smaller_position]].z - intsec_vector[intersection_pointer1].z;
//
// 					test_vector2.x = intsec_vector[int_index_vector[i]].x - intsec_vector[intersection_pointer1].x;
// 					test_vector2.y = intsec_vector[int_index_vector[i]].y - intsec_vector[intersection_pointer1].y;
// 					test_vector2.z = intsec_vector[int_index_vector[i]].z - intsec_vector[intersection_pointer1].z;
//
// 					distance1 = length(test_vector1);
// 					distance2 = length(test_vector2);
//
// 					if((angleChanged && (distance1 < distance2)) || (!angleChanged && (distance1 > distance2)))
// 					{
// 						// as long as the contour goes in a straight line away from intersection_pointer1,
// 						// prefer points closer to intersection_pointer1
// 						smaller_position = i;
// 					}
// 				}
// 				else if(cosine_vector[smaller_position] < cosine_vector[i]) // "cosine1 < cosine2" means "angle1 > angle2"
// 				{
// 					smaller_position = i;
// 				}
// 			}
//
// 			capping_contour.ring.push_back(int_index_vector[smaller_position]);
// 			angleChanged = angleChanged || (fabs(lastCosine-cosine_vector[smaller_position]) >= ANGLE_THRESHHOLD);
// 			lastCosine = cosine_vector[smaller_position];
//
// 			// erase
// 			int_index_vector.erase(int_index_vector.begin() + smaller_position);
// 			cosine_vector.erase(cosine_vector.begin() + smaller_position);
// 		}
// 		while(capping_contour.ring.size() < intsec_vector.size());
// 	}
//
// 	/* Finish Data Output Vector */
// 	/* Note:  data vector must not be ordered according to the Graham Scan */
// 	for(i = 0; i < intsec_vector.size(); i++)
// 	{
// 		data_vector.push_back(intsec_vector[i].data_vertex_int);
// 	}
//
// 	/*******************/
// 	/* Debugging Info  */
// 	/*******************/
//
// 	//sprintf(info, "Ring Size = %d", capping_contour.ring.size());
// 	//Covise::send_info(info);
// 	//sprintf(info, "Ring Index Size = %d", capping_contour.ring_index.size());
// 	//Covise::send_info(info);
// 	//sprintf(info, "Convex Hull of Intersections:");
// 	//Covise::send_info(info);
//
// 	//for(i = 0; i < capping_contour.ring.size(); i++)
// 	//{
// 	//	sprintf(info, "Intersection [%d ] = %d", i, capping_contour.ring[i]);
// 	//	Covise::send_info(info);
// 	//}
//
// 	//for(i = 0; i < data_vector.size(); i++)
// 	//{
// 	//	sprintf(info, "Intersection Data Value [%d ] = %f", i, data_vector[i]);
// 	//	Covise::send_info(info);
// 	//}
// }

// Explanation of cosine:
//   0 if vector has the same direction as plane_base_x
//   1 if vector has the same direction as plane_base_y
//   2 opposite to plane_base_x
//   3 opposite to normal plane_base_y
// Note:
//   plane_base_x has to be normalized
double get_cosine(EDGE_VECTOR &vector, EDGE_VECTOR &plane_base_x, EDGE_VECTOR &plane_base_y)
{
    double len = length(vector);
    if (len > 0.0)
    {
        double cosine = dot_product(vector, plane_base_x) / len;
        if (dot_product(vector, plane_base_y) > 0.0)
        {
            return 1.0 - cosine;
        }
        else
        {
            return 3.0 + cosine;
        }
    }
    else
    {
        return 0.0;
    }
}

void generate_capping_contour(CONTOUR &capping_contour, PLANE_EDGE_INTERSECTION_VECTOR &intsec_vector, EDGE_VECTOR &plane_base_x, EDGE_VECTOR &plane_base_y, vector<float> &u_data_vector, vector<float> &v_data_vector, vector<float> &w_data_vector)
{
    int size = (int)intsec_vector.size(); // we need this quite often
    if (size <= 3)
    {
        for (int i = 0; i < size; ++i)
        {
            capping_contour.ring.push_back(i);
        }
        capping_contour.ring_index.push_back(0);
    }
    else
    {

        // calculate the average position of all vertices
        EDGE_VECTOR average;
        average.x = 0.0;
        average.y = 0.0;
        average.z = 0.0;
        for (vector<PLANE_EDGE_INTERSECTION>::iterator intsec = intsec_vector.begin(); intsec < intsec_vector.end(); ++intsec)
        {
            average.x += intsec->intersection.x;
            average.y += intsec->intersection.y;
            average.z += intsec->intersection.z;
        }
        average.x /= double(size);
        average.y /= double(size);
        average.z /= double(size);

        // initiate scan (calculate angles)
        vector<int> vertex_index_vector;
        vector<double> vertex_cosine_vector;
        for (int i = 0; i < size; ++i)
        {
            EDGE_VECTOR current_vector = intsec_vector[i].intersection;
            current_vector.x -= average.x;
            current_vector.y -= average.y;
            current_vector.z -= average.z;
            vertex_cosine_vector.push_back(get_cosine(current_vector, plane_base_x, plane_base_y));
            vertex_index_vector.push_back(i);
        }

        // collect all vertices in ascending order (angles)
        capping_contour.ring_index.push_back(0);
        do
        {
            int smaller_position = 0;
            for (int i = 1; i < vertex_cosine_vector.size(); ++i)
            {
                if (vertex_cosine_vector[i] < vertex_cosine_vector[smaller_position])
                {
                    smaller_position = i;
                }
            }

            capping_contour.ring.push_back(vertex_index_vector[smaller_position]);

            // erase
            vertex_index_vector.erase(vertex_index_vector.begin() + smaller_position);
            vertex_cosine_vector.erase(vertex_cosine_vector.begin() + smaller_position);
        } while (capping_contour.ring.size() < size);

        // test if polygon is concave
        double lastCosine = 0.;
        bool concave = false;
        int concaveVertex;
        for (int i = 0; i <= capping_contour.ring.size(); ++i)
        {
            int vertex_index1 = capping_contour.ring[concaveVertex = (i) % capping_contour.ring.size()];
            int vertex_index2 = capping_contour.ring[(i + 1) % capping_contour.ring.size()];
            EDGE_VECTOR intersection1 = intsec_vector[vertex_index1].intersection;
            EDGE_VECTOR intersection2 = intsec_vector[vertex_index2].intersection;
            EDGE_VECTOR current_edge;
            current_edge.x = intersection2.x - intersection1.x;
            current_edge.y = intersection2.y - intersection1.y;
            current_edge.z = intersection2.z - intersection1.z;
            double cosine = get_cosine(current_edge, plane_base_x, plane_base_y);
            if (i != 0)
            {
                // check
                double diff = cosine - lastCosine;
                if (diff < 0.0)
                    diff = 4.0 + diff;
                if (diff > 2.0)
                {
                    concave = true;
                    break;
                }
            }
            lastCosine = cosine;
        }

        if (concave)
        {
            // we don't want concave polygons -> triangulate polygon
            tr_vertexVector vertices2d;
            for (vector<int>::iterator index = capping_contour.ring.begin(); index < capping_contour.ring.end(); ++index)
            {
                EDGE_VECTOR intersection = intsec_vector[*index].intersection;
                // go to 2D
                tr_vertex vertex2D;
                vertex2D.x = (float)dot_product(intersection, plane_base_x);
                vertex2D.y = (float)dot_product(intersection, plane_base_y);
                vertex2D.index = *index;
                // add
                vertices2d.push_back(vertex2D);
            }
            tr_intVector result;
            Triangulate::Process(vertices2d, result);
            capping_contour.ring.clear();
            capping_contour.ring_index.clear();
            for (int i = 0; i < result.size(); i += 3)
            {
                capping_contour.ring.push_back(result[i + 0]);
                capping_contour.ring.push_back(result[i + 1]);
                capping_contour.ring.push_back(result[i + 2]);
                capping_contour.ring_index.push_back(i);
            }
        }

    } // intsec_vector.size() > 3

    // Data output
    for (vector<PLANE_EDGE_INTERSECTION>::iterator intsec = intsec_vector.begin(); intsec < intsec_vector.end(); ++intsec)
    {
        u_data_vector.push_back(intsec->data_vertex_int.v[0]);
        if (intsec->data_vertex_int.dimension == 3)
        {
            v_data_vector.push_back(intsec->data_vertex_int.v[1]);
            w_data_vector.push_back(intsec->data_vertex_int.v[2]);
        }
    }
}
}
