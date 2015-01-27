/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2000 RUS   **
 **                                                                          **
 ** Description: Calculates normals and a point in the middle of a polygon   **
 **                                                                          **
 ** Name:        Normal                                                      **
 ** Category:    Tools                                                       **
 **                                                                          **
 ** Author: A. Heinchen		                                            **
 **                                                                          **
 ** History:  								    **
 ** Dec-00                                                                   **
 **   -- started writting module                                             **
 ** Jan-01                                                                   **
 **   -- working Version, strips and sets added                              **
 **                                                                          **
\****************************************************************************/

#include "ShowFaceNormal.h"
#include <util/coviseCompat.h>

ShowFaceNormal::ShowFaceNormal(int argc, char *argv[]) // this info appears in the module setup window
    : coSimpleModule(argc, argv, "Show the normals of surfaces")
{
    port_inPort = addInputPort("inPort", "Polygons|TriangleStrips", "surface consisting of polygons or triangle strips");
    point_outport = addOutputPort("points", "Points", "startpoints for normals (typically polygon center)");
    normal_outport = addOutputPort("vectors", "Vec3", "normals of the polygon or of the first triangle in tristrip");
}

int ShowFaceNormal::compute(const char *)
{

    int num_corners, num_polygons, num_strips;
    // number of corners        - the lenght of the corner_list
    // number of polygons       - the lenght of polygon_list
    // number of trianglestrips - the lenght of the triangle_list

    int cl_start, cl_end, cl_num;
    // start,end and number of points of the actuall polygon
    // used for accessing the corner_list

    float center_x, center_y, center_z;
    //the center of the polygon/first triangle of the strips
    float *vectorx, *vectory, *vectorz;
    //pointer to vector_values
    float *pointsx, *pointsy, *pointsz;
    //pointer to point_coordinates
    float *poly_x, *poly_y, *poly_z;
    //poiner to polygon_coordinates
    float *strip_x, *strip_y, *strip_z;
    //pointer to trinagle_strip_coordinates

    int *polygon_list;
    int *corner_list;
    int *triangle_list;
    //pointer to some arrays

    int point[3];
    //used for accesing the coords-arrays

    const coDistributedObject *obj = port_inPort->getCurrentObject();
    //is this an object?
    if (!obj)
    {
        sendError("No object at port '%s'", port_inPort->getName());
        return FAIL;
    }

    if (obj->isType("POLYGN"))
    //is it a ploygon?
    {
        coDoPolygons *polygon = (coDoPolygons *)obj;
        polygon->getAddresses(&poly_x, &poly_y, &poly_z, &corner_list, &polygon_list);
        num_corners = polygon->getNumVertices();
        num_polygons = polygon->getNumPolygons();
        // get the input obejct and some of its attributes

        coDoPoints *CentralPoints;
        CentralPoints = new coDoPoints(point_outport->getObjName(), num_polygons);
        point_outport->setCurrentObject(CentralPoints);
        CentralPoints->getAddresses(&pointsx, &pointsy, &pointsz);
        //create the Outputobject for the Points

        coDoVec3 *vectors;
        vectors = new coDoVec3(normal_outport->getObjName(), num_polygons);
        normal_outport->setCurrentObject(vectors);
        vectors->getAddresses(&vectorx, &vectory, &vectorz);
        //create the outputobjects for the normals

        for (int c_polygon = 0; c_polygon < num_polygons; c_polygon++)
        //step throung all polygons
        {
            //which part of the cornerlist is the Polygon
            cl_start = polygon_list[c_polygon];
            if (c_polygon == num_polygons - 1)
            //is this is the last polygon?
            {
                cl_end = num_corners;
            }
            else
            {
                cl_end = polygon_list[c_polygon + 1];
            }
            cl_num = cl_end - cl_start;

            //calculate the center of the polygon
            center_x = 0;
            center_y = 0;
            center_z = 0;
            for (int i = 0; i < cl_num; i++)
            {
                center_x = center_x + poly_x[corner_list[cl_start + i]];
                center_y = center_y + poly_y[corner_list[cl_start + i]];
                center_z = center_z + poly_z[corner_list[cl_start + i]];
            }
            pointsx[c_polygon] = center_x / cl_num;
            pointsy[c_polygon] = center_y / cl_num;
            pointsz[c_polygon] = center_z / cl_num;

            vectorx[c_polygon] = 0.0;
            vectory[c_polygon] = 0.0;
            vectorz[c_polygon] = 0.0;

            if (cl_num > 2)
            //at least three corners are needed to calc an normal
            {
                // go through all planes of the polygon
                for (int plane = 0; plane < cl_num; plane++)
                {
                    //get the next 3 points of the polygon
                    for (int point_num = 0; point_num < 3; point_num++)
                    {
                        point[point_num] = corner_list[cl_start + ((plane + point_num) % cl_num)];
                    }

                    //this huge block computes the vectorproduct of
                    //( (point1-point2) x (point3-point2) )
                    //this is the normal of the actuall plane
                    vectorx[c_polygon] = vectorx[c_polygon] + ((poly_y[point[1]] * poly_z[point[2]]) - (poly_y[point[0]] * poly_z[point[2]])
                                                               + (poly_y[point[0]] * poly_z[point[1]]) - (poly_z[point[1]] * poly_y[point[2]])
                                                               + (poly_z[point[0]] * poly_y[point[2]]) - (poly_z[point[0]] * poly_y[point[1]]));

                    vectory[c_polygon] = vectory[c_polygon] + ((poly_z[point[1]] * poly_x[point[2]]) - (poly_z[point[0]] * poly_x[point[2]])
                                                               + (poly_z[point[0]] * poly_x[point[1]]) - (poly_x[point[1]] * poly_z[point[2]])
                                                               + (poly_x[point[0]] * poly_z[point[2]]) - (poly_x[point[0]] * poly_z[point[1]]));

                    vectorz[c_polygon] = vectorz[c_polygon] + ((poly_x[point[1]] * poly_y[point[2]]) - (poly_x[point[0]] * poly_y[point[2]])
                                                               + (poly_x[point[0]] * poly_y[point[1]]) - (poly_y[point[1]] * poly_x[point[2]])
                                                               + (poly_y[point[0]] * poly_x[point[2]]) - (poly_y[point[0]] * poly_x[point[1]]));
                }
            }
            else
            {
                sendInfo("A polygon with less that three corners was skipped");
            }
        }
        return SUCCESS;
        //we are fini
    }
    else if (obj->isType("TRIANG"))
    //or is it a ploygon?
    {
        coDoTriangleStrips *strip = (coDoTriangleStrips *)obj;
        strip->getAddresses(&strip_x, &strip_y, &strip_z, &corner_list, &triangle_list);
        num_corners = strip->getNumVertices();
        num_strips = strip->getNumStrips();
        //get the inputobject and its attributes

        coDoPoints *CentralPoints;
        CentralPoints = new coDoPoints(point_outport->getObjName(), num_strips);
        point_outport->setCurrentObject(CentralPoints);
        CentralPoints->getAddresses(&pointsx, &pointsy, &pointsz);
        //create the point-object and give it to the port

        coDoVec3 *vectors;
        vectors = new coDoVec3(normal_outport->getObjName(), num_strips);
        normal_outport->setCurrentObject(vectors);
        vectors->getAddresses(&vectorx, &vectory, &vectorz);
        //create the normals-object and give it to the port

        for (int c_strip = 0; c_strip < num_strips; c_strip++)
        //step through all strips
        {

            //which part of the cornerlist is the Strip
            cl_start = triangle_list[c_strip];
            // if this is the last polygon
            if (c_strip == num_strips - 1)
            {
                cl_end = num_corners;
            }
            else
            {
                cl_end = triangle_list[c_strip + 1];
            }
            cl_num = cl_end - cl_start;

            if (cl_num > 2)
            // need at least three points for a normal
            {
                //calculate the center of the polygon
                center_x = 0;
                center_y = 0;
                center_z = 0;
                for (int i = 0; (i < 3); i++)
                {
                    center_x = center_x + strip_x[corner_list[cl_start + i]];
                    center_y = center_y + strip_y[corner_list[cl_start + i]];
                    center_z = center_z + strip_z[corner_list[cl_start + i]];
                }
                pointsx[c_strip] = center_x / 3;
                pointsy[c_strip] = center_y / 3;
                pointsz[c_strip] = center_z / 3;

                vectorx[c_strip] = 0.0;
                vectory[c_strip] = 0.0;
                vectorz[c_strip] = 0.0;

                //get the first 3 points of the strip
                for (int point_num = 0; point_num < 3; point_num++)
                {
                    point[point_num] = corner_list[cl_start + point_num];
                }

                //this huge block computes the vectorproduct of
                //( (point1-point2) x (point3-point2) )
                vectorx[c_strip] = vectorx[c_strip] + ((strip_y[point[1]] * strip_z[point[2]]) - (strip_y[point[0]] * strip_z[point[2]])
                                                       + (strip_y[point[0]] * strip_z[point[1]]) - (strip_z[point[1]] * strip_y[point[2]])
                                                       + (strip_z[point[0]] * strip_y[point[2]]) - (strip_z[point[0]] * strip_y[point[1]]));

                vectory[c_strip] = vectory[c_strip] + ((strip_z[point[1]] * strip_x[point[2]]) - (strip_z[point[0]] * strip_x[point[2]])
                                                       + (strip_z[point[0]] * strip_x[point[1]]) - (strip_x[point[1]] * strip_z[point[2]])
                                                       + (strip_x[point[0]] * strip_z[point[2]]) - (strip_x[point[0]] * strip_z[point[1]]));

                vectorz[c_strip] = vectorz[c_strip] + ((strip_x[point[1]] * strip_y[point[2]]) - (strip_x[point[0]] * strip_y[point[2]])
                                                       + (strip_x[point[0]] * strip_y[point[1]]) - (strip_y[point[1]] * strip_x[point[2]])
                                                       + (strip_y[point[0]] * strip_x[point[2]]) - (strip_y[point[0]] * strip_x[point[1]]));
            }
            else
            // panic - this was not realy an trianglestrip
            {
                sendInfo("A stripp with less than two corners was skipped");
                pointsx[c_strip] = 0;
                pointsy[c_strip] = 0;
                pointsz[c_strip] = 0;
                vectorx[c_strip] = 0;
                vectory[c_strip] = 0;
                vectorz[c_strip] = 0;
            }
        }
        return SUCCESS;
        //***************************************************************************************
    }
    else
    // panic - we recievend an unknown object
    {
        sendError("Recieved illegal type '%s' at port '%s'", obj->getType(), port_inPort->getName());
        return FAIL;
    }
}

MODULE_MAIN(Tools, ShowFaceNormal)
