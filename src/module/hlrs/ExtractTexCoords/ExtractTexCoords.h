/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EXTRACTTEXCOORDS_H
#define _EXTRACTTEXCOORDS_H

/****************************************************************************\
**                                                    (C) 2010 Stellba      **
**                                                                          **
** Description: Extract the texture coords of a polygons object             **
**              that fulfils certain requirements                           **
**                                                                          **
** Name:        ExtractTexCoords                                            **
** Category:    Stellba                                                     **
**                                                                          **
** Author: M. Becker                                                        **
**                                                                          **
**  10/2010                                                                 **
**                                                                          **
\****************************************************************************/

#include <do/coDoPolygons.h>
#include <do/coDoPoints.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <api/coSimpleModule.h>

using namespace covise;

class ExtractTexCoords : public coSimpleModule
{
private:
    //member functions
    //compute callback
    virtual int compute(const char *port);

    //member data
    //coDoPolygons *polygon(const char *name, int num_points, int num_corners, int num_polygons);

    //Ports
    coInputPort *geoInPort; // blade polygons
    coOutputPort *textureOutPort;
    coOutputPort *geoOutPort;

    int num_points, num_corners, num_polygons;

    int get_neighbour_node(int current_node, int *nodecounter, int *outer_poly_conn_list, int num_outer_polys, int *used_hub_shroud_nodes, float *z);
    float distance(int nodenr1, int nodenr2, float *x, float *y, float *z);
    int get_next_slice_node(int current_node, int *node_polygons, int *node_polygon_pointer, int *nodes_used, int num_corners, int num_points, int *corner_list, int *polygon_list);

public:
    ExtractTexCoords(int argc, char *argv[]); // Constructor: module set-up
};
#endif
