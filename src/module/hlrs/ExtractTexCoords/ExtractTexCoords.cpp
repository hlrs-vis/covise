/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include <img/coImage.h>
#include <do/coDoPixelImage.h>
#include <do/coDoTexture.h>

#include "ExtractTexCoords.h"
#include <util/coviseCompat.h>

#include "float.h"

#define sqr(x) ((x) * (x))

ExtractTexCoords::ExtractTexCoords(int argc, char *argv[])
    : coSimpleModule(argc, argv, "extract texture coords of polygon objects for LIC")
{
    // ports
    geoInPort = addInputPort("polygons", "Polygons", "surface consisting of polygons");
    geoInPort->setRequired(0);

    geoOutPort = addOutputPort("GridOut0", "Triangles", "surface converted to trianglestrips");
    textureOutPort = addOutputPort("TextureOut0", "Texture", "LIC Texture");
}

int ExtractTexCoords::compute(const char *)
{
    // this algorithm works for a structured mesh of quads on the blade
    // the blade must be an O-grid
    // blade-to blade-topology not yet implemented (can easily be done)
    // the blade polygons must be included in one polygon object

    fprintf(stderr, "entering ExtractTexCoords::compute\n");

    const coDistributedObject *geoObj = geoInPort->getCurrentObject();

    if (!geoObj)
    {
        sendError("No object at port '%s'", geoInPort->getName());
        return (FAIL);
    }

    // number of points         - the length of the coordinate list
    // number of corners        - the length of the corner list
    // number of polygons       - the length of polygonlist
    // number of data           - the length of velocity list

    // start,end and number of points of the actual polygon
    // used for accessing the corner_list

    float *x, *y, *z; // poiner to polygon_coordinates

    int *polygon_list; // polygon list
    int *corner_list; // connectivity list

    //is it a polygon object?
    if (geoObj->isType("POLYGN"))
    {
        const coDoPolygons *polygon = (const coDoPolygons *)geoObj;
        polygon->getAddresses(&x, &y, &z, &corner_list, &polygon_list);
        num_points = polygon->getNumPoints(); // length of coordinate array
        num_corners = polygon->getNumVertices(); // length of connectivity list
        num_polygons = polygon->getNumPolygons(); // number of polygons

        int *vl = new int[num_polygons * 2 * 3];

        for (int index = 0; index < num_polygons; index++)
        {
            int begin = polygon_list[index];
            int end;
            if (index < num_polygons - 1)
                end = polygon_list[index + 1] - 1;
            else
                end = num_corners - 1;

            vl[index * 3 * 2] = corner_list[begin];
            vl[index * 3 * 2 + 1] = corner_list[begin + 1];
            vl[index * 3 * 2 + 2] = corner_list[begin + 2];
            vl[index * 3 * 2 + 3] = corner_list[begin];
            vl[index * 3 * 2 + 4] = corner_list[begin + 2];
            vl[index * 3 * 2 + 5] = corner_list[begin + 3];
        }

        coDoTriangles *tri = new coDoTriangles(geoOutPort->getNewObjectInfo(),
                                               num_points, x, y, z,
                                               num_polygons * 2 * 3, vl);

        geoOutPort->setCurrentObject(tri);
        /*
      // conversion from quads to trianglestrips
      int *strip_corners = new int[num_corners];
      for (int index = 0; index < num_polygons; index ++) {

         int begin = polygon_list[index];
         int end;
         if (index < num_polygons - 1)
            end = polygon_list[index + 1] - 1;
         else
            end = num_corners - 1;
         
         strip_corners[begin] = corner_list[begin];
         strip_corners[begin + 1] = corner_list[begin + 1];
         int t = corner_list[begin + 2];
         strip_corners[begin + 2] = corner_list[begin + 3];
         strip_corners[begin + 3] = t;
      }

      coDoTriangleStrips *strips = new coDoTriangleStrips(geoOutPort->getNewObjectInfo(), num_points, x, y, z, num_corners, strip_corners, num_polygons, polygon_list);
      geoOutPort->setCurrentObject(strips);
                                             */
    }
    else
    {
        // received an unknown object
        sendError("Received illegal type '%s' at port '%s'. We need Polygons.", geoObj->getType(), geoInPort->getName());
        return (FAIL);
    }

    // count occurences of nodes in the connectivity-list
    int *nodecounter = new int[num_points];
    memset(nodecounter, 0, num_points * sizeof(int));

    for (int i = 0; i < num_corners; i++)
    {
        nodecounter[corner_list[i]]++;
    }

    // create a list of all nodes with nodecounter==2
    std::vector<int> hubshroud_nodes;
    int num_hubshroud_nodes = 0;
    for (int i = 0; i < num_points; i++)
    {
        if (nodecounter[i] == 2)
        {
            hubshroud_nodes.push_back(i);
            num_hubshroud_nodes++;
        }
    }
    fprintf(stderr, "we have %d nodes at hub and shroud!\n", num_hubshroud_nodes);

#ifdef DEBUG
    for (int i = 0; i < num_hubshroud_nodes; i++)
    {
        fprintf(stderr, "%d\n", hubshroud_nodes[i]);
    }
#endif

    // filter the polygon connectivity list to increase efficiency of search algorithm
    // we want to have all polygons containing exactly two nodes with nodecounter==2

    int polygon_nodes[4];
    int outernodes = 0;
    int j;
    std::vector<int> outer_poly_conn_list;
    // loop over all our polygons
    for (int i = 0; i < num_polygons; i++)
    {
        outernodes = 0;
        polygon_nodes[0] = corner_list[4 * i];
        polygon_nodes[1] = corner_list[4 * i + 1];
        polygon_nodes[2] = corner_list[4 * i + 2];
        polygon_nodes[3] = corner_list[4 * i + 3];

        // check whether two nodes have nodecounter==2
        for (j = 0; j < 4; j++)
        {
            if (nodecounter[polygon_nodes[j]] == 2)
            {
                outernodes++;
            }
        }

        if (outernodes == 2)
        {
            outer_poly_conn_list.push_back(polygon_nodes[0]);
            outer_poly_conn_list.push_back(polygon_nodes[1]);
            outer_poly_conn_list.push_back(polygon_nodes[2]);
            outer_poly_conn_list.push_back(polygon_nodes[3]);
        }
    }

    int num_outer_polys = outer_poly_conn_list.size() / 4;

    fprintf(stderr, "we have %d polygons at hub and shroud!\n", (int)outer_poly_conn_list.size() / 4);
    if (num_outer_polys != num_hubshroud_nodes) // O-Grid
    {
        fprintf(stderr, "error! num_outer_polys should be %d but is %d\n", num_hubshroud_nodes, num_outer_polys);
    }
    if (num_outer_polys == num_hubshroud_nodes - 1) // blade-to-blade
    {
        fprintf(stderr, "you might have a blade-to-blade mesh. Algorithm has to be adapted for that\n");
    }

    // divide this list based on connectivity into two separate lists
    // list which remembers whether a hub / shroud point has already been treated
    // values:
    // -1: node is not at hub or shroud
    //  0: node is at hub or shroud and is unused
    //  1: node is at hub or shroud and has already been treated for the list separation
    int *used_hub_shroud_nodes = new int[num_points];
    memset(used_hub_shroud_nodes, -1, num_points * sizeof(int));
    for (int i = 0; i < num_hubshroud_nodes; i++)
    {
        used_hub_shroud_nodes[hubshroud_nodes[i]] = 0;
    }

    // we simply take the first hub/shroud point and start with it
    int next_node = hubshroud_nodes[0];

    // now we go along the blade edge at the hub/shroud starting from startpoint
    // we reach the end when there are no more reachable neighbour nodes with used_hub_shroud_nodes==0
    // int num_neighbour_nodes;
    int current_node;

    std::vector<int> hubshroud1;
    do
    {
        current_node = next_node;
        hubshroud1.push_back(current_node);
        used_hub_shroud_nodes[current_node] = 1; // mark node as used
        // get "free" (unused) neighbour nodes
        next_node = get_neighbour_node(current_node, nodecounter, &outer_poly_conn_list[0], num_outer_polys, used_hub_shroud_nodes, z);
#ifdef DEBUG
        fprintf(stderr, "correct neighbour node of %d is %d\n", current_node, next_node);
#endif
    } while (next_node != -1);
    // close loop ...
    hubshroud1.push_back(hubshroud1[0]);

    // now do the same again for the other side (hub/shroud)
    // this time, we start with the first unused node
    int counter = 0;
    while (used_hub_shroud_nodes[hubshroud_nodes[counter]] != 0)
    {
        counter++;
    }
    if (counter > (num_points - 1))
    {
        fprintf(stderr, "no starting node found for second side (counter=%d, num_points=%d)!\n", counter, num_points);
        return (FAIL);
    }
    next_node = hubshroud_nodes[counter];

    std::vector<int> hubshroud2;
    do
    {
        current_node = next_node;
        hubshroud2.push_back(current_node);
        used_hub_shroud_nodes[current_node] = 1; // mark node as used
        // get "free" (unused) neighbour nodes
        next_node = get_neighbour_node(current_node, nodecounter, &outer_poly_conn_list[0], num_outer_polys, used_hub_shroud_nodes, z);
#ifdef DEBUG
        fprintf(stderr, "correct neighbour node of %d is %d\n", current_node, next_node);
#endif
    } while (next_node != -1);
    // close loop ...
    hubshroud2.push_back(hubshroud2[0]);

    // write out our lists (for debugging)
    fprintf(stderr, "writing hub and shroud blade margin to /tmp/hubshroud.txt\n");
    FILE *fp;
    char fn[200];

#ifdef _WIN32
    sprintf(fn, "%s\\hubshroud.txt", getenv("TMP"));
#else
    sprintf(fn, "/tmp/hubshroud.txt");
#endif
    if ((fp = fopen(fn, "w+")) == NULL)
    {
        fprintf(stderr, "couldn't open file '%s'!\n", fn);
    }
    fprintf(fp, "# hubshroud1\n");
    for (int i = 0; i < hubshroud1.size(); i++)
    {
        fprintf(fp, "%4d %d %f %f %f\n", i, hubshroud1[i], x[hubshroud1[i]], y[hubshroud1[i]], z[hubshroud1[i]]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "# hubshroud2\n");
    for (int i = 0; i < hubshroud2.size(); i++)
    {
        fprintf(fp, "%4d %d %f %f %f\n", i, hubshroud2[i], x[hubshroud2[i]], y[hubshroud2[i]], z[hubshroud2[i]]);
    }

    fclose(fp);

    // decide which list is hub and which is shroud (based on average radius)
    float r1 = 0.;
    float r2 = 0.;
    for (int i = 0; i < hubshroud1.size(); i++)
    {
        r1 += sqrt(sqr(x[hubshroud1[i]]) + sqr(y[hubshroud1[i]]));
    }
    r1 /= hubshroud1.size();
    for (int i = 0; i < hubshroud2.size(); i++)
    {
        r2 += sqrt(sqr(x[hubshroud2[i]]) + sqr(y[hubshroud2[i]]));
    }
    r2 /= hubshroud2.size();

    fprintf(stderr, "average radius of side1 is %8.5f\n", r1);
    fprintf(stderr, "average radius of side2 is %8.5f\n", r2);

    if (r1 < r2)
    {
        fprintf(stderr, "side1 is hub, side2 is shroud\n");
    }
    else
    {
        fprintf(stderr, "side1 is shroud, side2 is hub\n");
    }

    // now we need to extract all the slices between hub and shroud
    if (hubshroud1.size() != hubshroud2.size())
    {
        fprintf(stderr, "different number of nodes at hub and shroud! Error!\n");
        return (FAIL);
    }
    int num_slices = num_points / (hubshroud1.size() - 1);

    fprintf(stderr, "num mesh slices between hub and shroud: %d\n", num_slices);

    /*
   if (!( ((hubshroud1.size() * num_slices) == num_points)  || (((hubshroud1.size()-1) * num_slices) == num_points) ))
      // second part of OR is not logical, probably when reading CFX polygons num_points is not correct (check ReadCFX)
   {
      fprintf(stderr, "this is probably not a structured mesh on your blade! Error!\n");
      return(FAIL);
   }
   */
    fprintf(stderr, "number of slices between hub and shroud: %d\n", num_slices);

    // reorder hubshroud1 and hubshroud2 so that it runs from leading edge to trailing edge

    // find leading edge and trailing edge
    // first method: leading edge and trailing edge points are the points
    // that have max. distance from each other
    // other methods could consider the curvature ... no time for that detail at the moment, maybe later
    // we do that at the shroud because we expect to get better results here
    // (camber is smaller + thinner profiles)

    int *hub, *shroud;
    //fprintf(stderr,"r1=%8.5lf, r2=%8.5lf\n",r1,r2);
    if (r1 < r2)
    {
        hub = &hubshroud1[0];
        shroud = &hubshroud2[0];
    }
    else
    {
        hub = &hubshroud2[0];
        shroud = &hubshroud1[0];
    }

    float max_distance = 0.;
    int i;
    float dist;
    int p1, p2;
    int p1_i, p2_i;
    int p_le_i;
    for (i = 0; i < hubshroud1.size(); i++)
    {
        for (j = 0; j < hubshroud1.size(); j++)
        {
            dist = distance(shroud[i], shroud[j], x, y, z);
            if (dist > max_distance)
            {
                max_distance = dist;
                p1 = shroud[i];
                p1_i = i;
                p2 = shroud[j];
                p2_i = i;
            }
        }
    }

    int leshroud, teshroud;

    // one of those points is declred to be the leading edge, the other the trailing edge
    // could be distinguished by the pressure, but we do not want to depend on the pressure here
    // we just want to have texture coords, for that purpose it is wurschd which node is leading edge and
    // which is trailing edge

    fprintf(stderr, "node %d is leading edge point at shroud side!\n", p1);
    fprintf(stderr, "node %d is trailing edge point at shroud side\n", p2);
    leshroud = p1;
    p_le_i = p1_i;
    teshroud = p2;

    // automatic leading edge node correction could be done here
    // profile has a camber, that always leads to a faulty result
    // if we just consider the two nodes with the maximum distance
    // so far not implemented ;-) ...

    // manual leading edge correction
    // delta number of points (+-)

    // commented out ... not needed for texture coords extraction

    // now reorder shroud array
    // we want to have two arrays of nodes at shroud side
    // both running from le to te, one for ps, one for ss

    int *shroud2 = new int[hubshroud1.size() + 1];
    int pos = 0;
    int lepos = 0;
    while (shroud[lepos] != leshroud)
    {
        lepos++;
    }
    for (i = lepos; i < hubshroud1.size(); i++)
    {
        shroud2[pos] = shroud[i];
        pos++;
    }
    for (i = 0; i < lepos; i++)
    {
        shroud2[pos] = shroud[i];
        pos++;
    }
    shroud2[pos] = leshroud;

    std::vector<int> shroud_side1;
    std::vector<int> shroud_side2;

    pos = 0;
    while (shroud2[pos] != teshroud)
    {
        shroud_side1.push_back(shroud2[pos]);
        pos++;
    }
    shroud_side1.push_back(shroud2[pos]);

    pos = hubshroud1.size();
    while (shroud2[pos] != teshroud)
    {
        shroud_side2.push_back(shroud2[pos]);
        pos--;
    }
    shroud_side2.push_back(shroud2[pos]);

#ifdef _WIN32
    sprintf(fn, "%s\\shroud_ps_ss.txt", getenv("TMP"));
#else
    sprintf(fn, "/tmp/shroud_ps_ss.txt");
#endif
    if ((fp = fopen(fn, "w+")) == NULL)
    {
        fprintf(stderr, "couldn't open file '%s'!\n", fn);
    }
    float phi;
    for (int i = 0; i < shroud_side1.size(); i++)
    {
        phi = atan2(y[shroud_side1[i]], x[shroud_side1[i]]);
        fprintf(fp, "%f %f %f %f\n", phi, x[shroud_side1[i]], y[shroud_side1[i]], z[shroud_side1[i]]);
    }
    fprintf(fp, "\n\n");
    for (int i = 0; i < shroud_side2.size(); i++)
    {
        phi = atan2(y[shroud_side2[i]], x[shroud_side2[i]]);
        fprintf(fp, "%f %f %f %f\n", phi, x[shroud_side2[i]], y[shroud_side2[i]], z[shroud_side2[i]]);
    }
    fclose(fp);

    delete[] shroud2;

    // as we do not look at the pressure on each side,
    // the decision which side is pressure side and which suction side is a random decision
    int *shroud_ss;
    int *shroud_ps;
    int num_ps_nodes;
    int num_ss_nodes;

    //fprintf(stderr, "side 1 is pressure side, side 2 is suction side.\n");
    shroud_ps = &shroud_side1[0];
    shroud_ss = &shroud_side2[0];
    num_ps_nodes = shroud_side1.size();
    num_ss_nodes = shroud_side2.size();

    fprintf(stderr, "num nodes on pressure side: %d\n", num_ps_nodes);
    fprintf(stderr, "num nodes on suction side: %d\n", num_ss_nodes);

    // preparation:
    // we need an array that gives the adjacent polygons for each node
    // that is a sort of reoredered connectivity list (same length)

    int polygon_corners;
    // give polygons for a given node
    int *node_polygons = new int[num_corners];
    // points to node_polygons list (gives starting index for a node)
    int *node_polygon_pointer = new int[num_points];
    // counts how often a node has already been used to construct node_polygons list
    int *nodes_used = new int[num_points];
    memset(nodes_used, 0, num_points * sizeof(int));

    int nodenr;

    // construct node_polygon_pointer from nodecounter
    pos = 0;
    for (i = 0; i < num_points; i++)
    {
        node_polygon_pointer[i] = pos;
        pos += nodecounter[i];
    }

    for (i = 0; i < num_polygons; i++)
    {
        if (i < num_polygons - 1)
        {
            polygon_corners = polygon_list[i + 1] - polygon_list[i];
        }
        else
        {
            polygon_corners = num_corners - polygon_list[i];
        }
        if (polygon_corners != 4)
        {
            fprintf(stderr, "unsupported number of corners!\n");
        }

        //if (i<10)
        //   fprintf(stderr,"polygon %d has %d corners:",i,polygon_corners);

        for (j = 0; j < polygon_corners; j++)
        {
            // if (i==7314)
            //   fprintf(stderr," %d",corner_list[polygon_list[i]+j]);
            nodenr = corner_list[polygon_list[i] + j];
            //if (i==7314)
            //   fprintf(stderr,"node %d is in polygon %d\n",nodenr,i);

            // node_polygon_pointer[nodenr]+j zeigt auf die Polygonnr.-Liste
            node_polygons[node_polygon_pointer[nodenr] + nodes_used[nodenr]] = i;
            nodes_used[nodenr]++;
        }
        // if (i==7314)
        //    fprintf(stderr,"\n");
    }

    /*
    *      i=994;
    *      int num_nodepols = node_polygon_pointer[i+1] - node_polygon_pointer[i];
    *      fprintf(stderr,"node_polygon_pointer[%d]=%d\n",i,node_polygon_pointer[i]);
    *      fprintf(stderr,"node %d takes part in %d polygons:",i,num_nodepols);
    *
    *      for (j=0; j<num_nodepols; j++)
    *      {
    *         fprintf(stderr," %d", node_polygons[node_polygon_pointer[i]+j]);
    *      }
    *      fprintf(stderr,"\n");
    */

    // we will walk through the mesh starting at the shroud side,
    // sorting all the nodes to the different slices
    // node_polygons[node_polygon_pointer]  contains the fist polygon a node participates in

    int *sorted_ps = new int[num_ps_nodes * num_slices];
    int *sorted_ss = new int[num_ss_nodes * num_slices];
    memset(nodes_used, 0, num_points * sizeof(int));

    // sorted_ps / sorted_ss starts with shroud nodes
    for (i = 0; i < num_ps_nodes; i++)
    {
        sorted_ps[i] = shroud_ps[i];
    }
    for (i = 0; i < num_ss_nodes; i++)
    {
        sorted_ss[i] = shroud_ss[i];
    }

    // for each shroud node, walk through blade from hub to shroud
    // we start at the pressure side
    for (int slicenr = 0; slicenr < (num_slices - 1); slicenr++)
    {
        // mark this slice's nodes as used
        for (i = 0; i < num_ps_nodes; i++)
        {
            nodes_used[sorted_ps[slicenr * num_ps_nodes + i]] = 1;
        }
        for (i = 0; i < num_ss_nodes; i++)
        {
            nodes_used[sorted_ss[slicenr * num_ss_nodes + i]] = 1;
        }

        for (i = 0; i < num_ps_nodes; i++)
        {
            current_node = sorted_ps[slicenr * num_ps_nodes + i];
            sorted_ps[(slicenr + 1) * num_ps_nodes + i] = get_next_slice_node(current_node, node_polygons, node_polygon_pointer, nodes_used, num_corners, num_points, corner_list, polygon_list);
        }
        // do the same for suction side
        for (i = 0; i < num_ss_nodes; i++)
        {
            current_node = sorted_ss[slicenr * num_ss_nodes + i];
            sorted_ss[(slicenr + 1) * num_ss_nodes + i] = get_next_slice_node(current_node, node_polygons, node_polygon_pointer, nodes_used, num_corners, num_points, corner_list, polygon_list);
        }
    }

// evaluate ps
#ifdef _WIN32
    sprintf(fn, "%s\\ps.txt", getenv("TMP"));
#else
    sprintf(fn, "/tmp/ps.txt");
#endif
    if ((fp = fopen(fn, "w+")) == NULL)
    {
        fprintf(stderr, "couldn't open file '%s'!\n", fn);
    }

    for (int slicenr = 0; slicenr < num_slices; slicenr++)
    {
        for (i = 0; i < num_ps_nodes; i++)
        {
            nodenr = sorted_ps[slicenr * num_ps_nodes + i];
            fprintf(fp, "%d %f %f %f\n", nodenr, x[nodenr], y[nodenr], z[nodenr]);
        }
        fprintf(fp, "\n\n");
    }
    fclose(fp);

// evaluate ss
#ifdef _WIN32
    sprintf(fn, "%s\\ss.txt", getenv("TMP"));
#else
    sprintf(fn, "/tmp/ss.txt");
#endif
    if ((fp = fopen(fn, "w+")) == NULL)
    {
        fprintf(stderr, "couldn't open file '%s'!\n", fn);
    }

    for (int slicenr = 0; slicenr < num_slices; slicenr++)
    {
        for (i = 0; i < num_ss_nodes; i++)
        {
            nodenr = sorted_ss[slicenr * num_ss_nodes + i];
            fprintf(fp, "%d %f %f %f\n", nodenr, x[nodenr], y[nodenr], z[nodenr]);
        }
        fprintf(fp, "\n\n");
    }
    fclose(fp);

    // get distribution (mesh compression) between hub and shroud
    // 0 .. 1
    // we extract it from leading edge
    float *meridional_radius = new float[num_slices];
    float dr;
    meridional_radius[0] = 0.;
    int lenode[2];

    for (i = 1; i < num_slices; i++)
    {
        lenode[0] = sorted_ps[(i - 1) * num_ps_nodes];
        lenode[1] = sorted_ps[i * num_ps_nodes];
        r1 = sqrt(sqr(x[lenode[0]]) + sqr(y[lenode[0]]));
        r2 = sqrt(sqr(x[lenode[1]]) + sqr(y[lenode[1]]));
        dr = fabs(r1 - r2);
        meridional_radius[i] = meridional_radius[i - 1] + dr;
    }

    float compression_factor = 1. / meridional_radius[num_slices - 1];

    // normalize compression
    for (i = 0; i < num_slices; i++)
    {
        meridional_radius[i] *= compression_factor;
    }

// evaluate meridional compression
#ifdef _WIN32
    sprintf(fn, "%s\\compression.txt", getenv("TMP"));
#else
    sprintf(fn, "/tmp/compression.txt");
#endif
    if ((fp = fopen(fn, "w+")) == NULL)
    {
        fprintf(stderr, "couldn't open file '%s'!\n", fn);
    }

    for (i = 0; i < num_slices; i++)
    {
        fprintf(fp, "%d %f\n", i, meridional_radius[i]);
    }
    fclose(fp);

    // combine pressure side and suction side to one array (for texture coord extraction)
    int num_psss_nodes = num_ss_nodes - 2 + num_ps_nodes;
    int *sorted_blade = new int[num_psss_nodes * num_slices];

    for (int slicenr = 0; slicenr < num_slices; slicenr++)
    {
        pos = 0;
        for (i = 0; i < num_ps_nodes; i++)
        {
            sorted_blade[slicenr * num_psss_nodes + pos] = sorted_ps[slicenr * num_ps_nodes + i];
            pos++;
        }
        for (i = num_ss_nodes - 2; i > 0; i--)
        {
            sorted_blade[slicenr * num_psss_nodes + pos] = sorted_ss[slicenr * num_ss_nodes + i];
            pos++;
        }
    }

    // now we are able to interpolate equidistant slices from our ss / ps slices
    // length along blade streamwise
    float *blade_s = new float[num_psss_nodes * num_slices];
    // length along blade between hub and shroud
    float *blade_u = new float[num_psss_nodes * num_slices];
    // length along blade streamwise (will run from 0 .. 1)
    float *blade_s_norm = new float[num_psss_nodes * num_slices];
    float *slice_maxLength = new float[num_slices];

    // calc streamwise length
    float ds;
    int node0, node1;
    for (i = 0; i < num_slices; i++)
    {
        blade_s[i * num_psss_nodes + 0] = 0.;
        blade_u[i * num_psss_nodes + 0] = meridional_radius[i];
        for (j = 1; j < num_psss_nodes; j++)
        {
            blade_u[i * num_psss_nodes + j] = meridional_radius[i];

            node1 = sorted_blade[i * num_psss_nodes + j];
            node0 = sorted_blade[i * num_psss_nodes + j - 1];
            ds = sqr(x[node1] - x[node0]);
            ds += sqr(y[node1] - y[node0]);
            ds += sqr(z[node1] - z[node0]);
            ds = sqrt(ds);
            blade_s[i * num_psss_nodes + j] = blade_s[i * num_psss_nodes + j - 1] + ds;
        }
        slice_maxLength[i] = blade_s[i * num_psss_nodes + num_psss_nodes - 1];
    }

    // normalize lengths
    float normalizeFactor;
    for (i = 0; i < num_slices; i++)
    {
        normalizeFactor = 1. / slice_maxLength[i];
        for (j = 0; j < num_psss_nodes; j++)
        {
            blade_s_norm[i * num_psss_nodes + j] = blade_s[i * num_psss_nodes + j] * normalizeFactor;
        }
    }

    int numNodes = 0;
    for (i = 0; i < num_slices; i++)
        for (j = 0; j < num_psss_nodes; j++)
            if (sorted_blade[i * num_psss_nodes + j] > numNodes)
                numNodes = sorted_blade[i * num_psss_nodes + j];

    int *vert = new int[numNodes + 1];
    for (int index = 0; index < numNodes; index++)
        vert[index] = index;

    float **coords = new float *[2];
    coords[0] = new float[numNodes + 1];
    coords[1] = new float[numNodes + 1];

    for (i = 0; i < num_slices; i++)
    {
        for (j = 0; j < num_psss_nodes; j++)
        {
            nodenr = sorted_blade[i * num_psss_nodes + j];
            fprintf(fp, "%d %f %f %f %f %f\n", nodenr, x[nodenr], y[nodenr], z[nodenr], blade_s_norm[i * num_psss_nodes + j], blade_u[i * num_psss_nodes + j]);
            coords[0][nodenr] = blade_s_norm[i * num_psss_nodes + j];
            coords[1][nodenr] = blade_u[i * num_psss_nodes + j];
        }
    }

    coObjInfo info = textureOutPort->getNewObjectInfo();
    char fileName[256];
    getname(fileName, "share/covise/materials/noise.png", NULL);
    coImage *image = new coImage(fileName);
    char *objName = new char[strlen(info.getName()) + 5];
    sprintf(objName, "%s_PI", textureOutPort->getObjName());

    coDoPixelImage *pixelImage = new coDoPixelImage(objName, image->getWidth(), image->getHeight(), image->getNumChannels(), image->getNumChannels(), (const char *)image->getBitmap(0));
    delete[] objName;

    coDoTexture *texture = new coDoTexture(info, pixelImage, 0, image->getNumChannels(), 0, numNodes, vert, numNodes, coords);

    texture->addAttribute("SHADER", "bladelic");
    texture->addAttribute("WRAP_MODE", "repeat");
    texture->addAttribute("MIN_FILTER", "linear");
    texture->addAttribute("MAG_FILTER", "linear");

    textureOutPort->setCurrentObject(texture);
/*
   delete[] coords[0];
   delete[] coords[1];
   delete[] coords;
   delete[] vert;
*/
#ifdef _WIN32
    sprintf(fn, "%s\\textureCoords.txt", getenv("TMP"));
#else
    sprintf(fn, "/tmp/textureCoords.txt");
#endif
    if ((fp = fopen(fn, "w+")) == NULL)
    {
        fprintf(stderr, "couldn't open file '%s'!\n", fn);
    }

    fprintf(fp, "#nodenr  x[nodenr]  y[nodenr]  z[nodenr]  s[nodenr]  u[nodenr]\n");
    for (i = 0; i < num_slices; i++)
    {
        for (j = 0; j < num_psss_nodes; j++)
        {
            nodenr = sorted_blade[i * num_psss_nodes + j];
            fprintf(fp, "%d %f %f %f %f %f\n", nodenr, x[nodenr], y[nodenr], z[nodenr], blade_s_norm[i * num_psss_nodes + j], blade_u[i * num_psss_nodes + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    // free memory
    delete[] nodecounter;
    delete[] used_hub_shroud_nodes;
    delete[] node_polygons;
    delete[] node_polygon_pointer;
    delete[] nodes_used;
    delete[] sorted_ps;
    delete[] sorted_ss;
    delete[] meridional_radius;
    delete[] blade_s;
    delete[] blade_u;
    delete[] blade_s_norm;
    delete[] slice_maxLength;
    delete[] sorted_blade;

    hubshroud_nodes.clear();
    outer_poly_conn_list.clear();
    hubshroud1.clear();
    hubshroud2.clear();
    shroud_side1.clear();
    shroud_side2.clear();

    return (SUCCESS);
}

int ExtractTexCoords::get_neighbour_node(int current_node, int *nodecounter, int *outer_poly_conn_list, int num_outer_polys, int *used_hub_shroud_nodes, float *)
{
    // we are looking for neighbour nodes of current_node in outer_poly_conn_list
    // there are usually two polygons containing current_node
    // those polygons have one node with nodecounter[node]==2 which is not current_node
    // the polygon is the next polygon if the other node with nodecounter[node]==2 is unused
    // used_hub_shroud_nodes[node] == 0
    // in the beginning, there might be two nodes fulfilling this condition
    // we simply take the first one as is is unimportant whether we run clockwise or counter-clockwise around the blade

    // this will be the correct node (return value)
    int next_node;

    // look for polygons containing current_node
    int polygon[3];
    int hit = 0;
    int j;

    for (int i = 0; i < num_outer_polys; i++)
    {
        for (j = 0; j < 4; j++)
        {
            if (outer_poly_conn_list[4 * i + j] == current_node)
            {
                polygon[hit] = i;
                hit++;
                //continue;
            }
        }
    }

#ifdef DEBUG
    if (hit == 0)
    {
        fprintf(stderr, "error! current_node %d not found in outer_poly_conn_list!\n", current_node);
    }
    else if (hit == 1)
    {
        fprintf(stderr, "node %d is in outer_polygon %d\n", current_node, polygon[0]);
        fprintf(stderr, "%d %d %d %d\n", outer_poly_conn_list[4 * polygon[0] + 0],
                outer_poly_conn_list[4 * polygon[0] + 1],
                outer_poly_conn_list[4 * polygon[0] + 2],
                outer_poly_conn_list[4 * polygon[0] + 3]);
    }
    else if (hit == 2)
    {
        fprintf(stderr, "node %d is in outer_polygon %d and %d\n", current_node, polygon[0], polygon[1]);
        fprintf(stderr, "%d %d %d %d\n", outer_poly_conn_list[4 * polygon[0] + 0],
                outer_poly_conn_list[4 * polygon[0] + 1],
                outer_poly_conn_list[4 * polygon[0] + 2],
                outer_poly_conn_list[4 * polygon[0] + 3]);
        fprintf(stderr, "%d %d %d %d\n", outer_poly_conn_list[4 * polygon[1] + 0],
                outer_poly_conn_list[4 * polygon[1] + 1],
                outer_poly_conn_list[4 * polygon[1] + 2],
                outer_poly_conn_list[4 * polygon[1] + 3]);
    }
    else if (hit == 3)
    {
        fprintf(stderr, "node %d is in outer_polygon %d, %d and %d\n", current_node, polygon[0], polygon[1], polygon[1]);
        fprintf(stderr, "%d %d %d %d\n", outer_poly_conn_list[4 * polygon[0] + 0],
                outer_poly_conn_list[4 * polygon[0] + 1],
                outer_poly_conn_list[4 * polygon[0] + 2],
                outer_poly_conn_list[4 * polygon[0] + 3]);
        fprintf(stderr, "%d %d %d %d\n", outer_poly_conn_list[4 * polygon[0] + 0],
                outer_poly_conn_list[4 * polygon[1] + 1],
                outer_poly_conn_list[4 * polygon[1] + 2],
                outer_poly_conn_list[4 * polygon[1] + 3]);
        fprintf(stderr, "%d %d %d %d\n", outer_poly_conn_list[4 * polygon[1] + 0],
                outer_poly_conn_list[4 * polygon[2] + 1],
                outer_poly_conn_list[4 * polygon[2] + 2],
                outer_poly_conn_list[4 * polygon[2] + 3]);
    }
#endif

    int num_maybe_neighbours = 0;
    int maybe_neighbour[12];

    if (hit == 1)
    {
        num_maybe_neighbours = 4;
        maybe_neighbour[0] = outer_poly_conn_list[4 * polygon[0] + 0];
        maybe_neighbour[1] = outer_poly_conn_list[4 * polygon[0] + 1];
        maybe_neighbour[2] = outer_poly_conn_list[4 * polygon[0] + 2];
        maybe_neighbour[3] = outer_poly_conn_list[4 * polygon[0] + 3];
    }
    if (hit == 2)
    {
        num_maybe_neighbours = 8;
        maybe_neighbour[0] = outer_poly_conn_list[4 * polygon[0] + 0];
        maybe_neighbour[1] = outer_poly_conn_list[4 * polygon[0] + 1];
        maybe_neighbour[2] = outer_poly_conn_list[4 * polygon[0] + 2];
        maybe_neighbour[3] = outer_poly_conn_list[4 * polygon[0] + 3];
        maybe_neighbour[4] = outer_poly_conn_list[4 * polygon[1] + 0];
        maybe_neighbour[5] = outer_poly_conn_list[4 * polygon[1] + 1];
        maybe_neighbour[6] = outer_poly_conn_list[4 * polygon[1] + 2];
        maybe_neighbour[7] = outer_poly_conn_list[4 * polygon[1] + 3];
    }
    if (hit == 3)
    {
        num_maybe_neighbours = 12;
        maybe_neighbour[0] = outer_poly_conn_list[4 * polygon[0] + 0];
        maybe_neighbour[1] = outer_poly_conn_list[4 * polygon[0] + 1];
        maybe_neighbour[2] = outer_poly_conn_list[4 * polygon[0] + 2];
        maybe_neighbour[3] = outer_poly_conn_list[4 * polygon[0] + 3];
        maybe_neighbour[4] = outer_poly_conn_list[4 * polygon[1] + 0];
        maybe_neighbour[5] = outer_poly_conn_list[4 * polygon[1] + 1];
        maybe_neighbour[6] = outer_poly_conn_list[4 * polygon[1] + 2];
        maybe_neighbour[7] = outer_poly_conn_list[4 * polygon[1] + 3];
        maybe_neighbour[8] = outer_poly_conn_list[4 * polygon[2] + 0];
        maybe_neighbour[9] = outer_poly_conn_list[4 * polygon[2] + 1];
        maybe_neighbour[10] = outer_poly_conn_list[4 * polygon[2] + 2];
        maybe_neighbour[11] = outer_poly_conn_list[4 * polygon[2] + 3];
    }

    // check nodes in polygon[0] and polygon[1]
    // the correct node must fulfill the following conditions:
    // 1. neighbour node is not = current_node
    // 2. neighbour node's nodecounter must be 2
    // 3. used_hub_shroud_nodes must be 0 for neighbor node
    next_node = -1;

    for (int i = 0; i < num_maybe_neighbours; i++)
    {
        if ((maybe_neighbour[i] == current_node) || (nodecounter[maybe_neighbour[i]] != 2) || (used_hub_shroud_nodes[maybe_neighbour[i]] != 0))
        {
            continue; // check next maybe_neighbour node
        }
        next_node = maybe_neighbour[i];
    }

    return (next_node);
}

float ExtractTexCoords::distance(int nodenr1, int nodenr2, float *x, float *y, float *z)
{
    return (sqrt(sqr(x[nodenr2] - x[nodenr1]) + sqr(y[nodenr2] - y[nodenr1]) + sqr(z[nodenr2] - z[nodenr1])));
}

int ExtractTexCoords::get_next_slice_node(int current_node, int *node_polygons, int *node_polygon_pointer, int *nodes_used, int num_corners, int num_points, int *corner_list, int *polygon_list)
{
    int next_node;

    // get first polygon in which our node participates where not all nodes are used
    int num_nodepols;
    int polynr;

    int j, k;
    int used;

    int neighbour[2];

    // get number of polygons our node participates in
    if (current_node < (num_points - 1))
    {
        num_nodepols = node_polygon_pointer[current_node + 1] - node_polygon_pointer[current_node];
    }
    else
    {
        num_nodepols = num_corners - node_polygon_pointer[current_node];
    }

    /*
    * // DEBUG
    * if (current_node==-1)
    * {
    *    fprintf(stderr,"node_polygon_pointer[%d]=%d\n",current_node,node_polygon_pointer[current_node]);
    *    fprintf(stderr,"node %d takes part in %d polygons:",current_node,num_nodepols);
    * }
    */
    j = 0;
    do
    {
        /*
       *    // DEBUG
       *    if (current_node==-1)
       *    {
       *       fprintf(stderr," %d", node_polygons[node_polygon_pointer[current_node]+j]);
       *    }
       */
        polynr = node_polygons[node_polygon_pointer[current_node] + j];
        used = 0;
        for (k = 0; k < 4; k++) // polygon must have four nodes, we already checked that
        {
            // is 0 or 1
            if (polynr < num_polygons)
                used += nodes_used[corner_list[polygon_list[polynr] + k]];
        }
        if (j > num_nodepols)
        {
            fprintf(stderr, "no polygon with less than 4 unused nodes found! Error!\n");
        }
        j++;
    } while (used == 4);
    // DEBUG
    if (current_node == -1)
    {
        fprintf(stderr, "\n");
    }

    // polynr is the polygon we are looking for

    // the unused neighbour of current_node is the node we are looking for
    // get the two neighbours

    int pos = 0;
    while (corner_list[polygon_list[polynr] + pos] != current_node)
    {
        pos++;
    }
    neighbour[0] = corner_list[polygon_list[polynr] + (pos + 1) % 4];
    neighbour[1] = corner_list[polygon_list[polynr] + (pos - 1) % 4];

    if ((nodes_used[neighbour[0]] == 0) && (nodes_used[neighbour[1]] == 0))
    {
        fprintf(stderr, "Error! two unused neighbours found!\n");
    }
    if ((nodes_used[neighbour[0]] == 1) && (nodes_used[neighbour[1]] == 1))
    {
        fprintf(stderr, "Error! no unused neighbours found!\n");
    }
    if (nodes_used[neighbour[0]] == 0)
    {
        next_node = neighbour[0];
    }
    else
    {
        next_node = neighbour[1];
    }

    return (next_node);
}

MODULE_MAIN(Tools, ExtractTexCoords)
