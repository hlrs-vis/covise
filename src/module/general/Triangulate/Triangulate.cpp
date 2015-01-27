/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                   	      (C)2008 HLRS  **
**                                                                        **
** Description: Triangulate points to polygons (2D)                       **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include "Triangulate.h"
#include "tetgen.h"

#include "defs.h"
#include "decl.h"
#include "edge.h"
#include "dc.h"

extern point *p_array;

Triangulate::Triangulate(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Delaunay Triangulator")
{

    // the input ports
    p_inpoints = addInputPort("inmesh", "Points|TriangleStrips|Polygons|UnstructuredGrid|Vec3", "input points");
    p_boundaries1 = addInputPort("boundary_polygons_1", "Polygons", "domain boundary to be preserved");
    p_boundaries1->setRequired(0);
    p_boundaries2 = addInputPort("boundary_polygons_2", "Polygons", "domain boundary to be preserved");
    p_boundaries2->setRequired(0);
    p_boundaries3 = addInputPort("boundary_polygons_3", "Polygons", "domain boundary to be preserved");
    p_boundaries3->setRequired(0);
    p_boundaries4 = addInputPort("boundary_polygons_4", "Polygons", "domain boundary to be preserved");
    p_boundaries4->setRequired(0);

    // the output ports
    p_outmesh = addOutputPort("GridOut0", "Polygons|UnstructuredGrid", "output triangles");
    p_combinedPolys = addOutputPort("combined_polys", "Polygons", "combined polys");

    // parameters
    const char *case_labels[] = { "2D", "3D" };
    p_case = addChoiceParam("case", "triangulate (2D, points->polygons) or tetrahedronize (3D, points->tetrahedrons)");
    p_case->setValue(2, case_labels, 0);

    p_minAngle = addFloatParam("MinAngle", "minimum angle for outer triangles (if angle is smaller, triangles are removed)");
    p_minAngle->setValue(15.);

    const char *plane_labels[] = { "XY", "YZ", "ZX" };
    p_plane = addChoiceParam("plane", "plane to triangulate in (other axis will be neglected)");
    p_plane->setValue(3, plane_labels, 0);
}

Triangulate::~Triangulate()
{
}

void Triangulate::postInst()
{
    p_minAngle->enable();
    p_plane->enable();
}

void Triangulate::param(const char *paramname, bool /*inMapLoading*/)
{
    if (strcmp(paramname, p_case->getName()) == 0)
    {
        switch (p_case->getValue())
        {
        case TWOD:
            p_minAngle->enable();
            p_plane->enable();
            break;
        case THREED:
            p_minAngle->disable();
            p_plane->disable();
            break;
        }
    }
}

// =======================================================

int Triangulate::compute(const char *)
{
    tetgenio in, out;

    float *xCoord, *yCoord, *zCoord; // coordinate lists
    int nCoord; // number of vertices list (=number of measurement points)

    int *conn; // element, connectivity and type list
    coDoPolygons *polys = NULL;
    coDoUnstructuredGrid *grid = NULL;

    // for delaunay triangulation
    edge *l_cw, *r_ccw;
    point **p_sorted, **p_temp;

    // read coordinates
    const coDistributedObject *inObj = p_inpoints->getCurrentObject();

    if (inObj->isType("POINTS"))
    {
        nCoord = ((const coDoPoints *)inObj)->getNumPoints();
        ((coDoPoints *)inObj)->getAddresses(&xCoord, &yCoord, &zCoord);
    }
    else if (dynamic_cast<const coDoVec3 *>(inObj))
    {
        nCoord = ((coDoVec3 *)inObj)->getNumPoints();
        ((coDoVec3 *)inObj)->getAddresses(&xCoord, &yCoord, &zCoord);
    }
    else if ((inObj->isType("POLYGN")) || (inObj->isType("TRIANG")))
    {
        int *cl, *pl;
        ((coDoPolygons *)inObj)->getAddresses(&xCoord, &yCoord, &zCoord, &cl, &pl);
        nCoord = ((coDoPolygons *)inObj)->getNumPoints();
    }
    else if (inObj->isType("UNSGRD"))
    {
        int *dummy;
        ((coDoUnstructuredGrid *)inObj)->getAddresses(&dummy, &dummy, &xCoord, &yCoord, &zCoord);
        ((coDoUnstructuredGrid *)inObj)->getGridSize(dummy, dummy, &nCoord);
    }
    else
    {
        sendError("unsupported input type %s!\n", inObj->getType());
        return STOP_PIPELINE;
    }

    if (p_case->getValue() == TWOD)
    {

        // for triangulation
        alloc_memory(nCoord);

        if (p_plane->getValue() == XY)
        {
            // Initialise entry edge pointers and coordinates
            for (int i = 0; i < nCoord; i++)
            {
                p_array[i].x = xCoord[i];
                p_array[i].y = yCoord[i];
                p_array[i].entry_pt = NULL;
            }
        }
        else if (p_plane->getValue() == YZ)
        {
            // Initialise entry edge pointers and coordinates
            for (int i = 0; i < nCoord; i++)
            {
                p_array[i].x = yCoord[i];
                p_array[i].y = zCoord[i];
                p_array[i].entry_pt = NULL;
            }
        }
        else if (p_plane->getValue() == ZX)
        {
            // Initialise entry edge pointers and coordinates
            for (int i = 0; i < nCoord; i++)
            {
                p_array[i].x = zCoord[i];
                p_array[i].y = xCoord[i];
                p_array[i].entry_pt = NULL;
            }
        }

        // Sort for delaunay
        p_sorted = (point **)malloc((unsigned)nCoord * sizeof(point *));
        if (p_sorted == NULL)
            sendError("triangulate: not enough memory\n");
        p_temp = (point **)malloc((unsigned)nCoord * sizeof(point *));
        if (p_temp == NULL)
            sendError("triangulate: not enough memory\n");
        for (int i = 0; i < nCoord; i++)
            p_sorted[i] = p_array + i;
        merge_sort(p_sorted, p_temp, 0, nCoord - 1);

        free((char *)p_temp);

        // Triangulate
        divide(p_sorted, 0, nCoord - 1, &l_cw, &r_ccw);

        free((char *)p_sorted);

        int nElem;

        // remove triangles at outer edges (condition: 2 angles < min_angle)
        removeOuterEdges(nCoord);
        conn = get_triangles(nCoord, &nElem);

        cerr << "triangulated " << nElem << " polygons" << endl;

        int *elem = new int[nElem];

        for (int i = 0; i < nElem; i++)
        {
            elem[i] = 3 * i;
        }

        polys = new coDoPolygons(p_outmesh->getObjName(), nCoord, xCoord, yCoord, zCoord, nElem * 3, conn, nElem, elem);
        p_outmesh->setCurrentObject(polys);

        // free triangulator memory
        free_memory();
    }
    else // (p_case->getValue()==THREED)
    {
        // combining the four polygon input ports to one polygon object
        coDoPolygons *combinedPolys = combine_polygons(nCoord, xCoord, yCoord, zCoord);
        int nCombinedPolygons = combinedPolys->getNumPolygons();
        float *x, *y, *z;
        int *combinedCorners, *combinedPolygons;
        combinedPolys->getAddresses(&x, &y, &z, &combinedCorners, &combinedPolygons);

        p_combinedPolys->setCurrentObject(combinedPolys);

        // fill the tetgen data structures
        tetgenio::facet *f;
        tetgenio::polygon *p;

        in.numberofpoints = nCoord;
        in.pointlist = new double[in.numberofpoints * 3];

        in.firstnumber = 0;

        for (int i = 0; i < nCoord; i++)
        {
            in.pointlist[3 * i + 0] = xCoord[i];
            in.pointlist[3 * i + 1] = yCoord[i];
            in.pointlist[3 * i + 2] = zCoord[i];
        }

        if ((p_boundaries1->isConnected()) || (p_boundaries2->isConnected()) || (p_boundaries3->isConnected()) || (p_boundaries4->isConnected()))
        {
            if (nCombinedPolygons)
                in.numberoffacets = 1;

            // we try to pack everything in one facet, let's see what happens
            in.facetlist = new tetgenio::facet[in.numberoffacets];
            //in.facetmarkerlist = new int[in.numberoffacets];

            for (int i = 0; i < in.numberoffacets; i++)
            {
                f = &in.facetlist[0];
                f->numberofpolygons = 2 * nCombinedPolygons;
                f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
                f->numberofholes = 0;
                f->holelist = NULL;

                for (int j = 0; j < nCombinedPolygons; j++)
                {
                    // here we treat a 4-node polygon (translated to a triangle)

                    p = &f->polygonlist[2 * j + 0];
                    p->numberofvertices = 3;
                    p->vertexlist = new int[p->numberofvertices];
                    p->vertexlist[0] = combinedCorners[combinedPolygons[j] + 0];
                    p->vertexlist[1] = combinedCorners[combinedPolygons[j] + 1];
                    p->vertexlist[2] = combinedCorners[combinedPolygons[j] + 2];

                    p = &f->polygonlist[2 * j + 1];
                    p->numberofvertices = 3;
                    p->vertexlist = new int[p->numberofvertices];
                    p->vertexlist[0] = combinedCorners[combinedPolygons[j] + 0];
                    p->vertexlist[1] = combinedCorners[combinedPolygons[j] + 2];
                    p->vertexlist[2] = combinedCorners[combinedPolygons[j] + 3];
                }
            }
        }

        // calling tetrahedralize
        if (in.numberoffacets)
        {
            cerr << "calling tetrahedralize with a point cloud and a list of boundary polygons" << endl;
            tetrahedralize("pYY", &in, &out);
        }
        else
        {
            cerr << "calling tetrahedralize with a point cloud" << endl;
            tetrahedralize("", &in, &out);
        }

        int numElem = out.numberoftetrahedra;
        cerr << "triangulated " << numElem << " tetras" << endl;

        /*
      // what we have in "out"
      int *tetrahedronlist;
      REAL *tetrahedronattributelist;
      REAL *tetrahedronvolumelist;
      int *neighborlist;
      int numberoftetrahedra;
      int numberofcorners;      
*/

        int *conn = new int[numElem * 4];
        int *tl = new int[numElem];
        int *elem = new int[numElem];

        for (int i = 0; i < numElem; i++)
        {
            tl[i] = TYPE_TETRAHEDER;
            elem[i] = 4 * i;
            conn = out.tetrahedronlist;
        }

        grid = new coDoUnstructuredGrid(p_outmesh->getObjName(), numElem, numElem * 4, nCoord, elem, conn, xCoord, yCoord, zCoord, tl);
        p_outmesh->setCurrentObject(grid);
    }

    return SUCCESS;
}

int *Triangulate::get_triangles(int n, int *n_tria)
{
    edge *e_start, *e, *next;
    point *u, *v, *w;
    int i;
    point *t;

    int n_triangles = 0;

    int *triangles = new int[2 * 3 * n]; // we assume that n points do not produce more than 2*n triangles

    for (i = 0; i < n; i++)
    {
        u = &p_array[i];
        e_start = e = u->entry_pt;
        do
        {
            v = Other_point(e, u);
            if (u < v)
            {
                next = Next(e, u);
                w = Other_point(next, u);

                if (u < w)
                    if (Identical_refs(Next(next, w), Prev(e, v)))
                    {
                        // Triangle
                        if (v > w)
                        {
                            t = v;
                            v = w;
                            w = t;
                        }
                        //if (printf("%d %d %d\n", u - p_array, v - p_array, w - p_array) == EOF)
                        //    sendError("Error printing results\n");
                        if (n_triangles > 2 * n)
                        {
                            fprintf(stderr, "ooops, not enough memory allocated for triangles!\n");
                            *n_tria = n_triangles;
                            return triangles;
                        }
                        triangles[3 * n_triangles + 0] = u - p_array;
                        triangles[3 * n_triangles + 1] = v - p_array;
                        triangles[3 * n_triangles + 2] = w - p_array;
                        //cerr << "triangle " << n_triangles << ": " << u - p_array << ", " << v - p_array << ", " << w - p_array << endl;
                        n_triangles++;
                    }
            }

            // Next edge around u
            e = Next(e, u);
        } while (!Identical_refs(e, e_start));
    }

    *n_tria = n_triangles;

    return (triangles);
}

void Triangulate::removeOuterEdges(int n)
{
    edge *e_start, *e, *e2;
    point *u, *v, *w;
    int i;
    float min_angle = cos(p_minAngle->getValue() / 180.0 * M_PI);
    bool removed;
    do
    {
        removed = false;
        for (i = 0; i < n; i++)
        {
            u = &p_array[i];
            e_start = e = u->entry_pt;
            if (Next(Next(e, u), u) != e_start) // only remove one if we have more than two edges
            {
                do
                {
                    e2 = Next(e, u);
                    v = Other_point(e, u);
                    w = Other_point(e2, u);
                    if (!isConnected(v, w)) // this edge is an outer edge
                    {
                        if (checkAndRemove(u, e, Prev(e, u), min_angle))
                        {
                            removed = true;
                            if (e == e_start)
                                break;
                        }
                        else if (checkAndRemove(u, e2, Next(e, u), min_angle))
                        {
                            removed = true;
                            if (e == e_start)
                                break;
                        }
                    }
                    e = e2;
                } while (e != e_start);
            }
        }
    } while (removed);
}

coDoPolygons *Triangulate::combine_polygons(int nCoord, float *xCoord, float *yCoord, float *zCoord)
{
    coDoPolygons *boundaries1 = NULL;
    coDoPolygons *boundaries2 = NULL;
    coDoPolygons *boundaries3 = NULL;
    coDoPolygons *boundaries4 = NULL;

    coDoPolygons *Polys;

    float *boundx, *boundy, *boundz;
    int *bound1_conn, *bound1_elem;
    int bound1_npol = 0;
    int bound1_npoints = 0;
    int bound1_ncorners = 0;
    if (p_boundaries1->isConnected())
    {
        boundaries1 = (coDoPolygons *)p_boundaries1->getCurrentObject();
        boundaries1->getAddresses(&boundx, &boundy, &boundz, &bound1_conn, &bound1_elem);
        bound1_npol = boundaries1->getNumPolygons();
        bound1_npoints = boundaries1->getNumPoints();
        bound1_ncorners = boundaries1->getNumVertices();
    }

    int *bound2_conn, *bound2_elem;
    int bound2_npol = 0;
    int bound2_npoints = 0;
    int bound2_ncorners = 0;
    if (p_boundaries2->isConnected())
    {
        boundaries2 = (coDoPolygons *)p_boundaries2->getCurrentObject();
        boundaries2->getAddresses(&boundx, &boundy, &boundz, &bound2_conn, &bound2_elem);
        bound2_npol = boundaries2->getNumPolygons();
        bound2_npoints = boundaries2->getNumPoints();
        bound2_ncorners = boundaries2->getNumVertices();
    }

    int *bound3_conn, *bound3_elem;
    int bound3_npol = 0;
    int bound3_npoints = 0;
    int bound3_ncorners = 0;
    if (p_boundaries3->isConnected())
    {
        boundaries3 = (coDoPolygons *)p_boundaries3->getCurrentObject();
        boundaries3->getAddresses(&boundx, &boundy, &boundz, &bound3_conn, &bound3_elem);
        bound3_npol = boundaries3->getNumPolygons();
        bound3_npoints = boundaries3->getNumPoints();
        bound3_ncorners = boundaries3->getNumVertices();
    }
    int *bound4_conn, *bound4_elem;
    int bound4_npol = 0;
    int bound4_npoints = 0;
    int bound4_ncorners = 0;
    if (p_boundaries4->isConnected())
    {
        boundaries4 = (coDoPolygons *)p_boundaries4->getCurrentObject();
        boundaries4->getAddresses(&boundx, &boundy, &boundz, &bound4_conn, &bound4_elem);
        bound4_npol = boundaries4->getNumPolygons();
        bound4_npoints = boundaries4->getNumPoints();
        bound4_ncorners = boundaries4->getNumVertices();
    }
    // combined
    int nCombinedCorners = bound1_ncorners + bound2_ncorners + bound3_ncorners + bound4_ncorners;
    int nCombinedPolygons = bound1_npol + bound2_npol + bound3_npol + bound4_npol;
    int *combinedCorners;
    int *combinedPolygons;

    float *x, *y, *z;

    Polys = new coDoPolygons(p_combinedPolys->getObjName(), nCoord, nCombinedCorners, nCombinedPolygons);
    Polys->getAddresses(&x, &y, &z, &combinedCorners, &combinedPolygons);

    // let's waste some memory
    memcpy(x, xCoord, nCoord * sizeof(float));
    memcpy(y, yCoord, nCoord * sizeof(float));
    memcpy(z, zCoord, nCoord * sizeof(float));

    int polCounter = 0;
    int cornCounter = 0;
    int k;
    int pos = 0;
    if (p_boundaries1->isConnected())
    {
        for (k = 0; k < bound1_npol - 1; k++)
        {
            combinedPolygons[polCounter] = bound1_elem[k];
            for (int l = 0; l < bound1_elem[k + 1] - bound1_elem[k]; l++)
            {
                combinedCorners[cornCounter] = bound1_conn[bound1_elem[k] + l];
                cornCounter++;
            }
            polCounter++;
        }
        // and the last polygon
        combinedPolygons[polCounter] = bound1_elem[bound1_npol - 1];
        cerr << "nCorners(last)=" << bound1_ncorners - bound1_elem[bound1_npol - 1] << endl;
        for (int l = 0; l < (bound1_ncorners - bound1_elem[bound1_npol - 1]); l++)
        {
            combinedCorners[cornCounter] = bound1_conn[bound1_elem[k] + l];
            cornCounter++;
        }
        polCounter++;
    }
    pos = cornCounter;
    if (p_boundaries2->isConnected())
    {
        for (k = 0; k < bound2_npol - 1; k++)
        {
            combinedPolygons[polCounter] = pos + bound2_elem[k];
            for (int l = 0; l < bound2_elem[k + 1] - bound2_elem[k]; l++)
            {
                combinedCorners[cornCounter] = bound2_conn[bound2_elem[k] + l];
                cornCounter++;
            }
            polCounter++;
        }
        // and the last polygon
        combinedPolygons[polCounter] = pos + bound2_elem[bound2_npol - 1];
        cerr << "nCorners(last)=" << bound2_ncorners - bound2_elem[bound2_npol - 1] << endl;
        for (int l = 0; l < (bound2_ncorners - bound2_elem[bound2_npol - 1]); l++)
        {
            combinedCorners[cornCounter] = bound2_conn[bound2_elem[k] + l];
            cornCounter++;
        }
        polCounter++;
    }
    pos = cornCounter;
    if (p_boundaries3->isConnected())
    {
        for (k = 0; k < bound3_npol - 1; k++)
        {
            combinedPolygons[polCounter] = pos + bound3_elem[k];
            for (int l = 0; l < bound3_elem[k + 1] - bound3_elem[k]; l++)
            {
                combinedCorners[cornCounter] = bound3_conn[bound3_elem[k] + l];
                cornCounter++;
            }
            polCounter++;
        }
        // and the last polygon
        combinedPolygons[polCounter] = pos + bound3_elem[bound3_npol - 1];
        cerr << "nCorners(last)=" << bound3_ncorners - bound3_elem[bound3_npol - 1] << endl;
        for (int l = 0; l < (bound3_ncorners - bound3_elem[bound3_npol - 1]); l++)
        {
            combinedCorners[cornCounter] = bound3_conn[bound3_elem[k] + l];
            cornCounter++;
        }
        polCounter++;
    }
    pos = cornCounter;
    if (p_boundaries4->isConnected())
    {
        for (k = 0; k < bound4_npol - 1; k++)
        {
            combinedPolygons[polCounter] = pos + bound4_elem[k];
            for (int l = 0; l < bound4_elem[k + 1] - bound4_elem[k]; l++)
            {
                combinedCorners[cornCounter] = bound4_conn[bound4_elem[k] + l];
                cornCounter++;
            }
            polCounter++;
        }
        // and the last polygon
        combinedPolygons[polCounter] = pos + bound4_elem[bound4_npol - 1];
        cerr << "nCorners(last)=" << bound4_ncorners - bound4_elem[bound4_npol - 1] << endl;
        for (int l = 0; l < (bound4_ncorners - bound4_elem[bound4_npol - 1]); l++)
        {
            combinedCorners[cornCounter] = bound4_conn[bound4_elem[k] + l];
            cornCounter++;
        }
        polCounter++;
    }

    int *usedNode = new int[nCoord];
    memset(usedNode, 0, nCoord * sizeof(int));

    for (int i = 0; i < nCombinedCorners; i++)
    {
        usedNode[combinedCorners[i]] = 1;
    }

    cerr << "polCounter=" << polCounter << endl;
    cerr << "nCombinedPolygons=" << nCombinedPolygons << endl;
    cerr << "cornCounter=" << cornCounter << endl;
    cerr << "nCombinedCorners=" << nCombinedCorners << endl;

    if (polCounter != nCombinedPolygons)
    {
        cerr << "error! polCounter!=nCombinedPolygons" << endl;
        return NULL;
    }
    if (cornCounter != nCombinedCorners)
    {
        cerr << "error! cornCounter!=nCombinedCorners" << endl;
        return NULL;
    }

    return Polys;
}

MODULE_MAIN(Filter, Triangulate)
