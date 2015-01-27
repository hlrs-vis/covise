/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2001	  	  **
 ** Description: FixUSG delets points which are more than one time 		  **
 **				 in an UnstructuredGrid 											     **
 **                                                                        **
 ** Author:                                                                **
 **                            Karin Müller                             	  **
 **                				Vircinity               						  **
 **                            Technologiezentrum                          **
 **                            70550 Stuttgart                             **
 ** Date:  01.10.01		                                                  **
\**************************************************************************/

#include "DeleteUnusedPoints.h"
#include <do/coDoPolygons.h>
#include <do/coDoLines.h>
#include <do/coDoUnstructuredGrid.h>

using namespace covise;

namespace covise
{
coDistributedObject *checkUSG(coDistributedObject *DistrObj, const coObjInfo &outInfo)
{
    coDistributedObject *NewDistrObj = NULL;
    // temp
    int *elemList; //tl

    float bbx1, bby1, bbz1;
    float bbx2, bby2, bbz2;

    int *coordInBox;
    int numCoordInBox;

    // counters
    int num_elem, i;

    int num_coord;
    int num_conn;

    float *xcoord, *ycoord, *zcoord;
    int *connList; // vl

    int maxCoord;

    // check for errors
    if (DistrObj == NULL)
    {
        //Covise::sendError("No distributed object at checkUSG();");
        return 0;
    }

    // get parameters
    maxCoord = 50;

    // get input
    if (coDoPolygons *poly_in = dynamic_cast<coDoPolygons *>(DistrObj))
    {
        num_coord = poly_in->getNumPoints();
        num_conn = poly_in->getNumVertices();
        poly_in->getAddresses(&xcoord, &ycoord, &zcoord, &connList, &elemList);
    }
    else if (coDoLines *lines_in = dynamic_cast<coDoLines *>(DistrObj))
    {
        num_coord = lines_in->getNumPoints();
        num_conn = lines_in->getNumVertices();
        lines_in->getAddresses(&xcoord, &ycoord, &zcoord, &connList, &elemList);
    }
    else if (coDoUnstructuredGrid *grid_in = dynamic_cast<coDoUnstructuredGrid *>(DistrObj))
    {
        //   cells  connec     vertices
        grid_in->getGridSize(&num_elem, &num_conn, &num_coord);
        grid_in->getAddresses(&elemList, &connList, &xcoord, &ycoord, &zcoord);
    }
    else
    {
        fprintf(stderr, "FixUSG: wrong object-type\n");
        return (0);
    }

    /// create a Replace-List
    replBy = new int[num_coord];
    for (i = 0; i < num_coord; i++)
        replBy[i] = -2;

    numCoordToRemove = 0;
    coordInBox = new int[num_coord];
    if (!coordInBox)
    {
        fprintf(stderr, "FixUSG: failed to alloc array of %d ints (2)\n", num_coord);
        return (0);
    }
    numCoordInBox = 0;

    // the "starting" cell contains all USED coordinates
    // clear all flags -> no coordinates used at all
    for (i = 0; i < num_coord; i++)
        coordInBox[i] = 0;

    for (i = 0; i < num_conn; i++)
        coordInBox[connList[i]] = 1;

    // now remove the unused coordinates
    for (i = 0; i < num_coord; i++)
    {
        if (coordInBox[i])
        {
            // this one is used
            coordInBox[numCoordInBox] = i;
            numCoordInBox++;
        }
        else
        {
            // unused coordinate
            replBy[i] = -1;
            numCoordToRemove++;
        }
    }
    // what we do next depends on the chosen algorithm

    boundingBox(&xcoord, &ycoord, &zcoord, coordInBox, numCoordInBox,
                &bbx1, &bby1, &bbz1, &bbx2, &bby2, &bbz2);
    computeCell(xcoord, ycoord, zcoord,
                coordInBox, numCoordInBox, bbx1, bby1, bbz1,
                bbx2, bby2, bbz2, maxCoord, replBy, numCoordToRemove);

    // partially clean up
    delete[] coordInBox;

    // compute the working-lists
    computeWorkingLists(num_coord);

    // and finally remove the filtered coordinates
    NewDistrObj = filterCoordinates(DistrObj, outInfo, num_coord, xcoord, ycoord, zcoord);

    // clean up
    delete[] source2filtered;
    delete[] filtered2source;

    // status report
    //char buffer[64];
    //sprintf(buffer,"removed %d points",numCoordToRemove);
    //Covise::sendInfo(buffer);

    // done

    return NewDistrObj;
}

//=================================================================================

void boundingBox(float **x, float **y, float **z, int *c, int n,
                 float *bbx1, float *bby1, float *bbz1,
                 float *bbx2, float *bby2, float *bbz2)
{
    int i;
    float cx, cy, cz;

    *bbx1 = *bbx2 = (*x)[c[0]];
    *bby1 = *bby2 = (*y)[c[0]];
    *bbz1 = *bbz2 = (*z)[c[0]];

    for (i = 0; i < n; i++)
    {
        cx = (*x)[c[i]];
        cy = (*y)[c[i]];
        cz = (*z)[c[i]];

        // x
        if (cx < *bbx1)
            *bbx1 = cx;
        else if (cx > *bbx2)
            *bbx2 = cx;

        // y
        if (cy < *bby1)
            *bby1 = cy;
        else if (cy > *bby2)
            *bby2 = cy;

        // z
        if (cz < *bbz1)
            *bbz1 = cz;
        else if (cz > *bbz2)
            *bbz2 = cz;
    }
    return;
}

//=================================================================================
int isEqual(float x1, float y1, float z1,
            float x2, float y2, float z2, float dist)
{
    int r;
    float d;

    if (dist == 0.0)
        r = (x1 == x2 && y1 == y2 && z1 == z2);
    else
    {
        d = ((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)) + ((z2 - z1) * (z2 - z1));
        if (d <= dist)
            r = 1;
        else
            r = 0;
    }

    return (r);
}

//=================================================================================

int getOctant(float x, float y, float z, float ox, float oy, float oz)
{
    int r;

    // ... the origin

    if (x < ox) // behind

        if (y < oy) // below
            if (z < oz) // left
                r = 6;
            else // right
                r = 7;
        else // above
            if (z < oz) // left
            r = 4;
        else // right
            r = 5;

    else // in front of
        if (y < oy) // below
        if (z < oz) // left
            r = 2;
        else // right
            r = 3;
    else // above
        if (z < oz) // left
        r = 0;
    else // right
        r = 1;

    // done
    return (r);
}

//=================================================================================

void getOctantBounds(int o, float ox, float oy, float oz,
                     float bbx1, float bby1, float bbz1,
                     float bbx2, float bby2, float bbz2,
                     float *bx1, float *by1, float *bz1,
                     float *bx2, float *by2, float *bz2)
{
    switch (o)
    {
    case 0: // left, above, front
        *bx1 = bbx1;
        *by1 = oy;
        *bz1 = oz;
        *bx2 = ox;
        *by2 = bby2;
        *bz2 = bbz2;
        break;
    case 1: // right, above, front
        *bx1 = ox;
        *by1 = oy;
        *bz1 = oz;
        *bx2 = bbx2;
        *by2 = bby2;
        *bz2 = bbz2;
        break;
    case 2: // left, below, front
        *bx1 = bbx1;
        *by1 = bby1;
        *bz1 = oz;
        *bx2 = ox;
        *by2 = oy;
        *bz2 = bbz2;
        break;
    case 3: // right, below, front
        *bx1 = ox;
        *by1 = bby1;
        *bz1 = oz;
        *bx2 = bbx2;
        *by2 = oy;
        *bz2 = bbz2;
        break;
    case 4: // left, above, behind
        *bx1 = bbx1;
        *by1 = oy;
        *bz1 = bbz1;
        *bx2 = ox;
        *by2 = bby2;
        *bz2 = oz;
        break;
    case 5: // right, above, behind
        *bx1 = ox;
        *by1 = oy;
        *bz1 = bbz1;
        *bx2 = bbx2;
        *by2 = bby2;
        *bz2 = oz;
        break;
    case 6: // left, below, behind
        *bx1 = bbx1;
        *by1 = bby1;
        *bz1 = bbz1;
        *bx2 = ox;
        *by2 = oy;
        *bz2 = oz;
        break;
    case 7: // right, below, behind
        *bx1 = ox;
        *by1 = bby1;
        *bz1 = bbz1;
        *bx2 = bbx2;
        *by2 = oy;
        *bz2 = oz;
        break;
    }

    return;
}

//=================================================================================
void computeCell(float *xcoord, float *ycoord, float *zcoord,
                 int *coordInBox, int numCoordInBox,
                 float bbx1, float bby1, float bbz1,
                 float bbx2, float bby2, float bbz2,
                 int optimize, float maxDistanceSqr, int maxCoord)
{
    int i, j;
    int v, w;
    float rx, ry, rz;
    float obx1, oby1, obz1;
    float obx2, oby2, obz2;
    int numCoordInCell[8];
    int *coordInCell[8];

    // check if we have to go any further
    if (numCoordInBox > maxCoord)
    {
        // yes we have
        rx = (bbx1 + bbx2) / 2.0f;
        ry = (bby1 + bby2) / 2.0f;
        rz = (bbz1 + bbz2) / 2.0f;

        if (optimize)
        {
            // we have to mess with memory
            for (i = 0; i < 8; i++)
                numCoordInCell[i] = 0;
            for (i = 0; i < numCoordInBox; i++)
            {
                v = coordInBox[i];
                numCoordInCell[getOctant(xcoord[v], ycoord[v], zcoord[v], rx, ry, rz)]++;
            }
            for (i = 0; i < 8; i++)
            {
                if (numCoordInCell[i])
                    coordInCell[i] = new int[numCoordInCell[i]];
                else
                    coordInCell[i] = NULL;
            }
        }
        else
            for (i = 0; i < 8; i++)
                coordInCell[i] = new int[numCoordInBox];

        // go through the coordinates and sort them in the right cell
        for (i = 0; i < 8; i++)
            numCoordInCell[i] = 0;
        for (i = 0; i < numCoordInBox; i++)
        {
            v = coordInBox[i];
            w = getOctant(xcoord[v], ycoord[v], zcoord[v], rx, ry, rz);
            if (coordInCell[w] != NULL)
            {
                coordInCell[w][numCoordInCell[w]] = v;
                numCoordInCell[w]++;
            }
        }

        // we're recursive - hype
        for (i = 0; i < 8; i++)
        {
            if (numCoordInCell[i])
            {
                if (numCoordInCell[i] > numCoordInBox / 4)
                {
                    // we decide to compute the new BoundingBox instead of
                    // just splitting the parent-Box
                    boundingBox(&xcoord, &ycoord, &zcoord, coordInCell[i],
                                numCoordInCell[i], &obx1, &oby1, &obz1,
                                &obx2, &oby2, &obz2);
                }
                else
                    getOctantBounds(i, rx, ry, rz, bbx1, bby1, bbz1,
                                    bbx2, bby2, bbz2,
                                    &obx1, &oby1, &obz1, &obx2, &oby2, &obz2);
                computeCell(xcoord, ycoord, zcoord, coordInCell[i],
                            numCoordInCell[i], obx1, oby1, obz1,
                            obx2, oby2, obz2, optimize, maxDistanceSqr, maxCoord);
            }
            delete[] coordInCell[i];
        }
    }
    else if (numCoordInBox > 1)
    {
        // check these vertices
        for (i = 0; i < numCoordInBox - 1; i++)
        {
            v = coordInBox[i];
            rx = xcoord[v];
            ry = ycoord[v];
            rz = zcoord[v];
            // see if this one is doubled
            for (j = i + 1; j < numCoordInBox; j++)
            {
                w = coordInBox[j];
                if (isEqual(xcoord[w], ycoord[w], zcoord[w], rx, ry, rz, maxDistanceSqr))
                {
                    // this one is double
                    if (v < w)
                    {
                        replBy[w] = v;
                        //coordToRemove[0][numCoordToRemove] = w;  // was w
                        //coordToRemove[1][numCoordToRemove] = v;  // was v
                    }
                    else
                    {
                        replBy[v] = w;
                        //coordToRemove[0][numCoordToRemove] = w;  // was w
                        //coordToRemove[1][numCoordToRemove] = v;  // was v
                    }
                    numCoordToRemove++;
                    // break out
                    j = numCoordInBox;
                }
            }
        }
    }

    // done
    return;
}

//=================================================================================
void computeCell(float *xcoord, float *ycoord, float *zcoord,
                 int *coordInBox, int numCoordInBox,
                 float bbx1, float bby1, float bbz1,
                 float bbx2, float bby2, float bbz2,
                 int maxCoord, int *replBy, int &numCoordToRemove)
{
    int i, j;
    int v, w;
    float rx, ry, rz;
    float obx1, oby1, obz1;
    float obx2, oby2, obz2;
    int numCoordInCell[8];
    int *coordInCell[8];

    // too many Coords in my box -> split octree box deeper
    if (numCoordInBox > maxCoord)
    {
        // yes we have
        rx = (bbx1 + bbx2) / 2.0f;
        ry = (bby1 + bby2) / 2.0f;
        rz = (bbz1 + bbz2) / 2.0f;

        // go through the coordinates and sort them in the right cell

        for (i = 0; i < 8; i++)
        {
            coordInCell[i] = new int[numCoordInBox];
            numCoordInCell[i] = 0;
        }

        for (i = 0; i < numCoordInBox; i++)
        {
            v = coordInBox[i];
            w = getOctant(xcoord[v], ycoord[v], zcoord[v], rx, ry, rz);
            coordInCell[w][numCoordInCell[w]] = v;
            numCoordInCell[w]++;
        }

        // we're recursive - hype
        for (i = 0; i < 8; i++)
        {
            if (numCoordInCell[i])
            {
                if (numCoordInCell[i] > numCoordInBox / 4)
                {
                    // we decide to compute the new BoundingBox instead of
                    // just splitting the parent-Box
                    boundingBox(&xcoord, &ycoord, &zcoord, coordInCell[i],
                                numCoordInCell[i], &obx1, &oby1, &obz1,
                                &obx2, &oby2, &obz2);
                }
                else
                    getOctantBounds(i, rx, ry, rz, bbx1, bby1, bbz1,
                                    bbx2, bby2, bbz2,
                                    &obx1, &oby1, &obz1, &obx2, &oby2, &obz2);

                computeCell(xcoord, ycoord, zcoord, coordInCell[i],
                            numCoordInCell[i], obx1, oby1, obz1,
                            obx2, oby2, obz2, maxCoord, replBy, numCoordToRemove);
            }
            delete[] coordInCell[i];
        }
    }

    //// directly compare in box
    else if (numCoordInBox > 1)
    {
        // check these vertices
        for (i = 0; i < numCoordInBox - 1; i++)
        {
            v = coordInBox[i];
            rx = xcoord[v];
            ry = ycoord[v];
            rz = zcoord[v];
            // see if this one is doubled
            for (j = i + 1; j < numCoordInBox; j++)
            {
                w = coordInBox[j];

                if (xcoord[w] == rx && // @@@@ add distance fkt here if necessary
                    ycoord[w] == ry && zcoord[w] == rz)
                {
                    // this one is double
                    if (v < w)
                        replBy[w] = v;
                    else
                        replBy[v] = w;
                    numCoordToRemove++;
                    // break out
                    j = numCoordInBox;
                }
            }
        }
    }

    // done
    return;
}

//=================================================================================

void computeWorkingLists(int num_coord)
{
    int i, k;

    // now unlink the temporary list
    for (i = 0; i < num_coord; i++)
    {
        k = replBy[i];
        if (k >= 0)
        {
            // this one will be replaced, so unlink the list
            while (replBy[k] >= 0)
                k = replBy[k];

            if (replBy[k] == -1)
                // remove this one
                replBy[i] = -1;
            else
                // replace this one
                replBy[i] = k;
        }
    }

    // allocate mem
    source2filtered = new int[num_coord]; // original vertex i is replaced by s2f[i]

    // forward filter
    numFiltered = 0;
    for (i = 0; i < num_coord; i++)
    {
        // vertex untouched
        if (replBy[i] == -2)
        {
            source2filtered[i] = numFiltered;
            numFiltered++;
        }
        // vertex replaced: replacer < replacee
        else if (replBy[i] >= 0)
        {
            source2filtered[i] = source2filtered[replBy[i]];
        }
        else
        {
            source2filtered[i] = -1;
        }
    }

    // backward filter
    filtered2source = new int[numFiltered];
    for (i = 0; i < num_coord; i++)
    {
        if (source2filtered[i] >= 0)
        {
            filtered2source[source2filtered[i]] = i;
        }
    }
    // done
    return;
}

//=================================================================================
void computeReplaceLists(int num_coord, int *replBy,
                         int *&source2filtered, int *&filtered2source)
{
    int i, k;

    // now unlink the temporary list
    for (i = 0; i < num_coord; i++)
    {
        k = replBy[i];
        if (k >= 0)
        {
            // this one will be replaced, so unlink the list
            while (replBy[k] >= 0)
                k = replBy[k];

            if (replBy[k] == -1)
                // remove this one
                replBy[i] = -1;
            else
                // replace this one
                replBy[i] = k;
        }
    }

    // allocate mem
    source2filtered = new int[num_coord]; // original vertex i is replaced by s2f[i]

    // forward filter
    int numFiltered = 0;
    for (i = 0; i < num_coord; i++)
    {
        // vertex untouched
        if (replBy[i] == -2)
        {
            source2filtered[i] = numFiltered;
            numFiltered++;
        }
        // vertex replaced: replacer < replacee
        else if (replBy[i] >= 0)
        {
            source2filtered[i] = source2filtered[replBy[i]];
        }
        else
        {
            source2filtered[i] = -1;
        }
    }

    // backward filter
    filtered2source = new int[numFiltered];
    for (i = 0; i < num_coord; i++)
        if (source2filtered[i] >= 0)
            filtered2source[source2filtered[i]] = i;

    // done
    return;
}

//=============================================================================

coDistributedObject *filterCoordinates(coDistributedObject *obj_in, const coObjInfo &outInfo, int master_num_coord,
                                       float *xcoord, float *ycoord, float *zcoord)
{
    coDistributedObject *returnObject = NULL;

    int i, j;

    if (obj_in == NULL)
        return (NULL);

    if (coDoUnstructuredGrid *data_in = dynamic_cast<coDoUnstructuredGrid *>(obj_in))
    {
        coDoUnstructuredGrid *data_out;
        int num_elem, num_coord, num_conn;
        int *el_in, *cl_in, *tl_in;
        int *el_out, *cl_out, *tl_out;
        float *xcoord_out, *ycoord_out, *zcoord_out;

        // get input
        data_in->getGridSize(&num_elem, &num_conn, &num_coord);
        data_in->getAddresses(&el_in, &cl_in, &xcoord_out, &ycoord_out, &zcoord_out);
        data_in->getTypeList(&tl_in);

        // get output
        returnObject = data_out = new coDoUnstructuredGrid(
            // vo num_coord-numCoordToRemove, 1 );
            outInfo, num_elem, num_conn, numFiltered, 1);
        data_out->getAddresses(&el_out, &cl_out, &xcoord_out, &ycoord_out, &zcoord_out);
        data_out->getTypeList(&tl_out);

        // leave tl and el unchanged
        for (i = 0; i < num_elem; i++)
        {
            tl_out[i] = tl_in[i];
            el_out[i] = el_in[i];
        }

        // update coordinates
        for (i = 0; i < numFiltered; i++) // num_coord-numCoordToRemove; i++ )
        {
            j = filtered2source[i];
            xcoord_out[i] = xcoord[j];
            ycoord_out[i] = ycoord[j];
            zcoord_out[i] = zcoord[j];
        }

        // update cl
        for (i = 0; i < num_conn; i++)
            cl_out[i] = source2filtered[cl_in[i]];

        // done
    }
    else if (dynamic_cast<coDoLines *>(obj_in) || dynamic_cast<coDoPolygons *>(obj_in))
    {
        int num_coord = 0, num_poly = 0; // num_points,
        int *vl_in = NULL, *pl_in = NULL;
        int *vl_out = NULL, *pl_out = NULL;
        float *xcoord_out = NULL, *ycoord_out = NULL, *zcoord_out = NULL;

        if (coDoPolygons *pdata_in = dynamic_cast<coDoPolygons *>(obj_in))
        {
            num_coord = pdata_in->getNumVertices();
            num_poly = pdata_in->getNumPolygons();
            pdata_in->getAddresses(&xcoord_out, &ycoord_out, &zcoord_out, &vl_in, &pl_in);
            coDoPolygons *pdata_out = new coDoPolygons(outInfo, numFiltered, num_coord, num_poly);
            pdata_out->getAddresses(&xcoord_out, &ycoord_out, &zcoord_out, &vl_out, &pl_out);
            returnObject = pdata_out;
        }
        else if (coDoLines *ldata_in = dynamic_cast<coDoLines *>(obj_in))
        {
            num_coord = ldata_in->getNumVertices();
            num_poly = ldata_in->getNumLines();
            ldata_in->getAddresses(&xcoord_out, &ycoord_out, &zcoord_out, &vl_in, &pl_in);
            coDoLines *ldata_out = new coDoLines(outInfo, numFiltered, num_coord, num_poly);
            ldata_out->getAddresses(&xcoord_out, &ycoord_out, &zcoord_out, &vl_out, &pl_out);
            returnObject = ldata_out;
        }

        // leave pl unchanged
        for (i = 0; i < num_poly; i++)
            pl_out[i] = pl_in[i];

        // update coordinates
        for (i = 0; i < numFiltered; i++)
        {
            j = filtered2source[i];
            xcoord_out[i] = xcoord[j];
            ycoord_out[i] = ycoord[j];
            zcoord_out[i] = zcoord[j];
        }

        // update vl
        for (i = 0; i < num_coord; i++)
            vl_out[i] = source2filtered[vl_in[i]];

        // done
    }
    else if (coDoFloat *data_in = dynamic_cast<coDoFloat *>(obj_in))
    {
        coDoFloat *data_out;
        int num_points;
        float *in, *out;

        // get input
        data_in->getAddress(&in);
        num_points = data_in->getNumPoints();

        // get output
        if (num_points == master_num_coord)
        {
            // per_vert
            // num_points-numCoordToRemove );
            returnObject = data_out = new coDoFloat(outInfo, numFiltered);
            data_out->getAddress(&out);

            for (i = 0; i < numFiltered; i++)
                out[i] = in[filtered2source[i]];

            // done
        }
        else
        {
            // per_cell -> change nothing
            returnObject = data_out = new coDoFloat(outInfo, num_points);
            data_out->getAddress(&out);

            for (i = 0; i < num_points; i++)
                out[i] = in[i]; //in[ filtered2source[i] ];

            // done
        }

        // done
    }
    else if (coDoVec3 *data_in = dynamic_cast<coDoVec3 *>(obj_in))
    {
        coDoVec3 *data_out;
        int num_points;
        float *x_in, *y_in, *z_in;
        float *x_out, *y_out, *z_out;

        // get input
        data_in->getAddresses(&x_in, &y_in, &z_in);
        num_points = data_in->getNumPoints();

        // get output
        if (num_points == master_num_coord)
        {
            // per_vert
            // num_points-numCoordToRemove );
            returnObject = data_out = new coDoVec3(outInfo, numFiltered);
            data_out->getAddresses(&x_out, &y_out, &z_out);

            for (i = 0; i < numFiltered; i++)
            {
                j = filtered2source[i];
                x_out[i] = x_in[j];
                y_out[i] = y_in[j];
                z_out[i] = z_in[j];
            }

            // done
        }
        else
        {
            // per_cell -> change nothing
            returnObject = data_out = new coDoVec3(outInfo, num_points);
            data_out->getAddresses(&x_out, &y_out, &z_out);

            for (i = 0; i < num_points; i++)
            {
                j = i; //filtered2source[i];
                x_out[i] = x_in[j];
                y_out[i] = y_in[j];
                z_out[i] = z_in[j];
            }

            // done
        }

        // done
    }

    // done
    return (returnObject);
}
}
