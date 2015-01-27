/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coFixUsg.h"
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoData.h>

using namespace covise;

coFixUsg::coFixUsg()
{
    max_vertices_ = 50;
    delta_ = 0.;
    opt_mem_ = false;
};

coFixUsg::coFixUsg(int max_vertices, float delta, bool opt_mem)
{
    max_vertices_ = max_vertices;
    delta_ = delta;
    opt_mem_ = opt_mem;
};

int
coFixUsg::fixUsg(int num_elem, int num_conn, int num_coord,
                 const int *elem_list, const int *conn_list, const int *type_list,
                 const float *xcoord, const float *ycoord, const float *zcoord,

                 int &new_num_elem, int &new_num_conn, int &new_num_coord,
                 int **new_elem_list, int **new_conn_list, int **new_type_list,
                 float **new_xcoord, float **new_ycoord, float **new_zcoord,

                 int num_val, int *numComp, int *dataSize,
                 float **in_x, float **in_y, float **in_z,
                 float ***new_x, float ***new_y, float ***new_z)
{
    if (num_val != 0 && !dataSize)
    {
        return FIX_ERROR;
    }

    /// create a Replace-List
    int *replBy = new int[num_coord];
    for (int i = 0; i < num_coord; i++)
        replBy[i] = UNTOUCHED;

    int *coordInBox = new int[num_coord];
    if (!coordInBox)
    {
        //fprintf(stderr, "coFixUsg: failed to alloc array of %d ints (2)\n", num_coord);
        return (FIX_ERROR);
    }
    int numCoordInBox = 0;

    // the "starting" cell contains all USED coordinates
    // clear all flags -> no coordinates used at all
    for (int i = 0; i < num_coord; i++)
        coordInBox[i] = 0;

    // set flag if coordinate is used  // loop over connectivity

    for (int i = 0; i < num_conn; i++)
        coordInBox[conn_list[i]] = 1;

    // now remove the unused coordinates
    for (int i = 0; i < num_coord; i++)
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
            replBy[i] = REMOVE;
        }
    }
    // what we do next depends on the chosen algorithm

    float bbx1, bby1, bbz1;
    float bbx2, bby2, bbz2;

    boundingBox(&xcoord, &ycoord, &zcoord, coordInBox, numCoordInBox,
                &bbx1, &bby1, &bbz1, &bbx2, &bby2, &bbz2);
    computeCell(replBy, xcoord, ycoord, zcoord,
                coordInBox, numCoordInBox, bbx1, bby1, bbz1,
                bbx2, bby2, bbz2, opt_mem_, delta_ * delta_, max_vertices_);

    // partially clean up
    delete[] coordInBox;

    // compute the working-lists
    int *source2filtered;
    int *filtered2source;
    int num_target;
    computeWorkingLists(num_coord, replBy, &source2filtered, &filtered2source, num_target);

    // and finally remove the filtered coordinates
    // new data fields
    if (num_val > 0)
    {
        *new_x = new float *[num_val];
        *new_y = new float *[num_val];
        *new_z = new float *[num_val];
        if (!new_x || !new_y || !new_z)
        {
            return FIX_ERROR;
        }
        for (int i = 0; i < num_val; i++)
        {
            if (dataSize[i] == num_elem)
            {
                (*new_x)[i] = new float[num_elem];
                for (int j = 0; j < num_elem; j++)
                    (*new_x)[i][j] = in_x[i][j];
            }
            else if (dataSize[i] == num_coord)
            {
                (*new_x)[i] = new float[num_target];
                mapArray(in_x[i], (*new_x)[i], filtered2source, num_target);
            }
            else
            {
                return FIX_ERROR;
            }
            (*new_y)[i] = NULL;
            (*new_z)[i] = NULL;
            if (numComp[i] == 3)
            {
                if (dataSize[i] == num_elem)
                {
                    (*new_y)[i] = new float[num_elem];
                    (*new_z)[i] = new float[num_elem];
                    for (int j = 0; j < num_elem; j++)
                    {
                        (*new_y)[i][j] = in_y[i][j];
                        (*new_z)[i][j] = in_z[i][j];
                    }
                }
                else if (dataSize[i] == num_coord)
                {
                    (*new_y)[i] = new float[num_target];
                    (*new_z)[i] = new float[num_target];
                    mapArray(in_y[i], (*new_y)[i], filtered2source, num_target);
                    mapArray(in_z[i], (*new_z)[i], filtered2source, num_target);
                }
                else
                {
                    return FIX_ERROR;
                }
            }
        }
    }
    // new USG
    *new_xcoord = new float[num_target];
    *new_ycoord = new float[num_target];
    *new_zcoord = new float[num_target];
    *new_elem_list = new int[num_elem];
    if (type_list)
        *new_type_list = new int[num_elem];
    *new_conn_list = new int[num_conn];

    new_num_elem = num_elem;
    new_num_conn = num_conn;
    new_num_coord = num_target;

    if (!new_xcoord || !new_ycoord || !new_zcoord || !new_elem_list || !new_conn_list)
    {
        return FIX_ERROR;
    }

    for (int i = 0; i < num_elem; i++)
    {
        if (type_list)
            (*new_type_list)[i] = type_list[i];
        (*new_elem_list)[i] = elem_list[i];
    }
    mapArray(xcoord, (*new_xcoord), filtered2source, num_target);
    mapArray(ycoord, (*new_ycoord), filtered2source, num_target);
    mapArray(zcoord, (*new_zcoord), filtered2source, num_target);
    updateArray(conn_list, (*new_conn_list), source2filtered, num_conn);

    // remove degenerated Hexaeders
    if (*new_type_list)
    {
        new_num_conn = 0;
        for (int i = 0; i < new_num_elem; i++)
        {
            (*new_elem_list)[i] = new_num_conn;
            if ((*new_type_list)[i] == TYPE_HEXAEDER
                && conn_list[elem_list[i] + 2] == conn_list[elem_list[i] + 3]
                && conn_list[elem_list[i] + 4] == conn_list[elem_list[i] + 5]
                && conn_list[elem_list[i] + 4] == conn_list[elem_list[i] + 6]
                && conn_list[elem_list[i] + 4] == conn_list[elem_list[i] + 7])
            {
                (*new_type_list)[i] = TYPE_TETRAHEDER;
                (*new_conn_list)[new_num_conn] = conn_list[elem_list[i]];
                new_num_conn++;
                (*new_conn_list)[new_num_conn] = conn_list[elem_list[i] + 1];
                new_num_conn++;
                (*new_conn_list)[new_num_conn] = conn_list[elem_list[i] + 2];
                new_num_conn++;
                (*new_conn_list)[new_num_conn] = conn_list[elem_list[i] + 4];
                new_num_conn++;
            }
            else
            {
                int nc;
                if (i == num_elem - 1)
                    nc = num_conn - elem_list[i];
                else
                    nc = elem_list[i + 1] - elem_list[i];
                for (int k = 0; k < nc; k++)
                {
                    (*new_conn_list)[new_num_conn] = conn_list[elem_list[i] + k];
                    new_num_conn++;
                }
            }
        }
    }

    // clean up
    delete[] source2filtered;
    delete[] filtered2source;

    return (num_coord - new_num_coord);
}

void
coFixUsg::boundingBox(const float *const *x, const float *const *y, const float *const *z, const int *c, int n,
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

void
coFixUsg::computeCell(int *replBy, const float *xcoord, const float *ycoord, const float *zcoord,
                      const int *coordInBox, int numCoordInBox,
                      float bbx1, float bby1, float bbz1,
                      float bbx2, float bby2, float bbz2,
                      bool optimize, float maxDistanceSqr, int maxCoord,
                      int recurseLevel)
{
    int i, j;
    int v, w;
    float rx, ry, rz;
    float obx1, oby1, obz1;
    float obx2, oby2, obz2;
    int numCoordInCell[8];
    int *coordInCell[8];

    // check if we have to go any further:
    // CHANGE we 02/2003: make sure we terminate with same point >maxCoord times
    // recurse level criterion faster than bbox>0 comparison
    if (numCoordInBox > maxCoord && recurseLevel < 20)
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
                {
                    getOctantBounds(i, rx, ry, rz, bbx1, bby1, bbz1,
                                    bbx2, bby2, bbz2,
                                    &obx1, &oby1, &obz1, &obx2, &oby2, &obz2);
                }

                computeCell(replBy, xcoord, ycoord, zcoord, coordInCell[i],
                            numCoordInCell[i], obx1, oby1, obz1,
                            obx2, oby2, obz2, optimize, maxDistanceSqr, maxCoord,
                            recurseLevel + 1);
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
                    }
                    else
                    {
                        replBy[v] = w;
                    }
                    // break out
                    j = numCoordInBox;
                }
            }
        }
    }

    // done
    return;
}

bool coFixUsg::isEqual(float x1, float y1, float z1,
                       float x2, float y2, float z2, float dist)
{
    bool r;
    float d;

    if (dist == 0.0)
        r = (x1 == x2 && y1 == y2 && z1 == z2);
    else
    {
        d = ((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)) + ((z2 - z1) * (z2 - z1));
        if (d <= dist)
            r = true;
        else
            r = false;
    }

    return (r);
}

int coFixUsg::getOctant(float x, float y, float z, float ox, float oy, float oz)
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

void coFixUsg::getOctantBounds(int o, float ox, float oy, float oz,
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

void
coFixUsg::computeWorkingLists(int num_coord, int *replBy, int **src2fil, int **fil2src, int &num_target)
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

            if (replBy[k] == REMOVE)
                // remove this one
                replBy[i] = REMOVE;
            else
                // replace this one
                replBy[i] = k;
        }
    }

    // allocate mem
    int *source2filtered = new int[num_coord]; // original vertex i is replaced by s2f[i]
    *src2fil = source2filtered;

    // forward filter
    num_target = 0;
    for (i = 0; i < num_coord; i++)
    {
        // vertex untouched
        if (replBy[i] == UNTOUCHED)
        {
            source2filtered[i] = num_target;
            num_target++;
        }
        // vertex replaced: replacer < replacee
        else if (replBy[i] >= 0)
        {
            source2filtered[i] = source2filtered[replBy[i]];
        }
        else
        {
            source2filtered[i] = UNCHANGED;
        }
    }

    // backward filter
    int *filtered2source = new int[num_target];
    *fil2src = filtered2source;
    for (i = 0; i < num_coord; i++)
    {
        if (source2filtered[i] != UNCHANGED)
        {
            filtered2source[source2filtered[i]] = i;
        }
    }
    // done
    return;
}

template <class T>
void coFixUsg::mapArray(const T *src, T *target, const int *filtered2source, int num_target)
{
    int i, j;
    for (i = 0; i < num_target; i++)
    {
        j = filtered2source[i];
        target[i] = src[j];
    }
}

template <class T>
void coFixUsg::updateArray(const T *src, T *target, const int *source2filtered, int num_target)
{
    int i;
    for (i = 0; i < num_target; i++)
        target[i] = source2filtered[src[i]];
}

int
coFixUsg::fixUsg(const coDistributedObject *geo_in, coDistributedObject **geo_out, const char *geoObjName,
                 int num_val, const coDistributedObject *const *data_in, coDistributedObject **data_out, const char **objName)
{
    if (!geo_in || !geoObjName || !data_in || !objName)
        return FIX_ERROR;

    // geometry input
    int num_elem, num_conn, num_coord;
    int *elem_list, *conn_list;
    int *type_list = NULL;
    float *xcoord, *ycoord, *zcoord;
    enum
    {
        UNKNOWN,
        POLY,
        LINE,
        UNSTR
    } geoType = UNKNOWN;

    if (const coDoPolygons *poly_in = dynamic_cast<const coDoPolygons *>(geo_in))
    {
        geoType = POLY;
        // polygons
        num_elem = poly_in->getNumPolygons();
        num_coord = poly_in->getNumPoints();
        num_conn = poly_in->getNumVertices();
        poly_in->getAddresses(&xcoord, &ycoord, &zcoord, &conn_list, &elem_list);
    }
    else if (const coDoLines *lines_in = dynamic_cast<const coDoLines *>(geo_in))
    {
        geoType = LINE;
        // lines
        num_elem = lines_in->getNumLines();
        num_coord = lines_in->getNumPoints();
        num_conn = lines_in->getNumVertices();
        lines_in->getAddresses(&xcoord, &ycoord, &zcoord, &conn_list, &elem_list);
    }
    else if (const coDoUnstructuredGrid *grid_in = dynamic_cast<const coDoUnstructuredGrid *>(geo_in))
    {
        geoType = UNSTR;
        // unstructured grid
        grid_in->getGridSize(&num_elem, &num_conn, &num_coord);
        grid_in->getAddresses(&elem_list, &conn_list, &xcoord, &ycoord, &zcoord);
        grid_in->getTypeList(&type_list);
    }
    else
    {
        return (FIX_ERROR);
    }

    // data input
    int i, *dataSize = NULL;
    int *numComp = NULL;
    float **in_x = NULL;
    float **in_y = NULL;
    float **in_z = NULL;
    if (num_val > 0)
    {
        dataSize = new int[num_val];
        numComp = new int[num_val];
        in_x = new float *[num_val];
        in_y = new float *[num_val];
        in_z = new float *[num_val];

        for (i = 0; i < num_val; i++)
        {
            if (const coDoFloat *s_data_in = dynamic_cast<const coDoFloat *>(data_in[i]))
            {
                numComp[i] = 1;

                s_data_in->getAddress(&(in_x[i]));
                in_y[i] = NULL;
                in_z[i] = NULL;
                dataSize[i] = s_data_in->getNumPoints();
            }
            else if (const coDoVec3 *v_data_in = dynamic_cast<const coDoVec3 *>(data_in[i]))
            {
                numComp[i] = 3;
                v_data_in->getAddresses(&(in_x[i]), &(in_y[i]), &(in_z[i]));
                dataSize[i] = v_data_in->getNumPoints();
            }
            else
            {
                delete[] numComp;
                delete[] in_x;
                delete[] in_y;
                delete[] in_z;
                delete[] dataSize;
                return FIX_ERROR;
            }
        }
    }

    float **out_x = NULL, **out_y = NULL, **out_z = NULL;

    // geometry output
    int new_num_elem = 0, new_num_conn = 0, new_num_coord = 0;
    int *new_elem_list = NULL, *new_conn_list = NULL, *new_type_list = NULL;
    float *new_xcoord = NULL, *new_ycoord = NULL, *new_zcoord = NULL;
    int num_rem = 0;

    if (num_elem == 0 || num_conn == 0 || num_coord == 0 || (num_rem = fixUsg(num_elem, num_conn, num_coord, elem_list, conn_list, type_list, xcoord, ycoord, zcoord, new_num_elem, new_num_conn, new_num_coord, &new_elem_list, &new_conn_list, &new_type_list, &new_xcoord, &new_ycoord, &new_zcoord, num_val, numComp, dataSize, in_x, in_y, in_z, &out_x, &out_y, &out_z)) != FIX_ERROR)
    {
        if (out_x == NULL)
        {
            out_x = new float *[num_val + 1];
            if (out_y == NULL)
                out_y = new float *[num_val + 1];
            if (out_z == NULL)
                out_z = new float *[num_val + 1];
            for (i = 0; i < num_val; i++)
            {
                out_x[i] = NULL;
                out_y[i] = NULL;
                out_z[i] = NULL;
            }
        }
        if (geoType == POLY)
        {
            *geo_out = new coDoPolygons(geoObjName, new_num_coord, new_xcoord, new_ycoord, new_zcoord,
                                        new_num_conn, new_conn_list, new_num_elem, new_elem_list);
        }
        else if (geoType == LINE)
        {
            *geo_out = new coDoLines(geoObjName, new_num_coord, new_xcoord, new_ycoord, new_zcoord,
                                     new_num_conn, new_conn_list, new_num_elem, new_elem_list);
        }
        else if (geoType == UNSTR)
        {
            *geo_out = new coDoUnstructuredGrid(geoObjName, new_num_elem, new_num_conn, new_num_coord,
                                                new_elem_list, new_conn_list, new_xcoord, new_ycoord, new_zcoord, new_type_list);
            delete[] new_type_list;
        }

        // free
        delete[] new_elem_list;
        delete[] new_conn_list;
        delete[] new_xcoord;
        delete[] new_ycoord;
        delete[] new_zcoord;

        if (num_val > 0)
        {
            for (i = 0; i < num_val; i++)
            {
                if (dynamic_cast<const coDoFloat *>(data_in[i]))
                {
                    if (dataSize[i] == num_elem)
                    {
                        if (out_x[i] != NULL)
                            data_out[i] = new coDoFloat(objName[i], num_elem, out_x[i]);
                        else
                            data_out[i] = new coDoFloat(objName[i], num_elem);
                    }
                    else
                    {
                        if (out_x[i] != NULL)
                            data_out[i] = new coDoFloat(objName[i], new_num_coord, out_x[i]);
                        else
                            data_out[i] = new coDoFloat(objName[i], new_num_coord);
                    }
                    if (out_x[i] != NULL)
                        delete[] out_x[i];
                }
                else if (dynamic_cast<const coDoVec3 *>(data_in[i]))
                {
                    if (dataSize[i] == num_elem)
                    {
                        if (out_x[i] != NULL)
                            data_out[i] = new coDoVec3(objName[i], num_elem, out_x[i], out_y[i], out_z[i]);
                        else
                            data_out[i] = new coDoVec3(objName[i], num_elem);
                    }
                    else
                    {
                        if (out_x[i] != NULL)
                            data_out[i] = new coDoVec3(objName[i], new_num_coord, out_x[i], out_y[i], out_z[i]);
                        else
                            data_out[i] = new coDoVec3(objName[i], new_num_coord);
                    }
                    if (out_x[i] != NULL)
                        delete[] out_x[i];
                    if (out_y[i] != NULL)
                        delete[] out_y[i];
                    if (out_z[i] != NULL)
                        delete[] out_z[i];
                }
            }
            delete[] out_x;
            delete[] out_y;
            delete[] out_z;
        }
    }

    if (num_val > 0)
    {
        delete[] in_x;
        delete[] in_y;
        delete[] in_z;
    }
    return num_rem;
}
