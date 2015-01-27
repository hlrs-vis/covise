/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Filter program in COVISE API                           ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

#include "coBoundingBox.h"
#include <do/coDoSet.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoPoints.h>
#include <do/coDoLines.h>
#include <float.h>

using namespace covise;

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  recursively open all objects and find own min/max
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void coBoundingBox::calcBB(const coDistributedObject *obj, float min[3], float max[3], bool translate)
{
    if (obj)
    {
        // if it's a set: recurse
        if (const coDoSet *set = dynamic_cast<const coDoSet *>(obj))
        {
            int i, numElem;
            const coDistributedObject *const *objList = set->getAllElements(&numElem);

            if (numElem)
            {
                calcBB(objList[0], min, max);
                for (i = 1; i < numElem; i++)
                    if (strcmp(objList[i]->getName(), objList[i - 1]->getName()) != 0)
                        calcBB(objList[i], min, max);
            }
            for (i = 0; i < numElem; i++)
            {
                delete objList[i];
            }
            // sk: objList is now delete in the libary
            delete[] objList;
            return;
        }

        // otherwise: only valid data formats
        else if (dynamic_cast<const coDoUnstructuredGrid *>(obj) || dynamic_cast<const coDoPolygons *>(obj) || dynamic_cast<const coDoTriangleStrips *>(obj) || dynamic_cast<const coDoPoints *>(obj) || dynamic_cast<const coDoLines *>(obj) || dynamic_cast<const coDoStructuredGrid *>(obj))
        {
            // we only need numVert and vertex coordinates
            int numElem, numConn, numVert = 0;
            int *elemList, *connList;
            float *x = NULL, *y = NULL, *z = NULL;

            if (const coDoUnstructuredGrid *usg = dynamic_cast<const coDoUnstructuredGrid *>(obj))
            {
                usg->getGridSize(&numElem, &numConn, &numVert);
                usg->getAddresses(&elemList, &connList, &x, &y, &z);
            }
            else if (const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(obj))
            {
                numVert = poly->getNumPoints();
                poly->getAddresses(&x, &y, &z, &connList, &elemList);
            }
            else if (const coDoTriangleStrips *poly = dynamic_cast<const coDoTriangleStrips *>(obj))
            {
                numVert = poly->getNumPoints();
                poly->getAddresses(&x, &y, &z, &connList, &elemList);
            }
            else if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(obj))
            {
                numVert = points->getNumPoints();
                points->getAddresses(&x, &y, &z);
            }
            else if (const coDoLines *lines = dynamic_cast<const coDoLines *>(obj))
            {
                numVert = lines->getNumPoints();
                lines->getAddresses(&x, &y, &z, &connList, &elemList);
            }
            else if (const coDoStructuredGrid *str = dynamic_cast<const coDoStructuredGrid *>(obj))
            {
                int sx, sy, sz;
                str->getGridSize(&sx, &sy, &sz);
                numVert = sx * sy * sz;
                str->getAddresses(&x, &y, &z);
            }

            if (!translate)
            {
                for (int i = 0; i < numVert; i++)
                {
                    if (x[i] > max[0])
                        max[0] = x[i];
                    if (y[i] > max[1])
                        max[1] = y[i];
                    if (z[i] > max[2])
                        max[2] = z[i];

                    if (x[i] < min[0])
                        min[0] = x[i];
                    if (y[i] < min[1])
                        min[1] = y[i];
                    if (z[i] < min[2])
                        min[2] = z[i];
                }
            }
            else
            {
                for (int i = 0; i < numVert; i++)
                {
                    x[i] += min[0];
                    y[i] += min[1];
                    z[i] += min[2];
                }
            }
        }
        else if (const coDoUniformGrid *ug = dynamic_cast<const coDoUniformGrid *>(obj))
        {
            if (!translate)
            {
                ug->getMinMax(&min[0], &max[0],
                              &min[1], &max[1],
                              &min[2], &max[2]);
            }
        }
        else if (const coDoRectilinearGrid *rect = dynamic_cast<const coDoRectilinearGrid *>(obj))
        {
            int sx, sy, sz;
            rect->getGridSize(&sx, &sy, &sz);
            float *x, *y, *z;
            rect->getAddresses(&x, &y, &z);
            if (!translate)
            {
                for (int i = 0; i < sx; i++)
                {
                    if (x[i] > max[0])
                        max[0] = x[i];
                    if (x[i] < min[0])
                        min[0] = x[i];
                }
                for (int i = 0; i < sy; i++)
                {
                    if (y[i] > max[1])
                        max[1] = y[i];
                    if (y[i] < min[1])
                        min[1] = y[i];
                }
                for (int i = 0; i < sz; i++)
                {
                    if (z[i] > max[2])
                        max[2] = z[i];
                    if (z[i] < min[2])
                        min[2] = z[i];
                }
            }
            else
            {
                for (int i = 0; i < sx; i++)
                {
                    x[i] += min[0];
                }
                for (int i = 0; i < sy; i++)
                {
                    y[i] += min[1];
                }
                for (int i = 0; i < sz; i++)
                {
                    z[i] += min[2];
                }
            }
        }
    }

    return;
}

coDoLines *
coBoundingBox::createBox(const coObjInfo &objectName,
                         float ox, float oy, float oz, float size_x, float size_y, float size_z)
{
    float xCoords[8], yCoords[8], zCoords[8]; // the box coordinates
    int vertexList[24] = // the index list
        {
          0, 3, 7, 4, 3, 2, 6, 7, 0, 1, 2, 3, 0, 4, 5, 1, 1, 5, 6, 2, 7, 6, 5, 4
        };
    int linesList[6] = { 0, 4, 8, 12, 16, 20 };

    // compute the vertex coordinates

    //      5.......6
    //    .       . .
    //  4.......7   .  z
    //  .       .   .
    //  .   1   .   2
    //  .       . .  y
    //  0.......3
    //      x

    xCoords[0] = ox;
    yCoords[0] = oy;
    zCoords[0] = oz;

    xCoords[1] = ox;
    yCoords[1] = oy + size_y;
    zCoords[1] = oz;

    xCoords[2] = ox + size_x;
    yCoords[2] = oy + size_y;
    zCoords[2] = oz;

    xCoords[3] = ox + size_x;
    yCoords[3] = oy;
    zCoords[3] = oz;

    xCoords[4] = ox;
    yCoords[4] = oy;
    zCoords[4] = oz + size_z;

    xCoords[5] = ox;
    yCoords[5] = oy + size_y;
    zCoords[5] = oz + size_z;

    xCoords[6] = ox + size_x;
    yCoords[6] = oy + size_y;
    zCoords[6] = oz + size_z;

    xCoords[7] = ox + size_x;
    yCoords[7] = oy;
    zCoords[7] = oz + size_z;

    // create the lines data object
    coDoLines *linesObj = new coDoLines(objectName, 8,
                                        xCoords, yCoords, zCoords,
                                        24, vertexList,
                                        6, linesList);

    return (linesObj);
}
