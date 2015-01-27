/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description: Simple Geometry Generation Module                           **
 **              supports interaction from a COVER plugin                    **
 **              feedback style is later than COVISE 4.5.2                   **
 **                                                                          **
 ** Name:        cube                                                        **
 ** Category:    examples                                                    **
 **                                                                          **
 ** Author: D. Rainer                                                        **
 **                                                                          **
 ** History:                                                                 **
 ** September-99                                                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#include "Cube.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <api/coFeedback.h>
#include <do/coDoPolygons.h>

Cube::Cube(int argc, char *argv[]) // vvvv --- this info appears in the module setup window
    : coModule(argc, argv, "Simple Cube Generation Module")
{
    // output port
    // parameters:
    //   port name
    //   string to indicate connections, convention: name of the data type
    //   description
    p_polyOut = addOutputPort("polygons", "Polygons", "polygons which form the cubes");

    // input parameters
    // parameters:
    //   parameter name
    //   parameter type
    //   description
    p_center = addFloatVectorParam("center", "Center of the cube");
    cx = cy = cz = 0.0;
    p_center->setValue(cx, cy, cz);

    p_cusize = addFloatSliderParam("size", "Size of the cube");
    sMin = 1.0;
    sMax = 100.0;
    sVal = 10.0;
    p_cusize->setValue(sMin, sMax, sVal);
}

void Cube::postInst()
{
    p_center->show();
    p_cusize->show();
}

int Cube::compute(const char *port)
{
    printf("center: %f %f %f\n", p_center->getValue(0), p_center->getValue(1), p_center->getValue(2));

    (void)port;
    const char *polygonObjName;
    coDoPolygons *polygonObj;
    float xCoords[8], yCoords[8], zCoords[8];
    int vertexList[24] = { 0, 3, 7, 4, 3, 2, 6, 7, 0, 1, 2, 3, 0, 4, 5, 1, 1, 5, 6, 2, 7, 6, 5, 4 };
    int polygonList[6] = { 0, 4, 8, 12, 16, 20 };

    // compute the vertex coordinates

    //      5.......6
    //    .       . .
    //  4.......7   .
    //  .       .   .
    //  .   1   .   2
    //  .       . .
    //  0.......3

    xCoords[0] = cx - 0.5 * sVal;
    yCoords[0] = cy - 0.5 * sVal;
    zCoords[0] = cz - 0.5 * sVal;

    xCoords[1] = cx - 0.5 * sVal;
    yCoords[1] = cy + 0.5 * sVal;
    zCoords[1] = cz - 0.5 * sVal;

    xCoords[2] = cx + 0.5 * sVal;
    yCoords[2] = cy + 0.5 * sVal;
    zCoords[2] = cz - 0.5 * sVal;

    xCoords[3] = cx + 0.5 * sVal;
    yCoords[3] = cy - 0.5 * sVal;
    zCoords[3] = cz - 0.5 * sVal;

    xCoords[4] = cx - 0.5 * sVal;
    yCoords[4] = cy - 0.5 * sVal;
    zCoords[4] = cz + 0.5 * sVal;

    xCoords[5] = cx - 0.5 * sVal;
    yCoords[5] = cy + 0.5 * sVal;
    zCoords[5] = cz + 0.5 * sVal;

    xCoords[6] = cx + 0.5 * sVal;
    yCoords[6] = cy + 0.5 * sVal;
    zCoords[6] = cz + 0.5 * sVal;

    xCoords[7] = cx + 0.5 * sVal;
    yCoords[7] = cy - 0.5 * sVal;
    zCoords[7] = cz + 0.5 * sVal;

    // get the data object name from the controller
    polygonObjName = p_polyOut->getObjName();
    if (polygonObjName)
    {

        // create the polygons data object
        polygonObj = new coDoPolygons(polygonObjName, 8,
                                      xCoords, yCoords, zCoords,
                                      24, vertexList,
                                      6, polygonList);
        // interaction info for COVER
        coFeedback feedback("Cube");
        feedback.addPara(p_center);
        feedback.addPara(p_cusize);
        feedback.addString("Test the user string as well");
        feedback.apply(polygonObj);

        p_polyOut->setCurrentObject(polygonObj);
    }
    else
    {
        fprintf(stderr, "Covise::get_object_name failed\n");
        return FAIL;
    }

    return SUCCESS;
}

Cube::~Cube()
{
}

void Cube::param(const char *name, bool /*inMapLoading*/)
{
    if (strcmp(name, p_cusize->getName()) == 0)
    {
        sVal = p_cusize->getValue();
    }
    else if (strcmp(name, p_center->getName()) == 0)
    {
        cx = p_center->getValue(0);
        cy = p_center->getValue(1);
        cz = p_center->getValue(2);
    }
}

MODULE_MAIN(Examples, Cube)
