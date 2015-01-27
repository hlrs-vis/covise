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

#include "Test.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <api/coFeedback.h>
#include "ResultDataBase.h"
#include "ResultIntParam.h"
#include "ResultFloatParam.h"
#include "ResultEnumParam.h"

int main(int argc, char *argv[])
{
    Cube *application = new Cube;
    application->start(argc, argv);
}

Cube::Cube() // vvvv --- this info appears in the module setup window
    : coModule("Simple Cube Generation Module")
{
    // output port
    // parameters:
    //   port name
    //   string to indicate connections, convention: name of the data type
    //   description
    p_polyOut = addOutputPort("polygons", "coDoPolygons", "polygons which form the cubes");

    // input parameters
    // parameters:
    //   parameter name
    //   parameter type
    //   description
    p_center = addFloatVectorParam("center", "Center of the cube");
    p_center->setValue(0.0, 0.0, 0.0);

    p_cusize = addFloatParam("size", "Size of the cube");
    p_cusize->setValue(1.0);
}

void Cube::postInst()
{
    p_center->show();
    p_cusize->show();
}

int Cube::compute()
{

    ResultDataBase dataB("/usr/people/sk_te/");
    ResultIntParam *v1 = new ResultIntParam("noppenHoehe", 1);
    ResultFloatParam *v2 = new ResultFloatParam("noppenWinkel", 0.00012345, 3);

    char *labs[] = { "Raute", "Ellipse" };

    ResultEnumParam *v3 = new ResultEnumParam("noppenForm", 2, labs, 1);
    ia<ResultParam *> list;

    list.push_back(v1);
    list.push_back(v2);
    list.push_back(v3);

    cerr << dataB.getSaveDirName(3, list) << endl;

    char *new_labs[] = { "Raute", "Raus" };

    v3->setValue(2, new_labs, 1);

    float diff;
    cerr << dataB.searchForResult(diff, 3, list) << endl;
    cerr << "Diff: " << diff << endl;

    const char *polygonObjName;
    coDoPolygons *polygonObj;
    float cx, cy, cz; // coodinates of the cubes p_center
    float size; // edge length of the cube
    float xCoords[8], yCoords[8], zCoords[8];
    int vertexList[24] = { 0, 3, 7, 4, 3, 2, 6, 7, 0, 1, 2, 3, 0, 4, 5, 1, 1, 5, 6, 2, 7, 6, 5, 4 };
    int polygonList[6] = { 0, 4, 8, 12, 16, 20 };

    // get the values of the input parameters
    cx = p_center->getValue(0);
    cy = p_center->getValue(1);
    cz = p_center->getValue(2);
    size = p_cusize->getValue();

    // compute the vertex coordinates

    //      5.......6
    //    .       . .
    //  4.......7   .
    //  .       .   .
    //  .   1   .   2
    //  .       . .
    //  0.......3

    xCoords[0] = cx - 0.5 * size;
    yCoords[0] = cy - 0.5 * size;
    zCoords[0] = cz - 0.5 * size;

    xCoords[1] = cx - 0.5 * size;
    yCoords[1] = cy + 0.5 * size;
    zCoords[1] = cz - 0.5 * size;

    xCoords[2] = cx + 0.5 * size;
    yCoords[2] = cy + 0.5 * size;
    zCoords[2] = cz - 0.5 * size;

    xCoords[3] = cx + 0.5 * size;
    yCoords[3] = cy - 0.5 * size;
    zCoords[3] = cz - 0.5 * size;

    xCoords[4] = cx - 0.5 * size;
    yCoords[4] = cy - 0.5 * size;
    zCoords[4] = cz + 0.5 * size;

    xCoords[5] = cx - 0.5 * size;
    yCoords[5] = cy + 0.5 * size;
    zCoords[5] = cz + 0.5 * size;

    xCoords[6] = cx + 0.5 * size;
    yCoords[6] = cy + 0.5 * size;
    zCoords[6] = cz + 0.5 * size;

    xCoords[7] = cx + 0.5 * size;
    yCoords[7] = cy - 0.5 * size;
    zCoords[7] = cz + 0.5 * size;

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
        coFeedback feedback("CubePlugin");
        feedback.addPara(p_center);
        feedback.addPara(p_cusize);
        feedback.addString("Test the user string as well");
        feedback.apply(polygonObj);

        /// old-style Feedback
        // char feedbackInfo[1000]; // feedback information for COVER
        // sprintf(feedbackInfo,"%f %f %f %f", cx, cy, cz, size);
        // Covise::addInteractor(polygonObj,"Cube", feedbackInfo);

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
