/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TestSlider.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <api/coFeedback.h>

TestSlider::TestSlider(int argc, char *argv[])
    : coModule(argc, argv, "Simple TestSlider Generation Module")
{
    p_polyOut = addOutputPort("polygons", "Polygons", "polygons which form the cubes");

    p_center = addFloatVectorParam("center", "Center of the cube");
    cx = cy = cz = 0.0;
    p_center->setValue(cx, cy, cz);

    p_cusize = addFloatSliderParam("size", "Size of the cube");
    sMin = 1.0;
    sMax = 10.0;
    sVal = 2.0;
    p_cusize->setValue(sMin, sMax, sVal);
}

void TestSlider::postInst()
{
    p_center->show();
    p_cusize->show();
}

int TestSlider::compute(const char *port)
{
    (void)port;
    const char *polygonObjName;
    coDoPolygons *polygonObj;
    float xCoords[8], yCoords[8], zCoords[8];
    int vertexList[24] = { 0, 3, 7, 4,
                           3, 2, 6, 7,
                           0, 1, 2, 3,
                           0, 4, 5, 1,
                           1, 5, 6, 2,
                           7, 6, 5, 4 };

    int polygonList[6] = { 0, 4, 8, 12, 16, 20 };

    // compute the vertex coordinates

    //      5.......6
    //    .       . .
    //  4.......7   .
    //  .       .   .
    //  .   1   .   2
    //  .       . .
    //  0.......3

    xCoords[0] = cx - sVal;
    yCoords[0] = cy - sVal;
    zCoords[0] = cz - sVal;

    xCoords[1] = cx - sVal;
    yCoords[1] = cy + sVal;
    zCoords[1] = cz - sVal;

    xCoords[2] = cx + sVal;
    yCoords[2] = cy + sVal;
    zCoords[2] = cz - sVal;

    xCoords[3] = cx + sVal;
    yCoords[3] = cy - sVal;
    zCoords[3] = cz - sVal;

    xCoords[4] = cx - sVal;
    yCoords[4] = cy - sVal;
    zCoords[4] = cz + sVal;

    xCoords[5] = cx - sVal;
    yCoords[5] = cy + sVal;
    zCoords[5] = cz + sVal;

    xCoords[6] = cx + sVal;
    yCoords[6] = cy + sVal;
    zCoords[6] = cz + sVal;

    xCoords[7] = cx + sVal;
    yCoords[7] = cy - sVal;
    zCoords[7] = cz + sVal;

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
        coFeedback feedback("TestSlider");
        //feedback.addPara(p_center);
        feedback.addPara(p_cusize);
        feedback.addString("Test the user string as well");
        feedback.apply(polygonObj);

        ostringstream str;
        str << "V" << Covise::get_module() << "\n" << Covise::get_instance()
            << "\n" << Covise::get_host() << "\nfloat\nsize\n0.0\n10.0\n"
            << sVal << "\nSphereSegment\n0.01\n";

        int numpoints = 10;
        str << numpoints << "\n";
        float xpos = 0, ypos = 0, zpos = 0;
        for (int index = 0; index < numpoints; index++)
        {
            str << xpos << "\n" << ypos << "\n" << zpos << "\n";
            xpos += 1.0;
        }
        polygonObj->addAttribute("SLIDER0", str.str().c_str());

        p_polyOut->setCurrentObject(polygonObj);
    }
    else
    {
        fprintf(stderr, "Covise::get_object_name failed\n");
        return FAIL;
    }

    return SUCCESS;
}

TestSlider::~TestSlider()
{
}

void TestSlider::param(const char *name, bool /*inMapLoading*/)
{
    if (strcmp(name, p_cusize->getName()) == 0)
        sVal = p_cusize->getValue();
    else if (strcmp(name, p_center->getName()) == 0)
    {
        cx = p_center->getValue(0);
        cy = p_center->getValue(1);
        cz = p_center->getValue(2);
    }
}

MODULE_MAIN(Examples, TestSlider)
