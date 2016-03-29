/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2000 RUS   **
 **                                                                          **
 ** Description: Simple Example how to create a set of polygons              **
 **                                                                          **
 ** Name:        PolygonSet                                                  **
 ** Category:    examples                                                    **
 **                                                                          **
 ** Author: D. Rainer		                                            **
 **                                                                          **
 ** History:  								    **
 ** April-00     					       		    **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "PolygonSet.h"
#include <stdlib.h>
#include <stdio.h>
#include <do/coDoSet.h>
#include <do/coDoPolygons.h>

PolygonSet::PolygonSet(int argc, char *argv[]) // this info appears in the module setup window
    : coModule(argc, argv, "Polygon Set example Module")
{
    // output port
    // parameters:
    //   port name
    //   string to indicate connections, convention: name of the data type
    //   description
    outPort_polySet = addOutputPort("polygon_set", "Polygons", "set of polygons, each polygon is a cube");

    p_numObjects = addIntSliderParam("numObj", "Number of Objects");
    p_numObjects->setValue(1, 20, 5);

    p_timeSteps = addBooleanParam("timesteps", "Create Timesteps");
    p_timeSteps->setValue(true);
}

int PolygonSet::compute(const char *port)
{
    (void)port;
    const char *outputObjectName; // the name of the set object
    coDoSet *setObject; // the set object
    coDistributedObject **polygonObjects; // the  set elements

    // get the set object name from the controller
    outputObjectName = outPort_polySet->getObjName();

    // get and check # of objects to creeate
    int numObj = p_numObjects->getValue();
    if (numObj < 1)
    {
        sendError("must create at least 1 object");
        return STOP_PIPELINE;
    }

    // create list of object: NULL-terminated, thus 1 too long
    // specific obj class does not matter, therefore base class type
    polygonObjects = new coDistributedObject *[numObj + 1];
    polygonObjects[numObj] = NULL;

    if (outputObjectName)
    {

        // we need to create unique names for the set elements
        // to create a unique name we take the name of the set object which was
        // creted by the controller and append the set element index
        char *objName = new char[strlen(outputObjectName) + 5];

        /// Loop creating cubes
        int i;
        for (i = 0; i < numObj; i++)
        {
            // create a unique name for the object
            sprintf(objName, "%s_%d", outputObjectName, i);

            // create the first polygon object
            polygonObjects[i] = createCube(objName, i * 2.0, 0.0, 0.0, 1.0);
        }

        // create the set obejct
        setObject = new coDoSet(outputObjectName, (coDistributedObject **)polygonObjects);

        // now the only difference for time sets:
        if (p_timeSteps->getValue())
        {
            char attribValue[32];
            sprintf(attribValue, "1 %d", numObj);
            setObject->addAttribute("TIMESTEP", attribValue);
        }

        // assign output object to port
        outPort_polySet->setCurrentObject(setObject);

        // cleanup: do NOT delete polygonObjects[i] - coDoSet c'tor did it.
        //          do NOT delete setObject         - assigned to port
        delete polygonObjects;
        delete[] objName;

        return CONTINUE_PIPELINE;
    }
    else
    {
        fprintf(stderr, "Covise::get_object_name failed\n");

        // stop the execution of the covise pipeline below this module
        return STOP_PIPELINE;
    }
}

PolygonSet::~PolygonSet()
{
}

coDoPolygons *
PolygonSet::createCube(char *objectName, float cx, float cy, float cz, float size)
{
    coDoPolygons *polygonObj;
    float xCoords[8], yCoords[8], zCoords[8]; // the cube coordinates
    int vertexList[24] = // the index list
        {
          0, 3, 7, 4, 3, 2, 6, 7, 0, 1, 2, 3, 0, 4, 5, 1, 1, 5, 6, 2, 7, 6, 5, 4
        };
    int polygonList[6] = { 0, 4, 8, 12, 16, 20 };

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

    // create the polygons data object
    polygonObj = new coDoPolygons(objectName, 8,
                                  xCoords, yCoords, zCoords,
                                  24, vertexList,
                                  6, polygonList);

    return (polygonObj);
}

MODULE_MAIN(Examples, PolygonSet)
