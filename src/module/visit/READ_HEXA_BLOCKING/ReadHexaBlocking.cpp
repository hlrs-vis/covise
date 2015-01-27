/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)2000 RUS **
 **                                                                        **
 ** Description:  Reader for ICEMCFD Hexa Blocking Files	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Gabor Duroska                                                  **
 ** Date: April 2000                                                       **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadHexaBlocking.h"

#define LINE_SIZE 8192

int i, n, m, k, h; // counters
int GridElements; // number of grid elements

void main(int argc, char *argv[])
{
    ReadHexaBlocking *application = new ReadHexaBlocking();
    application->start(argc, argv);
}

ReadHexaBlocking::ReadHexaBlocking()
{
    // this information appears in the module setup windows
    set_module_description("Reader for ICEMCFD Hexa Blocking Files");

    // select the file name with a file browser
    param_file = addFileBrowserParam("blockingFile", "ICEMCFD Hexa Blocking File");
    param_file->setValue("data/blocking", "*");

    outPort_polySet = addOutputPort("ploygon_set", "coDoPolygons", "set of polygons, each polygon is a cube");
    outPort_lineSet = addOutputPort("line_set", "coDoLines", "set of lines");

    p_min = addInt32Param("min", "Minimal Material ID");
    p_min->setValue(0);

    p_max = addInt32Param("max", "Maximal Material ID");
    p_max->setValue(9999999);

    p_scale = addFloatSliderParam("scale", "size of cubes");
    p_scale->setMin(0.1);
    p_scale->setMax(1.0);
    p_scale->setValue(1.0);

    const char *selectChoice[] = { "as ploygon", "as block" };
    set_type = addChoiceParam("set type", "select what a set should be");
    set_type->setValue(2, selectChoice, 0);
}

ReadHexaBlocking::~ReadHexaBlocking()
{
}

void ReadHexaBlocking::quit()
{
}

int ReadHexaBlocking::openFile()
{
    char infobuf[500];

    strcpy(infobuf, "opening file ");
    strcat(infobuf, filename);
    sendInfo(infobuf);

    // open the block file
    if ((fp = fopen((char *)filename, "r")) == NULL)
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, filename);
        sendError(infobuf);
        return (false);
    }
    else
    {
        return (true);
    }
}

void ReadHexaBlocking::readFile()
{
    char infobuf[500];
    char line[LINE_SIZE];
    char key[LINE_SIZE];
    char *first;
    int numScanned;
    int material;
    int param_selection;

    numNodes = 0;
    numGridElements = 0;
    GridElements = 0;

    // count NodeElements and GridElements
    strcpy(infobuf, "counting nodes and grid elements");
    sendInfo(infobuf);

    // read one line after another
    while (fgets(line, LINE_SIZE, fp) != NULL)
    {
        first = line;
        sscanf(first, "%s%n", key, &numScanned);
        first += numScanned;

        // count nodes
        if (strcasecmp("NODES", key) == 0)
        {
            // skip one line
            fgets(line, LINE_SIZE, fp);

            do
            {
                fgets(line, LINE_SIZE, fp);
                first = line;
                sscanf(first, "%s%n", key, &numScanned);
                first += numScanned;

                if (strcasecmp("}", key) != 0)
                    numNodes++;
            } while (strcasecmp("}", key) != 0);
        }

        // count grid elements which materials are not '-1'
        if (strcasecmp("GRID_ELEMENTS", key) == 0)
        {
            // skip one line
            fgets(line, LINE_SIZE, fp);

            do
            {
                fgets(line, LINE_SIZE, fp);
                first = line;
                sscanf(first, "%s%n", key, &numScanned);
                first += numScanned;

                if (strcasecmp("}", key) != 0)
                    sscanf(first, "%*d %*d %d", &material);
                if (material != -1)
                {
                    numGridElements++;
                    GridElements++;
                }
                else
                    GridElements++;
            } while (strcasecmp("}", key) != 0);
        }
    }

    // set array dimensions
    nodesTab = new NodesTabEntry[numNodes];
    gridElementsTab = new GridElementsTabEntry[numGridElements];

    // open file again for data input
    strcpy(infobuf, "reading nodes and grid elements");
    sendInfo(infobuf);
    openFile();

    while (fgets(line, LINE_SIZE, fp) != NULL)
    {
        first = line;
        sscanf(first, "%s%n", key, &numScanned);
        first += numScanned;

        // goto NODES
        if (strcasecmp("NODES", key) == 0)
        {
            //skip one line
            fgets(line, LINE_SIZE, fp);

            //get data now
            for (i = 0; i < numNodes; i++)
            {
                fgets(line, LINE_SIZE, fp);
                first = line;
                sscanf(first, "%f %f %f %*d %*d %d",
                       &nodesTab[i].x,
                       &nodesTab[i].y,
                       &nodesTab[i].z,
                       &nodesTab[i].number);
            }
        }

        // goto GRID_ELEMENTS
        if (strcasecmp("GRID_ELEMENTS", key) == 0)
        {
            // skip one line
            fgets(line, LINE_SIZE, fp);

            // get data now
            k = 0;
            for (i = 0; i < GridElements; i++)
            {
                fgets(line, LINE_SIZE, fp);
                first = line;
                sscanf(first, "%*d %*d %d", &material);
                if (material != -1)
                {
                    sscanf(first, "%*d %d %d %d %d %d %d %d %d %d %d",
                           &gridElementsTab[k].output_block_nr,
                           &gridElementsTab[k].material_id,
                           &gridElementsTab[k].nodes[0],
                           &gridElementsTab[k].nodes[1],
                           &gridElementsTab[k].nodes[2],
                           &gridElementsTab[k].nodes[3],
                           &gridElementsTab[k].nodes[4],
                           &gridElementsTab[k].nodes[5],
                           &gridElementsTab[k].nodes[6],
                           &gridElementsTab[k].nodes[7]);

                    k++;
                }
            }
        }
    }

    // get the choice parameter's value and decide which algotithm to use
    param_selection = set_type->getValue();

    // compute the coordinates
    computeCube(param_selection);
    computeWireframe(param_selection);
}

int ReadHexaBlocking::computeCube(const int mark)
{
    char infobuf[500];
    char **polygonObjectNames = new char *[numGridElements];
    coDoPolygons **polygonObjects = new coDoPolygons *[numGridElements];
    const char *outputObjectName_poly;
    coDoSet *setObject_poly;
    int block_counter = 0;
    int cntr;
    const int minimum = p_min->getValue();
    const int maximum = p_max->getValue();
    const float scale = p_scale->getValue();

    // make cube sets
    switch (mark)
    {
    case 0: // a set is a ploygon
        strcpy(infobuf, "creating polygons");
        sendInfo(infobuf);

        // get the set name from the controller
        outputObjectName_poly = outPort_polySet->getObjName();

        if (outputObjectName_poly)
        {
            block_counter = 0;
            for (n = 0; n < numGridElements; n++)
            {
                if (gridElementsTab[n].material_id >= minimum && gridElementsTab[n].material_id <= maximum)
                {
                    // create a unique name for the object
                    polygonObjectNames[block_counter] = new char[strlen(outputObjectName_poly) + 100];
                    sprintf(polygonObjectNames[block_counter], "%s %d", outputObjectName_poly, block_counter);

                    // create the ploygon objects
                    polygonObjects[block_counter] = createCube(polygonObjectNames[block_counter], n, scale);

                    block_counter++;
                }
            }

            // create the set object
            setObject_poly = new coDoSet(outputObjectName_poly, block_counter, (coDistributedObject **)polygonObjects);

            outPort_polySet->setCurrentObject(setObject_poly);

            // clean up
            for (n = 0; n < block_counter; n++)
                delete polygonObjectNames[n];

            return CONTINUE_PIPELINE;
        }
        else
        {
            fprintf(stderr, "Covise::get_object_name failed\n");

            // stop execution here
            return STOP_PIPELINE;
        }
        break;

    case 1: // a set is a block
        strcpy(infobuf, "creating polygons");
        sendInfo(infobuf);

        // get the set name from the controller
        outputObjectName_poly = outPort_polySet->getObjName();

        if (outputObjectName_poly)
        {
            for (n = minimum; n <= maximum; n++)
            {
                cntr = 0;
                for (k = 0; k < numGridElements; k++)
                    if (gridElementsTab[k].material_id == n)
                        cntr++;

                if (cntr != 0)
                {
                    // create a unique name for the object
                    polygonObjectNames[block_counter] = new char[strlen(outputObjectName_poly) + 100];
                    sprintf(polygonObjectNames[block_counter], "%s %d", outputObjectName_poly, block_counter);

                    // create the polygon objects
                    polygonObjects[block_counter] = createCube(polygonObjectNames[block_counter], n, cntr, scale);

                    block_counter++;
                }
            }

            // create the set object
            setObject_poly = new coDoSet(outputObjectName_poly, block_counter, (coDistributedObject **)polygonObjects);

            outPort_polySet->setCurrentObject(setObject_poly);

            // clean up
            for (n = 0; n < block_counter; n++)
                delete polygonObjectNames[n];

            return CONTINUE_PIPELINE;
        }
        else
        {
            fprintf(stderr, "Covise::get_object_name failed\n");

            // stop execution here
            return STOP_PIPELINE;
        }
        break;

    default:
        // never reach this point
        return STOP_PIPELINE;
        break;
    }
}

int ReadHexaBlocking::computeWireframe(const int mark)
{
    char infobuf[500];
    char **wireframeObjectNames = new char *[numGridElements];
    coDoLines **wireframeObjects = new coDoLines *[numGridElements];
    const char *outputObjectName_wireframe;
    coDoSet *setObject_wireframe;
    int block_counter = 0;
    int cntr;
    const int minimum = p_min->getValue();
    const int maximum = p_max->getValue();
    const float scale = p_scale->getValue();

    // make wireframe sets
    switch (mark)
    {
    case 0: // a set is a hexahedron
        strcpy(infobuf, "creating wireframe");
        sendInfo(infobuf);

        // get the set name from the controller
        outputObjectName_wireframe = outPort_lineSet->getObjName();

        if (outputObjectName_wireframe)
        {
            block_counter = 0;
            for (n = 0; n < numGridElements; n++)
            {
                if (gridElementsTab[n].material_id >= minimum && gridElementsTab[n].material_id <= maximum)
                {
                    // create a unique name for the object
                    wireframeObjectNames[block_counter] = new char[strlen(outputObjectName_wireframe) + 100];
                    sprintf(wireframeObjectNames[block_counter], "%s %d", outputObjectName_wireframe, block_counter);

                    // create the wireframe objects
                    wireframeObjects[block_counter] = createWireframe(wireframeObjectNames[block_counter], n, scale);

                    block_counter++;
                }
            }

            // create the set object
            setObject_wireframe = new coDoSet(outputObjectName_wireframe, block_counter, (coDistributedObject **)wireframeObjects);

            outPort_lineSet->setCurrentObject(setObject_wireframe);

            // clean up
            for (n = 0; n < block_counter; n++)
                delete wireframeObjectNames[n];

            return CONTINUE_PIPELINE;
        }
        else
        {
            fprintf(stderr, "Covise::get_object_name failed\n");

            // stop execution here
            return STOP_PIPELINE;
        }
        break;

    case 1: // a set is a block
        strcpy(infobuf, "creating wireframe");
        sendInfo(infobuf);

        // get the set name from the controller
        outputObjectName_wireframe = outPort_lineSet->getObjName();

        if (outputObjectName_wireframe)
        {
            for (n = minimum; n <= maximum; n++)
            {
                cntr = 0;
                for (k = 0; k < numGridElements; k++)
                    if (gridElementsTab[k].material_id == n)
                        cntr++;

                if (cntr != 0)
                {
                    // create a unique name for the object
                    wireframeObjectNames[block_counter] = new char[strlen(outputObjectName_wireframe) + 100];
                    sprintf(wireframeObjectNames[block_counter], "%s %d", outputObjectName_wireframe, block_counter);

                    // create the wireframe objects
                    wireframeObjects[block_counter] = createWireframe(wireframeObjectNames[block_counter], n, cntr, scale);

                    block_counter++;
                }
            }

            // create the set object
            setObject_wireframe = new coDoSet(outputObjectName_wireframe, block_counter, (coDistributedObject **)wireframeObjects);

            outPort_lineSet->setCurrentObject(setObject_wireframe);

            // clean up
            for (n = 0; n < block_counter; n++)
                delete wireframeObjectNames[n];

            return CONTINUE_PIPELINE;
        }
        else
        {
            fprintf(stderr, "Covise::get_object_name failed\n");

            // stop execution here
            return STOP_PIPELINE;
        }
        break;

    default:
        // never reach this point
        return STOP_PIPELINE;
        break;
    }
}

coDoLines *ReadHexaBlocking::createWireframe(char *objectName, const int counter, const float scale)
{
    float xCoords[8], yCoords[8], zCoords[8];
    int lineList[4] = { 0, 5, 10, 15 };
    int cornerList[20] = { 0, 4, 5, 1, 0, 2, 3, 7, 6, 2, 6, 4, 0, 2, 6, 5, 7, 3, 1, 5 };
    coDoLines *wireframeObj;
    int dataList[8];
    int pos;
    int found;
    float center_x, center_y, center_z;

    // make dataList
    for (i = 0; i < 8; i++)
        dataList[i] = gridElementsTab[counter].nodes[i];

    // make coordinates
    for (i = 0; i < 8; i++)
    {
        pos = 0;
        found = 0;
        do
        {
            if (nodesTab[pos].number == dataList[i])
            {
                xCoords[i] = nodesTab[pos].x;
                yCoords[i] = nodesTab[pos].y;
                zCoords[i] = nodesTab[pos].z;
                found = 1;
            }
            pos++;
        } while (found != 1 || pos == numNodes);
    }

    // count gravity-center
    center_x = ((xCoords[0] + xCoords[1] + xCoords[2] + xCoords[3] + xCoords[4] + xCoords[5] + xCoords[6] + xCoords[7]) / 8);
    center_y = ((yCoords[0] + yCoords[1] + yCoords[2] + yCoords[3] + yCoords[4] + yCoords[5] + yCoords[6] + yCoords[7]) / 8);
    center_z = ((zCoords[0] + zCoords[1] + zCoords[2] + zCoords[3] + zCoords[4] + zCoords[5] + zCoords[6] + zCoords[7]) / 8);

    // scale coordinates
    for (i = 0; i < 8; i++)
    {
        xCoords[i] = scale * xCoords[i] + (1 - scale) * center_x;
        yCoords[i] = scale * yCoords[i] + (1 - scale) * center_y;
        zCoords[i] = scale * zCoords[i] + (1 - scale) * center_z;
    }

    // create the wireframe data object
    wireframeObj = new coDoLines(objectName, 8, xCoords, yCoords, zCoords, 20, cornerList, 4, lineList);
    return (wireframeObj);
}

coDoLines *ReadHexaBlocking::createWireframe(char *objectName, const int this_block, const int counter, const float scale)
{
    const int num_coords = 8 * counter;
    const int num_lines = 4 * counter;
    const int num_corners = 20 * counter;
    float *xCoords = new float[num_coords];
    float *yCoords = new float[num_coords];
    float *zCoords = new float[num_coords];
    int *lineList = new int[num_lines];
    int *cornerList = new int[num_corners];
    int dataList[8];
    coDoLines *wireframeObj;
    int pos;
    int found;
    int cntr = 0;
    float center_x, center_y, center_z;

    for (i = 0; i < numGridElements; i++)
    {
        if (gridElementsTab[i].material_id == this_block)
        {
            // make dataList
            for (k = 0; k < 8; k++)
                dataList[k] = gridElementsTab[i].nodes[k];

            // make coordinates
            for (k = 0; k < 8; k++)
            {
                pos = 0;
                found = 0;
                do
                {
                    if (nodesTab[pos].number == dataList[k])
                    {
                        xCoords[cntr] = nodesTab[pos].x;
                        yCoords[cntr] = nodesTab[pos].y;
                        zCoords[cntr] = nodesTab[pos].z;
                        found = 1;
                        cntr++;
                    }
                    pos++;
                } while (found != 1 || pos == numNodes);
            }
        }
    }

    for (k = 0; k < num_coords; k += 8)
    {
        // count gravity-center
        center_x = ((xCoords[k + 0] + xCoords[k + 1] + xCoords[k + 2] + xCoords[k + 3] + xCoords[k + 4] + xCoords[k + 5] + xCoords[k + 6] + xCoords[k + 7]) / 8);
        center_y = ((yCoords[k + 0] + yCoords[k + 1] + yCoords[k + 2] + yCoords[k + 3] + yCoords[k + 4] + yCoords[k + 5] + yCoords[k + 6] + yCoords[k + 7]) / 8);
        center_z = ((zCoords[k + 0] + zCoords[k + 1] + zCoords[k + 2] + zCoords[k + 3] + zCoords[k + 4] + zCoords[k + 5] + zCoords[k + 6] + zCoords[k + 7]) / 8);

        // scale coordinates
        for (m = k; m < k + 8; m++)
        {
            xCoords[m] = scale * xCoords[m] + (1 - scale) * center_x;
            yCoords[m] = scale * yCoords[m] + (1 - scale) * center_y;
            zCoords[m] = scale * zCoords[m] + (1 - scale) * center_z;
        }
    }

    // make corner list
    cntr = 0;
    for (k = 0; k < num_corners; k += 20)
    {
        cornerList[k] = 0 + cntr;
        cornerList[k + 1] = 4 + cntr;
        cornerList[k + 2] = 5 + cntr;
        cornerList[k + 3] = 1 + cntr;
        cornerList[k + 4] = 0 + cntr;

        cornerList[k + 5] = 2 + cntr;
        cornerList[k + 6] = 3 + cntr;
        cornerList[k + 7] = 7 + cntr;
        cornerList[k + 8] = 6 + cntr;
        cornerList[k + 9] = 2 + cntr;

        cornerList[k + 10] = 6 + cntr;
        cornerList[k + 11] = 4 + cntr;
        cornerList[k + 12] = 0 + cntr;
        cornerList[k + 13] = 2 + cntr;
        cornerList[k + 14] = 6 + cntr;

        cornerList[k + 15] = 5 + cntr;
        cornerList[k + 16] = 7 + cntr;
        cornerList[k + 17] = 3 + cntr;
        cornerList[k + 18] = 1 + cntr;
        cornerList[k + 19] = 5 + cntr;

        cntr = cntr + 8;
    }

    // make line list
    cntr = 0;
    for (k = 0; k < num_lines; k++)
    {
        lineList[k] = cntr;
        cntr += 5;
    }

    // create the wireframe data object
    wireframeObj = new coDoLines(objectName, num_coords, xCoords, yCoords, zCoords, num_corners, cornerList, num_lines, lineList);
    return (wireframeObj);
}

coDoPolygons *ReadHexaBlocking::createCube(char *objectName, const int counter, const float scale)
{
    float xCoords[8], yCoords[8], zCoords[8];
    int vertexList[24] = { 0, 1, 5, 4, 1, 3, 7, 5, 0, 2, 3, 1, 0, 4, 6, 2, 2, 6, 7, 3, 5, 7, 6, 4 };
    int dataList[8];
    int polygonList[6] = { 0, 4, 8, 12, 16, 20 };
    coDoPolygons *polygonObj;
    int pos;
    int found;
    float center_x, center_y, center_z;

    // make dataList
    for (i = 0; i < 8; i++)
        dataList[i] = gridElementsTab[counter].nodes[i];

    // make coordinates
    for (i = 0; i < 8; i++)
    {
        pos = 0;
        found = 0;
        do
        {
            if (nodesTab[pos].number == dataList[i])
            {
                xCoords[i] = nodesTab[pos].x;
                yCoords[i] = nodesTab[pos].y;
                zCoords[i] = nodesTab[pos].z;
                found = 1;
            }
            pos++;
        } while (found != 1 || pos == numNodes);
    }

    // count gravity-center
    center_x = ((xCoords[0] + xCoords[1] + xCoords[2] + xCoords[3] + xCoords[4] + xCoords[5] + xCoords[6] + xCoords[7]) / 8);
    center_y = ((yCoords[0] + yCoords[1] + yCoords[2] + yCoords[3] + yCoords[4] + yCoords[5] + yCoords[6] + yCoords[7]) / 8);
    center_z = ((zCoords[0] + zCoords[1] + zCoords[2] + zCoords[3] + zCoords[4] + zCoords[5] + zCoords[6] + zCoords[7]) / 8);

    // scale coordinates
    for (i = 0; i < 8; i++)
    {
        xCoords[i] = scale * xCoords[i] + (1 - scale) * center_x;
        yCoords[i] = scale * yCoords[i] + (1 - scale) * center_y;
        zCoords[i] = scale * zCoords[i] + (1 - scale) * center_z;
    }

    // create the polygons data object
    polygonObj = new coDoPolygons(objectName, 8, xCoords, yCoords, zCoords, 24, vertexList, 6, polygonList);
    return (polygonObj);
}

coDoPolygons *ReadHexaBlocking::createCube(char *objectName, const int this_block, const int counter, const float scale)
{
    const int num_coords = 8 * counter;
    const int num_vertices = 24 * counter;
    const int num_polys = 6 * counter;
    float *xCoords = new float[num_coords];
    float *yCoords = new float[num_coords];
    float *zCoords = new float[num_coords];
    int *vertexList = new int[num_vertices];
    int dataList[8];
    int *polygonList = new int[num_polys];
    coDoPolygons *polygonObj;
    int pos;
    int found;
    int cntr = 0;
    float center_x, center_y, center_z;

    for (i = 0; i < numGridElements; i++)
    {
        if (gridElementsTab[i].material_id == this_block)
        {
            // make dataList
            for (k = 0; k < 8; k++)
                dataList[k] = gridElementsTab[i].nodes[k];

            // make coordinates
            for (k = 0; k < 8; k++)
            {
                pos = 0;
                found = 0;
                do
                {
                    if (nodesTab[pos].number == dataList[k])
                    {
                        xCoords[cntr] = nodesTab[pos].x;
                        yCoords[cntr] = nodesTab[pos].y;
                        zCoords[cntr] = nodesTab[pos].z;
                        found = 1;
                        cntr++;
                    }
                    pos++;
                } while (found != 1 || pos == numNodes);
            }
        }
    }

    for (k = 0; k < num_coords; k += 8)
    {
        // count gravity-center
        center_x = ((xCoords[k + 0] + xCoords[k + 1] + xCoords[k + 2] + xCoords[k + 3] + xCoords[k + 4] + xCoords[k + 5] + xCoords[k + 6] + xCoords[k + 7]) / 8);
        center_y = ((yCoords[k + 0] + yCoords[k + 1] + yCoords[k + 2] + yCoords[k + 3] + yCoords[k + 4] + yCoords[k + 5] + yCoords[k + 6] + yCoords[k + 7]) / 8);
        center_z = ((zCoords[k + 0] + zCoords[k + 1] + zCoords[k + 2] + zCoords[k + 3] + zCoords[k + 4] + zCoords[k + 5] + zCoords[k + 6] + zCoords[k + 7]) / 8);

        // scale coordinates
        for (m = k; m < k + 8; m++)
        {
            xCoords[m] = scale * xCoords[m] + (1 - scale) * center_x;
            yCoords[m] = scale * yCoords[m] + (1 - scale) * center_y;
            zCoords[m] = scale * zCoords[m] + (1 - scale) * center_z;
        }
    }

    // make vertex list
    cntr = 0;
    for (k = 0; k < num_vertices; k += 24)
    {
        vertexList[k] = 0 + cntr;
        vertexList[k + 1] = 1 + cntr;
        vertexList[k + 2] = 5 + cntr;
        vertexList[k + 3] = 4 + cntr;

        vertexList[k + 4] = 1 + cntr;
        vertexList[k + 5] = 3 + cntr;
        vertexList[k + 6] = 7 + cntr;
        vertexList[k + 7] = 5 + cntr;

        vertexList[k + 8] = 0 + cntr;
        vertexList[k + 9] = 2 + cntr;
        vertexList[k + 10] = 3 + cntr;
        vertexList[k + 11] = 1 + cntr;

        vertexList[k + 12] = 0 + cntr;
        vertexList[k + 13] = 4 + cntr;
        vertexList[k + 14] = 6 + cntr;
        vertexList[k + 15] = 2 + cntr;

        vertexList[k + 16] = 2 + cntr;
        vertexList[k + 17] = 6 + cntr;
        vertexList[k + 18] = 7 + cntr;
        vertexList[k + 19] = 3 + cntr;

        vertexList[k + 20] = 5 + cntr;
        vertexList[k + 21] = 7 + cntr;
        vertexList[k + 22] = 6 + cntr;
        vertexList[k + 23] = 4 + cntr;

        cntr = cntr + 8;
    }

    // make polygon list
    cntr = 0;
    for (k = 0; k < num_polys; k++)
    {
        polygonList[k] = cntr;
        cntr += 4;
    }

    // create the polygons data object
    polygonObj = new coDoPolygons(objectName, num_coords, xCoords, yCoords, zCoords, num_vertices, vertexList, num_polys, polygonList);
    return (polygonObj);
}

int ReadHexaBlocking::compute()
{
    char infobuf[500]; // buffer for COVISE info and error message

    // get the file name
    filename = param_file->getValue();

    if (filename != NULL)
    {
        // open the file
        if (openFile())
        {
            sprintf(infobuf, "File %s open", filename);
            sendInfo(infobuf);

            // read the file, create the lists and create a COVISE set object
            readFile();
        }
        else
        {
            sprintf(infobuf, "Error opening file %s", filename);
            sendError(infobuf);
            return STOP_PIPELINE;
        }
    }
    else
    {
        sendError("ERROR: fileName is NULL");
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

void ReadHexaBlocking::postInst()
{
    param_file->show();
    p_min->show();
    p_max->show();
    p_scale->show();
    set_type->show();
}
