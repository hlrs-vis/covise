/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadHyperMesh module                                      ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  6.2006                                                      ++
// ++**********************************************************************/

#include "stdio.h"
#include "ReadHyperMesh.h"
#include <util/coRestraint.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>

#include <float.h>
#include <limits.h>
#include <string.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ReadHyperMesh::ReadHyperMesh(int argc, char **argv)
    : coSimpleModule(argc, argv, "Read Altair HyperMesh files")
{
    // module parameters

    fileName = addFileBrowserParam("MeshFileName", "dummy");
    fileName->setValue("./", "*.hm*");
    resultsFileName = addFileBrowserParam("ReslutFileName", "dummy");
    resultsFileName->setValue("./", "*.hm*;*.fma");

    subdivideParam = addBooleanParam("subdivide", "Subdivide tet10 and hex20 elements");
    subdivideParam->setValue(true);

    p_numt = addInt32Param("numt", "Nuber of Timesteps to read");
    p_numt->setValue(1000);
    p_skip = addInt32Param("skip", "Nuber of Timesteps to skip");
    p_skip->setValue(0);

    p_Selection = addStringParam("Selection", "Parts to load");
    p_Selection->setValue("0-9999999");

    // Output ports
    mesh = addOutputPort("mesh", "UnstructuredGrid", "Unstructured Grid");
    mesh->setInfo("Unstructured Grid");
    char buf[1000];
    int i;
    for (i = 0; i < NUMRES; i++)
    {
        sprintf(buf, "data%d", i);
        dataPort[i] = addOutputPort(buf, "Float|Vec3", buf);
        dataPort[i]->setInfo(buf);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void ReadHyperMesh::getLine()
{
    if (!pushedBack)
    {
        if (fgets(line, LINE_SIZE, file) == NULL)
        {
            //   fprintf(stderr, "ReadHyperMesh::getLine(): fgets failed\n" );
        }
    }
    else
        pushedBack = false;

    c = line;
    while (*c != '\0' && isspace(*c))
    {
        c++;
    }
}

void ReadHyperMesh::pushBack()
{
    pushedBack = true;
}

coDistributedObject *ReadHyperMesh::readVec(const char *name)
{
    while (!feof(file))
    {
        getLine();
        if (c[0] == 'V') // vector
        {

            int nodeNum;
            float xc, yc, zc;
            int iret = sscanf(c + 1, "%d %f %f %f", &nodeNum, &xc, &yc, &zc);
            if (iret != 4)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            xData[nodeNum - 1] = xc;
            yData[nodeNum - 1] = yc;
            zData[nodeNum - 1] = zc;
        }
        else if (strncmp(c, "DISPLACE", 8) == 0) // vector
        {

            int nodeNum;
            float xc, yc, zc;
            int iret = sscanf(c + 8, "%d %f %f %f", &nodeNum, &xc, &yc, &zc);
            if (iret != 4)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            xData[nodeNum - 1] = xc;
            yData[nodeNum - 1] = yc;
            zData[nodeNum - 1] = zc;
        }
        else
        {
            break;
        }
    }
    if (doSkip)
        return NULL;
    coDoVec3 *dataObj = new coDoVec3(name, numPoints, &xData[0], &yData[0], &zData[0]);
    if (strncmp(names[numData], "Displace", 8) == 0)
    {
        displacementData = numData;
        haveDisplacements = true;
    }
    return dataObj;
}
coDistributedObject *ReadHyperMesh::readScal(const char *name)
{
    while (!feof(file))
    {
        getLine();
        if (c[0] == 'S') // scalar
        {

            int nodeNum;
            float xc;
            int iret = sscanf(c + 1, "%d %f", &nodeNum, &xc);
            if (iret != 2)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            xData[nodeNum - 1] = xc;
        }
        else if (strncmp(c, "NSTRESS", 7) == 0) // scalar
        {

            int nodeNum;
            float xc;
            int iret = sscanf(c + 7, "%d %f", &nodeNum, &xc);
            if (iret != 2)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            xData[nodeNum - 1] = xc;
        }
        else
        {
            break;
        }
    }
    if (doSkip)
        return NULL;
    coDoFloat *dataObj = new coDoFloat(name, numPoints, &xData[0]);
    return dataObj;
}

int ReadHyperMesh::compute(const char *)
{

    // compute parameters
    if ((file = fopen(fileName->getValue(), "r")) <= 0)
    {
        sendError("ERROR: can't open file %s", fileName->getValue());
        return FAIL;
    }
    pushedBack = false;
    //xPoints.reserve(1000);
    //yPoints.reserve(1000);
    //zPoints.reserve(1000);
    vertices.reserve(1000);
    xPoints.clear();
    yPoints.clear();
    zPoints.clear();
    vertices.clear();
    elements.clear();
    types.clear();

    coRestraint sel;
    // Get the selection
    const char *selection;
    // Covise::get_string_param("Selection",&selection);
    selection = p_Selection->getValue();
    sel.add(selection);

    numberOfTimesteps = 0;
    numPoints = 0;
    numVert = 0;
    numElem = 0;
    subdivide = subdivideParam->getValue();
    while (!feof(file))
    {
        getLine();
        if (strncmp(c, "*component(", 11) == 0) // skip unselected components
        {
            int componentID;
            sscanf(c + 11, "%d", &componentID);
            while (*c && (*c != '"'))
                c++;
            if (*c)
            {
                c++;
                const char *componentName = c;
                while (*c && (*c != '"'))
                    c++;
                if (*c == '"')
                {
                    *c = '\0';
                }
                sendInfo("component %d %s\n", componentID, componentName);
            }
            while (!feof(file) && !sel(componentID))
            {
                getLine();
                if (strncmp(c, "*component(", 11) == 0)
                {
                    sscanf(c + 11, "%d", &componentID);
                    while (*c && (*c != '"'))
                        c++;
                    if (*c)
                    {
                        c++;
                        const char *componentName = c;
                        while (*c && (*c != '"'))
                            c++;
                        if (*c == '"')
                        {
                            *c = '\0';
                        }
                        sendInfo("component %d %s\n", componentID, componentName);
                    }
                }
            }
        }
        if (strncmp(c, "*tetra10(", 9) == 0)
        { // new line node
            int tetNum;
            int vert[10];
            int iret = sscanf(c + 9, "%d,1,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &tetNum, vert, vert + 1, vert + 2, vert + 3, vert + 4, vert + 5, vert + 6, vert + 7, vert + 8, vert + 9);
            if (iret != 11)
            {
                cerr << "parse error reading tetra10 " << iret << endl;
                break;
            }
            if (subdivide)
            {
                if (types.capacity() < (numElem - 20))
                {
                    types.reserve(types.capacity() + 1000);
                }
                elements.push_back(numVert);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_PYRAMID;
                numElem++;
                elements.push_back(elements[numElem - 1] + 5);
                types[numElem] = TYPE_PYRAMID;
                numElem++;

                vertices.push_back(vert[4] - 1);
                numVert++;
                vertices.push_back(vert[7] - 1);
                numVert++;
                vertices.push_back(vert[6] - 1);
                numVert++;
                vertices.push_back(vert[0] - 1);
                numVert++;

                vertices.push_back(vert[4] - 1);
                numVert++;
                vertices.push_back(vert[5] - 1);
                numVert++;
                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[1] - 1);
                numVert++;

                vertices.push_back(vert[5] - 1);
                numVert++;
                vertices.push_back(vert[6] - 1);
                numVert++;
                vertices.push_back(vert[9] - 1);
                numVert++;
                vertices.push_back(vert[2] - 1);
                numVert++;

                vertices.push_back(vert[7] - 1);
                numVert++;
                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[9] - 1);
                numVert++;
                vertices.push_back(vert[3] - 1);
                numVert++;

                //innen pyramids

                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[7] - 1);
                numVert++;
                vertices.push_back(vert[6] - 1);
                numVert++;
                vertices.push_back(vert[5] - 1);
                numVert++;
                vertices.push_back(vert[4] - 1);
                numVert++;

                vertices.push_back(vert[5] - 1);
                numVert++;
                vertices.push_back(vert[6] - 1);
                numVert++;
                vertices.push_back(vert[7] - 1);
                numVert++;
                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[9] - 1);
                numVert++;
            }
            else
            {
                elements.push_back(numVert);
                types.push_back(TYPE_TETRAHEDER);
                int i;
                for (i = 0; i < 4; i++)
                {
                    vertices.push_back(vert[i] - 1);
                    numVert++;
                }
                numElem++;
            }
        }
        if (strncmp(c, "*hexa20(", 8) == 0)
        { // new line node
            int tetNum;
            int vert[20];
            int iret = sscanf(c + 8, "%d,1,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &tetNum, vert, vert + 1, vert + 2, vert + 3, vert + 4, vert + 5, vert + 6, vert + 7, vert + 8, vert + 9, vert + 10, vert + 11, vert + 12, vert + 13, vert + 14, vert + 15, vert + 16, vert + 17, vert + 18, vert + 19);
            if (iret != 21)
            {
                cerr << "parse error reading hexa20 " << iret << endl;
                break;
            }
            if (subdivide)
            {
                if (types.capacity() < (numElem - 20))
                {
                    types.reserve(types.capacity() + 1000);
                }
                elements.push_back(numVert);
                types[numElem] = TYPE_HEXAEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 8);
                types[numElem] = TYPE_PYRAMID;
                numElem++;
                elements.push_back(elements[numElem - 1] + 5);
                types[numElem] = TYPE_PYRAMID;
                numElem++;
                elements.push_back(elements[numElem - 1] + 5);
                types[numElem] = TYPE_PYRAMID;
                numElem++;
                elements.push_back(elements[numElem - 1] + 5);
                types[numElem] = TYPE_PYRAMID;
                numElem++;
                elements.push_back(elements[numElem - 1] + 5);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;
                elements.push_back(elements[numElem - 1] + 4);
                types[numElem] = TYPE_TETRAHEDER;
                numElem++;

                // hex
                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[9] - 1);
                numVert++;
                vertices.push_back(vert[10] - 1);
                numVert++;
                vertices.push_back(vert[11] - 1);
                numVert++;
                vertices.push_back(vert[16] - 1);
                numVert++;
                vertices.push_back(vert[17] - 1);
                numVert++;
                vertices.push_back(vert[18] - 1);
                numVert++;
                vertices.push_back(vert[19] - 1);
                numVert++;

                //pyramids
                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[16] - 1);
                numVert++;
                vertices.push_back(vert[19] - 1);
                numVert++;
                vertices.push_back(vert[11] - 1);
                numVert++;
                vertices.push_back(vert[12] - 1);
                numVert++;

                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[9] - 1);
                numVert++;
                vertices.push_back(vert[17] - 1);
                numVert++;
                vertices.push_back(vert[16] - 1);
                numVert++;
                vertices.push_back(vert[13] - 1);
                numVert++;

                vertices.push_back(vert[9] - 1);
                numVert++;
                vertices.push_back(vert[10] - 1);
                numVert++;
                vertices.push_back(vert[18] - 1);
                numVert++;
                vertices.push_back(vert[17] - 1);
                numVert++;
                vertices.push_back(vert[14] - 1);
                numVert++;

                vertices.push_back(vert[10] - 1);
                numVert++;
                vertices.push_back(vert[11] - 1);
                numVert++;
                vertices.push_back(vert[19] - 1);
                numVert++;
                vertices.push_back(vert[18] - 1);
                numVert++;
                vertices.push_back(vert[15] - 1);
                numVert++;

                // tetrahedra

                vertices.push_back(vert[11] - 1);
                numVert++;
                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[12] - 1);
                numVert++;
                vertices.push_back(vert[0] - 1);
                numVert++;

                vertices.push_back(vert[12] - 1);
                numVert++;
                vertices.push_back(vert[16] - 1);
                numVert++;
                vertices.push_back(vert[19] - 1);
                numVert++;
                vertices.push_back(vert[4] - 1);
                numVert++;

                vertices.push_back(vert[8] - 1);
                numVert++;
                vertices.push_back(vert[9] - 1);
                numVert++;
                vertices.push_back(vert[13] - 1);
                numVert++;
                vertices.push_back(vert[1] - 1);
                numVert++;

                vertices.push_back(vert[13] - 1);
                numVert++;
                vertices.push_back(vert[17] - 1);
                numVert++;
                vertices.push_back(vert[16] - 1);
                numVert++;
                vertices.push_back(vert[5] - 1);
                numVert++;

                vertices.push_back(vert[9] - 1);
                numVert++;
                vertices.push_back(vert[10] - 1);
                numVert++;
                vertices.push_back(vert[14] - 1);
                numVert++;
                vertices.push_back(vert[2] - 1);
                numVert++;

                vertices.push_back(vert[17] - 1);
                numVert++;
                vertices.push_back(vert[14] - 1);
                numVert++;
                vertices.push_back(vert[18] - 1);
                numVert++;
                vertices.push_back(vert[6] - 1);
                numVert++;

                vertices.push_back(vert[10] - 1);
                numVert++;
                vertices.push_back(vert[11] - 1);
                numVert++;
                vertices.push_back(vert[15] - 1);
                numVert++;
                vertices.push_back(vert[3] - 1);
                numVert++;

                vertices.push_back(vert[15] - 1);
                numVert++;
                vertices.push_back(vert[19] - 1);
                numVert++;
                vertices.push_back(vert[18] - 1);
                numVert++;
                vertices.push_back(vert[7] - 1);
                numVert++;
            }
            else
            {

                elements[numElem] = numVert;
                types[numElem] = TYPE_HEXAEDER;
                int i;
                for (i = 0; i < 8; i++)
                {
                    vertices.push_back(vert[i] - 1);
                    numVert++;
                }
                numElem++;
            }
        }
        if (strncmp(c, "*node(", 6) == 0)
        { // new line node
            int nodeNum;
            float xc, yc, zc;
            int iret = sscanf(c + 6, "%d,%f,%f,%f", &nodeNum, &xc, &yc, &zc);
            if (iret != 4)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            if ((numPoints % 1000) == 0)
            {
                xPoints.reserve(xPoints.capacity() + 1000);
                yPoints.reserve(yPoints.capacity() + 1000);
                zPoints.reserve(zPoints.capacity() + 1000);
            }
            if (nodeNum > xData.capacity())
            {
                xPoints[nodeNum - 1] = xc;
                yPoints[nodeNum - 1] = yc;
                zPoints[nodeNum - 1] = zc;
            }
            else
            {
                fprintf(stderr, "error reading coordinates.\n");
                return STOP_PIPELINE;
            }
            numPoints++;
        }
    }

    fclose(file);

    xData.reserve(numPoints);
    yData.reserve(numPoints);
    zData.reserve(numPoints);
    coDoUnstructuredGrid *gridObj;
    // construct the output grid
    if (numPoints)
    {
        char buf[1000];
        sprintf(buf, "%s_Grid", mesh->getObjName());
        gridObj = new coDoUnstructuredGrid(buf, numElem, numVert, numPoints, &elements[0], &vertices[0], &xPoints[0], &yPoints[0], &zPoints[0], &types[0]);
    }
    else
    {
        sendError("could not read points from file %s", fileName->getValue());
        return FAIL;
    }

    if ((file = fopen(resultsFileName->getValue(), "r")) <= 0)
    {
        sendError("ERROR: can't open file %s", resultsFileName->getValue());
        return FAIL;
    }

    numTimeSteps = 0;
    skipped = 0;

    haveDisplacements = false;
    dx = dy = dz = NULL;
    while (!feof(file))
    {
        getLine();
        if (strncmp(c, "SUBCASE", 7) == 0)
        { // new line node
            numData = 0;
            if (skipped < p_skip->getValue())
            {
                skipped++;
                doSkip = true;
                sendInfo("Skipping %d", skipped);
            }
            else
            {
                skipped = 0;
                doSkip = false;
                sendInfo("Reading Timestep %d", numTimeSteps);
            }
            while (!feof(file))
            {
                getLine();
                if (strncmp(c, "RESULTS", 7) == 0)
                {
                    names[numData] = new char[strlen(c + 8) + 1];
                    strcpy(names[numData], c + 8);
                    getLine();

                    char buf[1000];
                    sprintf(buf, "%s_%d_%d", dataPort[numData]->getObjName(), numData, numTimeSteps);
                    if (c[0] == 'V')
                    {
                        pushBack();
                        dataObjects[numData][numTimeSteps] = readVec(buf);
                    }
                    else if (c[0] == 'S')
                    {
                        pushBack();
                        dataObjects[numData][numTimeSteps] = readScal(buf);
                    }
                    else if (strncmp(c, "DISPLACE", 8) == 0) // vector
                    {
                        pushBack();
                        dataObjects[numData][numTimeSteps] = readVec(buf);
                    }
                    else if (strncmp(c, "NSTRESS", 7) == 0) // scalar
                    {
                        pushBack();
                        dataObjects[numData][numTimeSteps] = readScal(buf);
                    }
                    numData++;
                }
                else if (strncmp(c, "SUBCASE", 7) == 0) // new timestep, so quit this loop
                {
                    pushBack();
                    break;
                }
            }
            if (dataObjects[0][numTimeSteps]) // if we actually have data for this timestep go to next timestep
                numTimeSteps++;
        }
        if (numTimeSteps >= p_numt->getValue())
            break;
    }
    fclose(file);
    int n;
    int i;
    for (n = 0; n < numData; n++)
    {
        sendInfo("DataValue %d :%s", n, names[n]);
        coDistributedObject **dataObjs = new coDistributedObject *[numTimeSteps + 1];
        for (i = 0; i < numTimeSteps; i++)
        {
            dataObjs[i] = dataObjects[n][i];
        }
        dataObjs[numTimeSteps] = NULL;
        coDoSet *dataSet = new coDoSet(dataPort[n]->getObjName(), dataObjs);
        if (numTimeSteps > 1)
            dataSet->addAttribute("TIMESTEP", "1 10");
        delete[] dataObjs;
        // Assign sets to output ports:
        dataPort[n]->setCurrentObject(dataSet);
    }
    if (numTimeSteps == 0)
        numTimeSteps = 1;
    coDistributedObject **gridObjects = new coDistributedObject *[numTimeSteps + 1];
    gridObjects[0] = gridObj;
    for (i = 1; i < numTimeSteps; i++)
    {
        if (haveDisplacements)
        {
            char buf[1000];
            sprintf(buf, "%s_Grid_%d", mesh->getObjName(), i);
            coDoVec3 *dataObj = (coDoVec3 *)dataObjects[displacementData][i];
            dataObj->getAddresses(&dx, &dy, &dz);
            int n;
            for (n = 0; n < numPoints; n++) // displaycements are relative to the original point, so add that to the displacements
            {
                dx[n] += xPoints[n];
                dy[n] += yPoints[n];
                dz[n] += zPoints[n];
            }
            gridObj = new coDoUnstructuredGrid(buf, numElem, numVert, numPoints, &elements[0], &vertices[0], dx, dy, dz, &types[0]);
        }
        else
        {
            gridObj->incRefCount();
        }
        gridObjects[i] = gridObj;
    }
    gridObjects[numTimeSteps] = NULL;
    coDoSet *gridSet = new coDoSet(mesh->getObjName(), gridObjects);

    if (numTimeSteps > 1)
        gridSet->addAttribute("TIMESTEP", "1 10");
    delete[] gridObjects;
    // Assign sets to output ports:
    mesh->setCurrentObject(gridSet);

    return SUCCESS;
}

MODULE_MAIN(IO, ReadHyperMesh)
