/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*****************************************************
Description:	Data reader for Ensight format
Author:		B.Teplitski
Created:	19.09.2000
To do:		22.11.00: Timestep file names saved still in a file -> change to ENV-variables???
      24.11 Still not implemented: geometry changing
      15.12 change data directory data_bo
      30.01.2001: delete parameter paramLenghOfCornerLineElement and hardcode this value for different versions
*****************************************************/

#include "ReadEnsight.h"
#include <api/coFeedback.h>

#include <util/coviseCompat.h>
#include <stdlib.h>
#include <ctype.h>

/******************************************************************
Function			Constructor
Parameter
Return
Description			Creates output ports for all data objects, set parameters for data files input
               (file browser window) and defines input files
******************************************************************/
ReadEnsight::ReadEnsight(int argc, char *argv[])
    : coModule(argc, argv, "Ensight")
    , caseFileFlg_(0)
    , translList_(NULL)
{
    geoFileParam = addFileBrowserParam("geoFile", "OBJ file");
    geoFileParam->setValue("data/geoFile", "*.*");

    caseFileParam = addFileBrowserParam("caseFile", "Case file");
    caseFileParam->setValue("data/caseFile", "*.case");

    resFileParam = addFileBrowserParam("resFile", "Result file");
    resFileParam->setValue("data/resFile", "*.res");

    gridOutPort = addOutputPort("grid", "UnstructuredGrid", "Grid object");
    polygonOutPort = addOutputPort("polygons", "Polygons", "Polygon object");
    gridScalarOutPort = addOutputPort("GridScalarOutPort", "Float", "Scalar data mapped to grids");
    gridVectorOutPort = addOutputPort("GridVectorOutPort", "Vec3", "Vector data mapped to grids");
    polygonScalarOutPort = addOutputPort("PolygonScalarOutPort", "Float", "Scalar data mapped to polygons");
    polygonVectorOutPort = addOutputPort("PolygonVectorOutPort", "Vec3", "Vector data mapped to polygons");

    paramGridScalarFile = addFileBrowserParam("gridScalarFile", "Grid Scalar File");
    paramGridScalarFile->setValue("data/gridScalarFile", "*.*");

    paramGridVectorFile = addFileBrowserParam("gridVectorFile", "Grid Vector File");
    paramGridVectorFile->setValue("data/gridVectorFile", "*.*");

    paramPolygonScalarFile = addFileBrowserParam("polygonScalarFile", "Polygon Scalar File");
    paramPolygonScalarFile->setValue("data/polygonScalarFile", "*.*");

    paramPolygonVectorFile = addFileBrowserParam("polygonVectorFile", "Polygon Vector File");
    paramPolygonVectorFile->setValue("data/polygonVectorFile", "*.*");

    paramLenghOfLineElement = addInt32Param("LineElementLength", "Length of the line element in a data file");
    paramLenghOfLineElement->setValue(12); //standard settings, ensight format

    //this sparameter later changed to const in program
    //paramLenghOfCornerLineElement=addInt32Param("CornerLineElementLengh","Length of the corner line element in a geometry file");
    //paramLenghOfCornerLineElement->setValue(10);

    paramNumberValuesInLine = addInt32Param("NumberValuesInLine", "Number of values in a line of a data file");
    paramNumberValuesInLine->setValue(6);
}

ReadEnsight::~ReadEnsight() {}
void ReadEnsight::postInst() {}

int
ReadEnsight::computeGold()
{

    sendInfo("ensight version gold");
    char strLine[250];
    char strFigureName[20];
    char strTemp[20];

    char buf[500];

    intElementCount = 0;
    intElementCornerCount = 0;
    intElementNumber = 0;
    intElementCornerNumber = 0;
    intCoordsNumber = 0;
    intNodeIdSectionExist = 0;
    intElementIDSectionExist = 0;
    intLineCounter = 0;
    intFileMapIndex = 0;
    bool isPoly = false, isGri = false;
    int i, j, k;
    int tmp;
    char strSetElementName[100];
    for (i = 0; i < 20; i++)
    {
        strTemp[i] = 0;
    }
    LINE_ELEMENT_LENGTH = paramLenghOfLineElement->getValue();
    //CORNER_LINE_ELEMENT_LENGTH = paramLenghOfCornerLineElement->getValue();
    CORNER_LINE_ELEMENT_LENGTH = 10;
    NUMBER_VALUES_IN_LINE = paramNumberValuesInLine->getValue();

    coDistributedObject **objects = new coDistributedObject *[intNumberOfTimeSteps + 1];
    if (objects == NULL)
    {
        sendInfo("Memory allocation for timesteps object failed");
        return FAIL;
    }

    objects[intNumberOfTimeSteps] = NULL;

    ifstream ifstrFile(geoFileParam->getValue(), ios::in);
    if (ifstrFile.good())
    {
        //first 2 lines - description
        for (i = 0; i < 3; i++)
        {
            ifstrFile.get(strLine, 250);
            ifstrFile.seekg(1L, ios::cur);
            intLineCounter++;
        }

        //third line - node id info
        if ((strcmp(strLine, "node id given") == 0) || (strcmp(strLine, "node id ignore") == 0))
        {
            intNodeIdSectionExist = 1;
        }

        //forth line - element id
        ifstrFile.get(strLine, 250);
        ifstrFile.seekg(1L, ios::cur);
        intLineCounter++;
        if ((strcmp(strLine, "element id given") == 0) || (strcmp(strLine, "element id ignore") == 0))
        {
            intElementIDSectionExist = 1;
        }

        //5th line and next 3 lines - optional extens block
        ifstrFile.get(strLine, 250);
        ifstrFile.seekg(1L, ios::cur);
        intLineCounter++;
        if (strcmp(strLine, "extents") == 0)
        {
            for (i = 0; i < 3; i++)
            {
                ifstrFile.get(strLine, 250);
                ifstrFile.seekg(1L, ios::cur);
                intLineCounter++;
            }
        }
        initGridMap();
        initPolygonMap();

        while (!ifstrFile.eof() && !ifstrFile.bad() && (strlen(strLine) != 0))
        {
            ifstrFile.get(strLine, 250);
            ifstrFile.seekg(1L, ios::cur);
            intLineCounter++;

            for (i = 0; i < 4; i++) //read in coordinates header up to line with coordinates number
            {
                ifstrFile.get(strLine, 250);
                ifstrFile.seekg(1L, ios::cur);
                intLineCounter++;
            }
            intCoordsNumber = atoi(strLine);
            if (!intCoordsNumber)
            {
                break;
            }

            fltXCoords = new float[intCoordsNumber + 1];
            if (fltXCoords == NULL)
            {
                sendInfo("Memory allocation for list of X-coordinates  failed");
                return FAIL;
            }
            fltYCoords = new float[intCoordsNumber + 1];
            if (fltYCoords == NULL)
            {
                sendInfo("Memory allocation for list of Y-coordinates failed");
                return FAIL;
            }
            fltZCoords = new float[intCoordsNumber + 1];
            if (fltZCoords == NULL)
            {
                sendInfo("Memory allocation for list of Z-coordinates failed");
                return FAIL;
            }

            ifstream &iFile = ifstrFile;

            //if(!read3DCoordsSequent(iFile, intCoordsNumber, &(*fltXCoords), &(*fltYCoords), &(*fltZCoords)))
            if (!read3DCoordsSequent(iFile, intCoordsNumber, fltXCoords, fltYCoords, fltZCoords))
            {
                sendInfo("Failed to read in coordinates from geometry file.");
                return FAIL;
            }
            initFileMap();

            while (!ifstrFile.eof() && !ifstrFile.bad() && (strlen(strLine) != 0))
            {
                ifstrFile.get(strLine, 250);
                ifstrFile.seekg(1L, ios::cur);
                intLineCounter++;
                strcpy(strFigureName, strLine); //save element type
                if (isPolygon(strFigureName))
                {
                    isPoly = true;
                    isGri = false;
                }
                else if (isGrid(strFigureName))
                {
                    isGri = true;
                    isPoly = false;
                }
                else //we found the word "part" or eof, close the file and create objects
                {
                    break;
                }
                //line with number of elements
                ifstrFile.get(strLine, 250);
                ifstrFile.seekg(1L, ios::cur);
                intLineCounter++;

                tmp = atoi(strLine);

                intElementNumber += tmp;
                intElementCornerNumber += tmp * getNumberOfCorner(strFigureName);

                if (isPoly)
                {
                    fillPolygonMap(tmp);
                }
                else if (isGri)
                {
                    fillGridMap(tmp);
                }

                //if numbering exists
                if (intElementIDSectionExist)
                {
                    fillFileMap(intLineCounter + tmp, getNumberOfCorner(strFigureName), tmp, getType(strFigureName));
                    tmp *= 2;
                }
                else
                {
                    fillFileMap(intLineCounter, getNumberOfCorner(strFigureName), tmp, getType(strFigureName));
                }

                for (i = 0; i < tmp; i++)
                {
                    ifstrFile.get(strLine, 250);
                    ifstrFile.seekg(1L, ios::cur);
                    intLineCounter++;
                }
            }

            intLineCounter = 0;

            intElementList = new int[intElementNumber + 1];
            if (intElementList == NULL)
            {
                sendInfo("Memory allocation for element list failed");
                return FAIL;
            }
            int i;
            for (i = 0; i <= intElementNumber; i++)
            {
                intElementList[i] = 0;
            }
            intElementCornerList = new int[intElementCornerNumber + 1];
            if (intElementCornerList == NULL)
            {
                sendInfo("Memory allocation for vertex list failed");
                return FAIL;
            }
            for (i = 0; i <= intElementCornerNumber; i++)
            {
                intElementCornerList[i] = 0;
            }
            intTypeList = new int[intElementNumber + 1];
            if (intTypeList == NULL)
            {
                sendInfo("Memory allocation for type list failed");
                return FAIL;
            }

            for (i = 0; i < 100; i++) //for the hole map
            {
                if (intFileMap[i][0] == 0)
                {
                    break;
                }
                ifstrFile.close(); //to set the file pointer to file begin was not possible
                ifstrFile.open(geoFileParam->getValue(), ios::in);
                ifstrFile.seekg(0L, ios::beg); //go to begin of file

                intLineCounter = 0;
                for (j = 0; j < intFileMap[i][0]; j++) //remote forward up to the line with corners
                {
                    ifstrFile.get(strLine, 250);
                    ifstrFile.seekg(1L, ios::cur);
                    intLineCounter++;
                }
                for (j = 0; j < intFileMap[i][2]; j++) //throught all the lines with corners
                {
                    ifstrFile.get(strLine, 250);
                    ifstrFile.seekg(1L, ios::cur);
                    intLineCounter++;
                    for (k = 0; k < intFileMap[i][1]; k++)
                    {
                        //copy line element number j to strTemp
                        strncpy(strTemp, strLine + CORNER_LINE_ELEMENT_LENGTH * k, CORNER_LINE_ELEMENT_LENGTH);
                        //sometimes we must substract 1;
                        intElementCornerList[intElementCornerCount] = atoi(strTemp) - 1;
                        intElementCornerCount++;
                    }
                    intElementList[intElementCount + 1] = intElementList[intElementCount] + intFileMap[i][1];
                    intTypeList[intElementCount] = intFileMap[i][3];
                    intElementCount++;
                }
            }

            if (isPoly)
            {

                for (i = 0; i < intNumberOfTimeSteps; i++)
                {
                    objects[i] = NULL;
                }
                if (intNumberOfTimeSteps > 1)
                {

                    for (i = 0; i < intNumberOfTimeSteps; i++)
                    {
                        strcpy(strSetElementName, polygonOutPort->getObjName());
                        sprintf(buf, "%d", i);
                        strcat(strSetElementName, buf);
                        objects[i] = new coDoPolygons(strSetElementName,
                                                      intCoordsNumber,
                                                      fltXCoords,
                                                      fltYCoords,
                                                      fltZCoords,
                                                      intElementCornerNumber,
                                                      intElementCornerList,
                                                      intElementNumber,
                                                      intElementList);
                        objects[i]->addAttribute("vertexOrder", "2");
                        if (objects[i] == NULL)
                        {
                            strcpy(buf, "Cannot create object ");
                            strcat(buf, polygonOutPort->getObjName());
                            sendInfo(buf);
                            return FAIL;
                        }
                        //save for scalar values
                        intPolygonNumber = intElementNumber;
                        intPolygonCoordsNumber = intCoordsNumber;
                    }
                    coDoSet *polygonSet = new coDoSet(polygonOutPort->getObjName(), objects);
                    if (polygonSet == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, polygonOutPort->getObjName());
                        sendInfo(buf);
                        return FAIL;
                    }
                    sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
                    polygonSet->addAttribute("TIMESTEP", buf);
                    polygonSet->addAttribute("vertexOrder", "2");
                    polygonOutPort->setCurrentObject(polygonSet);
                    sendInfoSuccess(polygonOutPort->getName());

                    for (i = 0; i < intNumberOfTimeSteps; i++)
                    {
                        if (objects[i] != NULL)
                        {
                            delete objects[i];
                        }
                    }
                }
                else if (intNumberOfTimeSteps == 1)
                {
                    polygonObj = new coDoPolygons(polygonOutPort->getObjName(),
                                                  intCoordsNumber,
                                                  fltXCoords,
                                                  fltYCoords,
                                                  fltZCoords,
                                                  intElementCornerNumber,
                                                  intElementCornerList,
                                                  intElementNumber,
                                                  intElementList);
                    polygonObj->addAttribute("vertexOrder", "2");
                    if (polygonObj == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, polygonOutPort->getObjName());
                        sendInfo(buf);
                        return FAIL;
                    }

                    intPolygonNumber = intElementNumber; //save for scalar values
                    intPolygonCoordsNumber = intCoordsNumber;
                    polygonObj->addAttribute("vertexOrder", "2");
                    polygonOutPort->setCurrentObject(polygonObj);

                    sendInfoSuccess(polygonOutPort->getName());
                }
            }

            else if (isGri)
            {
                for (i = 0; i < intNumberOfTimeSteps; i++)
                {
                    objects[i] = NULL;
                }

                if (intNumberOfTimeSteps > 1)
                {
                    for (i = 0; i < intNumberOfTimeSteps; i++)
                    {
                        strcpy(strSetElementName, gridOutPort->getObjName());
                        sprintf(buf, "%d", i);
                        strcat(strSetElementName, buf);
                        objects[i] = new coDoUnstructuredGrid(strSetElementName,
                                                              intElementNumber,
                                                              intElementCornerNumber,
                                                              intCoordsNumber,
                                                              intElementList,
                                                              intElementCornerList,
                                                              fltXCoords,
                                                              fltYCoords,
                                                              fltZCoords,
                                                              intTypeList);
                        if (objects[i] == NULL)
                        {
                            strcpy(buf, "Cannot create object ");
                            strcat(buf, gridOutPort->getObjName());
                            sendInfo(buf);
                            return FAIL;
                        }
                        intGridNumber = intElementNumber; //save for scalar values
                        intGridCoordsNumber = intCoordsNumber;
                    }
                    coDoSet *gridSet = new coDoSet(gridOutPort->getObjName(), objects);
                    if (gridSet == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, gridOutPort->getObjName());
                        sendInfo(buf);
                        return FAIL;
                    }
                    sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
                    gridOutPort->setCurrentObject(gridSet);
                    gridSet->addAttribute("TIMESTEP", buf);
                    sendInfoSuccess(gridOutPort->getName());

                    for (i = 0; i < intNumberOfTimeSteps; i++)
                    {
                        if (objects[i] != NULL)
                        {
                            delete objects[i];
                        }
                    }
                }
                else if (intNumberOfTimeSteps == 1)
                {
                    gridObj = new coDoUnstructuredGrid(gridOutPort->getObjName(),
                                                       intElementNumber,
                                                       intElementCornerNumber,
                                                       intCoordsNumber,
                                                       intElementList,
                                                       intElementCornerList,
                                                       fltXCoords,
                                                       fltYCoords,
                                                       fltZCoords,
                                                       intTypeList);

                    if (gridObj == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, gridOutPort->getObjName());
                        sendInfo(buf);
                        return FAIL;
                    }

                    intGridNumber = intElementNumber; //save for scalar values
                    intGridCoordsNumber = intCoordsNumber;

                    gridOutPort->setCurrentObject(gridObj);
                    sendInfoSuccess(gridOutPort->getName());
                }
            }

            delete[] intElementList;
            delete[] intElementCornerList;
            delete[] intTypeList;
            delete[] fltXCoords;
            delete[] fltYCoords;
            delete[] fltZCoords;

            intElementNumber = 0;
            intElementCornerNumber = 0;
            intCoordsNumber = 0;
            intElementCount = 0;
            intElementCornerCount = 0;
        }
    }
    else
    {
        sendError("file '%s' not found or is corrupt", geoFileParam->getValue());
        return FAIL;
    }
    ifstrFile.close();
    /*************************************************************
   SCALAR DATA POLYGON
   *************************************************************/
    int size = 0;
    if (intKindOfMeasuring == PER_NODE)
    {
        size = intPolygonCoordsNumber;
    }
    else
    {
        size = intPolygonNumber;
    }
    fltData = new float[size + 1];
    if (fltData == NULL)
    {
        sendInfo("Memory allocation for scalar data failed");
        return FAIL;
    }
    for (i = 0; i < intNumberOfTimeSteps; i++)
    {
        objects[i] = NULL;
    }
    //if( strlen(paramPolygonScalarFile->getValue()) > 0 )//if smth in filebrowser choosen
    //if smth in filebrowser choosen
    if (strcmp(paramPolygonScalarFile->getValue(), "data/polygonScalarFile") != 0)
    {
        if (intNumberOfTimeSteps > 1)
        {
            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                char *TFN = buildTimestepFileName((char *)paramPolygonScalarFile->getValue(), intNumberOfTimeSteps, i);
                if (readScalarValuesParallel(TFN, fltData, size))
                {
                    strcpy(strSetElementName, polygonScalarOutPort->getObjName());
                    sprintf(buf, "%d", i);
                    strcat(strSetElementName, buf);
                    objects[i] = new coDoFloat(strSetElementName, size, fltData);
                    if (objects[i] == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, strSetElementName);
                        sendInfo(buf);
                        return FAIL;
                    }
                }
                delete[] TFN;
            }
            coDoSet *polygonScalarSet = new coDoSet(polygonScalarOutPort->getObjName(), objects);
            if (polygonScalarSet == NULL)
            {
                strcpy(buf, "Cannot create object ");
                strcat(buf, polygonScalarOutPort->getObjName());
                sendInfo(buf);
                return FAIL;
            }

            sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
            polygonScalarSet->addAttribute("TIMESTEP", buf);
            polygonScalarOutPort->setCurrentObject(polygonScalarSet);
            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                if (objects[i] != NULL)
                {
                    delete objects[i];
                }
            }
        }
        else if (intNumberOfTimeSteps == 1)
        {

            if (readScalarValuesParallel((char *)paramPolygonScalarFile->getValue(), fltData, size))
            {
                polygonScalarObj = new coDoFloat(polygonScalarOutPort->getObjName(), size, fltData);

                if (polygonScalarObj == NULL)
                {
                    strcpy(buf, "Cannot create object ");
                    strcat(buf, polygonScalarOutPort->getObjName());
                    sendInfo(buf);
                    return FAIL;
                }
                polygonScalarOutPort->setCurrentObject(polygonScalarObj);
            }
        }
    }
    delete[] fltData;

    /*************************************************************
    VECTOR DATA POLYGON
   *************************************************************/
    fltXData = new float[size + 1];
    if (fltXData == NULL)
    {
        sendInfo("Memory allocation for X-component of vector data failed");
        return FAIL;
    }
    fltYData = new float[size + 1];
    if (fltYData == NULL)
    {
        sendInfo("Memory allocation for Y-component of vector data failed");
        return FAIL;
    }
    fltZData = new float[size + 1];
    if (fltZData == NULL)
    {
        sendInfo("Memory allocation for Z-component of vector data failed");
        return FAIL;
    }
    for (i = 0; i < intNumberOfTimeSteps; i++)
    {
        objects[i] = NULL;
    }

    //if(  (strlen(paramPolygonVectorFile->getValue())  ) >0 )
    //if smth in filebrowser choosen
    if (strcmp(paramPolygonVectorFile->getValue(), "data/polygonVectorFile") != 0)
    {
        if (intNumberOfTimeSteps > 1)
        {
            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                char *TFN = buildTimestepFileName((char *)paramPolygonVectorFile->getValue(), intNumberOfTimeSteps, i);
                if (readVectorValuesParallel(TFN, fltXData, fltYData, fltZData, size))
                {
                    strcpy(strSetElementName, polygonVectorOutPort->getObjName());
                    sprintf(buf, "%d", i);
                    strcat(strSetElementName, buf);
                    objects[i] = new coDoVec3(strSetElementName, size, fltXData, fltYData, fltZData);

                    if (objects[i] == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, strSetElementName);
                        sendInfo(buf);
                        return FAIL;
                    }
                }
                delete[] TFN;
            }
            coDoSet *polygonVectorSet = new coDoSet(polygonVectorOutPort->getObjName(), objects);
            if (polygonVectorSet == NULL)
            {
                strcpy(buf, "Cannot create object ");
                strcat(buf, polygonVectorOutPort->getObjName());
                sendInfo(buf);
                return FAIL;
            }

            sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
            polygonVectorSet->addAttribute("TIMESTEP", buf);
            polygonVectorOutPort->setCurrentObject(polygonVectorSet);

            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                if (objects[i] != NULL)
                {
                    delete objects[i];
                }
            }
        }
        else if (intNumberOfTimeSteps == 1)
        {
            if (readVectorValuesParallel((char *)paramPolygonVectorFile->getValue(), fltXData, fltYData, fltZData, size))
            {
                polygonVectorObj = new coDoVec3(polygonVectorOutPort->getObjName(), size, fltXData, fltYData, fltZData);
                if (polygonVectorObj == NULL)
                {
                    strcpy(buf, "Cannot create object ");
                    strcat(buf, polygonVectorOutPort->getObjName());
                    sendInfo(buf);
                    return FAIL;
                }
                polygonVectorOutPort->setCurrentObject(polygonVectorObj);
            }
        }
    }
    delete[] fltXData;
    delete[] fltYData;
    delete[] fltZData;
    /*************************************************************
   SCALAR DATA GRID
   *************************************************************/
    if (intKindOfMeasuring == PER_NODE)
    {
        size = intGridCoordsNumber;
    }
    else
    {
        size = intGridNumber;
    }
    fltData = new float[size + 1];
    if (fltData == NULL)
    {
        sendInfo("Memory allocation for scalar data failed");
        return FAIL;
    }
    for (i = 0; i < intNumberOfTimeSteps; i++)
    {
        objects[i] = NULL;
    }
    //if(  (strlen(paramGridScalarFile->getValue())  ) >0 )
    //if smth in filebrowser choosen
    if (strcmp(paramGridScalarFile->getValue(), "data/gridScalarFile") != 0)
    {
        if (intNumberOfTimeSteps > 1)
        {
            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                char *TFN = buildTimestepFileName((char *)paramGridScalarFile->getValue(), intNumberOfTimeSteps, i);
                if (readScalarValuesParallel(TFN, fltData, size))
                {
                    strcpy(strSetElementName, gridScalarOutPort->getObjName());
                    sprintf(buf, "%d", i);
                    strcat(strSetElementName, buf);
                    objects[i] = new coDoFloat(strSetElementName, size, fltData);
                    if (objects[i] == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, strSetElementName);
                        sendInfo(buf);
                        return FAIL;
                    }
                }
                delete[] TFN;
            }
            coDoSet *gridScalarSet = new coDoSet(gridScalarOutPort->getObjName(), objects);
            if (gridScalarSet == NULL)
            {
                strcpy(buf, "Cannot create object ");
                strcat(buf, gridScalarOutPort->getObjName());
                sendInfo(buf);
                return FAIL;
            }

            sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
            gridScalarSet->addAttribute("TIMESTEP", buf);
            gridScalarOutPort->setCurrentObject(gridScalarSet);
            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                if (objects[i] != NULL)
                {
                    delete objects[i];
                }
            }
        }
        else if (intNumberOfTimeSteps == 1)
        {
            if (readScalarValuesParallel((char *)paramGridScalarFile->getValue(), fltData, size))
            {
                gridScalarObj = new coDoFloat(gridScalarOutPort->getObjName(), size, fltData);
                if (gridScalarObj == NULL)
                {
                    strcpy(buf, "Cannot create object ");
                    strcat(buf, gridScalarOutPort->getObjName());
                    sendInfo(buf);
                    return FAIL;
                }
                gridScalarOutPort->setCurrentObject(gridScalarObj);
            }
        }
    }
    delete[] fltData;
    /*************************************************************
   VECTOR DATA GRID
   *************************************************************/
    fltXData = new float[size + 1];
    if (fltXData == NULL)
    {
        sendInfo("Memory allocation for X-component of vector data failed");
        return FAIL;
    }
    fltYData = new float[size + 1];
    if (fltYData == NULL)
    {
        sendInfo("Memory allocation for Y-component of vector data failed");
        return FAIL;
    }
    fltZData = new float[size + 1];
    if (fltZData == NULL)
    {
        sendInfo("Memory allocation for Z-component of vector data failed");
        return FAIL;
    }
    for (i = 0; i < intNumberOfTimeSteps; i++)
    {
        objects[i] = NULL;
    }

    //if(  (strlen(paramGridVectorFile->getValue())  ) >0 )
    //if smth in filebrowser choosen
    if (strcmp(paramGridVectorFile->getValue(), "data/gridVectorFile") != 0)
    {
        if (intNumberOfTimeSteps > 1)
        {

            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                char *TFN = buildTimestepFileName((char *)paramGridVectorFile->getValue(), intNumberOfTimeSteps, i);
                if (readVectorValuesParallel(TFN, fltXData, fltYData, fltZData, size))
                {
                    strcpy(strSetElementName, gridVectorOutPort->getObjName());
                    sprintf(buf, "%d", i);
                    strcat(strSetElementName, buf);
                    objects[i] = new coDoVec3(strSetElementName, size, fltXData, fltYData, fltZData);
                    if (objects[i] == NULL)
                    {
                        strcpy(buf, "Cannot create object ");
                        strcat(buf, strSetElementName);
                        sendInfo(buf);
                        return FAIL;
                    }
                }
                delete[] TFN;
            }
            coDoSet *gridVectorSet = new coDoSet(gridVectorOutPort->getObjName(), objects);
            if (gridVectorSet == NULL)
            {
                strcpy(buf, "Cannot create object ");
                strcat(buf, gridVectorOutPort->getObjName());
                sendInfo(buf);
                return FAIL;
            }
            sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
            gridVectorSet->addAttribute("TIMESTEP", buf);
            gridVectorOutPort->setCurrentObject(gridVectorSet);
            for (i = 0; i < intNumberOfTimeSteps; i++)
            {
                if (objects[i] != NULL)
                {
                    delete objects[i];
                }
            }
        }
        else if (intNumberOfTimeSteps == 1)
        {
            if (readVectorValuesParallel((char *)paramGridVectorFile->getValue(), fltXData, fltYData, fltZData, size))
            {
                gridVectorObj = new coDoVec3(gridVectorOutPort->getObjName(), size, fltXData, fltYData, fltZData);
                if (gridVectorObj == NULL)
                {
                    strcpy(buf, "Cannot create object ");
                    strcat(buf, gridVectorOutPort->getObjName());
                    sendInfo(buf);
                    return FAIL;
                }
                gridVectorOutPort->setCurrentObject(gridVectorObj);
            }
        }
    }
    delete[] fltXData;
    delete[] fltYData;
    delete[] fltZData;
    delete[] objects;
    return SUCCESS;
}

int
ReadEnsight::compute5()
{
    initFileMap();

    sendInfo("ensight version 5");
    intPolygonNumber = 0; //number of all polygons
    intGridNumber = 0;

    intPolygonCoordsNumber = 0;
    intGridCoordsNumber = 0;

    intPolygonCornerNumber = 0;
    intGridCornerNumber = 0;
    intLineCounter = 0; //line number in the file

    LINE_ELEMENT_LENGTH = paramLenghOfLineElement->getValue();
    //CORNER_LINE_ELEMENT_LENGTH = paramLenghOfCornerLineElement->getValue();
    CORNER_LINE_ELEMENT_LENGTH = 8;
    NUMBER_VALUES_IN_LINE = paramNumberValuesInLine->getValue();
    //CORNER_LINE_ELEMENT_LENGTH = paramLenghOfCornerLineElement->setValue(8);
    //cout << "CORNER_LINE_ELEMENT_LENGTH 5.0: " << CORNER_LINE_ELEMENT_LENGTH << endl;
    char strFileName[100]; //name of the file to be read
    char strLine[300]; //line in the file
    //char strFigureName[20];
    char buf[500];
    int intHeaderFound;
    intAncor = 0;

    strcpy(strFileName, geoFileParam->getValue());
    ifstrFile.open(strFileName, ios::in); //file strFileName will be opened from constructor ifstream,

    if (ifstrFile.good())
    {
        strcpy(buf, "Geometry file ");
        strcat(buf, strFileName);
        strcat(buf, " opened.");
        sendInfo(buf);

        //      ifstream& iFile=ifstrFile;

        // indexed or not indexed coordinates cells ???
        nodeIdFlg_ = isThereNodeIDSection(ifstrFile);
        elementIdFlg_ = isThereElementIDSection(ifstrFile);

        /******************************************************************************
          LOOK FOR COORDINATES HEADER
      ******************************************************************************/
        intHeaderFound = lookForHeader(ifstrFile, "coordinates");
        if (intHeaderFound == 0) //header not found
        {
            sendError("Header in the file '%s' not found.", strFileName);
            return FAIL;
        }
        /******************************************************************************
             GET THE NUMBER OF COORDINATESS
      ******************************************************************************/
        //we stay now at the line with number of grids coordinates
        ifstrFile.get(strLine, 250);
        ifstrFile.seekg(1L, ios::cur);
        intLineCounter++;
        intGridCoordsNumber = atoi(strLine); //number of coords for all grid elements
        strcpy(buf, "Found ");
        char temp[20];
        sprintf(temp, "%d", intGridCoordsNumber);
        strcat(buf, temp);
        strcat(buf, " coordinates, line ");
        sprintf(temp, "%d", intLineCounter);
        strcat(buf, temp);
        sendInfo(buf);

        if (!intGridCoordsNumber)
        {
            sprintf(temp, "%d", intLineCounter);
            sendError("failed to read in the number of grid coordinates, line '%s'.", temp);
            return FAIL;
        }
        /******************************************************************************
             ALLOCATE MEMORY FOR ARRAYS WITH GRIDS COORDINATES
      ******************************************************************************/
        fltGridXCoords = new float[intGridCoordsNumber + 1];
        if (fltGridXCoords == NULL)
        {
            sendError("Cannot allocate memory for X-coordinates array.");
            return FAIL;
        }
        fltGridYCoords = new float[intGridCoordsNumber + 1];
        if (fltGridYCoords == NULL)
        {
            sendError("Cannot allocate memory for Y-coordinates array.");
            return FAIL;
        }
        fltGridZCoords = new float[intGridCoordsNumber + 1];
        if (fltGridZCoords == NULL)
        {
            sendError("Cannot allocate memory for Z-coordinates array.");
            return FAIL;
        }

        index_ = new int[intGridCoordsNumber + 1];
        if (index_ == NULL)
        {
            sendError("Cannot allocate memory for index array.");
            return FAIL;
        }
        maxIndex_ = intGridCoordsNumber + 1;
        /******************************************************************************
            FILL IN COORDS ARRAYS FOR GRIDS
      ******************************************************************************/

        if (!read3DCoordsParallel(ifstrFile, intGridCoordsNumber, fltGridXCoords, fltGridYCoords, fltGridZCoords))
        {
            sendError("failed to read in grid coordinates");
            return FAIL;
        }

        /******************************************************************************
             GET NUMBER OF ALL GRIDS AND OF ALL GRID CORNERS
      ******************************************************************************/
        int &refGN = intGridNumber;
        int &refPN = intPolygonNumber;
        int &refGKN = intGridCornerNumber;
        int &refPKN = intPolygonCornerNumber;
        intAncor = intLineCounter; //to this point we come later, wenn we read corners in the array in function fillCornerArrays

        //here the intAncor will be set
        if (!getNumberOfElements(ifstrFile, refPN, refGN, refPKN, refGKN))
        {
            sendError("Failed to determine the number of elements.");
            return FAIL;
        }
        strcpy(buf, "Number of all grid elements: ");
        sprintf(temp, "%d", intGridNumber);
        strcat(buf, temp);
        sendInfo(buf);

        strcpy(buf, "Number of all grid corners: ");
        sprintf(temp, "%d", intGridCornerNumber);
        strcat(buf, temp);
        sendInfo(buf);

        strcpy(buf, "Number of all polygon elements: ");
        sprintf(temp, "%d", intPolygonNumber);
        strcat(buf, temp);
        sendInfo(buf);

        strcpy(buf, "Number of all polygon corners: ");
        sprintf(temp, "%d", intPolygonCornerNumber);
        strcat(buf, temp);
        sendInfo(buf);
        /******************************************************************************
             ALLOCATE MEMORY FOR GRIDS AND GRID CORNERS
      ******************************************************************************/
        intGridList = new int[intGridNumber + 1];
        if (intGridList == NULL)
        {
            sendError("Cannot allocate memory for grid list.");
            return FAIL;
        }
        intGridList[0] = 0; //very-very important, because recursion!!!

        intGridCornerList = new int[intGridCornerNumber + 1];
        if (intGridCornerList == NULL)
        {
            sendError("Cannot allocate memory for grid corner list.");
            return FAIL;
        }
        intTypeList = new int[intGridNumber + 1];
        if (intTypeList == NULL)
        {
            sendError("Cannot allocate memory for element type list.");
            return FAIL;
        }
        intPolygonList = new int[intPolygonNumber + 1];
        if (intPolygonList == NULL)
        {
            sendError("Cannot allocate memory for polygon list.");
            return FAIL;
        }
        intPolygonList[0] = 0; //very-very important, because recursion!!!

        intPolygonCornerList = new int[intPolygonCornerNumber + 1];
        if (intPolygonCornerList == NULL)
        {
            sendError("Cannot allocate memory for polygon corner list.");
            return FAIL;
        }
        /******************************************************************************
        GO BACK TO THE BEGIN OF THE GRID CORNER LIST IN THE FILE
       ******************************************************************************/
        ifstrFile.close();

        ifstream nFile(strFileName, ios::in);
        if (!nFile)
        {
            cerr << "ReadEnsight::compute5() : Ifstream error for file " << strFileName << endl;
            return FAIL;
        }
        //       ifstrFile.open(strFileName, ios::in );
        //       ifstrFile.seekg(0L,ios::beg);
        intLineCounter = 0;

        for (int i = 0; i < intAncor; i++)
        {
            nFile.get(strLine, 250);
            nFile.seekg(1L, ios::cur);
            intLineCounter++;
        }

        /******************************************************************************
                 FILL IN ARRAYS WITH GRIDS AND GRID CORNERS
      ******************************************************************************/
        if (!fillCornerArrays(nFile, &(*intGridList), &(*intGridCornerList), &(*intTypeList), &(*intPolygonList), &(*intPolygonCornerList)))
        {
            sendError("cannot fill corner lists.");
            nFile.close();
            return FAIL;
        }
        nFile.close();
    }
    else
    {

        sendError("file '%s' not found or is corrupt", strFileName);
        return FAIL;
    }

    /*************************************************************
   RESULT FILE
   *************************************************************/
    strcpy(strFileName, resFileParam->getValue());
    //select last 7 bytes from file name - we want know
    strncpy(buf, strFileName + strlen(strFileName) - 7, 7);
    //whether user has choosen a result file
    buf[7] = '\0';

    int v6Flg = 0;
    if (strcmp(buf, "resFile") != 0)
    {
        if (!readResultFile(strFileName))
        {
            cerr << "ReadEnsight::compute5(..) res-file NOT found " << caseFileFlg_ << endl;
            if (caseFileFlg_)
            {
                sendInfo("Case File found assume ENSIGHT 6");
                v6Flg = 1;
            }
            else
            {
                sendInfo("Result file not found or is corrupt. No display of scalar or vector data");
                //cerr << "ReadEnsight::compute5(..) case-file NOT found" << endl;
            }
        }
        //       cerr << "ReadEnsight::compute5(..) try to find case-file" << endl;
        // if we have a case-file we have ensight version 6 and should try to read result files
    }

    coDistributedObject **objects = new coDistributedObject *[intNumberOfTimeSteps + 1];
    ;
    objects[intNumberOfTimeSteps] = NULL;

    char strSetElementName[100];
    int i;
    for (i = 0; i < intNumberOfTimeSteps; i++)
    {
        strcpy(strSetElementName, gridOutPort->getObjName());
        sprintf(buf, "%d", i);
        strcat(strSetElementName, buf);
        objects[i] = new coDoUnstructuredGrid(strSetElementName,
                                              intGridNumber,
                                              intGridCornerNumber,
                                              intGridCoordsNumber,
                                              intGridList,
                                              intGridCornerList,
                                              fltGridXCoords,
                                              fltGridYCoords,
                                              fltGridZCoords,
                                              intTypeList);
        if (objects[i] == NULL)
        {
            strcpy(buf, "Cannot create object ");
            strcat(buf, gridOutPort->getObjName());
            sendInfo(buf);
            return FAIL;
        }
    }

    coDoSet *gridSet = new coDoSet(gridOutPort->getObjName(), objects);
    if (gridSet == NULL)
    {
        strcpy(buf, "Cannot create object ");
        strcat(buf, gridOutPort->getObjName());
        sendInfo(buf);
        return FAIL;
    }
    sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
    gridSet->addAttribute("TIMESTEP", buf);
    gridOutPort->setCurrentObject(gridSet);
    sendInfoSuccess(gridOutPort->getName());

    for (i = 0; i < intNumberOfTimeSteps; i++)
    {
        strcpy(strSetElementName, polygonOutPort->getObjName());
        sprintf(buf, "%d", i);
        strcat(strSetElementName, buf);
        objects[i] = new coDoPolygons(strSetElementName,
                                      intGridCoordsNumber,
                                      fltGridXCoords,
                                      fltGridYCoords,
                                      fltGridZCoords,
                                      intPolygonCornerNumber,
                                      intPolygonCornerList,
                                      intPolygonNumber,
                                      intPolygonList);
        objects[i]->addAttribute("vertexOrder", "2");
        if (objects[i] == NULL)
        {
            strcpy(buf, "Cannot create object ");
            strcat(buf, polygonOutPort->getObjName());
            sendInfo(buf);
            return FAIL;
        }
    }

    coDoSet *polygonSet = new coDoSet(polygonOutPort->getObjName(), objects);
    if (polygonSet == NULL)
    {
        strcpy(buf, "Cannot create object ");
        strcat(buf, polygonOutPort->getObjName());
        sendInfo(buf);
        return FAIL;
    }
    sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
    polygonSet->addAttribute("TIMESTEP", buf);
    polygonOutPort->setCurrentObject(polygonSet);
    sendInfoSuccess(polygonOutPort->getName());

    // it seem to be ensight version 6 we try to read variables
    if (v6Flg)
    {

        //*************************************************************
        //  SCALAR DATA GRID
        //*************************************************************
        int size;
        if (intKindOfMeasuring == PER_NODE)
        {
            size = intGridCoordsNumber;
        }
        else
        {
            size = intGridNumber;
        }
        fltData = new float[size + 1];
        if (fltData == NULL)
        {
            sendInfo("Memory allocation for scalar data failed");
            return FAIL;
        }
        for (i = 0; i < intNumberOfTimeSteps; i++)
        {
            objects[i] = NULL;
        }

        if (strcmp(paramGridScalarFile->getValue(), "data/gridScalarFile") != 0)
        {

            if (intNumberOfTimeSteps > 0)
            {
                for (i = 0; i < intNumberOfTimeSteps; i++)
                {
                    char *TFN = buildTimestepFileName((char *)paramGridScalarFile->getValue(), intNumberOfTimeSteps, i);
                    if (readScalarValuesV6(TFN, fltData, size))
                    {
                        strcpy(strSetElementName, gridScalarOutPort->getObjName());
                        sprintf(buf, "%d", i);
                        strcat(strSetElementName, buf);
                        objects[i] = new coDoFloat(strSetElementName, size, fltData);
                        if (objects[i] == NULL)
                        {
                            strcpy(buf, "Cannot create object ");
                            strcat(buf, strSetElementName);
                            sendInfo(buf);
                            return FAIL;
                        }
                    }
                    delete[] TFN;
                }
                coDoSet *gridScalarSet = new coDoSet(gridScalarOutPort->getObjName(), objects);
                if (gridScalarSet == NULL)
                {
                    strcpy(buf, "Cannot create object ");
                    strcat(buf, gridScalarOutPort->getObjName());
                    sendInfo(buf);
                    return FAIL;
                }

                sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
                gridScalarSet->addAttribute("TIMESTEP", buf);
                gridScalarOutPort->setCurrentObject(gridScalarSet);
                // 	       for (i=0;i<intNumberOfTimeSteps;i++) {
                // 		   if(objects[i]!=NULL) {
                // 		       delete objects[i];
                // 		   }
                //	       }
            }
        }
        delete[] fltData;

        //*************************************************************
        //   VECTOR DATA GRID
        //*************************************************************/
        fltXData = new float[size + 1];
        if (fltXData == NULL)
        {
            sendInfo("Memory allocation for X-component of vector data failed");
            return FAIL;
        }
        fltYData = new float[size + 1];
        if (fltYData == NULL)
        {
            sendInfo("Memory allocation for Y-component of vector data failed");
            return FAIL;
        }
        fltZData = new float[size + 1];
        if (fltZData == NULL)
        {
            sendInfo("Memory allocation for Z-component of vector data failed");
            return FAIL;
        }
        for (i = 0; i < intNumberOfTimeSteps; i++)
        {
            objects[i] = NULL;
        }

        //if(  (strlen(paramGridVectorFile->getValue())  ) >0 )
        //if smth in filebrowser choosen
        if (strcmp(paramGridVectorFile->getValue(), "data/gridVectorFile") != 0)
        {

            if (intNumberOfTimeSteps > 0)
            {

                for (i = 0; i < intNumberOfTimeSteps; i++)
                {
                    char *TFN = buildTimestepFileName((char *)paramGridVectorFile->getValue(), intNumberOfTimeSteps, i);
                    if (readVectorValuesV6(TFN, fltXData, fltYData, fltZData, size))
                    {
                        strcpy(strSetElementName, gridVectorOutPort->getObjName());
                        sprintf(buf, "%d", i);
                        strcat(strSetElementName, buf);
                        objects[i] = new coDoVec3(strSetElementName, size, fltXData, fltYData, fltZData);
                        if (objects[i] == NULL)
                        {
                            strcpy(buf, "Cannot create object ");
                            strcat(buf, strSetElementName);
                            sendInfo(buf);
                            return FAIL;
                        }
                    }
                    delete[] TFN;
                }
                coDoSet *gridVectorSet = new coDoSet(gridVectorOutPort->getObjName(), objects);
                if (gridVectorSet == NULL)
                {
                    strcpy(buf, "Cannot create object ");
                    strcat(buf, gridVectorOutPort->getObjName());
                    sendInfo(buf);
                    return FAIL;
                }
                sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
                gridVectorSet->addAttribute("TIMESTEP", buf);
                gridVectorOutPort->setCurrentObject(gridVectorSet);
                // 	       for (i=0;i<intNumberOfTimeSteps;i++) {
                // 		   if(objects[i]!=NULL) {
                // 		       delete objects[i];
                // 		   }
                // 	       }
            }
        }

        delete[] fltXData;
        delete[] fltYData;
        delete[] fltZData;
    }

    ifstrFile.close();

    for (i = 0; i < intNumberOfTimeSteps; i++)
    {
        delete objects[i];
    }
    delete[] objects;

    delete[] fltGridXCoords;
    delete[] fltGridYCoords;
    delete[] fltGridZCoords;

    delete[] intPolygonList;
    delete[] intPolygonCornerList;
    delete[] intTypeList;

    delete[] intGridList;
    delete[] intGridCornerList;
    return SUCCESS;
}

/******************************************************************
Function				compute()
Parameter
Return
Description				called by controller if the module will be executed.
                  Creates data objects from data read from the disk
                  data files.
                  This function manages the workflow and call all
                  the rest functions.
******************************************************************/
int ReadEnsight::compute()
{
    initFileMap();
    sendInfo("compute callback enter... ");
    intKindOfMeasuring = 0;
    if (strlen(caseFileParam->getValue()) > 0)
    {
        //if( strcmp(caseFileParam->getValue(),"data/caseFile") != 0 )//if smth in filebrowser choosen

        ifstream caseFile(caseFileParam->getValue(), ios::in);
        if (caseFile.good())
        {
            caseFileFlg_ = 1;
            strcpy(ENSIGHT_VERSION, getEnsightVersion(caseFile));
            intKindOfMeasuring = getPerNodeOrPerElement(caseFile);
            intNumberOfTimeSteps = getNumberOfTimeSteps(caseFile);
        }
        else
        {
            sendInfo("cannot open case file. Per default ensight version 5.");
            strcpy(ENSIGHT_VERSION, "ensight");
            intKindOfMeasuring = PER_ELEMENT;
            intNumberOfTimeSteps = 1;
        }
    }
    else
    {
        strcpy(ENSIGHT_VERSION, "ensight");
        intKindOfMeasuring = PER_NODE;
        intNumberOfTimeSteps = 1;
    }

    if (strcmp(caseFileParam->getValue(), "data/caseFile") == 0)
    {
        strcpy(ENSIGHT_VERSION, "ensight");
        intKindOfMeasuring = PER_NODE;
    }
    int retVal;
    if (strcmp(ENSIGHT_VERSION, "ensight gold") == 0)
    {
        retVal = computeGold();
    }
    else //version is ensight5
    {
        retVal = compute5();
    }

    //caseFile.close();
    sendInfo("compute callback left... ");
    return retVal;
}

/******************************************************************
 ********************	 F U N C T I O N S	***********************
 ******************************************************************/

/******************************************************************
Function	 int isPolygon(char*)
Argument	 strPolygonType - name of polygon element
Return		 1 if strPolygonType is recognized as polygon, otherwise 0
Description	 check whether element found in the geometry file corresponds to currently
       supported by covise polygon types
******************************************************************/
int
ReadEnsight::isPolygon(char *strPolygonType)
{
    //     string str(strPolygonType);

    //     if (str.find("bar2") != string::npos )  {  return 1; }
    //     if (str.find("bar3") != string::npos )  {  return 1; }
    //     if (str.find("tria3") != string::npos ) {  return 1; }
    //     if (str.find("tria6") != string::npos ) {  return 1; }
    //     if (str.find("quad4") != string::npos ) {  return 1; }
    //     if (str.find("quad8") != string::npos ) {  return 1; }

    if (strncmp(strPolygonType, "bar2", 4) == 0)
    {
        return 1;
    }
    if (strncmp(strPolygonType, "bar3", 4) == 0)
    {
        return 1;
    }
    if (strncmp(strPolygonType, "tria3", 5) == 0)
    {
        return 1;
    }
    if (strncmp(strPolygonType, "tria6", 5) == 0)
    {
        return 1;
    }
    if (strncmp(strPolygonType, "quad4", 5) == 0)
    {
        return 1;
    }
    if (strncmp(strPolygonType, "quad8", 5) == 0)
    {
        return 1;
    }

    return 0;
}

/******************************************************************
Function			int isGrid(char*)
Arguments			strGridType - name of grid element
Return				1 if strGridType is recognized as grid, otherwise 0
Description			check whether element found in the geometry file corresponds to currently
               supported by covise grid types
******************************************************************/
int
ReadEnsight::isGrid(char *strGridType)
{
    //     string str(strGridType);

    //     if (str.find("tetra4") != string::npos ) { return 1; }
    //     if (str.find("tetra10") != string::npos ) { return 1; }
    //     if (str.find("hexa8") != string::npos ) { return 1; }
    //     if (str.find("hexa20") != string::npos ) { return 1; }
    //     if (str.find("penta6") != string::npos ) { return 1; }
    //     if (str.find("pyramid5") != string::npos ) { return 1; }

    if (strstr(strGridType, "tetra4") != NULL)
    {
        return 1;
    }
    if (strstr(strGridType, "tetra10") != NULL)
    {
        return 1;
    }
    if (strstr(strGridType, "hexa8") != NULL)
    {
        return 1;
    }
    if (strstr(strGridType, "hexa20") != NULL)
    {
        return 1;
    }
    if (strstr(strGridType, "penta6") != NULL)
    {
        return 1;
    }
    if (strstr(strGridType, "pyramid5") != NULL)
    {
        return 1;
    }

    //    if( strncmp(strGridType,"tetra4",6) == 0 )		{ return 1; }
    //    if( strncmp(strGridType,"tetra10",7) == 0)	        { return 1; }
    //    if( strncmp(strGridType,"hexa8",5) == 0)		{ return 1; }
    //    if( strncmp(strGridType,"hexa20",6) == 0)		{ return 1; }
    //    if( strncmp(strGridType,"penta6",6) == 0)		{ return 1; }
    //    if( strncmp(strGridType,"pyramid5",8) == 0)         { return 1; }

    return 0;
}

/******************************************************************
Function			int isLine(char*)
Arguments			strLineType - name of line element
Return				1 if strLineType is recognized as line, otherwise 0
Description			check whether element found in the geometry file corresponds to currently
               supported by covise line types
******************************************************************/
int ReadEnsight::isLine(char *strLineType)
{
    if (strcmp(strLineType, "bar2") == 0)
    {
        return 1;
    }
    if (strcmp(strLineType, "bar3") == 0)
    {
        return 1;
    }
    return 0;
}

/******************************************************************
Function			int getNumberOfCorner(char*)
Arguments			strType - element name
Return				number of corners
Description			returns number of corners for element
******************************************************************/
int
ReadEnsight::getNumberOfCorner(char *strType)
{
    //     string str(strType);

    //     if (str.find("bar2") != string::npos )     { return 2; }
    //     if (str.find("bar3") != string::npos )     { return 3; }
    //     if (str.find("tria3") != string::npos )    { return 3; }
    //     if (str.find("quad4") != string::npos )    { return 4; }
    //     if (str.find("tetra4") != string::npos )   { return 4; }
    //     if (str.find("pyramid5") != string::npos ) { return 5; }
    //     if (str.find("tria6") != string::npos )    { return 6; }
    //     if (str.find("penta6") != string::npos )   { return 6; }
    //     if (str.find("quad8") != string::npos )    { return 8; }
    //     if (str.find("hexa8") != string::npos )    { return 8; }
    //     if (str.find("hexa20") != string::npos )   { return 20; }

    if (strstr(strType, "bar2") != NULL)
    {
        return 2;
    }
    if (strstr(strType, "bar3") != NULL)
    {
        return 3;
    }
    if (strstr(strType, "tria3") != NULL)
    {
        return 3;
    }
    if (strstr(strType, "quad4") != NULL)
    {
        return 4;
    }
    if (strstr(strType, "tetra4") != NULL)
    {
        return 4;
    }
    if (strstr(strType, "pyramid5") != NULL)
    {
        return 5;
    }
    if (strstr(strType, "tria6") != NULL)
    {
        return 6;
    }
    if (strstr(strType, "penta6") != NULL)
    {
        return 6;
    }
    if (strstr(strType, "quad8") != NULL)
    {
        return 8;
    }
    if (strstr(strType, "hexa8") != NULL)
    {
        return 8;
    }
    if (strstr(strType, "hexa20") != NULL)
    {
        return 20;
    }

    cerr << "ReadEnsight::getNumberOfCorner(..) no possibility found " << strType << endl;

    return 0;
}

/******************************************************************
Function			int getType(char*)
Arguments			strType - element name
Return				covise type of element from .../covise_unstrgrd.h
Description			returns element type
******************************************************************/
int
ReadEnsight::getType(char *strType)
{
    //     string str(strType);

    //     if (str.find("bar2") != string::npos )     { return TYPE_BAR; }//1
    //     if (str.find("bar3") != string::npos )     { return TYPE_BAR; }//1
    //     if (str.find("tria3") != string::npos )    { return TYPE_TRIANGLE; }//
    //     if (str.find("quad4") != string::npos )    { return TYPE_QUAD;     }//
    //     if (str.find("tetra4") != string::npos )   { return TYPE_TETRAHEDER; }
    //     if (str.find("pyramid5") != string::npos ) { return TYPE_PYRAMID; }//5
    //     if (str.find("tria6") != string::npos )    { return TYPE_TRIANGLE; }//
    //     if (str.find("penta6") != string::npos )   { return TYPE_PRISM; }
    //     if (str.find("quad8") != string::npos )    { return TYPE_QUAD; }//3
    //     if (str.find("hexa8") != string::npos )    { return TYPE_HEXAEDER; }//
    //     if (str.find("hexa20") != string::npos )   { return TYPE_HEXAEDER;}//7

    if (strstr(strType, "bar2") != NULL) //1
    {
        return TYPE_BAR;
    }
    if (strstr(strType, "bar3") != NULL) //1
    {
        return TYPE_BAR;
    }
    if (strstr(strType, "tria3") != NULL) //2
    {
        return TYPE_TRIANGLE;
    }
    if (strstr(strType, "quad4") != NULL) //3
    {
        return TYPE_QUAD;
    }
    if (strstr(strType, "tetra4") != NULL) //4
    {
        return TYPE_TETRAHEDER;
    }
    if (strstr(strType, "pyramid5") != NULL) //5
    {
        return TYPE_PYRAMID;
    }
    if (strstr(strType, "tria6") != NULL) //2
    {
        return TYPE_TRIANGLE;
    }
    if (strstr(strType, "penta6") != NULL)
    {
        return TYPE_PRISM;
    }
    if (strstr(strType, "quad8") != NULL) //3
    {
        return TYPE_QUAD;
    }
    if (strstr(strType, "hexa8") != NULL) //7
    {
        return TYPE_HEXAEDER;
    }
    if (strstr(strType, "hexa20") != NULL) //7
    {
        return TYPE_HEXAEDER;
    }

    cerr << "ReadEnsight::getType(..) no possibility found <" << strType << "> " << endl;

    return 0;
}

/******************************************************************
Function			readScalarValuesSequent()
Arguments			strFilename - name of the file with scalar values
               fltKindOfData - pointer to data arrays that will be filled in
               intNumberElements - number of elements
Return				1 (success) or 0 (failure)
Description			read in scalar data from disc file to the array. File has such a structure:
               1 line: description line, 2 line till eof - scalar values. The line has
               such a structure: number of elements in the line - NUMBER_VALUES_IN_LINE,
               length of an line element - LINE_ELEMENT_LENGTH. Used for version 5 .
******************************************************************/
int
ReadEnsight::readScalarValuesSequent(char *strFilename, float *fltKindOfData, int intNumberOfElements)
{
    //sendInfo("Begin reading scalar data... ");
    int i = 0, bad = 0;
    char strLine[250];
    char strTemp[200];
    char buf[500];
    ifstream iFile;
    iFile.open(strFilename, ios::in);
    if (iFile.good())
    {
        strcpy(buf, "Reading data from the file ");
        strcat(buf, strFilename);
        sendInfo(buf);
        iFile.getline(strLine, 250); //description line, no use

        while (iFile.good() && bad == 0)
        {
            iFile.getline(strLine, 250);
            strTemp[0] = '\0';
            for (int j = 0; j < NUMBER_VALUES_IN_LINE; j++)
            {
                strncpy(strTemp, &strLine[LINE_ELEMENT_LENGTH * j], LINE_ELEMENT_LENGTH);
                strTemp[12] = '\0';
                fltKindOfData[i] = atof(strTemp);
                i++; //counter of all scalar elements
                if (i >= intNumberOfElements)
                {
                    bad = 1;
                    break; //all elements are read
                }
            }
        }
    }
    else
    {
        strcpy(buf, "File ");
        strcat(buf, strFilename);
        strcat(buf, " not found or is corrupt.");
        sendInfo(buf);
        return 0;
    }
    iFile.close();
    sendInfo("Done.");
    return 1;
}

/******************************************************************
Function			readScalarValuesParallel()
Arguments			strFilename - name of the file with scalar values
               fltKindOfData - pointer to data arrays that will be filled in
               intNumberElements - number of elements
Return				1 (success) or 0 (failure)
Description			read in scalar data from disc file to the array. Used for version gold .
******************************************************************/
int
ReadEnsight::readScalarValuesParallel(char *strFilename, float *fltKindOfData, int intNumberOfElements)
{
    sendInfo("Begin reading scalar data... ");
    int j = 0;
    char strLine[250];
    char buf[500];
    ifstream iFile;
    iFile.open(strFilename, ios::in);
    if (iFile.good())
    {
        while (iFile.good())
        {
            iFile.getline(strLine, 250);
            if (isGrid(strLine) || isPolygon(strLine))
            {
                break; //we go out if we found a figure or eof
            }
        }
        while ((iFile.good()) && (j < intNumberOfElements))
        {
            iFile.getline(strLine, 250);
            //if valid value
            if ((!isGrid(strLine)) && (!isPolygon(strLine)))
            {
                fltKindOfData[j] = atof(strLine);
                j++;
            }
        }
    }
    else
    {
        strcpy(buf, "File ");
        strcat(buf, strFilename);
        strcat(buf, " not found or is corrupt.");
        sendInfo(buf);
        return 0;
    }
    iFile.close();
    sendInfo("Done.");
    return 1;
}

int
ReadEnsight::readScalarValuesV6(char *strFilename, float *fltKindOfData, int intNumberOfElements)
{
    sendInfo("Begin reading scalar data... ");
    int j = 0;
    int i;
    char *strLine = new char[251];
    char *resetPtr = strLine;
    char *num = new char[FLOAT_LENG + 1];

    char buf[500];
    ifstream iFile;
    iFile.open(strFilename, ios::in);
    if (iFile.good())
    {

        // one line comment
        iFile.getline(strLine, 250);
        iFile.seekg(0L, ios::cur);

        // strLine has to be initialized
        for (i = 0; i < 251; ++i)
            strLine[i] = '\0';

        while ((iFile.good()) && (j < intNumberOfElements))
        {
            strLine = resetPtr;
            iFile.getline(strLine, 250, '\n');
            iFile.seekg(0L, ios::cur);
            //	   cerr << "ReadEnsight::readScalarValuesV6(..) LINE: <" << strLine << ">" << endl;
            if (strncmp(strLine, "part", 4) == 0)
            {
                sendInfo("found structured part  - NOT SUPPORTED yet");
                break;
            }

            //if valid value
            if ((!isGrid(strLine)) && (!isPolygon(strLine)))
            {

                while (strlen(strLine) > 0)
                {

                    strncpy(num, strLine, FLOAT_LENG);
                    num[FLOAT_LENG] = '\0';
                    strLine = &strLine[FLOAT_LENG];
                    float val = atof(num);
                    fltKindOfData[j] = val;
                    // 		   cerr << "ReadEnsight::ReadEnsight(..) <"
                    // 			<< num
                    // 			<< "> "
                    // 			<< val << " : " <<  strlen(strLine) << endl;
                    j++;
                }
            }
        }
    }
    else
    {
        strcpy(buf, "File ");
        strcat(buf, strFilename);
        strcat(buf, " not found or is corrupt.");
        sendInfo(buf);
        return 0;
    }
    iFile.close();
    delete[] resetPtr;
    delete[] num;
    sendInfo("Done.");
    return 1;
}

/******************************************************************
Function			readVectorValuesSequent()
Parameter			strFilename - name of the file with vector values
               fltXData,fltYData,fltZData - pointers to data arrays
               intNumberElements - number of vector values
Return				1 (success) or 0 (failure)
Description			read in vector data from disk file to arrays given. Used for version 5.
******************************************************************/
int ReadEnsight::readVectorValuesSequent(char *strFilename, float *fltXData, float *fltYData, float *fltZData, int intNumberElements)
{
    sendInfo("Begin reading vector data... ");
    //6 elements in the line
    int j = 0, k = 0, n = 0; //n is the current coordinate number
    char strLine[250];
    char strTemp[200];
    char buf[500];
    ifstream iFile;
    //ofstream test("test.txt");
    iFile.open(strFilename, ios::in);
    if (iFile.good())
    {
        strcpy(buf, "Reading data from the file ");
        strcat(buf, strFilename);
        sendInfo(buf);

        iFile.getline(strLine, 250); //read in the first description line
        while (iFile.good())
        {
            iFile.getline(strLine, 250); //read in the current coordinate line
            j = 0;
            //twice 3-elements strings (or probably sometime 3*3...)
            for (int p = 0; p < NUMBER_VALUES_IN_LINE / 3; p++)
            {
                if (n >= intNumberElements * 3)
                {
                    break;
                }
                //in strTemp stays line element number j
                strncpy(strTemp, strLine + LINE_ELEMENT_LENGTH * j, LINE_ELEMENT_LENGTH);
                j++;
                fltXData[k] = atof(strTemp);
                n++;
                if (n >= intNumberElements * 3)
                {
                    break;
                }

                //in strTemp stays line element number j
                strncpy(strTemp, strLine + LINE_ELEMENT_LENGTH * j, LINE_ELEMENT_LENGTH);
                j++;
                fltYData[k] = atof(strTemp);
                n++;
                if (n >= intNumberElements * 3)
                {
                    break;
                }

                //in strTmep stays line element number j
                strncpy(strTemp, strLine + LINE_ELEMENT_LENGTH * j, LINE_ELEMENT_LENGTH);
                j++;
                fltZData[k] = atof(strTemp);
                n++; //counter for every value
                k++; //common counter for 3 data arrays
            }
        }
    }
    else
    {
        strcpy(buf, "File ");
        strcat(buf, strFilename);
        strcat(buf, " not found or is corrupt.");
        sendInfo(buf);

        return 0;
    }

    iFile.close();
    sendInfo("Done.");
    return 1;
}

/******************************************************************
Function			readVectorValuesParallel()
Parameter			strFilename - name of the file with vector values
               fltXData,fltYData,fltZData - pointers to data arrays
               intNumberElements - number of vector values
Return				1 (success) or 0 (failure)
Description			read in vector data from disk file to arrays given. Used for version gold.
******************************************************************/
int
ReadEnsight::readVectorValuesParallel(char *strFilename,
                                      float *fltXData,
                                      float *fltYData,
                                      float *fltZData,
                                      int intNumberElements)
{
    //cout << "function readVectorValue() enter... " << endl;
    int i = 0, j = 0, k = 0, l = 0, m = 0, isGri = 0, isPoly = 0;
    char strLine[250];
    char buf[500];
    ifstream iFile;
    iFile.open(strFilename, ios::in);
    if (iFile.good())
    {
        while (iFile.good())
        {
            iFile.getline(strLine, 250);
            if (isGrid(strLine) || isPolygon(strLine))
            {
                isGri = 1;
                break; //we go out if we found a figure or eof
            }
            else if (isPolygon(strLine))
            {
                isPoly = 1;
                break;
            }
        }

        if (isGri == 1)
        {
            i = 0;
            j = 0;
            k = 0;
            l = 0;
            m = 0;
            while (intGridMap[i] != 0)
            {
                if (iFile.bad())
                {
                    break;
                }
                for (j = 0; j < intGridMap[i] && k < intNumberElements; j++)
                {
                    iFile.getline(strLine, 250);
                    fltXData[k] = atof(strLine);
                    //debug << k << fltXData[k] << endl;
                    k++;
                }
                for (j = 0; j < intGridMap[i] && l < intNumberElements; j++)
                {
                    iFile.getline(strLine, 250);
                    fltYData[l] = atof(strLine);
                    l++;
                }
                for (j = 0; j < intGridMap[i] && m < intNumberElements; j++)
                {
                    iFile.getline(strLine, 250);
                    fltZData[m] = atof(strLine);
                    m++;
                }
                if (iFile.good())
                {
                    iFile.getline(strLine, 250); //do not read the line with the name of the body
                }
                i++;
            }
        }
        else if (isPoly == 1)
        {
            i = 0;
            j = 0;
            k = 0;
            l = 0;
            m = 0;
            while (intPolygonMap[i] != 0)
            {
                if (iFile.bad())
                {
                    break;
                }
                for (j = 0; j < intPolygonMap[i] && k < intNumberElements; j++)
                {
                    iFile.getline(strLine, 250);
                    fltXData[k] = atof(strLine);
                    k++;
                }
                for (j = 0; j < intPolygonMap[i] && l < intNumberElements; j++)
                {
                    iFile.getline(strLine, 250);
                    fltYData[l] = atof(strLine);
                    l++;
                }
                for (j = 0; j < intPolygonMap[i] && m < intNumberElements; j++)
                {
                    iFile.getline(strLine, 250);
                    fltZData[m] = atof(strLine);
                    m++;
                }
                if (iFile.good())
                {
                    iFile.getline(strLine, 250); //do not read the line with the name of the body
                }
                i++;
            }
        }
    }
    else
    {
        strcpy(buf, "File ");
        strcat(buf, strFilename);
        strcat(buf, " not found or is corrupt.");
        sendInfo(buf);
        return 0;
    }

    iFile.close();
    return 1;
}

////////////////////////////////////////////////////////////////////////////////////

int
ReadEnsight::readVectorValuesV6(char *strFilename,
                                float *fltXData,
                                float *fltYData,
                                float *fltZData,
                                int intNumberOfElements)
{
    //cout << "function readVectorValue() enter... " << endl;
    long j = 0, k = 0;
    char *strLine = new char[250];
    char *resetPtr = strLine;
    char *num = new char[FLOAT_LENG + 1];

    char buf[500];
    ifstream iFile;
    iFile.open(strFilename, ios::in);
    if (iFile.good())
    {

        // one line comment
        iFile.getline(strLine, 250);
        iFile.seekg(0L, ios::cur);

        // strLine has to be initialized
        strLine[0] = '\0';

        int cnt(0);
        while ((iFile.good()) && (j < 3 * intNumberOfElements))
        {
            strLine = resetPtr;
            strLine[0] = '\0';
            iFile.getline(strLine, 250, '\n');
            iFile.seekg(0L, ios::cur);

            if (strncmp(strLine, "part", 4) == 0)
            {
                sendInfo("found structured part  - NOT SUPPORTED yet");
                break;
            }

            //int len = strlen(strLine);
            //	    if ((cnt >= 0) && (cnt < 20)) cerr << "ReadEnsight::readVectorValuesV6(..) <" << strLine << "> " << len << endl;

            //if valid value
            if ((!isGrid(strLine)) && (!isPolygon(strLine)))
            {

                while (strlen(strLine) > 0)
                {
                    // we have to be carefull reading the first float of the line
                    // if the float is signed we can read it like usual if not we have
                    // to read FLOAT_LENG-1 chars and convert it to a float
                    strncpy(num, strLine, FLOAT_LENG);
                    num[FLOAT_LENG] = '\0';
                    strLine = &strLine[FLOAT_LENG];

                    float val = atof(num);
                    //		    if ((cnt >= 0) && (cnt < 20)) cerr << "ReadEnsight::readVectorValuesV6(..) num : <" << num << "> val: " << val << " len: " << strlen(strLine) << endl;
                    ++cnt;
                    // we can only read vertex based data
                    if (j % 3 == 0)
                    {
                        fltXData[k] = val;
                    }

                    if (j % 3 == 1)
                    {
                        fltYData[k] = val;
                    }

                    if (j % 3 == 2)
                    {
                        fltZData[k] = val;
                        k++;
                    }
                    j++;
                }
            }
        }
    }
    else
    {
        strcpy(buf, "File ");
        strcat(buf, strFilename);
        strcat(buf, " not found or is corrupt.");
        sendInfo(buf);
        return 0;
    }
    iFile.close();
    delete[] resetPtr; // resetPtr is the real address of strLine - sorry but that's C-style string handling
    return 1;
}

/******************************************************************
Function	read3DCoordsParallel()
Paramete	iFile - reference to the ifstream object, represents a opened data file.
         intCoordsNumber - how many coords we have, how long coords arrays are
         fltXCoords,fltYCoords,fltZCoords - pointers to coords arrays
Return		1 (success) or 0 (failure)
Descript	Read in 3D coordinates from geometry file. Used for ensight 5
******************************************************************/
int
ReadEnsight::read3DCoordsParallel(ifstream &iFile,
                                  int intCoordsNumber,
                                  float *fltXCoords, float *fltYCoords, float *fltZCoords)
{
    sendInfo("Begin reading coordinates from the geometry file...");
    char *strLine = new char[251];
    char strTemp[200];
    char buf[500];
    int i;
    char *num = new char[FLOAT_LENG + 1];

    char *resetPtr = strLine;

    if (nodeIdFlg_)
    {

        for (i = 0; i < intCoordsNumber; i++)
        {
            strLine = resetPtr;
            iFile.get(strLine, 250);
            if (iFile.bad())
            {
                sendError("Unexpected end of file by reading coordinates.");
                return 0;
            }

            iFile.seekg(1L, ios::cur);
            intLineCounter++;
            int idx;
            int recCnt = 0;

            while (strlen(strLine) > 1) // ifstream::get(..) copies the \n into strLine
            {
                float val;
                // Index
                if (recCnt == 0)
                {
                    strncpy(num, strLine, INT_LENG);
                    num[INT_LENG] = '\0';
                    strLine = &strLine[INT_LENG];
                    idx = atoi(num);
                    // do we have to reallocate momory?
                    if (idx >= maxIndex_)
                    {
                        int *tmp = new int[idx + 10];
                        int ii;
                        for (ii = 0; ii < maxIndex_; ++ii)
                        {
                            tmp[ii] = index_[ii];
                        }
                        delete index_;
                        index_ = tmp;
                        maxIndex_ = idx + 10;
                    }
                    index_[idx] = i;
                }
                // coordinate
                else
                {
                    strncpy(num, strLine, FLOAT_LENG);
                    num[FLOAT_LENG] = '\0';
                    strLine = &strLine[FLOAT_LENG];
                    val = atof(num);
                    if (recCnt == 1)
                        fltXCoords[i] = val;
                    if (recCnt == 2)
                        fltYCoords[i] = val;
                    if (recCnt == 3)
                        fltZCoords[i] = val;
                }
                recCnt++;
            }
        }

        strcpy(buf, "Done. ");
        sprintf(strTemp, "%d", i);
        strcat(buf, strTemp);
        strcat(buf, " coordinates read.");
        sendInfo(buf);
    }
    else
    {

        for (i = 0; i < intCoordsNumber; i++)
        {
            iFile.get(strLine, 250);
            if (iFile.bad())
            {
                sendError("Unexpected end of file by reading coordinates.");
                return 0;
            }

            iFile.seekg(1L, ios::cur);
            intLineCounter++;

            //x-coord
            strncpy(strTemp, strLine + LINE_ELEMENT_LENGTH * 0, LINE_ELEMENT_LENGTH);
            fltXCoords[i] = atof(strTemp);

            //y-coord
            strncpy(strTemp, strLine + LINE_ELEMENT_LENGTH * 1, LINE_ELEMENT_LENGTH);
            fltYCoords[i] = atof(strTemp);

            //z-coord
            strncpy(strTemp, strLine + LINE_ELEMENT_LENGTH * 2, LINE_ELEMENT_LENGTH);
            fltZCoords[i] = atof(strTemp);
        }
        strcpy(buf, "Done. ");
        sprintf(strTemp, "%d", i);
        strcat(buf, strTemp);
        strcat(buf, " coordinates read.");
        sendInfo(buf);
    }

    delete[] resetPtr;
    delete[] num;

    return 1;
}

/******************************************************************
Function			read3DCoordsParallel()
Parameter			ifFile - reference to the ifstream object, represents a opened data file.
               intCoordsNumber - how many coords we have, how long coords arrays are
               fltXCoords,fltYCoords,fltZCoords - pointers to coords arrays
Return				1 (success) or 0 (failure)
Description			Read in 3D coordinates from geometry file. Used for Ensight gold.
******************************************************************/
int ReadEnsight::read3DCoordsSequent(ifstream &iFile, int intCoordsNumber, float *fltXCoords, float *fltYCoords, float *fltZCoords)
{
    sendInfo("Begin reading coordinates from the geometry file...");
    char strLine[200];
    char strTemp[200];
    char buf[500];
    int i;
    for (i = 0; i < intCoordsNumber; i++)
    {
        iFile.get(strLine, 250);
        if (iFile.bad())
        {
            sendError("Unexpected end of file by reading coordinates.");
            return 0;
        }
        iFile.seekg(1L, ios::cur);
        intLineCounter++;
        fltXCoords[i] = atof(strLine);
    }
    for (i = 0; i < intCoordsNumber; i++)
    {
        iFile.get(strLine, 250);
        if (iFile.bad())
        {
            sendError("Unexpected end of file by reading coordinates.");
            return 0;
        }
        iFile.seekg(1L, ios::cur);
        intLineCounter++;
        fltYCoords[i] = atof(strLine);
    }
    for (i = 0; i < intCoordsNumber; i++)
    {
        iFile.get(strLine, 250);
        if (iFile.bad())
        {
            sendError("Unexpected end of file by reading coordinates.");
            return 0;
        }
        iFile.seekg(1L, ios::cur);
        intLineCounter++;
        fltZCoords[i] = atof(strLine);
    }

    strcpy(buf, "Done. ");
    sprintf(strTemp, "%d", i);
    strcat(buf, strTemp);
    strcat(buf, " coordinates read.");
    sendInfo(buf);

    return 1;
}

/******************************************************************
Function			lookForHeader()
Parameter			iFile - reference to the ifstream object, represents an opened data file.
               strHeader - string to find in the file
Return				1 (success) or 0 (failure)
Description			search for the string given till we find it or eof occures
******************************************************************/
int ReadEnsight::lookForHeader(ifstream &iFile, char *strHeader)
{
    char strLine[250];
    char buf[500];
    char strTemp[200];

    strcpy(buf, "Searching for header *");
    strcat(buf, strHeader);
    strcat(buf, "* in the geometry file...");
    sendInfo(buf);

    while (iFile.good()) //read all lines in the file (in worst case) till we find the header with patches
    {
        iFile.get(strLine, 250); //we use here get(), because getline deletes delimiters from the stream.
        //it is bad for us, because we shell search the file once more
        iFile.seekg(1L, ios::cur);
        intLineCounter++;
        if ((strstr(strLine, strHeader) != NULL))
        {
            strcpy(buf, "Header *");
            strcat(buf, strHeader);
            strcat(buf, "* found at the line ");
            sprintf(strTemp, "%d", intLineCounter);
            strcat(buf, strTemp);
            sendInfo(buf);
            return 1;
        }
    }
    strcpy(buf, "Header *");
    strcat(buf, strHeader);
    strcat(buf, "* not found in the geometry file, exit. ");
    sendInfo(buf);
    return 0;
}

/******************************************************************
Function			getNumberOfElements()
Parameter			iFile - reference to the ifstream object, represents a opened data file.
               intElementNumber - common number of all elements
               intElementCornerNumber - common number of all element corners
Return				1 (success) or 0 (failure)
Description			at the begin we stay at the first element name. Till we find a delimiter, we only
               count, how many element we have, we don't need smth know about element types.
               We need this function to know, how long our arrays must be
******************************************************************/
//cursor intLineCounter stays at the first element name
int
ReadEnsight::getNumberOfElements(ifstream &iFile,
                                 int &intPolygonNumber,
                                 int &intGridNumber,
                                 int &intPolygonCornerNumber,
                                 int &intGridCornerNumber)
{
    sendInfo("Begin getting number of geometry elements in the geometry file...");
    char strLine[500];
    char strFigureName[200];

    char buf[500];
    int isPoly = 0, isGri = 0;
    char temp[200];
    int tmp, i;
    iFile.get(strLine, 250); //read line with "part..."
    iFile.seekg(1L, ios::cur);
    intLineCounter++;

    while (iFile.good())
    {
        if (strstr(strLine, "part") == NULL) //if we don't found the header
        {
            strcpy(buf, "Element ");
            strcat(buf, strLine);
            strcat(buf, " not recognized as begin of a part (line ");
            sprintf(temp, "%d", intLineCounter);
            strcat(buf, temp);
            strcat(buf, ") in the geometry file.");
            sendInfo(buf);
            return 0;
        }
        iFile.get(strLine, 250);
        iFile.seekg(1L, ios::cur);
        intLineCounter++;
        isPoly = 0;
        isGri = 0;
        while (1)
        {
            iFile.get(strLine, 250);
            iFile.seekg(1L, ios::cur);
            intLineCounter++;

            if (isPolygon(strLine)) //if we can recognize word as a polygon
            {
                strcpy(buf, "Geometry element ");
                strcat(buf, strLine);
                strcat(buf, " recognized as a polygon, line ");
                sprintf(temp, "%d", intLineCounter);
                strcat(buf, temp);
                sendInfo(buf);
                isPoly = 1;
            }
            else if (isGrid(strLine)) //if we can recognize word as a grid
            {
                strcpy(buf, "Geometry element ");
                strcat(buf, strLine);
                strcat(buf, " recognized as a grid, line ");
                sprintf(temp, "%d", intLineCounter);
                strcat(buf, temp);
                sendInfo(buf);
                isGri = 1;
            }
            else
            {
                break;
            }
            strcpy(strFigureName, strLine); //save element type

            iFile.get(strLine, 250); //get number of elements
            iFile.seekg(1L, ios::cur);
            intLineCounter++;

            if (!atoi(strLine)) //if number of polygons of this type not recognized
            {
                sprintf(temp, "%d", intLineCounter);
                strcpy(buf, "Failed to read in the number of elements at the line  ");
                strcat(buf, temp);
                sendInfo(buf);
                return 0;
            }
            strcpy(buf, "Found ");
            strcat(buf, strLine);
            strcat(buf, " elements of type ");
            strcat(buf, strFigureName);
            sendInfo(buf);

            if (isPoly == 1)
            {
                intPolygonNumber += atoi(strLine); //number of all polygons
                //number of all polygon corners
                intPolygonCornerNumber += atoi(strLine) * getNumberOfCorner(strFigureName);
            }

            if (isGri == 1)
            {
                intGridNumber += atoi(strLine); //number of all grids
                //number of all grid corners
                intGridCornerNumber += atoi(strLine) * getNumberOfCorner(strFigureName);
            }
            tmp = atoi(strLine);
            //if numbering exists
            if (intElementIDSectionExist)
            {
                tmp *= 2;
            }
            for (i = 0; i < tmp; i++)
            {
                iFile.get(strLine, 250);
                iFile.seekg(1L, ios::cur);
                intLineCounter++;
                if (iFile.bad())
                {
                    strcpy(buf, "unexpected end of file by getting number of elements, line ");
                    sprintf(temp, "%d", intLineCounter);
                    strcat(buf, temp);
                    sendInfo(buf);
                    return 0;
                }
            }
        } //end of while(1)
    } //end of while(iFile.good())

    strcpy(buf, "Searchig results: ");
    strcat(buf, "\n");
    strcat(buf, "Full number of polygons: ");
    sprintf(temp, "%d", intPolygonNumber);
    strcat(buf, temp);
    strcat(buf, "\n");
    strcat(buf, "Full number of polygon corners: ");
    sprintf(temp, "%d", intPolygonCornerNumber);
    strcat(buf, temp);
    strcat(buf, "\n");
    strcat(buf, "Full number of grid elements: ");
    sprintf(temp, "%d", intGridNumber);
    strcat(buf, temp);
    strcat(buf, "\n");
    strcat(buf, "Full number of grid element corners: ");
    sprintf(temp, "%d", intGridCornerNumber);
    strcat(buf, temp);
    sendInfo(buf);

    return 1;
}

/******************************************************************
Function			fillCornerArrays()
Parameter			iFile - reference to the ifstream object, represented
               an opened data file.
               intGridList - array with pointers to grid elements
               intGridCornerList - array with links to elements of grid corner list
               intTypeList - pointer to array with element types (see f-n getType())
               intElementCornerList - pointer to array with corners
               intPolygonList - array with pointers to polygon elements
               intPolygonCornerList - array with links to elements of polygon corner list
Return				1 (success) or 0 (failure)
Description			fills in arrays given. Function use line counter intLineCounter
and will be called from the position in front of first element name. Used for ensight 5
******************************************************************/
int
ReadEnsight::fillCornerArrays(ifstream &iFile,
                              int *intGridList,
                              int *intGridCornerList,
                              int *intTypeList,
                              int *intPolygonList,
                              int *intPolygonCornerList)
{
    sendInfo("Begin reading corner indexes from the geometry file...");
    char strLine[250];
    char strFigureName[20];
    int isPoly = 0, isGri = 0;
    int intPolygonCount = 0;
    int intPolygonCornerCount = 0;
    int intGridCount = 0;
    int intGridCornerCount = 0;
    int i, j;
    int tmp;
    char strTemp[200];

    translList_ = new int[intGridNumber + intPolygonNumber];
    maxTranslL_ = intGridNumber + intPolygonNumber;

    //    cerr << "ReadEnsight::fillCornerArrays(..) : intGridNumber " << intGridNumber << endl;

    iFile.get(strLine, 250);
    iFile.seekg(1L, ios::cur);
    intLineCounter++;
    while (iFile.good())
    {
        iFile.get(strLine, 250);
        iFile.seekg(1L, ios::cur);
        intLineCounter++;
        isPoly = 0;
        isGri = 0;
        while (1)
        {
            iFile.get(strLine, 250);
            iFile.seekg(1L, ios::cur);
            intLineCounter++;

            if (isPolygon(strLine)) //if we can recognize word as a polygon
            {
                //strcpy(buf, "Geometry element "); strcat(buf, strLine); strcat(buf, " recognized as a polygon."); sendInfo(buf);
                isPoly = 1;
            }
            else if (isGrid(strLine)) //if we can recognize word as a grid
            {
                //strcpy(buf, "Geometry element "); strcat(buf, strLine); strcat(buf, " recognized as a grid."); sendInfo(buf);
                isGri = 1;
            }
            else
            {
                break;
            }
            strcpy(strFigureName, strLine); //save element type

            iFile.get(strLine, 250); //get number of elements
            iFile.seekg(1L, ios::cur);
            intLineCounter++;
            //strcpy(buf, strLine); strcat(buf, " elements of type "); strcat(buf, strFigureName); strcat(buf, " found."); sendInfo(buf);

            if (isPoly == 1)
            {
                tmp = atoi(strLine); //we need it as upper bound in the loop
                //if we have the numbering before the corner description comes, we cut it out
                if (intElementIDSectionExist)
                {
                    for (i = 0; i < tmp; i++)
                    {
                        iFile.get(strLine, 250);
                        iFile.seekg(1L, ios::cur);
                        intLineCounter++;
                    }
                }
                for (i = 0; i < tmp; i++)
                {
                    iFile.get(strLine, 250); //read first line with corners
                    iFile.seekg(1L, ios::cur);
                    intLineCounter++;

                    int end = getNumberOfCorner(strFigureName) + elementIdFlg_;

                    for (j = 0; j < end; j++) //for every all element in the line
                    {
                        if ((j == 0) && elementIdFlg_)
                        {
                            // read element id
                            // 			   strncpy(strTemp,strLine+CORNER_LINE_ELEMENT_LENGTH*j,CORNER_LINE_ELEMENT_LENGTH);//copy line element number j to strTemp
                            // 			   idx =atoi(strTemp)-1;
                            // 			   // do we have to reallocate momory?
                            // 			   if (idx >= maxTranslL_) {
                            // 			       int *tmp = new int[idx+overRealloc];
                            // 			       int ii;
                            // 			       for (ii=0; ii < maxTranslL_; ++ii) {
                            // 				   tmp[ii] = translList_[ii];
                            // 			       }
                            // 			       delete translList_;
                            // 			       translList_ = tmp;
                            // 			       maxTranslL_ = idx+overRealloc;
                            // 			       cerr << " fillCornerArrays(..) memory reallocated  " << endl;
                            //			   }

                            //			   translList_[idx] = intPolygonCount+1;
                        }
                        else
                        {
                            //copy line element number j to strTemp
                            strncpy(strTemp, strLine + CORNER_LINE_ELEMENT_LENGTH * j, CORNER_LINE_ELEMENT_LENGTH);
                            if (nodeIdFlg_)
                            {
                                //sometimes we must substract 1;
                                intPolygonCornerList[intPolygonCornerCount] = index_[atoi(strTemp)];
                            }
                            else
                            {
                                //sometimes we must substract 1;
                                intPolygonCornerList[intPolygonCornerCount] = atoi(strTemp) - 1;
                            }
                            intPolygonCornerCount++;
                        }
                    }
                    intPolygonList[intPolygonCount + 1] = intPolygonList[intPolygonCount] + getNumberOfCorner(strFigureName);
                    //here we should not forget to initialize the first element of intElementList with 0
                    intPolygonCount++;
                }
            }

            if (isGri == 1)
            {
                tmp = atoi(strLine); //we need it as upper bound in the loop
                //if we have the numbering before the corner description comes, we cut it out
                if (intElementIDSectionExist)
                {
                    for (i = 0; i < tmp; i++)
                    {
                        iFile.get(strLine, 250);
                        iFile.seekg(1L, ios::cur);
                        intLineCounter++;
                    }
                }
                for (i = 0; i < tmp; i++)
                {

                    iFile.get(strLine, 250); //read first line with corners
                    iFile.seekg(1L, ios::cur);
                    intLineCounter++;

                    int end = getNumberOfCorner(strFigureName) + elementIdFlg_;
                    //		int idx;
                    for (j = 0; j < end; j++) //for every all element in the line
                    {
                        if ((j == 0) && elementIdFlg_)
                        {
                            // read element id
                            //copy line element number j to strTemp
                            strncpy(strTemp, strLine + CORNER_LINE_ELEMENT_LENGTH * j, CORNER_LINE_ELEMENT_LENGTH);
                            //idx =atoi(strTemp)-1;

                            // 			// do we have to reallocate momory?
                            // 			if (idx >= maxTranslL_) {
                            // 			    int *tmp = new int[idx+overRealloc];
                            // 			    int ii;
                            // 			    for (ii=0; ii < maxTranslL_; ++ii) {
                            // 				tmp[ii] = translList_[ii];
                            // 			    }
                            // 			    delete translList_;
                            // 			    translList_ = tmp;
                            // 			    maxTranslL_ = idx+overRealloc;
                            // 			    cerr << " fillCornerArrays(..) memory reallocated  " << endl;
                            // 			}

                            // 			translList_[idx] = intPolygonCount+1;
                        }
                        else
                        {
                            //copy line element number j to strTemp
                            strncpy(strTemp, strLine + CORNER_LINE_ELEMENT_LENGTH * j, CORNER_LINE_ELEMENT_LENGTH);
                            if (nodeIdFlg_)
                            {
                                int idx = index_[atoi(strTemp)];
                                if (idx > intGridCoordsNumber)
                                {
                                    cerr << "ReadEnsight::fillCornerArrays(..) index out of range "
                                         << idx << " | " << strTemp << " | " << intGridCornerCount << endl;
                                }
                                intGridCornerList[intGridCornerCount] = idx;
                            }
                            else
                            {
                                intGridCornerList[intGridCornerCount] = atoi(strTemp) - 1;
                            }
                            intGridCornerCount++;
                        }
                    }
                    intGridList[intGridCount + 1] = intGridList[intGridCount] + getNumberOfCorner(strFigureName);
                    intTypeList[intGridCount] = getType(strFigureName);
                    //here we should not forget to initialize the first element of intElementList with 0
                    intGridCount++;
                }
            }
        } //end of while(1)
    } //end of while(iFile.good())
    sendInfo("Done.");
    return 1;
}

/******************************************************************
Function			sendInfoSuccess()
Parameter			const char *strInfo - name of the object created
Return
Description			shows the message, that object "strObject" is created
******************************************************************/
void ReadEnsight::sendInfoSuccess(const char *strObject)
{
    char buf[500];
    strcpy(buf, "Object ");
    strcat(buf, strObject);
    strcat(buf, " created.");
    sendInfo(buf);
}

/******************************************************************
Function			readResultFile()
Parameter			name of the result file
Return
Description			analyse the result file. Ensight 5
******************************************************************/
int ReadEnsight::readResultFile(const char *strResultFileName)
{
    char strLine[250];

    char buf[500];

    int intNumberOfScalarVariables;
    int intNumberOfVectorVariables;

    int intGeometryChangingFlag = 0;
    char ctmp[200];
    int intLineCounter = 0;

    strcpy(buf, "Begin getting information about scalar and vector data from the result file ");
    strcat(buf, strResultFileName);
    sendInfo(buf);

    ifstream iFile;
    iFile.open(strResultFileName, ios::in);
    if (iFile.good())
    {
        strcpy(buf, "Result file opened.");
        sendInfo(buf);
        //READ NUMBER OF SCALAR AND VECTOR VARIABLES AND GEOMETRY CHANGING FLAG - line 1

        iFile.getline(strLine, 250); //number of scalar and vector values and geometry changing flag, separated throught a blank: 1 1 0
        intLineCounter++;

        int ii = 0;
        int jj = 0;
        int kk = 0;
        int retVal = sscanf(strLine, "%d %d %d", &ii, &jj, &kk);

        intNumberOfScalarVariables = 0;
        intNumberOfVectorVariables = 0;
        intGeometryChangingFlag = 0;

        if (retVal == 3)
        {
            intNumberOfScalarVariables = ii;
            intNumberOfVectorVariables = jj;
            intGeometryChangingFlag = kk;
        }
        else
        {
            strcpy(buf, "Error reading Line 1 of result-file ");
            sendInfo(buf);
            return 0;
        }

        strcpy(buf, "Line 1: ");
        sprintf(ctmp, "%d", intNumberOfScalarVariables);
        strcat(buf, ctmp);
        strcat(buf, " scalar and ");
        sprintf(ctmp, "%d", intNumberOfVectorVariables);
        strcat(buf, ctmp);
        strcat(buf, " vector variables are to be processed.");
        sendInfo(buf);

        //READ NUMBER OF TIME STEPS - line 2

        iFile.getline(strLine, 250); //second line - number of time steps
        intLineCounter++;
        intNumberOfTimeSteps = atoi(strLine);

        strcpy(buf, "Line 2: ");
        strcat(buf, strLine);
        strcat(buf, " timesteps");
        sendInfo(buf);

        //create a set, we use it later
        setObject = new coDistributedObject *[intNumberOfTimeSteps + 1];
        setObject[intNumberOfTimeSteps] = NULL; //they want so

        //READ TIMES FOR EACH STEP - line 3 (can be several lines in one, according to steps number)

        //if intNumberOfTimeSteps=23 and  NUMBER_VALUES_IN_LINE=6, we get 23/6~=3.9 => 3 and not 4.
        //Therefore we add 1 to get the hole number of lines
        int i;
        for (i = 0; i < intNumberOfTimeSteps / NUMBER_VALUES_IN_LINE + 1; i++)
        {
            iFile.getline(strLine, 250); //we don't process this lines yet
            intLineCounter++;
        }

        //READ OFFSET BETWEEN FILES (see Docu) - line 4 (only if more then 1 timestep in line 2)

        if (intNumberOfTimeSteps > 1)
        {
            iFile.getline(strLine, 250); //we don't process this line yet
            intLineCounter++;
        }

        //READ NAME OF GEOMETRY FILES - only if geometry changing flaf at the line 1 is set to 1

        if (intGeometryChangingFlag)
        {
            iFile.getline(strLine, 250); //we don't process this line yet, reserved for the future use
            intLineCounter++;
        }

        //READ SCALAR TIMESTEP FILE NAMES

        if (intNumberOfScalarVariables > 0)
        {
            //setObject will be here filled in
            if (!readVarFiles(iFile, intNumberOfScalarVariables, intNumberOfTimeSteps, 1))
            {
                sendInfo("readTimestepFiles (scalar) failed");
                return 0;
            }
            sendInfo("readTimestepFiles (scalar) success");
            coDoSet *scalarSet = new coDoSet(gridScalarOutPort->getObjName(), setObject);
            if (scalarSet == NULL)
            {
                sendInfo("cannot create scalarSet");
                return 0;
            }

            sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
            scalarSet->addAttribute("TIMESTEP", buf);
            gridScalarOutPort->setCurrentObject(scalarSet);
            for (int i = 0; i < intNumberOfTimeSteps; i++)
            {
                delete setObject[i];
            }
        }

        //READ VECTOR TIMESTEP FILE NAMES

        if (intNumberOfVectorVariables > 0)
        {
            if (!readVarFiles(iFile, intNumberOfVectorVariables, intNumberOfTimeSteps, 0))
            {
                sendInfo("readTimestepFiles (vector) failed");
                return 0;
            }
            sendInfo("readTimestepFiles (vector) success");
            coDoSet *vectorSet = new coDoSet(gridVectorOutPort->getObjName(), setObject);
            if (vectorSet == NULL)
            {
                sendInfo("cannot create vectorSet");
                return 0;
            }

            sprintf(buf, "%d %d", 1, intNumberOfTimeSteps);
            vectorSet->addAttribute("TIMESTEP", buf);
            gridVectorOutPort->setCurrentObject(vectorSet);
            for (int i = 0; i < intNumberOfTimeSteps; i++)
            {
                delete setObject[i];
            }
        }
        delete[] setObject;
    }
    else
    {
        strcpy(buf, "Result file ");
        strcat(buf, strResultFileName);
        strcat(buf, " not found or is corrupt.");
        sendInfo(buf);
        return 0;
    }
    iFile.close();
    strcpy(buf, "End reading data");
    sendInfo(buf);
    return 1;
}

/******************************************************************
Function			int readVarFiles(ifstream&, int, int, int)
Arguments			iFile - opened res file
               intNumberOfValues - number of values to read in
               intNumberOfTimeSteps - number of timesteps
               isScalar - whether we must read scalar or vector data: 1 - we have scalar, otherwise - vector data
Return				1 by success, otherwise 0
Description			Build timestep file names (if there are more as 1) and read data from these files
******************************************************************/
int
ReadEnsight::readVarFiles(ifstream &iFile, int intNumberOfValues, int intNumberOfTimeSteps, int isScalar)
{
    char strLine[250];
    char strTemp[250];
    char buf[500];
    char *ptrFound = NULL;
    int result = 0;
    int intNumberOfAsterisks;
    char strRoot[20];
    char strTimestepFileName[30];
    char ctmp[200];
    float *fltKindOfGridScalarData;

    strcpy(buf, "Building timestep file names...");
    sendInfo(buf);

    int size;
    if (intKindOfMeasuring == PER_NODE)
    {
        size = intGridCoordsNumber;
    }
    else
    {
        size = intGridNumber;
    }

    int nVal;
    for (nVal = 0; nVal < intNumberOfValues; nVal++)
    {
        iFile.getline(strLine, 250);
        //ptrFound=strchr(strLine,' ');//look for the first blank
        int corr = 0;
        while (strLine[corr] == ' ')
            corr++;
        ptrFound = strchr(strLine + corr, ' '); //look for the first blank
        if (ptrFound == NULL)
        {
            strcpy(buf, "Delimiter at the line with file names in the result file not found.");
            sendInfo(buf);
            return 0;
        }
        //result=ptrFound-strLine+1;//at the position result we found the first blank
        //strncpy(strTemp,strLine,result);//select first number - up to blank
        ptrFound += corr;
        result = ptrFound - 2 * corr - strLine + 1; //at the position result we found the first blank
        strncpy(strTemp, strLine + corr, result); //select first number - up to blank
        strTemp[result] = '\0'; //in strTemp we have e.g. plastic***

        //GET NUMBER OF ASTERISKS AND SELECT THE ROOT OF THE FILENAME

        sprintf(buf, "%d", intNumberOfTimeSteps);
        intNumberOfAsterisks = strlen(buf) + 1; //if 21 timestep, then plastic***, 1 asterisk more
        if (intNumberOfTimeSteps == 1) //special case - for 1 we have no asterisks
        {
            intNumberOfAsterisks = 0;
        }
        strncpy(strRoot, strTemp, strlen(strTemp) - intNumberOfAsterisks);
        strRoot[strlen(strTemp) - intNumberOfAsterisks - 1] = '\0';

        //BUILD FILENAMES AND OPEN FILES

        strcpy(strTimestepFileName, strRoot);
        strTimestepFileName[strlen(strRoot)] = '\0';
        if ((isScalar && (strstr(paramGridScalarFile->getValue(), strRoot) != NULL || strstr(paramPolygonScalarFile->getValue(), strRoot) != NULL)) || (!isScalar && (strstr(paramGridVectorFile->getValue(), strRoot) == 0 || strstr(paramPolygonVectorFile->getValue(), strRoot) == 0)))
        {
            int intNumberOfNulls = 0;
            for (int i = 0; i < intNumberOfTimeSteps; i++)
            {
                strTimestepFileName[strlen(strRoot)] = '\0';
                sprintf(buf, "%d", i);
                //number of nulls to be added to the root, plastic00
                intNumberOfNulls = intNumberOfAsterisks - strlen(buf);
                if (intNumberOfTimeSteps == 1) //special case - for 1 we have no asterisks
                {
                    intNumberOfNulls = 0;
                }
                for (int j = 0; j < intNumberOfNulls; j++)
                {
                    strcat(strTimestepFileName, "0"); //plastic0, plastic00, ...
                }
                if (intNumberOfTimeSteps > 1)
                {
                    strcat(strTimestepFileName, buf); //plastic019
                }
                strcpy(buf, geoFileParam->getValue());
                strcpy(ctmp, "dirname ");
                strcat(ctmp, buf);
                strcat(ctmp, " > tempFile.txt"); //change to ENV-variable!!
                system(ctmp);
                ifstream iResultFile;
                iResultFile.open("tempFile.txt", ios::in);
                if (iResultFile.good())
                {
                    iResultFile.getline(strLine, 250);
                    strcat(strLine, "/");
                    strcat(strLine, strTimestepFileName);
                }
                else
                {
                    sendInfo("Error by reading of a temporary file");
                    return 0;
                }
                //READ DATA FROM SCALAR TIMESTEP FILES
                if (isScalar == 1)
                {
                    fltKindOfGridScalarData = new float[size + 1];
                    if (!readScalarValuesSequent(strLine, fltKindOfGridScalarData, size))
                    {
                        strcpy(buf, "Failed to read scalar timestep values by file ");
                        strcat(buf, strLine);
                        sendInfo(buf);
                        delete[] fltKindOfGridScalarData;
                        return 0;
                    }

                    //CREATE OBJECTS AND INSERT IT IN THE SET

                    char *setElemName = new char[10 + strlen(strTimestepFileName) + strlen(gridScalarOutPort->getObjName())];
                    char num[3];
                    sprintf(num, "%d%d", i, nVal);
                    strcpy(setElemName, gridScalarOutPort->getObjName());
                    strcat(setElemName, strTimestepFileName);
                    strcat(setElemName, "S");
                    strcat(setElemName, num);

                    setObject[i] = new coDoFloat(setElemName, size, fltKindOfGridScalarData);

                    if (!setObject[i]->objectOk())
                    {
                        cerr << "ReadEnsight::readVarFiles(..) failed to create Obj " << setElemName << endl;
                    }

                    delete[] setElemName;

                    if (setObject[i] == NULL)
                    {
                        strcpy(buf, "Failed to create object ");
                        strcat(buf, strTimestepFileName);
                        sendInfo(buf);
                        delete[] fltKindOfGridScalarData;
                        return 0;
                    }

                    setObject[i]->addAttribute("SPECIES", strRoot);
                    sendInfoSuccess(strTimestepFileName);
                    delete[] fltKindOfGridScalarData;
                }
                else //vector data
                {
                    fltXData = new float[size + 1];
                    if (fltXData == NULL)
                    {
                        strcpy(buf, "Cannot allocate memory for X-component of vector data.");
                        sendInfo(buf);
                        return 0;
                    }
                    fltYData = new float[size + 1];
                    if (fltYData == NULL)
                    {
                        strcpy(buf, "Cannot allocate memory for Y-component of vector data.");
                        sendInfo(buf);
                        delete[] fltXData;
                        return 0;
                    }
                    fltZData = new float[size + 1];
                    if (fltZData == NULL)
                    {
                        strcpy(buf, "Cannot allocate memory for Y-component of vector data.");
                        sendInfo(buf);
                        delete[] fltXData;
                        delete[] fltYData;
                        return 0;
                    }

                    if (!readVectorValuesSequent(strLine, fltXData, fltYData, fltZData, size))
                    {
                        strcpy(buf, "Vector data is not valid.");
                        sendInfo(buf);
                        return 0;
                    }

                    // create unique name of setelem
                    char *setElemName = new char[10 + strlen(strTimestepFileName) + strlen(gridVectorOutPort->getObjName())];
                    char num[3];
                    sprintf(num, "%d", i);
                    strcpy(setElemName, gridVectorOutPort->getObjName());
                    strcat(setElemName, strTimestepFileName);
                    strcat(setElemName, "V");
                    strcat(setElemName, num);

                    setObject[i] = new coDoVec3(setElemName, size, fltXData, fltYData, fltZData);

                    delete[] setElemName;

                    if (setObject[i] == NULL)
                    {
                        strcpy(buf, "Failed to create object ");
                        strcat(buf, strTimestepFileName);
                        sendInfo(buf);
                        delete[] fltXData;
                        delete[] fltYData;
                        delete[] fltZData;
                        return 0;
                    }

                    setObject[i]->addAttribute("SPECIES", strRoot);
                    sendInfoSuccess(strTimestepFileName);
                    delete[] fltXData;
                    delete[] fltYData;
                    delete[] fltZData;
                }
            }
        }
    }
    remove("tempFile.txt"); //in this file we saved the absolute path to current directory
    strcpy(buf, "End reading timestep values.");
    sendInfo(buf);

    return 1;
}

/******************************************************************
Function			int getEnsightVersion(ifstream&)
Arguments			iFile - opened case file
Return				"ensight gold" for version gold, otherwise "ensight"
Description			searchs case file for version string. As default version 5
******************************************************************/
//get version of ensight, true==ensight gold
char *ReadEnsight::getEnsightVersion(ifstream &iFile)
{
    char strLine[250];
    char *pos;
    iFile.seekg(0L, ios::beg);
    while (iFile.good())
    {
        iFile.getline(strLine, 250);
        if (strcmp(strLine, "FORMAT") == 0)
        {
            iFile.getline(strLine, 250);
            pos = strstr(strLine, "ensight gold");
            if (pos != NULL)
            {
                return ("ensight gold");
            }
            return ("ensight");
        }
    }
    return ("ensight");
}

/******************************************************************
Function			int getNumberOfTimeSteps(ifstream&)
Arguments			iFile - opened case file
Return				number of timesteps
Description			searchs case file for timesteps number
******************************************************************/
int ReadEnsight::getNumberOfTimeSteps(ifstream &iFile)
{
    char strLine[250];
    char *pos;
    iFile.seekg(0L, ios::beg);
    while (iFile.good())
    {
        iFile.getline(strLine, 250);
        pos = strstr(strLine, "number of steps:");
        if (pos != NULL)
        {
            strncpy(strLine, strLine + strlen("number of steps:") + 1, strlen(strLine) - strlen("number of steps:"));
            strLine[strlen(strLine) - strlen("number of steps:") + 1] = '\0';
            return (atoi(strLine));
        }
    }
    return 1;
}

/******************************************************************
Function			int isThereNodeIDSection(ifstream&)
Arguments			iFile - opened geometry file
Return				1 if node ID section exists (see Docu), otherwise 0
Description
******************************************************************/
//in 3th line we find info about node id - is there such a section
int
ReadEnsight::isThereNodeIDSection(ifstream &iFile)
{
    char strLine[250];
    int i;
    iFile.seekg(0L, ios::beg);
    for (i = 0; i < 3; i++)
    {
        iFile.getline(strLine, 250); //in strLine we have "node id ..."
    }
    if ((strcmp(strLine, "node id given") == 0) || (strcmp(strLine, "node id ignore") == 0))
    {
        iFile.seekg(0L, ios::beg);
        return 1;
    }
    iFile.seekg(0L, ios::beg);
    return 0;
}

/******************************************************************
Function			int isThereElementIDSection(ifstream&)
Arguments			iFile - opened geometry file
Return				1 if element ID section exists (see Docu), otherwise 0
Description
******************************************************************/
//in 4th line we find info about element id - is there an id section for each element type of each part
int ReadEnsight::isThereElementIDSection(ifstream &iFile)
{
    char strLine[250];
    int i;
    iFile.seekg(0L, ios::beg);
    for (i = 0; i < 4; i++)
    {
        iFile.getline(strLine, 250); //in strLine we have "node id ..."
    }
    if ((strcmp(strLine, "element id given") == 0) || (strcmp(strLine, "element id ignore") == 0))
    {
        iFile.seekg(0L, ios::beg);
        return 1;
    }
    iFile.seekg(0L, ios::beg);
    return 0;
}

/******************************************************************
Function			int initFileMap()
Arguments
Return
Description
******************************************************************/
void ReadEnsight::initFileMap()
{
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            intFileMap[i][j] = 0;
        }
    }
    intFileMapIndex = 0;
}

/******************************************************************
Function			int fillFileMap(int,int,int,int)
Arguments			intLineNumber - line number in geometry file
               intCornersNumber - number of corners for figure found in the line
               intNumberOfLines - how many lines with indexes are to read for the element (number of elements)
               intType - type of element (see getType() function)
Return
Description			intFileMap save info about all element found in geometry file. During first searching
               process this structure will be filled in, later we use it for more effectiv working
******************************************************************/
void ReadEnsight::fillFileMap(int intLineNumber, int intCornersNumber, int intNumberOfLines, int intType)
{
    intFileMap[intFileMapIndex][0] = intLineNumber;
    intFileMap[intFileMapIndex][1] = intCornersNumber;
    intFileMap[intFileMapIndex][2] = intNumberOfLines;
    intFileMap[intFileMapIndex][3] = intType;

    intFileMapIndex++;
}

/******************************************************************
Function			showFileMap()
Arguments
Return
Description			only debug info
******************************************************************/
void ReadEnsight::showFileMap()
{
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (intFileMap[i][j] == 0)
            {
                goto ex;
            }
            cout << intFileMap[i][j] << '\t';
        }
        cout << endl;
    }
ex:
    ;
}

/******************************************************************
Function			int isPolygon(char*)
Arguments			strPolygonType - string with polygon name
Return				1 if strPolygonType is recognized as polygon, otherwise 0
Description			check whether element found in the geometry file corresponds to currently
               supported by covise polygon types
******************************************************************/
void ReadEnsight::initPolygonMap()
{
    for (int i = 0; i < 100; i++)
    {
        intPolygonMap[i] = 0;
    }
    intPolygonMapIndex = 0;
}

/******************************************************************
Function			int fillPolygonMap(int)
Arguments			intNumberOfElements - number of elements for each polygon type found
Return
Description			save numbers for all polygons found in geo file
******************************************************************/
void ReadEnsight::fillPolygonMap(int intNumberOfElements)
{
    intPolygonMap[intPolygonMapIndex] = intNumberOfElements;
    intPolygonMapIndex++;
}

/******************************************************************
Function			showPolygonMap()
Arguments
Return
Description			only debug info
******************************************************************/
void ReadEnsight::showPolygonMap()
{
    for (int i = 0; i < 100; i++)
    {
        if (intPolygonMap[i] == 0)
        {
            break;
        }
        cout << intPolygonMap[i] << endl;
    }
}

/******************************************************************
Function			void initGridMap()
Arguments
Return
Description			initialize array with grid elements found in geometry file
******************************************************************/
void ReadEnsight::initGridMap()
{
    for (int i = 0; i < 100; i++)
    {
        intGridMap[i] = 0;
    }
    intGridMapIndex = 0;
}

/******************************************************************
Function			int fillGridMap(int)
Arguments			intNumberOfElements - number of elements for each grid type found
Return
Description			save numbers for all grid elements found in geo file
******************************************************************/
void ReadEnsight::fillGridMap(int intNumberOfElements)
{
    intGridMap[intGridMapIndex] = intNumberOfElements;
    intGridMapIndex++;
}

/******************************************************************
Function			showGridMap()
Arguments
Return
Description			only debug info
******************************************************************/
void ReadEnsight::showGridMap()
{
    for (int i = 0; i < 100; i++)
    {
        if (intGridMap[i] == 0)
        {
            break;
        }
        cout << intGridMap[i] << endl;
    }
}

/******************************************************************
Function			char* buildTimestepFileName(char*,int,int)
Arguments			strFileName - name of the first timestep file
               intNumberOfTimeSteps
               intStep - current timestep
Return
Description			build name of timestep file to be opened
******************************************************************/
char *ReadEnsight::buildTimestepFileName(char *strFileName, int intNumberOfTimeSteps, int intStep)
{
    char buf[500];
    int intNumberOfNulls = 0;
    int intNumberOfAsterisks = 0;
    char strTimestepFileName[100];
    char strRoot[100];
    int j;

    sprintf(buf, "%d", intNumberOfTimeSteps);
    intNumberOfAsterisks = strlen(buf) + 1; //if 21 timestep, then plastic***, 1 asterisk more
    if (intNumberOfTimeSteps == 1) //special case - for 1 we have no asterisks
    {
        intNumberOfAsterisks = 0;
    }
    strcpy(strTimestepFileName, strFileName);
    strncpy(strRoot, strTimestepFileName, strlen(strTimestepFileName) - intNumberOfAsterisks);
    strRoot[strlen(strTimestepFileName) - intNumberOfAsterisks] = '\0';
    strcpy(strTimestepFileName, strRoot);
    strTimestepFileName[strlen(strRoot)] = '\0';
    sprintf(buf, "%d", intStep);
    //number of nulls to be added to the root, plastic00
    intNumberOfNulls = intNumberOfAsterisks - strlen(buf);
    if (intNumberOfTimeSteps == 1) //special case - for 1 we have no asterisks
    {
        intNumberOfNulls = 0;
    }
    for (j = 0; j < intNumberOfNulls; j++)
    {
        strcat(strTimestepFileName, "0"); //plastic0, plastic00, ...
    }
    if (intNumberOfTimeSteps > 1)
    {
        strcat(strTimestepFileName, buf); //plastic019
    }
    char *rtn = new char[strlen(strTimestepFileName) + 1];
    strcpy(rtn, strTimestepFileName);
    return rtn;
}

/******************************************************************
Function			int getPerNodeOrPerElement(ifstream&)
Arguments			iFile - opened case file
Return				PER_NODE or PER_ELEMENT
Description			art of mesuaring - do we have data per node or per element in data files
******************************************************************/
int ReadEnsight::getPerNodeOrPerElement(ifstream &iFile)
{
    char strLine[250];
    char *pos;
    iFile.seekg(0L, ios::beg);
    while (iFile.good())
    {
        iFile.getline(strLine, 250);
        pos = strstr(strLine, "per node:");
        if (pos != NULL)
        {
            return PER_NODE;
        }
        pos = strstr(strLine, "per element:");
        if (pos != NULL)
        {
            return PER_ELEMENT;
        }
    }
    return PER_NODE;
}

MODULE_MAIN(Obsolete, ReadEnsight)
