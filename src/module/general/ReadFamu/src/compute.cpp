/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file compute.cpp
 * top-level management of mesh and results file input.
 */

#include "ReadFamu.h"
#include "BuilderGrid.h" // a builder for grids.
#include "FactoryResultsFileParser.h" // a factory for results file parsers.
#include "FactoryMeshFileTransParser.h" // a factory for results file parsers.
#include "errorinfo.h" // a container for error data.
#include "PeriodicRotResults.h" // completes results/meshes to a full 360 degree rotation.

#include "Plane.h"
#include "Node.h"
#include "Functions.h"
#include "Tools.h" // some helpful tools.

#include <api/coFeedback.h>
#include <math.h>
#include <string.h>
//#include <mbstring.h>

#ifndef _MSC_VER
#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#define strncpy_s(a, b, c, d) strncpy(a, c, min(d, b))
#define strncat_s(a, b, c, d) strncat(a, c, min(d, b))
#endif

//#include "builderfamu.hxx"
//#include "famu.hxx"

//#pragma warning( disable :  4101 )   // "'error' : unreferenced local variable"

int ReadFamu::compute(const char * /*port*/)
{

    if (reset->getValue())
    {
        bottomLeft->setValue(origBottomLeft[0], origBottomLeft[1], origBottomLeft[2]);
        bottomRight->setValue(origBottomRight[0], origBottomRight[1], origBottomRight[2]);
        topRight->setValue(origTopRight[0], origTopRight[1], origTopRight[2]);
        topLeft->setValue(origTopLeft[0], origTopLeft[1], origTopLeft[2]);

        //reset params
        moveDist->setValue(0.0, 0.0, 0.0);
        scaleFactor->setValue(1.0);
        XYDegree->setValue(0.0);
        YZDegree->setValue(0.0);
        ZXDegree->setValue(0.0);

        reset->setValue(false);
    }

    const char *firstFile = _in_FirstFile->getValue();
    const char *secondFile = _in_SecondFile->getValue();

    if (FileExists(firstFile) && FileExists(secondFile) && _startSim->getValue())
    {
        Plane myPlane;
        //move the plane
        float xDist = 0, yDist = 0, zDist = 0;
        xDist = moveDist->getValue(0);
        yDist = moveDist->getValue(1);
        zDist = moveDist->getValue(2);

        //scale the plane
        float scale = 1;
        scale = sqrt(scaleFactor->getValue());
        float origCenter[3];
        origCenter[0] = (bottomLeft->getValue(0) + topRight->getValue(0)) / 2;
        origCenter[1] = (bottomLeft->getValue(1) + topRight->getValue(1)) / 2;
        origCenter[2] = (bottomLeft->getValue(2) + topRight->getValue(2)) / 2;

        float xDiff = 0, yDiff = 0, zDiff = 0;

        if (scale != 1)
        {

            xDiff = (1 - scale) * origCenter[0];
            yDiff = (1 - scale) * origCenter[1];
            zDiff = (1 - scale) * origCenter[2];
        }

        //initialize the reference points
        myPlane._referenceNode[0].x = bottomLeft->getValue(0) * scale + xDist + xDiff;
        myPlane._referenceNode[0].y = bottomLeft->getValue(1) * scale + yDist + yDiff;
        myPlane._referenceNode[0].z = bottomLeft->getValue(2) * scale + zDist + zDiff;

        myPlane._referenceNode[1].x = scale * (bottomLeft->getValue(0) + bottomRight->getValue(0)) / 2 + xDist + xDiff;
        myPlane._referenceNode[1].y = scale * (bottomLeft->getValue(1) + bottomRight->getValue(1)) / 2 + yDist + yDiff;
        myPlane._referenceNode[1].z = scale * (bottomLeft->getValue(2) + bottomRight->getValue(2)) / 2 + zDist + zDiff;

        myPlane._referenceNode[2].x = bottomRight->getValue(0) * scale + xDist + xDiff;
        myPlane._referenceNode[2].y = bottomRight->getValue(1) * scale + yDist + yDiff;
        myPlane._referenceNode[2].z = bottomRight->getValue(2) * scale + zDist + zDiff;

        myPlane._referenceNode[3].x = scale * (bottomRight->getValue(0) + topRight->getValue(0)) / 2 + xDist + xDiff;
        myPlane._referenceNode[3].y = scale * (bottomRight->getValue(1) + topRight->getValue(1)) / 2 + yDist + yDiff;
        myPlane._referenceNode[3].z = scale * (bottomRight->getValue(2) + topRight->getValue(2)) / 2 + zDist + zDiff;

        myPlane._referenceNode[4].x = topRight->getValue(0) * scale + xDist + xDiff;
        myPlane._referenceNode[4].y = topRight->getValue(1) * scale + yDist + yDiff;
        myPlane._referenceNode[4].z = topRight->getValue(2) * scale + zDist + zDiff;

        myPlane._referenceNode[5].x = scale * (topRight->getValue(0) + topLeft->getValue(0)) / 2 + xDist + xDiff;
        myPlane._referenceNode[5].y = scale * (topRight->getValue(1) + topLeft->getValue(1)) / 2 + yDist + yDiff;
        myPlane._referenceNode[5].z = scale * (topRight->getValue(2) + topLeft->getValue(2)) / 2 + zDist + zDiff;

        myPlane._referenceNode[6].x = topLeft->getValue(0) * scale + xDist + xDiff;
        myPlane._referenceNode[6].y = topLeft->getValue(1) * scale + yDist + yDiff;
        myPlane._referenceNode[6].z = topLeft->getValue(2) * scale + zDist + zDiff;

        myPlane._referenceNode[7].x = scale * (bottomLeft->getValue(0) + topLeft->getValue(0)) / 2 + xDist + xDiff;
        myPlane._referenceNode[7].y = scale * (bottomLeft->getValue(1) + topLeft->getValue(1)) / 2 + yDist + yDiff;
        myPlane._referenceNode[7].z = scale * (bottomLeft->getValue(2) + topLeft->getValue(2)) / 2 + zDist + zDiff;

        //rotate the electrode
        Node center1, center2, center3;
        float degree1 = 0, degree2 = 0, degree3 = 0;
        degree1 = XYDegree->getValue();

        //rotate eletrode around the axis Z

        if (XYDegree->getValue() > 0)
        {
            degree1 = XYDegree->getValue();
            center1.x = (myPlane._referenceNode[2].x + myPlane._referenceNode[0].x) / 2;
            center1.y = (myPlane._referenceNode[2].y + myPlane._referenceNode[0].y) / 2;
            center1.z = (myPlane._referenceNode[2].z + myPlane._referenceNode[0].z) / 2;

            rotatePoint(&(myPlane._referenceNode[0]), center1, degree1, 1);
            rotatePoint(&(myPlane._referenceNode[2]), center1, degree1, 1);
            rotatePoint(&(myPlane._referenceNode[3]), center1, degree1, 1);
            rotatePoint(&(myPlane._referenceNode[4]), center1, degree1, 1);
            rotatePoint(&(myPlane._referenceNode[5]), center1, degree1, 1);
            rotatePoint(&(myPlane._referenceNode[6]), center1, degree1, 1);
            rotatePoint(&(myPlane._referenceNode[7]), center1, degree1, 1);
        }

        //rotate eletrode around the axis X
        if (YZDegree->getValue() > 0)
        {
            degree2 = YZDegree->getValue();
            center2.x = (myPlane._referenceNode[0].x + myPlane._referenceNode[4].x) / 2;
            center2.y = (myPlane._referenceNode[0].y + myPlane._referenceNode[4].y) / 2;
            center2.z = (myPlane._referenceNode[0].z + myPlane._referenceNode[4].z) / 2;

            rotatePoint(&(myPlane._referenceNode[0]), center2, degree2, 2);
            rotatePoint(&(myPlane._referenceNode[1]), center2, degree2, 2);
            rotatePoint(&(myPlane._referenceNode[2]), center2, degree2, 2);
            rotatePoint(&(myPlane._referenceNode[3]), center2, degree2, 2);
            rotatePoint(&(myPlane._referenceNode[4]), center2, degree2, 2);
            rotatePoint(&(myPlane._referenceNode[5]), center2, degree2, 2);
            rotatePoint(&(myPlane._referenceNode[6]), center2, degree2, 2);
            rotatePoint(&(myPlane._referenceNode[7]), center2, degree2, 2);
        }
        //rotate eletrode around the axis Y
        if (ZXDegree->getValue() > 0)
        {
            degree3 = ZXDegree->getValue();
            center3.x = (myPlane._referenceNode[6].x + myPlane._referenceNode[0].x) / 2;
            center3.y = (myPlane._referenceNode[6].y + myPlane._referenceNode[0].y) / 2;
            center3.z = (myPlane._referenceNode[6].z + myPlane._referenceNode[0].z) / 2;

            rotatePoint(&(myPlane._referenceNode[0]), center3, degree3, 3);
            rotatePoint(&(myPlane._referenceNode[1]), center3, degree3, 3);
            rotatePoint(&(myPlane._referenceNode[2]), center3, degree3, 3);
            rotatePoint(&(myPlane._referenceNode[3]), center3, degree3, 3);
            rotatePoint(&(myPlane._referenceNode[4]), center3, degree3, 3);
            rotatePoint(&(myPlane._referenceNode[5]), center3, degree3, 3);
            rotatePoint(&(myPlane._referenceNode[6]), center3, degree3, 3);
        }

        //set the new values to the Param

        bottomLeft->setValue(myPlane._referenceNode[0].x, myPlane._referenceNode[0].y, myPlane._referenceNode[0].z);
        bottomRight->setValue(myPlane._referenceNode[2].x, myPlane._referenceNode[2].y, myPlane._referenceNode[2].z);
        topRight->setValue(myPlane._referenceNode[4].x, myPlane._referenceNode[4].y, myPlane._referenceNode[4].z);
        topLeft->setValue(myPlane._referenceNode[6].x, myPlane._referenceNode[6].y, myPlane._referenceNode[6].z);

        myPlane.controllPlaneCreating();
        myPlane.targetPlaneCreating();

        exportHmo(myPlane, _planeFile->getValue());
        //exportHmascii(myPlane);

        //load and transform the block file
        float blockMoveVec3[3] = { moveIsol->getValue(0), moveIsol->getValue(1), moveIsol->getValue(2) };
        float blockScaleVec3[3] = { scaleIsol->getValue(0), scaleIsol->getValue(1), scaleIsol->getValue(2) };
        const char *thirdFile = _in_ThirdFile->getValue();
        char tempFile[256];
        if (FileExists(thirdFile) && _startSim->getValue())
        {
            size_t len = strlen(thirdFile);
            if (len >= 10)
            {
                if (len > 255)
                    len = 255;
                memcpy(tempFile, thirdFile, len-10);
                tempFile[len] = '\0';
            }
            else
            {
                len = 0;
                strcpy(tempFile, "");
            }
            if (len < 255)
                strncpy(tempFile+len, "tempBlock.hmo", 256-len);

            transformBlock(thirdFile, tempFile, blockMoveVec3, blockScaleVec3);
        }
        else
        {
            sendError("files %s and / or %s not existing, we do not execute simulation", firstFile, secondFile);
            return FAIL;
        }

        filesBinding(_in_FirstFile->getValue(), _in_SecondFile->getValue(), (const char *)tempFile, _targetFile->getValue());
        char command[1024];
        sprintf(command, "%s %s", _FamuExePath->getValue(), _FamuArgs->getValue());
        if (system(command) == -1)
            sendError("execution of %s failed", command);
        moveDist->setValue(0.0, 0.0, 0.0);
        scaleFactor->setValue(1.0);
        XYDegree->setValue(0);
        YZDegree->setValue(0);
        ZXDegree->setValue(0);
    }
    else
    {
        if (_startSim->getValue())
        {
            sendWarning("files %s and / or %s not existing, we do not execute simulation", firstFile, secondFile);
        }
    }

    try
    {
        // load mesh
        MeshDataTrans *meshDataTrans = NULL;
        getMeshDataTrans(&meshDataTrans);

        // get results and send them to output ports
        FactoryResultsFileParser *facParser = FactoryResultsFileParser::getInstance();
        ResultsFileParser *resFileParser = facParser->create(_resultsFileName->getValue(), meshDataTrans, this);

        if (_startSim->getValue())
        {
            std::string filename = _resultsFileName->getValue();
            // filename = parserTools::replace(filename, ".", "_.");
            resFileParser->parseResultsFile(filename.c_str(), _p_skip->getValue(), _p_numt->getValue(), meshDataTrans, std::string(_dataPort[0]->getObjName()));
        }
        else
        {
            resFileParser->parseResultsFile(_resultsFileName->getValue(), _p_skip->getValue(), _p_numt->getValue(), meshDataTrans, std::string(_dataPort[0]->getObjName()));
        }

        // ------------------
        ResultsFileData *resFileData = NULL;
        float symmAngle = _periodicAngle->getValue();
        int noOfSymmTimeSteps = _periodicTimeSteps->getValue();
        const bool isPeriodicRot = symmAngle != 0 && noOfSymmTimeSteps > 0;
        if (isPeriodicRot)
        {
            resFileData = new PeriodicRotResults(symmAngle, noOfSymmTimeSteps, resFileParser, this);
        }
        else
        {
            resFileData = resFileParser;
        }
        sendResultsToPorts(resFileData, meshDataTrans);
        // ------------------

        // create grids of all time steps and send them to output ports
        BuilderGrid builderGrid(this);
        coDoSet *gridSet = builderGrid.construct(_mesh->getObjName(), _scaleDisplacements->getValue(),
                                                 _periodicAngle->getValue(), (int)_periodicTimeSteps->getValue(),
                                                 meshDataTrans, resFileParser);
        _mesh->setCurrentObject(gridSet);

        coFeedback feedback("Famu");
        //feedback.addPara(moveDist);

        feedback.addPara(bottomLeft); //first Point   Param.0
        feedback.addPara(bottomRight); // second Point Param.1
        feedback.addPara(topRight); //third Point  Param.2
        feedback.addPara(topLeft); //fourth Point Param.3
        feedback.addPara(scaleFactor); //Param.4
        feedback.addPara(XYDegree); //Param.5
        feedback.addPara(YZDegree); //Param.6
        feedback.addPara(ZXDegree); //Param.7
        feedback.addPara(reset); //Param.8
        feedback.addPara(moveIsol); //Param.9
        feedback.addPara(scaleIsol); //Param.10

        feedback.addString("important information");
        feedback.apply(gridSet);

        delete meshDataTrans;
        delete resFileParser;
        //return SUCCESS;
        return CONTINUE_PIPELINE;
    }
    catch (ErrorInfo error)
    {
        error.outputError();
        return FAIL;
    }
    catch (int error)
    {
        sendError("sorry, error %d occured.", error);
        return FAIL;
    }
}

/** 
 * returns a container with results file data
 */
void ReadFamu::sendResultsToPorts(ResultsFileData *resultsFileData,
                                  MeshDataTrans * /*meshDataTrans*/)
{
    int noOfTimeSteps = resultsFileData->getNoOfTimeSteps();
    int noOfDataTypes = resultsFileData->getNoOfDataTypes(0);
    noOfDataTypes = noOfDataTypes > NUMRES ? NUMRES : noOfDataTypes;
    int n;
    for (n = 0; n < noOfDataTypes; n++)
    {
        // collect all timesteps of a DataType in a dataSet
        // and send the dataSet to  _dataPort[dataTypeNo]
        std::string dtName = resultsFileData->getDataTypeName(0, n);
        const char *cName = dtName.c_str();
        sendInfo("DataValue %d :%s", n, cName);

        coDistributedObject **dataObjs = new coDistributedObject *[noOfTimeSteps + 1];
        int i;
        for (i = 0; i < noOfTimeSteps; i++)
        {
            dataObjs[i] = resultsFileData->getDataObject(n, i);
        }
        dataObjs[noOfTimeSteps] = NULL;

        // send the DataTypes to _dataPort[]
        coDoSet *dataSet = new coDoSet(_dataPort[n]->getObjName(), dataObjs);
        if (noOfTimeSteps > 1)
        {
            dataSet->addAttribute("TIMESTEP", "1 10");
            if (_periodicAngle->getValue() > 0 && _periodicTimeSteps->getValue() == 0)
            {
                int num = (int)(360.0 / _periodicAngle->getValue());
                char attrValue[200];
                sprintf(attrValue, "%d %f", num - 1, _periodicAngle->getValue());
                dataSet->addAttribute("MULTIROT", attrValue);
            }
        }
        delete[] dataObjs;

        // Assign sets to output ports:
        _dataPort[n]->setCurrentObject(dataSet);
    }
}

/** 
 * returns a container with mesh file data
 */
void ReadFamu::getMeshDataTrans(MeshDataTrans **meshDataTrans)
{
    // get collector selections
    coRestraint sel;
    const char *selection;
    selection = _p_selection->getValue();
    sel.add(selection);

    MeshFileTransParser *meshFileParser;

    if (_startSim->getValue())
    {
        std::string filename = _meshFileName->getValue();
        //filename = parserTools::replace(filename, ".", "_.");

        FactoryMeshFileTransParser *facMesh = FactoryMeshFileTransParser::getInstance();
        meshFileParser = facMesh->create(filename.c_str(), this);
        meshFileParser->parseMeshFile(filename.c_str(), sel,
                                      _subdivideParam->getValue(),
                                      _p_skip->getValue(), _p_numt->getValue());
    }
    else
    {
        FactoryMeshFileTransParser *facMesh = FactoryMeshFileTransParser::getInstance();
        meshFileParser = facMesh->create(_meshFileName->getValue(), this);
        meshFileParser->parseMeshFile(_meshFileName->getValue(), sel,
                                      _subdivideParam->getValue(),
                                      _p_skip->getValue(), _p_numt->getValue());
    }

    *meshDataTrans = meshFileParser->getMeshDataTrans();
}
