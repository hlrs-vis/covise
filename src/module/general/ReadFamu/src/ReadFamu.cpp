/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadFamu module                                        ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  6.2006                                                       ++
// ++**********************************************************************/

#include "ReadFamu.h"
#include <util/coRestraint.h>
#include <iostream>

ReadFamu::ReadFamu(int argc, char **argv)
    : coSimpleModule(argc, argv, "Read ITE FAMU")
    , _mesh(NULL)
    , _meshFileName(NULL)
    , _resultsFileName(NULL)
    , _subdivideParam(NULL)
    ,

    _scaleDisplacements(NULL)
    , _p_numt(NULL)
    , _p_skip(NULL)
    , _p_selection(NULL)
{

    // module parameters
    _meshFileName = addFileBrowserParam("MeshFileName", "dummy");
    _meshFileName->setValue("C:/temp/readfamu/results/test1.cvm", "*.hmascii;*.cvm");

    _resultsFileName = addFileBrowserParam("ResultFileName", "dummy");
    _resultsFileName->setValue("C:/temp/readfamu/results/test1.cvr", "*.hm*;*.fma;*.fmb;*.cvr");

    _startSim = addBooleanParam("StartSimulation", "Start simulation?");
    _startSim->setValue(true);

    _planeFile = addFileBrowserParam("PlaneFileName", "dummy");
    _planeFile->setValue("C:/temp/readfamu/results/plane.hmo", "*.hmo");

    _in_FirstFile = addFileBrowserParam("FirstFile", "Container File");
    _in_FirstFile->setValue("C:/temp/readfamu/results/mesh_ohne_elektrode1.hmo", "*.hmo");

    _in_SecondFile = addFileBrowserParam("SecondFile", "Eletrode File");
    _in_SecondFile->setValue("C:/temp/readfamu/results/plane.hmo", "*.hmo");

    _in_ThirdFile = addFileBrowserParam("ThirdFile", "Isolator File");
    _in_ThirdFile->setValue("C:/temp/readfamu/results/block1.hmo", "*.hmo");

    _targetFile = addFileBrowserParam("TargetFile", "dummy");
    _targetFile->setValue("C:/temp/readfamu/results/test1.hmo", "*.hmo");

    _FamuExePath = addFileBrowserParam("FamuExePath", "path to Famu Executable");
    _FamuExePath->setValue("C:/temp/readfamu/results/famud", "*.exe");

    _FamuArgs = addStringParam("FamuArguments", "arguments for Famu exe");
    _FamuArgs->setValue("C:/temp/readfamu/results/block1");

    _subdivideParam = addBooleanParam("SubdivideElements", "Subdivide Tetra10 and Hex20 elements");
    _subdivideParam->setValue(true);

    _scaleDisplacements = addBooleanParam("DisplacementTimes1000", "Multiplies displacements with 1000");
    _scaleDisplacements->setValue(true);

    _periodicAngle = addFloatParam("SymmetryRotAngle", "Symmetry angle for periodic rotations (deg.)");
    _periodicAngle->setValue(45);
    _periodicTimeSteps = addInt32Param("OriginalSymmSteps", "number of original time steps computed when periodic rot. is used.");
    _periodicTimeSteps->setValue(0);

    _p_numt = addInt32Param("NoOfTimestepsToRead", "Number of time steps to read");
    _p_numt->setValue(1);
    _p_skip = addInt32Param("NoOfTimestepsToSkip", "Number of time steps to skip");
    _p_skip->setValue(0);

    _p_selection = addStringParam("CollectorsToLoad", "Collectors to load");
    _p_selection->setValue("0-999");

    //set the original parameter for the plane

    origBottomLeft[0] = -6;
    origBottomLeft[1] = -2.25;
    origBottomLeft[2] = 5.25;

    origBottomRight[0] = -6;
    origBottomRight[1] = 2.25;
    origBottomRight[2] = 5.25;

    origTopRight[0] = -6;
    origTopRight[1] = 2.25;
    origTopRight[2] = 10.75;

    origTopLeft[0] = -6;
    origTopLeft[1] = -2.25;
    origTopLeft[2] = 10.75;

    //Add the Coordinates parameter for the Plane

    bottomLeft = addFloatVectorParam("BottomLeftPoint", "Coordinates of the bottomleft Point of the Plane (No. 1)");
    bottomLeft->setValue(3, origBottomLeft);

    bottomRight = addFloatVectorParam("BottomRightPoint", "Coordinates of the bottomright Point of the Plane (No. 2)");
    bottomRight->setValue(3, origBottomRight);

    topRight = addFloatVectorParam("TopRightPoint", "Coordinates of the topright Point of the Plane (No. 3)");
    topRight->setValue(3, origTopRight);

    topLeft = addFloatVectorParam("TopLeftPoint", "Coordinates of the topleft Point of the Plane (No. 4)");
    topLeft->setValue(3, origTopLeft);

    moveDist = addFloatVectorParam("MoveDistances", "Moving Distances of the four points");
    moveDist->setValue(0.0, 0.0, 0.0);

    scaleFactor = addFloatSliderParam("ScaleFactor", "Scale the size of the electrode");
    scaleFactor->setValue(0.1f, 5.0, 1.0);

    XYDegree = addFloatSliderParam("RotateDegXY", "Rotate eletrode on XY");
    XYDegree->setValue(0.0, 180.0f, 0.0);
    YZDegree = addFloatSliderParam("RotateDegYZ", "Rotate eletrode on YZ");
    YZDegree->setValue(0.0, 180.0f, 0.0);
    ZXDegree = addFloatSliderParam("RotateDegXZ", "Rotate eletrode on XZ");
    ZXDegree->setValue(0.0, 180.0f, 0.0);

    reset = addBooleanParam("ResetElectrode", "Choose this and click on \"Execute\"");
    reset->setValue(false);

    moveIsol = addFloatVectorParam("MoveIsolator", "Move Isolator relative to original poisition");
    moveIsol->setValue(0.0, 0.0, 0.0);
    scaleIsol = addFloatVectorParam("ScaleIsolator", "Scale Isolator relative to original size");
    scaleIsol->setValue(1.0, 1.0, 1.0);
    // Output ports
    _mesh = addOutputPort("mesh", "UnstructuredGrid", "Unstructured Grid");
    _mesh->setInfo("Unstructured Grid");
    char buf[1000];
    int i;
    for (i = 0; i < NUMRES; i++)
    {
        sprintf(buf, "data%d", i);
        _dataPort[i] = addOutputPort(buf, "Float|Vec3", buf);
        _dataPort[i]->setInfo(buf);
    }
}

/** 
 *   DESTRUCTOR IS NEVER CALLED ??? 
 */
ReadFamu::~ReadFamu()
{
}

/**
 * What's left to do for the Main program: create the module and start it
 */
int main(int argc, char *argv[])
{
#ifdef YAC
    coDispatcher *dispatcher = coDispatcher::Instance();
    ReadFamu *application = new ReadFamu(argc, argv);
    dispatcher->add(application);
    while (dispatcher->dispatch(1000))
        ;
    coDispatcher::deleteDispatcher();
#else
    // create the module according to name in Compatibility modes
    ReadFamu *application = NULL;

    application = new ReadFamu(argc, argv);

    // this call leaves with exit()
    if (application)
    {
        application->start(argc, argv);
    }
#endif

    return 0;
}

void ReadFamu::displayString(const char *s)
{
    sendInfo("%s", s);
}

void ReadFamu::displayError(const char *c)
{
    sendError("%s", c);
}

void ReadFamu::displayError(std::string s)
{
    sendError("%s", s.c_str());
}

void ReadFamu::displayError(std::string s, std::string strQuotes)
{
    s = s + "\"";
    s = s + strQuotes;
    s = s + "\"";
    displayError(s);
}
