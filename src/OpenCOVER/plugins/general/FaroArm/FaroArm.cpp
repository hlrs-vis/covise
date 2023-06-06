/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//**********************************************************
// Plugin FaroArm
// obtain coordinates from the FaroArm
// Date: 2008-05-20
//**********************************************************

#include "FaroArm.h"

FaroArm *plugin = NULL;

FaroArm::FaroArm()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "FaroArm::FaroArm\n");

    currentSample = 100;
    maxSamples = 0;
    refFrameMode = false;
    calibrationMode = false;
    samplePointMode = false;
    haveRefFrame = false;
    haveSamplePoints = false;
    haveObjectPoints = false;
    haveCalibration = false;
    havePatternPos = false;

    //--- TEST SIMULATION start ---
    simulateData();
    //--- TEST SIMULATION end ---
}

// this is called if the plugin is removed at runtime
// which currently never happens
FaroArm::~FaroArm()
{
    fprintf(stderr, "FaroArm::~FaroArm\n");

    m_FaroLaserScanner.StopDevice();

    delete myFrame;
    delete myTab;
    delete stopButton;
    delete calibButton;
    delete samplePointButton;
    delete setPointButton;
    delete refFrameButton;
    delete directStartButton;
    delete coordLabel;
    delete refPointSampled_Label;
    delete calibrationLabel;
    delete imageFrameLabel;
    delete imageFrameNo;
    delete debugLabel;

    for (int i = 0; i < 3; i++)
    {
        delete transLabel[i];
        delete rotLabel[i];
        delete transValue[i];
        delete rotValue[i];
        delete refPointSampled_pointLabel[i];
        delete transPointSampled_Label[i];
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            delete transPointSampled_Value[i][j];
            delete object_transField[i][j];
        }
    }
}

bool FaroArm::init()
{
    myTab = new coTUITab("FaroArm", coVRTui::instance()->mainFolder->getID());
    myTab->setPos(0, 0);
    myFrame = new coTUIFrame("FaroArm", myTab->getID());
    myFrame->setPos(0, 1);
    myFrame->setSize(18, 18);

    calibButton = new coTUIButton("Calibration", myTab->getID());
    calibButton->setPos(12, 0);
    calibButton->setEventListener(this);
    samplePointButton = new coTUIButton("Sample points", myTab->getID());
    samplePointButton->setPos(12, 1);
    samplePointButton->setEventListener(this);
    stopButton = new coTUIButton("Stop and Reset", myTab->getID());
    stopButton->setPos(12, 2);
    stopButton->setEventListener(this);
    refFrameButton = new coTUIButton("Reference frame", myTab->getID());
    refFrameButton->setPos(14, 1);
    refFrameButton->setEventListener(this);
    setPointButton = new coTUIButton("Set points", myTab->getID());
    setPointButton->setPos(12, 10);
    setPointButton->setEventListener(this);
    directStartButton = new coTUIButton("Direct START", myTab->getID());
    directStartButton->setPos(12, 18);
    directStartButton->setEventListener(this);

    debugLabel = new coTUILabel("Debug: ", myTab->getID());
    debugLabel->setPos(14, 0);

    coordLabel = new coTUILabel("Coordinates: ", myTab->getID());
    coordLabel->setPos(0, 0);
    transLabel[0] = new coTUILabel("x:", myTab->getID());
    transLabel[1] = new coTUILabel("y:", myTab->getID());
    transLabel[2] = new coTUILabel("z:", myTab->getID());
    rotLabel[0] = new coTUILabel("h:", myTab->getID());
    rotLabel[1] = new coTUILabel("p:", myTab->getID());
    rotLabel[2] = new coTUILabel("r:", myTab->getID());

    for (int i = 0; i < 3; i++)
    {
        transValue[i] = new coTUILabel("-", myTab->getID());
        rotValue[i] = new coTUILabel("-", myTab->getID());
    }

    for (int i = 0; i < 3; i++)
    {
        transLabel[i]->setPos(1, i);
        rotLabel[i]->setPos(3, i);
    }

    for (int i = 0; i < 3; i++)
    {
        transValue[i]->setPos(2, i);
        rotValue[i]->setPos(4, i);
    }

    refPointSampled_Label = new coTUILabel("Samples: ", myTab->getID());
    refPointSampled_Label->setPos(0, 6);
    refPointSampled_pointLabel[0] = new coTUILabel("Point 0: ", myTab->getID());
    refPointSampled_pointLabel[1] = new coTUILabel("Point 1: ", myTab->getID());
    refPointSampled_pointLabel[2] = new coTUILabel("Point 2: ", myTab->getID());
    transPointSampled_Label[0] = new coTUILabel("x:", myTab->getID());
    transPointSampled_Label[1] = new coTUILabel("y:", myTab->getID());
    transPointSampled_Label[2] = new coTUILabel("z:", myTab->getID());

    for (int j = 0; j < 3; j++)
    {
        refPointSampled_pointLabel[j]->setPos(0, j + 7);
        transPointSampled_Label[j]->setPos(j + 1, 6);
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            transPointSampled_Value[i][j] = new coTUILabel("-", myTab->getID());

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            transPointSampled_Value[i][j]->setPos(j + 1, 7 + i);

    objectCoords_transLabel[0] = new coTUILabel("X:", myTab->getID());
    objectCoords_transLabel[1] = new coTUILabel("Y:", myTab->getID());
    objectCoords_transLabel[2] = new coTUILabel("Z:", myTab->getID());

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            object_transField[i][j] = new coTUIEditFloatField("ee", myTab->getID());

    for (int i = 0; i < 3; i++)
        objectCoords_transLabel[i]->setPos(i + 12, 6);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            object_transField[i][j]->setPos(i + 12, 7 + j);
            object_transField[i][j]->setEventListener(this);
        }
    }

    calibrationLabel = new coTUILabel("Calibration:", myTab->getID());
    calibrationLabel->setPos(0, 12);
    imageFrameLabel = new coTUILabel("Frames:", myTab->getID());
    imageFrameLabel->setPos(0, 13);
    imageFrameNo = new coTUILabel("0", myTab->getID());
    imageFrameNo->setPos(1, 13);

    updateTUI();
    m_FaroLaserScanner.SetDataCaputreCallBack(this);

    if (m_FaroLaserScanner.StartDevice())
    {
        startThread();
        cerr << " FARO device started ...  " << endl;
    }
    else
        cerr << " FARO device could not be started.  " << endl;

    return true;
}

void FaroArm::tabletEvent(coTUIElement *tuiItem)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (tuiItem == object_transField[i][j])
                objectPoint[i][j] = object_transField[i][j]->getValue();
}

void FaroArm::tabletPressEvent(coTUIElement *tuiItem)
{
    if (tuiItem == stopButton)
    {
        m_FaroLaserScanner.StopDevice();
        refFrameMode = false;
        samplePointMode = false;
        calibrationMode = false;
        haveRefFrame = false;
        haveSamplePoints = false;
        haveObjectPoints = false;
        havePatternPos = false;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                transPointSampled_Value[i][j]->setLabel("-");

        cerr << " FARO device stopped ...  " << endl;
    }

    if (tuiItem == calibButton)
    {
        if (!refFrameMode && !samplePointMode)
        {
            calibrationMode = true;
            myMarker = new ARToolKitMarker("calibMarker");

            if (!havePatternPos)
            {
                currentSample = 0;
                maxSamples = 4;
                cerr << " " << endl;
                cerr << " Now pick " << maxSamples << " pattern corner points.  " << endl;
            }
            else if (havePatternPos)
            {
                currentSample = 0;
                maxSamples = 10;
                cerr << " " << endl;
                cerr << " Now pick " << maxSamples << " image frames.  " << endl;
            }
        }
        else if (refFrameMode || samplePointMode)
            cerr << " Currently in another mode!  " << endl;
    }

    if (tuiItem == samplePointButton)
    {
        if (!calibrationMode && !refFrameMode)
        {
            samplePointMode = true;
            currentSample = 0;
            maxSamples = 3;
            cerr << " " << endl;
            cerr << " Now pick " << maxSamples << " reference points.  " << endl;
        }
        else if (calibrationMode || refFrameMode)
            cerr << " Currently in another mode!  " << endl;
    }

    if (tuiItem == setPointButton)
    {
        if (haveSamplePoints && !calibrationMode && !refFrameMode && !samplePointMode)
            registerObject();
        else if (!haveSamplePoints)
            cerr << " No sample points available - pick them first.  " << endl;
        else if (calibrationMode || refFrameMode || samplePointMode)
            cerr << " Currently in another mode!  " << endl;
    }

    if (tuiItem == refFrameButton)
    {
        if (!calibrationMode && !samplePointMode)
        {
            refFrameMode = true;
            currentSample = 0;
            maxSamples = 3;
            cerr << " " << endl;
            cerr << " Now pick " << maxSamples << " reference points.  " << endl;
        }
        else if (calibrationMode || samplePointMode)
            cerr << " Currently in another mode!  " << endl;
    }

    if (tuiItem == directStartButton)
    {
        if (!refFrameMode && !samplePointMode && !calibrationMode)
        {
            //registerObject();
            haveCalibration = true;
        }
        else
            cerr << " Currently in another mode!  " << endl;
    }
}

// Called when the position of FARO probe has changed
void FaroArm::OnPositionChanged(const CScanData &ScanData)
{
    coCoord probeCoords;
    probeCoords.xyz.set(ScanData.m_ArmXYZ[0], ScanData.m_ArmXYZ[1], ScanData.m_ArmXYZ[2]);
    probeCoords.hpr.set(ScanData.m_ArmABC[0], ScanData.m_ArmABC[1], ScanData.m_ArmABC[2]);
    probeCoords.makeMat(probeMat);
}

// Coordinates of a point picked when the green button on the FARO device is pressed
void FaroArm::OnPositionSampled(const CScanData &ScanData)
{
    if (samplePointMode && !refFrameMode)
    {
        if (currentSample < maxSamples)
        {
            cerr << " " << endl;
            cerr << " Sample point " << currentSample + 1 << ".  " << endl;

            transPointSampled[currentSample][0] = ScanData.m_ArmXYZ[0];
            transPointSampled[currentSample][1] = ScanData.m_ArmXYZ[1];
            transPointSampled[currentSample][2] = ScanData.m_ArmXYZ[2];
            currentSample++;
        }
        if (currentSample == maxSamples)
        {
            haveSamplePoints = true;
            samplePointMode = false;
            displaySamplePoints();
        }
    }
    else if (calibrationMode && !havePatternPos)
    {
        if (currentSample < maxSamples)
        {
            cerr << " " << endl;
            cerr << " Sample point " << currentSample + 1 << " on pattern-marker.  " << endl;

            transPointSampled[currentSample][0] = ScanData.m_ArmXYZ[0];
            transPointSampled[currentSample][1] = ScanData.m_ArmXYZ[1];
            transPointSampled[currentSample][2] = ScanData.m_ArmXYZ[2];

            // if reference frame has been made before, transform the picked sample points into the reference frame
            if (haveRefFrame)
                transPointSampled[currentSample] = transPointSampled[currentSample] * transformMat;

            currentSample++;
        }
        if (currentSample == maxSamples)
        {
            calcPatternPosition();
            havePatternPos = true;
            currentSample = 0;
            maxSamples = 10;
            cerr << " " << endl;
            cerr << " Now pick " << maxSamples << " images of the pattern-marker.  " << endl;
        }
    }
    else if (calibrationMode && havePatternPos)
    {
        bool newOrientation = false;
        bool newTranslation = false;
        osg::Matrix sampledMat, oldSampledMat, cameraMat, cameraTransMat, cameraProbeMat;
        // transformation of pattern-marker into camera coordinates
        const osg::Matrix patternRotMat(0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1);

        if (currentSample < maxSamples)
        {
            if (myMarker->isVisible())
            {
                cerr << " " << endl;
                cerr << " Sample " << currentSample + 1 << ".  " << endl;

                transPointSampled[currentSample][0] = ScanData.m_ArmXYZ[0];
                transPointSampled[currentSample][1] = ScanData.m_ArmXYZ[1];
                transPointSampled[currentSample][2] = ScanData.m_ArmXYZ[2];
                rotPointSampled[currentSample][0] = ScanData.m_ArmABC[0];
                rotPointSampled[currentSample][1] = ScanData.m_ArmABC[1];
                rotPointSampled[currentSample][2] = ScanData.m_ArmABC[2];

                //--- DEBUG start ---
                cerr << " ---------------------------" << endl;
                cerr << " T and R: " << endl;
                cerr << "  Tx:  " << transPointSampled[currentSample][0];
                cerr << "  Ty:  " << transPointSampled[currentSample][1];
                cerr << "  Tz:  " << transPointSampled[currentSample][2] << endl;
                cerr << "  Rh:  " << rotPointSampled[currentSample][0];
                cerr << "  Rp:  " << rotPointSampled[currentSample][1];
                cerr << "  Rr:  " << rotPointSampled[currentSample][2] << endl;
                //--- DEBUG end ---

                //--- TEST SIMULATION start ---
                coCoord patternDistCoords;
                patternDistCoords.xyz.set(transTestVec[currentSample][0], transTestVec[currentSample][1], transTestVec[currentSample][2]);
                patternDistCoords.hpr.set(rotTestVec[currentSample][0], rotTestVec[currentSample][1], rotTestVec[currentSample][2]);
                patternDistCoords.makeMat(patternDistMat);
                cameraMat = patternDistMat;
                //--- TEST SIMULATION end ---

                //cameraMat = myMarker->getCameraTrans();   // use for values from ARToolKit
                cameraTransMat = (cameraMat * patternRotMat) * patternMat;

                // if reference frame has been made before, transform the picked sample points into the reference frame
                if (haveRefFrame)
                    transPointSampled[currentSample] = transPointSampled[currentSample] * transformMat;

                coCoord currentSampleCoords;
                currentSampleCoords.xyz.set(transPointSampled[currentSample][0], transPointSampled[currentSample][1], transPointSampled[currentSample][2]);
                currentSampleCoords.hpr.set(rotPointSampled[currentSample][0], rotPointSampled[currentSample][1], rotPointSampled[currentSample][2]);
                currentSampleCoords.makeMat(sampledMat);

                /*
            //--- DEBUG start ---
            cerr << " ---------------------------" << endl;
            cerr << " PROBE parameters (matrix in faro frame): " << endl;
            cerr << "  x:  " << sampledMat.getTrans().x();
            cerr << "  y:  " << sampledMat.getTrans().y();
            cerr << "  z:  " << sampledMat.getTrans().z() << endl;
            cerr << "  h:  " << sampledMat.getRotate().x();
            cerr << "  p:  " << sampledMat.getRotate().y();
            cerr << "  r:  " << sampledMat.getRotate().z() << endl;
            //--- DEBUG end ---
            */

                // compute translation calibration offset
                for (int i = 0; i < 3; i++)
                    cameraProbeMat(3, i) = sampledMat(3, i) - cameraTransMat(3, i);

                // compute rotation calibration offset: V1
                osg::Matrix tmpMat;
                tmpMat.invert(sampledMat);
                osg::Matrix probeRotMat = tmpMat * patternMat;
                probeRotMat.invert(probeRotMat);
                cameraMat = cameraTransMat * probeRotMat;

                /*
            //--- DEBUG start ---
            cerr << " ---------------------------" << endl;
            cerr << " CAMERA parameters (matrix in faro frame): " << endl;
            //cerr << "  x:  " << cameraMat.getTrans().x();
            //cerr << "  y:  " << cameraMat.getTrans().y();
            //cerr << "  z:  " << cameraMat.getTrans().z() << endl;
            cerr << "  h:  " << cameraMat.getRotate().x();
            cerr << "  p:  " << cameraMat.getRotate().y();
            cerr << "  r:  " << cameraMat.getRotate().z() << endl;
            //--- DEBUG end ---
            */

                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        cameraProbeMat(i, j) = sampledMat(i, j) - probeRotMat(i, j);

                /*
            //--- DEBUG start ---
            cerr << " ---------------------------" << endl;
            cerr << " CAMERA-PROBE offset (matrix in pattern frame): " << endl;
            //cerr << "  x:  " << cameraProbeMat.getTrans().x();
            //cerr << "  y:  " << cameraProbeMat.getTrans().y();
            //cerr << "  z:  " << cameraProbeMat.getTrans().z() << endl;
            cerr << "  h:  " << cameraProbeMat.getRotate().x();
            cerr << "  p:  " << cameraProbeMat.getRotate().y();
            cerr << "  r:  " << cameraProbeMat.getRotate().z() << endl;
            //--- DEBUG end ---
            

            //--- DEBUG start ---
            coCoord cameraProbeMatCoords = cameraProbeMat;
            cerr << " ---------------------------" << endl;
            cerr << " CAMERA-PROBE offset (in coCoord): " << endl;
            //cerr << " x: " << cameraProbeMatCoords.xyz[0];
            //cerr << " y: " << cameraProbeMatCoords.xyz[1];
            //cerr << " z: " << cameraProbeMatCoords.xyz[2] << endl;
            cerr << " h: " << cameraProbeMatCoords.hpr[0];
            cerr << " p: " << cameraProbeMatCoords.hpr[1];
            cerr << " r: " << cameraProbeMatCoords.hpr[2] << endl;
            //--- DEBUG end ---
            */

                // check if orientation has changed
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        if (sampledMat(i, j) != oldSampledMat(i, j))
                        {
                            newOrientation = true;
                            break;
                        }
                    }
                }
                // check if translation has changed
                for (int j = 0; j < 3; j++)
                {
                    if (fabs(sampledMat(3, j) - oldSampledMat(3, j)) > 0.0)
                    {
                        newTranslation = true;
                        break;
                    }
                }
                if (newOrientation || newTranslation)
                {
                    if (currentSample == 0)
                        summMat = cameraProbeMat;
                    else
                    {
                        for (int i = 0; i < 4; i++)
                            for (int j = 0; j < 3; j++)
                                summMat(i, j) += cameraProbeMat(i, j);
                    }
                    char buf[16];
                    sprintf(buf, " %d", currentSample + 1);
                    imageFrameNo->setLabel(buf);
                    oldSampledMat = sampledMat;
                    currentSample++;
                }
            }
            else if (!myMarker->isVisible())
                cerr << " *** Pattern-marker was not visible ***  " << endl;
        }

        if (currentSample == maxSamples)
        {
            calcOffset();
            haveCalibration = true;
            calibrationMode = false;
        }
    } // fi(calibrationMode && havePatternPos)
    else if (refFrameMode && !haveRefFrame)
    {
        if (currentSample < maxSamples)
        {
            cerr << " " << endl;
            cerr << " Sample point " << currentSample + 1 << " for reference frame.  " << endl;

            transPointSampled[currentSample][0] = ScanData.m_ArmXYZ[0];
            transPointSampled[currentSample][1] = ScanData.m_ArmXYZ[1];
            transPointSampled[currentSample][2] = ScanData.m_ArmXYZ[2];
            currentSample++;
        }

        if (currentSample == maxSamples)
        {
            makeReferenceFrame();
            haveSamplePoints = true;
            haveRefFrame = true;
            refFrameMode = false;

            // transform the picked sample points into the reference frame to display correct values
            for (int i = 0; i < currentSample; i++)
                transPointSampled[i] = transPointSampled[i] * transformMat;

            displaySamplePoints();
        }
    }
    else
    {
        cerr << " Usage: " << endl;
        cerr << " " << endl;
        cerr << " Do a camera-to-probe calibration: " << endl;
        cerr << " 1. Press the 'Calibration' button." << endl;
        cerr << " 2. Get position of the pattern by picking the 4 corners." << endl;
        cerr << " 3. Get 10 image frames by picking 10 points." << endl;
        cerr << " " << endl;
        cerr << " Transform into object coordinate system: " << endl;
        cerr << " 1. Pick 3 sample points." << endl;
        cerr << " 2. Enter coordinates of 3 points according to the previously chosen sample points." << endl;
        cerr << " 3. Press the 'Set values' button." << endl;
        cerr << " Make a reference frame: " << endl;
        cerr << " 1. Press the 'Reference Frame' button." << endl;
        cerr << " 2. Pick 3 sample points to set the reference frame." << endl;
        cerr << " " << endl;
    }
}

void FaroArm::makeReferenceFrame()
{
    osg::Vec3 refPoints[3];
    osg::Vec3 v, v1, v2, v3;
    // transformation matrix for the reference points given in the config-file
    osg::Matrix refMat;
    // matrix for the sampled points picked by the Faro device
    osg::Matrix refMatSampled;

    char configName[1000];
    // reference points from config-file
    for (int i = 0; i < 3; i++)
    {
        sprintf(configName, "COVER.Input.Faro.ReferencePoint%d", i);
        refPoints[i][0] = coCoviseConfig::getFloat("x", configName, 0);
        refPoints[i][1] = coCoviseConfig::getFloat("y", configName, 0);
        refPoints[i][2] = coCoviseConfig::getFloat("z", configName, 0);
    }

    // make a cartesian reference frame
    v1 = refPoints[0] - refPoints[1];
    v = refPoints[0] - refPoints[2];
    v2 = v ^ v1;
    v3 = v1 ^ v2;
    v1.normalize();
    v2.normalize();
    v3.normalize();

    for (int i = 0; i < 3; i++)
    {
        refMat(0, i) = v1[i];
        refMat(1, i) = v2[i];
        refMat(2, i) = v3[i];
        refMat(3, i) = refPoints[0][i];
    }

    // make a cartesian frame for the picked samle points
    v1 = transPointSampled[0] - transPointSampled[1];
    v = transPointSampled[0] - transPointSampled[2];
    v2 = v ^ v1;
    v3 = v1 ^ v2;
    v1.normalize();
    v2.normalize();
    v3.normalize();

    for (int i = 0; i < 3; i++)
    {
        refMatSampled(0, i) = v1[i];
        refMatSampled(1, i) = v2[i];
        refMatSampled(2, i) = v3[i];
        refMatSampled(3, i) = transPointSampled[0][i];
    }

    refMatSampled.invert(refMatSampled);
    transformMat = refMatSampled * refMat;

    cerr << " Reference frame made.  " << endl;
}

// Determination of the initial pattern position
void FaroArm::calcPatternPosition()
{
    // make a matrix of distances for unsorted sample points
    // matrix: 0 for distance = 0, 1 for distance to neighbor, 2 for distance across
    float distance, oldDistance;
    int indexTo = 3, centerNo = 0;
    osg::Matrix distMat, tmpMat;

    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            if (row == col)
                distMat(row, col) = 0;
            else
                distMat(row, col) = 1;
        }
    }
    for (int j = 0; j < 4; j++)
    {
        oldDistance = 0.0;
        if (j != indexTo && centerNo < 2)
        {
            for (int i = 0; i < 4; i++)
            {
                distance = (transPointSampled[i] - transPointSampled[j]).length();
                if (distance > oldDistance)
                {
                    oldDistance = distance;
                    indexTo = i;
                }
            }
            distMat(indexTo, j) = 2;
            distMat(j, indexTo) = 2;
        }
    }

    // center of pattern-marker
    osg::Vec3 central;
    central = transPointSampled[0] / maxSamples + transPointSampled[1] / maxSamples + transPointSampled[2] / maxSamples + transPointSampled[3] / maxSamples;

    // frame for pattern-marker with lower right corner as first picked sample
    osg::Vec3 v1, v2, v3, v, vX, vY, vZ;
    int counter = 0;

    for (int n = 0; n < 4; n++)
    {
        if (distMat(0, n) == 1 && counter == 0)
        {
            v1 = transPointSampled[n] - transPointSampled[0];
            counter++;
        }
        else if (distMat(0, n) == 1 && counter == 1)
            v2 = transPointSampled[n] - transPointSampled[0];
    }

    v3 = v1 ^ v2;
    v1.normalize();
    v2.normalize();
    v3.normalize();
    v = central - transPointSampled[0];

    for (int i = 0; i < 3; i++)
    {
        patternMat(0, i) = v1[i];
        patternMat(1, i) = v2[i];
        patternMat(2, i) = v3[i];
        patternMat(3, i) = transPointSampled[0][i];
    }

    // tmpMat for translation from corner to center vector
    tmpMat.makeTranslate(v);
    patternMat = patternMat * tmpMat;

    /*
   //--- DEBUG start ---
   cerr << " FARO basis coordinate system: " << endl;
   cerr << "  (0,0):  " << patternMat(0,0);
   cerr << "  (0,1):  " << patternMat(0,1);
   cerr << "  (0,2):  " << patternMat(0,2);
   cerr << "  (0,3):  " << patternMat(0,3) << endl;
   cerr << "  (1,0):  " << patternMat(1,0);
   cerr << "  (1,1):  " << patternMat(1,1);
   cerr << "  (1,2):  " << patternMat(1,2);
   cerr << "  (1,3):  " << patternMat(1,3) << endl;
   cerr << "  (2,0):  " << patternMat(2,0);
   cerr << "  (2,1):  " << patternMat(2,1);
   cerr << "  (2,2):  " << patternMat(2,2);
   cerr << "  (2,3):  " << patternMat(2,3) << endl;
   cerr << "  (3,0):  " << patternMat(3,0);
   cerr << "  (3,1):  " << patternMat(3,1);
   cerr << "  (3,2):  " << patternMat(3,2);
   cerr << "  (3,3):  " << patternMat(3,3) << endl;
   //--- DEBUG end ---
   */

    cerr << " Position of pattern determined.  " << endl;
}

// Transformation from camera to FARO probe
void FaroArm::calcOffset()
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            summMat(i, j) /= currentSample;

    coCoord coordinates = summMat;
    /* Do this later when definitely correct
   osg::Vec3 trans(coordinates.xyz[0],coordinates.xyz[1],coordinates.xyz[2]);
	sg::Vec3 rot(coordinates.hpr[0],coordinates.hpr[1],coordinates.hpr[2]);
   coVRTrackingUtil::instance()->setDeviceOffset(coVRTrackingUtil::cameraDev,trans,rot);
   */

    //--- DEBUG start ---
    cerr << " ---------------------------" << endl;
    cerr << " FINAL calibration offset: " << endl;
    //cerr << " x: " << coordinates.xyz[0];
    //cerr << " y: " << coordinates.xyz[1];
    //cerr << " z: " << coordinates.xyz[2] << endl;
    cerr << " h: " << coordinates.hpr[0];
    cerr << " p: " << coordinates.hpr[1];
    cerr << " r: " << coordinates.hpr[2] << endl;
    //--- DEBUG end ---

    /*
   //--- DEBUG start ---
   cerr << " ---------------------------" << endl;
   cerr << " FINAL calibration offset: " << endl;
   //cerr << "  x:  " << summMat.getTrans().x();
   //cerr << "  y:  " << summMat.getTrans().y();
   //cerr << "  z:  " << summMat.getTrans().z() << endl;
   cerr << "  h:  " << summMat.getRotate().x();
   cerr << "  p:  " << summMat.getRotate().y();
   cerr << "  r:  " << summMat.getRotate().z() << endl;
   //--- DEBUG end ---
   */

    cerr << " Calibration done. Now export parameters into config.faro.com, CameraDevice.  " << endl;
}

// Transformation between points entered in tabletUI and object origin
void FaroArm::registerObject()
{
    /*
   // TO DO: solve linear equations using non-linear methods like levenberg-m. for numerical stabilty and convergence
   // transformObjectMat * transPointSampled = objectPoint;
   
   //coCoord coordinates(transformObjectMat);
   //coVRTrackingUtil::instance()->setDeviceOffset(coVRTrackingUtil::trackingSys,coordinates.xyz,coordinates.hpr);
   */

    // transformation given in config-file
    //transformObjectMat = VRTracker::instance()->getDeviceMat(coVRTrackingUtil::trackingSys);
    transformObjectMat = coVRTrackingUtil::instance()->computeDeviceOffsetMat(coVRTrackingUtil::trackingSys);

    /*
   //--- DEBUG start ---
   coCoord objectCoords = transformObjectMat;
   cerr << " ---------------------------" << endl;
   cerr << " Object coordinates: " << endl;
   cerr << " x: " << objectCoords.xyz[0];
   cerr << " y: " << objectCoords.xyz[1];
   cerr << " z: " << objectCoords.xyz[2] << endl;
   cerr << " h: " << objectCoords.hpr[0];
   cerr << " p: " << objectCoords.hpr[1];
   cerr << " r: " << objectCoords.hpr[2] << endl;
   //--- DEBUG end ---
   */

    haveObjectPoints = true;

    cerr << " Given points transformed.  " << endl;
}

void FaroArm::displaySamplePoints()
{
    char buf[16];

    for (int i = 0; i < 3; i++)
    {
        sprintf(buf, " %f", transPointSampled[i].x());
        transPointSampled_Value[i][0]->setLabel(buf);
        sprintf(buf, " %f", transPointSampled[i].y());
        transPointSampled_Value[i][1]->setLabel(buf);
        sprintf(buf, " %f", transPointSampled[i].z());
        transPointSampled_Value[i][2]->setLabel(buf);
    }
    updateTUI();
}

void FaroArm::updateTUI()
{
    char buf[32];
    osg::Matrix newCoordMat;

    if (haveRefFrame && !haveObjectPoints)
        newCoordMat = probeMat * transformMat;

    else if (haveRefFrame && haveObjectPoints)
        newCoordMat = probeMat * transformMat * transformObjectMat;

    else if (!haveRefFrame && !haveObjectPoints)
        newCoordMat = probeMat;

    else if (!haveRefFrame && haveObjectPoints)
        newCoordMat = probeMat * transformObjectMat;

    sprintf(buf, " %f", newCoordMat.getTrans().x());
    transValue[0]->setLabel(buf);
    sprintf(buf, " %f", newCoordMat.getTrans().y());
    transValue[1]->setLabel(buf);
    sprintf(buf, " %f", newCoordMat.getTrans().z());
    transValue[2]->setLabel(buf);
    sprintf(buf, " %f", newCoordMat.getRotate().x());
    rotValue[0]->setLabel(buf);
    sprintf(buf, " %f", newCoordMat.getRotate().y());
    rotValue[1]->setLabel(buf);
    sprintf(buf, " %f", newCoordMat.getRotate().z());
    rotValue[2]->setLabel(buf);
}

void FaroArm::run()
{
    CCloudData CloudData;
    m_FaroLaserScanner.CaptureData(CloudData);
}

int FaroArm::cancel()
{
    m_FaroLaserScanner.EndDataCapture();
    return 0;
}

void FaroArm::getMatrix(int station, osg::Matrix &mat)
{
    (void)station;
    mat = probeMat;
}

void FaroArm::simulateData()
{
    // fixed position of camera in ARToolKit coordinates
    transTestVec[0][0] = 0;
    transTestVec[0][1] = -600;
    transTestVec[0][2] = -82;
    transTestVec[1][0] = 0;
    transTestVec[1][1] = -500;
    transTestVec[1][2] = -82;
    transTestVec[2][0] = 0;
    transTestVec[2][1] = -400;
    transTestVec[2][2] = -82;
    transTestVec[3][0] = 0;
    transTestVec[3][1] = -300;
    transTestVec[3][2] = -82;

    transTestVec[4][0] = 50;
    transTestVec[4][1] = -600;
    transTestVec[4][2] = -82;
    transTestVec[5][0] = 0;
    transTestVec[5][1] = -600;
    transTestVec[5][2] = -82;
    transTestVec[6][0] = -50;
    transTestVec[6][1] = -600;
    transTestVec[6][2] = -82;
    transTestVec[7][0] = -100;
    transTestVec[7][1] = -600;
    transTestVec[7][2] = -82;

    transTestVec[8][0] = 0;
    transTestVec[8][1] = -600;
    transTestVec[8][2] = -82;
    transTestVec[9][0] = 0;
    transTestVec[9][1] = -600;
    transTestVec[9][2] = -32;
    transTestVec[10][0] = 0;
    transTestVec[10][1] = -600;
    transTestVec[10][2] = 18;
    transTestVec[11][0] = 0;
    transTestVec[11][1] = -600;
    transTestVec[11][2] = 68;

    // fixed rotation of camera in ARToolKit coordinates
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 3; j++)
            rotTestVec[i][j] = 0;
}

void FaroArm::preFrame()
{
    static double oldTime = 0;
    if (cover->frameTime() > oldTime + 1.0)
    {
        updateTUI();
        oldTime = cover->frameTime();
    }
}

COVERPLUGIN(FaroArm)

// EOF
