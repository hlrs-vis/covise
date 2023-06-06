/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//**********************************************************
// Plugin HMDCalibration
// calibration of the HMD device using approximated matrices
// Date: 2008-04-18
//**********************************************************

#include "HMDCalibration.h"
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/input/dev/legacy/coVRTrackingUtil.h>

HMDCalibration *plugin = NULL;

HMDCalibration::HMDCalibration()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "HMDCalibration::HMDCalibration\n");

    myMarker = new ARToolKitMarker("HMDCalib");
    cycle = 1000000;
    cameraMat.makeIdentity();
    systemMat.makeIdentity();
    oldTargetMat.makeIdentity();
}

// this is called if the plugin is removed at runtime
// which currently never happens
HMDCalibration::~HMDCalibration()
{
    fprintf(stderr, "HMDCalibration::~HMDCalibration\n");

    delete myFrame;
    delete myTab;
    delete startButton;
    delete noOfCyclesLabel;
    delete noOfCyclesInt;
    delete cameraLabel;
    delete patternLabel;
    delete myMarker;

    for (int i = 0; i < 3; i++)
    {
        delete transLabel[i];
        delete rotLabel[i];
    }

    for (int i = 0; i < 12; i++)
    {
        delete transValue[i];
        delete rotValue[i];
    }
}

bool HMDCalibration::init()
{
    char buf[32];

    myTab = new coTUITab("HMDCalibration", coVRTui::instance()->mainFolder->getID());
    myTab->setPos(0, 0);
    myFrame = new coTUIFrame("HMDCalibration", myTab->getID());
    myFrame->setPos(0, 1);
    myFrame->setSize(18, 18);

    startButton = new coTUIButton("Start", myTab->getID());
    startButton->setPos(10, 4);
    startButton->setEventListener(this);

    noOfCyclesLabel = new coTUILabel("No. of cycles:", myTab->getID());
    noOfCyclesLabel->setPos(10, 5);
    noOfCyclesInt = new coTUILabel("  0", myTab->getID());
    noOfCyclesInt->setPos(10, 6);
    sprintf(buf, "      %d", cycle);
    noOfCyclesInt->setLabel(buf);

    targetLabel = new coTUILabel("Target: ", myTab->getID());
    targetLabel->setPos(1, 1);
    cameraLabel = new coTUILabel("Camera: ", myTab->getID());
    cameraLabel->setPos(3, 1);
    patternLabel = new coTUILabel("Pattern: ", myTab->getID());
    patternLabel->setPos(5, 1);
    systemLabel = new coTUILabel("Average: ", myTab->getID());
    systemLabel->setPos(7, 1);

    transLabel[0] = new coTUILabel("x:", myTab->getID());
    transLabel[1] = new coTUILabel("y:", myTab->getID());
    transLabel[2] = new coTUILabel("z:", myTab->getID());
    rotLabel[0] = new coTUILabel("h:", myTab->getID());
    rotLabel[1] = new coTUILabel("p:", myTab->getID());
    rotLabel[2] = new coTUILabel("r:", myTab->getID());

    for (int i = 0; i < 12; i++)
    {
        transValue[i] = new coTUILabel("  0", myTab->getID());
        rotValue[i] = new coTUILabel("  0", myTab->getID());
    }

    // set labels x,y,z,h,p,r
    int row = 2;
    for (int v = 0; v < 3; v++)
    {
        transLabel[v]->setPos(0, row);
        rotLabel[v]->setPos(0, row + 3);
        row++;
    }

    // set values for target, camera, pattern, system
    int c = 0, r = 2;
    for (int v = 0; v < 12; v++)
    {
        transValue[v]->setPos(c + 1, r);
        rotValue[v]->setPos(c + 1, r + 3);

        if (r % 4 != 0)
            r++;
        else
        {
            c += 2;
            r = 2;
        }
    }
    updateTUI();

    return true;
}

void HMDCalibration::doMatApproximation()
{
    fprintf(stderr, "HMDCalibration::doMatApproximation\n");

    bool newOrientation = false;

    if (myMarker->isVisible())
        patternMat = myMarker->getMarkerTrans();
    else
        patternMat.makeIdentity();

    //vld: VRTracker usage. CameraMat. Add method and non-person tracking body?
    //targetMat = VRTracker::instance()->getCameraMat();
    targetMat.makeIdentity();
    cameraMat = targetMat * patternMat;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (targetMat(i, j) != oldTargetMat(i, j))
            {
                newOrientation = true;
                break;
            }
        }
    }

    coCoord coord = cameraMat;
    osg::Vec3 posDiff = oldTargetMat.getTrans() - targetMat.getTrans();

    if (newOrientation && posDiff.length() > 30 && fabs(cameraMat(3, 0)) < 200 && fabs(cameraMat(3, 1)) < 200 && fabs(cameraMat(3, 2)) < 200 && fabs(coord.hpr[0]) < 20 && fabs(coord.hpr[1]) < 20 && fabs(coord.hpr[2]) < 20)
    {
        oldTargetMat = targetMat;
        if (cycle == 0)
        {
            cameraSumm = cameraMat;
        }
        else
        {
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    cameraSumm(i, j) += cameraMat(i, j);
        }
        cycle++;
    }
}

void HMDCalibration::tabletPressEvent(coTUIElement *tuiItem)
{
    if (tuiItem == startButton)
    {
        osg::Vec3 trans(0, 0, 0);
        osg::Vec3 rot(0, 0, 0);

        coVRTrackingUtil::instance()->setDeviceOffset(coVRTrackingUtil::cameraDev, trans, rot);
        cycle = 0;
    }
}

void HMDCalibration::tabletEvent(coTUIElement * /*tuiItem*/)
{
}

void HMDCalibration::updateTUI()
{
    char buf[32];

    sprintf(buf, "      %d", cycle);
    noOfCyclesInt->setLabel(buf);

    sprintf(buf, "  %f", targetMat.getTrans().x());
    transValue[0]->setLabel(buf);
    sprintf(buf, "  %f", targetMat.getTrans().y());
    transValue[1]->setLabel(buf);
    sprintf(buf, "  %f", targetMat.getTrans().z());
    transValue[2]->setLabel(buf);
    sprintf(buf, "  %f", cameraMat.getTrans().x());
    transValue[3]->setLabel(buf);
    sprintf(buf, "  %f", cameraMat.getTrans().y());
    transValue[4]->setLabel(buf);
    sprintf(buf, "  %f", cameraMat.getTrans().z());
    transValue[5]->setLabel(buf);
    sprintf(buf, "  %f", patternMat.getTrans().x());
    transValue[6]->setLabel(buf);
    sprintf(buf, "  %f", patternMat.getTrans().y());
    transValue[7]->setLabel(buf);
    sprintf(buf, "  %f", patternMat.getTrans().z());
    transValue[8]->setLabel(buf);
    sprintf(buf, "  %f", cameraSumm.getTrans().x());
    transValue[9]->setLabel(buf);
    sprintf(buf, "  %f", cameraSumm.getTrans().y());
    transValue[10]->setLabel(buf);
    sprintf(buf, "  %f", cameraSumm.getTrans().z());
    transValue[11]->setLabel(buf);

    sprintf(buf, "  %f", targetMat.getRotate().x());
    rotValue[0]->setLabel(buf);
    sprintf(buf, "  %f", targetMat.getRotate().y());
    rotValue[1]->setLabel(buf);
    sprintf(buf, "  %f", targetMat.getRotate().z());
    rotValue[2]->setLabel(buf);
    sprintf(buf, "  %f", cameraMat.getRotate().x());
    rotValue[3]->setLabel(buf);
    sprintf(buf, "  %f", cameraMat.getRotate().y());
    rotValue[4]->setLabel(buf);
    sprintf(buf, "  %f", cameraMat.getRotate().z());
    rotValue[5]->setLabel(buf);
    sprintf(buf, "  %f", patternMat.getRotate().x());
    rotValue[6]->setLabel(buf);
    sprintf(buf, "  %f", patternMat.getRotate().y());
    rotValue[7]->setLabel(buf);
    sprintf(buf, "  %f", patternMat.getRotate().z());
    rotValue[8]->setLabel(buf);
    sprintf(buf, "  %f", cameraSumm.getRotate().x());
    rotValue[9]->setLabel(buf);
    sprintf(buf, "  %f", cameraSumm.getRotate().y());
    rotValue[10]->setLabel(buf);
    sprintf(buf, "  %f", cameraSumm.getRotate().z());
    rotValue[11]->setLabel(buf);
}

void HMDCalibration::testCalibrate()
{
    //char buf[32];
}

void HMDCalibration::preFrame()
{
    if (cycle < 100)
    {
        static double oldTime = 0;
        if (cover->frameTime() > oldTime + 1.0)
        {
            updateTUI();
            oldTime = cover->frameTime();
        }
        doMatApproximation();
    }
    if (cycle == 100)
    {
        for (int i = 0; i < 3; i++)
        {
            osg::Vec3 v;
            v.set(cameraSumm(i, 0), cameraSumm(i, 1), cameraSumm(i, 2));
            v.normalize();
            cameraSumm(i, 0) = v[0];
            cameraSumm(i, 1) = v[1];
            cameraSumm(i, 2) = v[2];
        }
        cameraSumm(3, 0) /= 100;
        cameraSumm(3, 1) /= 100;
        cameraSumm(3, 2) /= 100;
        cameraSumm.invert(cameraSumm);
        coCoord coord = cameraSumm;

        cerr << " x: " << coord.xyz[0];
        cerr << " y: " << coord.xyz[1];
        cerr << " z: " << coord.xyz[2] << endl;
        cerr << " h: " << coord.hpr[0];
        cerr << " p: " << coord.hpr[1];
        cerr << " r: " << coord.hpr[2] << endl;

        osg::Vec3 trans(coord.xyz[0], coord.xyz[1], coord.xyz[2]);
        osg::Vec3 rot(coord.hpr[0], coord.hpr[1], coord.hpr[2]);

        coVRTrackingUtil::instance()->setDeviceOffset(coVRTrackingUtil::cameraDev, trans, rot);

        updateTUI();
        cycle++;
    }
}

COVERPLUGIN(HMDCalibration)

// EOF
