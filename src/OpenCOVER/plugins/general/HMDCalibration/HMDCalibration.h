/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//**********************************************************
// Plugin HMDCalibration
// calibration of the HMD device using approximated matrices
// Date: 2008-03-13
//**********************************************************

#ifndef _HMDCALIBRATION_H
#define _HMDCALIBRATION_H

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/coAction.h>
#include <osg/Matrix>
#include <cover/ARToolKit.h>

class HMDCalibration : public coVRPlugin, public coTUIListener
{
public:
    HMDCalibration();
    ~HMDCalibration();

    bool init();

    virtual void tabletEvent(coTUIElement *tuiItem);
    virtual void tabletPressEvent(coTUIElement *tuiItem);
    // this will be called in PreFrame
    void preFrame();

private:
    // for the HMDCalibration TUI plugin
    coTUITab *myTab;
    coTUIFrame *myFrame;
    coTUIButton *startButton;
    coTUILabel *noOfCyclesLabel;
    coTUILabel *noOfCyclesInt;
    coTUILabel *targetLabel;
    coTUILabel *cameraLabel;
    coTUILabel *patternLabel;
    coTUILabel *systemLabel;
    coTUILabel *transLabel[12];
    // label to display translation values
    coTUILabel *transValue[12];
    coTUILabel *rotLabel[12];
    // label to display rotation values
    coTUILabel *rotValue[12];

    // just for testing
    coTUILabel *startButtonState;
    coTUILabel *startButtonStateLabel;

    // ARToolKit pattern marker to determine the position of the camera
    ARToolKitMarker *myMarker;
    // known position of the IR-target in the tracking coordinate system
    osg::Matrix targetMat;
    osg::Matrix oldTargetMat;
    // unknown position of the camera in the IR-target coordinate system
    osg::Matrix cameraMat;
    osg::Matrix cameraSumm;
    // ARToolKit pattern position in the camera coordinate system
    osg::Matrix patternMat;
    // unknown transformation matrix for the pattern in the tracking coordinate system
    osg::Matrix systemMat;
    int cycle;
    // approximates two unknown transformation matrices
    void doMatApproximation();
    void updateTUI();
    void testCalibrate();
};
#endif
