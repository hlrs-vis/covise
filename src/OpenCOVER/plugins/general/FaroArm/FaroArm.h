/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//**********************************************************
// Plugin FaroArm
// obtain coordinates from the FaroArm
// Date: 2008-05-20
//**********************************************************

#ifndef _FAROARM_H
#define _FAROARM_H

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/coAction.h>
#include <osg/Matrix>
#include <device/VRTracker.h>
#include "device/coVRTrackingUtil.h"
#include <cover/ARToolKit.h>
#include "config/CoviseConfig.h"

#include "FaroScannerAPI.h"
#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>

class FaroArm : public coVRPlugin, public coTUIListener, public CDataCaptureCallBack, public OpenThreads::Thread
{
public:
    FaroArm();
    ~FaroArm();

    bool init();

    virtual void tabletEvent(coTUIElement *tuiItem);
    virtual void tabletPressEvent(coTUIElement *tuiItem);
    virtual void run();
    virtual int cancel();
    virtual void OnPositionChanged(const CScanData &ScanData);
    virtual void OnPositionSampled(const CScanData &ScanData);
    // this will be called in PreFrame
    void preFrame();
    virtual void getMatrix(int station, osg::Matrix &mat);

private:
    coTUITab *myTab;
    coTUIFrame *myFrame;
    coTUIButton *stopButton;
    coTUIButton *calibButton;
    coTUIButton *samplePointButton;
    coTUIButton *setPointButton;
    coTUIButton *refFrameButton;
    coTUIButton *directStartButton;

    coTUILabel *debugLabel;
    coTUILabel *coordLabel;
    coTUILabel *refPointSampled_Label;
    coTUILabel *calibrationLabel;
    coTUILabel *imageFrameLabel;
    coTUILabel *imageFrameNo;

    // labels x, y, z, h, p, r for the current translation and rotation parameters of FARO probe
    coTUILabel *transLabel[3];
    coTUILabel *rotLabel[3];
    // value-labels for x, y, z, h, p, r
    coTUILabel *transValue[3];
    coTUILabel *rotValue[3];

    // labels for sample points from Faro device
    coTUILabel *refPointSampled_pointLabel[3];
    coTUILabel *transPointSampled_Label[3];
    // value-labels for the translation parameters of the sample points
    coTUILabel *transPointSampled_Value[3][3];

    // point coordinates (translation only) from tabletUI
    coTUILabel *objectCoords_transLabel[3];
    // edit fields for object coordinates
    coTUIEditFloatField *object_transField[3][3];

    CFaroLaserScanner m_FaroLaserScanner;
    // translation parameters of sample points picked by the Faro device
    osg::Vec3 transPointSampled[10];
    // rotation parameters of sample points picked by the Faro device
    osg::Vec3 rotPointSampled[10];
    // point coordinates
    osg::Vec3 objectPoint[3];
    // transformation into reference frame
    osg::Matrix transformMat;
    // current position of the Faro probe
    osg::Matrix probeMat;
    // transformation into object coordinates
    osg::Matrix transformObjectMat;
    // matrix to calculate average values
    osg::Matrix summMat;
    // transformation of pattern from faro into camera coordinates
    osg::Matrix patternMat;
    // pattern-marker to do the camera-to-probe calibration
    ARToolKitMarker *myMarker;

    int currentSample;
    int maxSamples;

    bool refFrameMode;
    bool calibrationMode;
    bool samplePointMode;
    bool haveRefFrame;
    bool haveSamplePoints;
    bool haveObjectPoints;
    bool haveCalibration;
    bool havePatternPos;

    void makeReferenceFrame();
    void registerObject();
    void calcPatternPosition();
    void calcOffset();
    void displaySamplePoints();
    void updateTUI();

    //--- TEST SIMULATION start ---
    void simulateData();
    osg::Matrix patternDistMat;
    osg::Vec3 transTestVec[12];
    osg::Vec3 rotTestVec[12];
    //--- TEST SIMULATION end ---
};
#endif
