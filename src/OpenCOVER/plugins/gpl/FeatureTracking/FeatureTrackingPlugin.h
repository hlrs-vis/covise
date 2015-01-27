/*************************************************************************
// Plugin: FeatureTrackingPlugin                                           
// Description: Natural Feature Tracking
// Date: 2010-07-01
// Author: RTW
//***********************************************************************/

#ifndef _FEATURETRACKINGPLUGIN_H
#define _FEATURETRACKINGPLUGIN_H

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ARToolKit.h>
#include <cover/coVRPlugin.h>

using namespace covise;
using namespace opencover;

#include "SIFTApplication.h"
#include "EpipolarGeometry.h"
#include "TrackingObject.h"

#ifdef HAVE_AR
#include <AR/param.h>
#include <AR/ar.h>
#include <AR/video.h>

class FeatureTrackingPlugin : public coVRPlugin, public coTUIListener, public ARToolKitInterface
{
public:
    FeatureTrackingPlugin();
    ~FeatureTrackingPlugin();

    static FeatureTrackingPlugin *plugin;
    SIFTApplication siftApp;
    EpipolarGeometry epiGeo;

    bool init();
    void preFrame();
    virtual void tabletPressEvent(coTUIElement *tuiItem);
    virtual void tabletEvent(coTUIElement *tuiItem);
    void updateTUI();

    // for evaluation
    void writeResultsToFile(const char *inPath, const char *inAlgoDesc, const int inFrameNum,
                            const int inRotX_1, const int inRotX_2,
                            const int inRotY_1, const int inRotY_2,
                            const int inRotZ_1, const int inRotZ_2,
                            const int inTrlX_1, const int inTrlX_2,
                            const int inTrlY_1, const int inTrlY_2,
                            const int inTrlZ_1, const int inTrlZ_2,
                            const int inRefPoints, const int inCapPoints, const int inMatches, const int inCorrectMatches,
                            const float inQualityTh, const float inMatchingTh, const int inKdLeaves);

private:
    // time evaluation
    timeval startTime, endTime;
    int convImg, findKPs, findMatches, camPosTime;

    // frames and images
    unsigned char *dataPtr;
    double oldFrameTime;
    int frameMod;
    int frameNum;
    bool doInit;
    bool haveRefFrame;
    bool haveCapFrame;
    bool useSingleFrames;
    std::string refPath;
    std::string capPath;

    // video capturing
    bool flipBufferH;
    bool flipBufferV;
    char *vconf;
    char *vconf2;
    char *cconf;
    char *pconf;
    void captureRightVideo();
    int xsize;
    int ysize;
    ARParam cparam;

    // natural feature tracking
    TrackingObject *trackObj_R;
    TrackingObject *trackObj_C;
    TrackingObject *trackObj_P;
    bool runTracking;
    bool isDebugMode;

    // ARToolKit tracking
    int markerIDs[500];
    char *markerNames[500];
    osg::Matrix camMarkerMat;
    list<ARToolKitMarker *> objectMarkers;
    int markerNum;
    int markerThresh;
    ARMarkerInfo *markerInfo;
    bool useMarkers;
    int msgQueue;

    virtual int loadPattern(const char *);
    virtual osg::Matrix getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4]);
    virtual bool isVisible(int);
    osg::Matrix getCameraPoseFromMarker();

    // tablet plugin UI
    coTUITab *featureTrackingTab;
    coTUIButton *startButton;
    coTUIButton *stopButton;
    coTUIButton *camCalibButton;
    coTUIButton *registerButton;
    coTUIToggleButton *debugButton;
    coTUIToggleButton *singleFrameButton;
    coTUIToggleButton *artoolkitButton;
    coTUIToggleButton *arThresholdButton;
    coTUIFileBrowserButton *loadSingleFrameButton;
    coTUILabel *qualityThresholdLabel;
    coTUILabel *matchingThresholdLabel;
    coTUILabel *noOfMatchesLabel;
    coTUILabel *noOfMatchesValue;
    coTUILabel *noOfCorrectMatchesLabel;
    coTUILabel *noOfCorrectMatchesValue;
    coTUILabel *kdLeavesLabel;
    coTUILabel *spaceLabel;
    coTUILabel *camLabel;
    coTUILabel *camFeatureLabel;
    coTUILabel *rotLabel[3];
    coTUILabel *trlLabel[3];
    coTUILabel *rotValue[3];
    coTUILabel *trlValue[3];
    coTUILabel *camARTKLabel;
    coTUILabel *rotARTKLabel[3];
    coTUILabel *trlARTKLabel[3];
    coTUILabel *rotARTKValue[3];
    coTUILabel *trlARTKValue[3];
    coTUIFloatSlider *qualityThresholdSlider;
    coTUIFloatSlider *matchingThresholdSlider;
    coTUISlider *kdLeavesSlider;
    coTUISlider *arThresholdSlider;
};

#endif
#endif

// EOF
