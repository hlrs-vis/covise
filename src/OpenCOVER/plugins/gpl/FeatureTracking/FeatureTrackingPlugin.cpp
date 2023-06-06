/*************************************************************************
// Plugin: FeatureTrackingPlugin
// Description: Natural Feature Tracking 
// Date: 2010-06-11
// Author: RTW
//***********************************************************************/

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#endif

#include "FeatureTrackingPlugin.h"

#include "config/CoviseConfig.h"
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRMSController.h>

#ifdef __MINGW32__
#include <GL/glext.h>
#endif

#ifdef HAVE_AR
#include <AR/gsub.h>
#include <AR/video.h>
#define MODE_1280x960_MONO 130

#ifdef __linux__
#include <asm/ioctls.h>
#define sigset signal
#endif

#ifndef _WIN32
#include <sys/ipc.h>
#include <sys/msg.h>
#endif

// only for debug mode: description of used method, maximal 30 characters
#define DESC "SIFT 8PA RANSAC"
#define DATA_RESULT "E:/NFT/Results/result_tracking_data.csv"

FeatureTrackingPlugin *FeatureTrackingPlugin::plugin = NULL;
extern int arDebug;

FeatureTrackingPlugin::FeatureTrackingPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "-----\nFeatureTrackingPlugin::FeatureTrackingPlugin\n");

    // frames and images
    dataPtr = NULL;
    static double oldFrameTime = 0.0;
    frameMod = 10; // (camera frameRate / 3)
    frameNum = 0;
    doInit = false;
    haveRefFrame = false;
    haveCapFrame = false;
    useSingleFrames = false;

    // natural feature tracking
    trackObj_R = new TrackingObject();
    trackObj_C = new TrackingObject();
    trackObj_P = new TrackingObject();
    runTracking = false;
    isDebugMode = false;

    // ARToolKit tracking
    ARToolKit::instance()->m_artoolkitVariant = "NFT";
    ARToolKit::instance()->arInterface = this;
    camMarkerMat.makeIdentity();
    markerNum = 0;
    markerThresh = 0;
    markerInfo = NULL;
    useMarkers = false;

    // camera properties
    vconf = 0;
    vconf2 = 0;
    cconf = 0;
    pconf = 0;
}

// this is called if the plugin is removed at runtime
FeatureTrackingPlugin::~FeatureTrackingPlugin()
{
    fprintf(stderr, "\nFeatureTrackingPlugin::~FeatureTrackingPlugin\n");

    ARToolKit::instance()->arInterface = NULL;
    ARToolKit::instance()->running = false;

    arVideoCapStop();
    arVideoClose();

    //delete[] vconf2;
    //delete[] cconf;
    //delete[] vconf;
    //delete[] pconf;
    //delete[] markerNames;

    //delete trackObj_R;
    //delete trackObj_C;
    //delete trackObj_P;

    if (msgQueue >= 0)
    {
#ifndef _WIN32
        msgctl(msgQueue, IPC_RMID, NULL);
#endif
    }
}

bool FeatureTrackingPlugin::init()
{
    // read entries from configuration file
    if (coCoviseConfig::isOn("COVER.Plugin.NFT.Capture", false))
    {
        if (coCoviseConfig::isOn("COVER.Plugin.NFT.MirrorRight", false))
        {
            ARToolKit::instance()->videoMirrorRight = true;
        }
        if (coCoviseConfig::isOn("COVER.Plugin.NFT.MirrorLeft", false))
        {
            ARToolKit::instance()->videoMirrorLeft = true;
        }
        ARToolKit::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.NFT.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.NFT.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.NFT.FlipBufferV", true);
        std::string vconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.NFT.VideoConfig", "-mode=PAL");
        std::string cconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.NFT.Camera", "/mnt/raid/data/ARToolKit/camera_para.dat");
        std::string pconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.NFT.ViewpointMarker", "/mnt/raid/data/ARToolKit/patt.hiro");
        std::string vconf2s = coCoviseConfig::getEntry("value", "COVER.Plugin.NFT.VideoConfigRight", "-mode=PAL");
        markerThresh = coCoviseConfig::getInt("COVER.Plugin.NFT.Threshold", 100);
        msgQueue = -1;

        cconf = new char[cconfs.length() + 1];
        pconf = new char[pconfs.length() + 1];
        vconf = new char[vconfs.length() + 1];
        vconf2 = new char[vconf2s.length() + 1];

        strcpy(cconf, cconfs.c_str());
        strcpy(pconf, pconfs.c_str());
        strcpy(vconf, vconfs.c_str());
        strcpy(vconf2, vconf2s.c_str());

        if (coCoviseConfig::isOn("COVER.Plugin.NFT.Stereo", false))
        {
            captureRightVideo();
        }
        ARParam wparam;

        if (arVideoOpen(vconf) < 0)
        {
            ARToolKit::instance()->running = false;
            fprintf(stderr, "\n Error: Video path could not be opened! \n");
        }
        else
        {
            ARToolKit::instance()->running = true;

#ifdef WIN32
            arVideoSetFlip(flipBufferH, flipBufferV);
#endif

            if (arVideoInqSize(&xsize, &ysize) < 0)
            {
                exit(0);
            }
            printf("\n ARToolKitImage size (x,y): (%d,%d) \n", xsize, ysize);
#ifdef __APPLE__
            ARToolKit::instance()->videoMode = GL_RGB;
            ARToolKit::instance()->videoDepth = 3;
            ARToolKit::instance()->videoData = new unsigned char[xsize * ysize * 3];
#else
            int mode;
            arVideoInqMode(&mode);
            int driver = arVideoInqDriver();
            ARToolKit::instance()->videoDepth = 3;
            printf("\n ARToolKitImage node driver: (%d,%d) \n", mode, driver);
            if (driver == VIDEO_LINUX_DV)
            {
                fprintf(stderr, "\n colorRGB MODE. \n");
                ARToolKit::instance()->videoMode = GL_RGB;
                ARToolKit::instance()->videoDepth = 3;
                ARToolKit::instance()->videoData = new unsigned char[xsize * ysize * 3];
            }
#if !defined(_WIN32)
            else if (driver == VIDEO_LINUX_V4L)
            {
                fprintf(stderr, "\n\n\ncolorBGR MODE\n");
                ARToolKit::instance()->videoMode = GL_BGR;
                ARToolKit::instance()->videoDepth = 3;
                //ARToolKit::instance()->videoData=new unsigned char[xsize*ysize*3];
            }
#else
            /* required for win32 */
            else if (driver == VIDEO_LINUX_FF)
            {
                fprintf(stderr, "\n Mode: colorBGRA. \n");
                if (mode == AR_FORMAT_GRAY)
                {
                    ARToolKit::instance()->videoMode = GL_LUMINANCE;
                    ARToolKit::instance()->videoDepth = 1;
                }
                else if (mode == AR_FORMAT_BGR)
                {
                    ARToolKit::instance()->videoMode = GL_BGR_EXT;
                    ARToolKit::instance()->videoDepth = 3;
                }
                else
                {
                    ARToolKit::instance()->videoMode = GL_RGB;
                    ARToolKit::instance()->videoDepth = 3;
                }
            }
            else if (driver == VIDEO_LINUX_V4L)
            {
                fprintf(stderr, "\n Mode: colorBGRA. \n");
                ARToolKit::instance()->videoMode = GL_BGRA_EXT;
                ARToolKit::instance()->videoDepth = 4;
            }

            else if (driver == VIDEO_LINUX_DS)
            {
                fprintf(stderr, "\n Mode: DSVideoLib. \n");
                ARToolKit::instance()->videoMode = GL_BGR_EXT;
                ARToolKit::instance()->videoDepth = 3;
            }
            else if (driver == VIDEO_LINUX_VI)
            {
                fprintf(stderr, "\n Mode: VideoInput. \n");
                ARToolKit::instance()->videoMode = GL_BGR_EXT;
                ARToolKit::instance()->videoDepth = 3;
            }
/*    else if(driver==VIDEO_LINUX_V4W)
         {
         fprintf(stderr,"\n colorBGR MODE. \n");
         ARToolKit::instance()->videoMode = GL_BGR_EXT;
         ARToolKit::instance()->videoDepth = 3;
         }*/
#endif
            else if (driver == VIDEO_LINUX_1394CAM && mode == MODE_1280x960_MONO)
            {
                fprintf(stderr, "\n Mode: grayscale. \n");
                ARToolKit::instance()->videoMode = GL_LUMINANCE;
                //ARToolKit::instance()->videoData = new unsigned char[xsize*ysize];
                ARToolKit::instance()->videoDepth = 1;
            }
            else
            {
                fprintf(stderr, "\n Mode: color MODE \n");
                ARToolKit::instance()->videoMode = GL_RGB;
                //ARToolKit::instance()->videoData = new unsigned char[xsize*ysize*3];
                ARToolKit::instance()->videoDepth = 3;
            }
#endif
            ARToolKit::instance()->videoWidth = xsize;
            ARToolKit::instance()->videoHeight = ysize;

            // set the initial camera parameters
            if (arParamLoad(cconf, 1, &wparam) < 0)
            {
                printf("\n Error: Camera parameters could not be loaded! \n");
                ARToolKit::instance()->running = false;
            }
            arParamChangeSize(&wparam, xsize, ysize, &cparam);
            arParamDisp(&cparam);

            if (coCoviseConfig::isOn("COVER.Plugin.NFT.AdjustCameraParameter", false))
            {
                osg::Vec3 viewPos;
                float sxsize;
                float sysize;

                if ((coVRConfig::instance()->screens[0].viewportXMax - coVRConfig::instance()->screens[0].viewportXMin) == 0)
                {
                    sxsize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sx;
                    sysize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sy;
                }
                else
                {
                    sxsize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sx * (coVRConfig::instance()->screens[0].viewportXMax - coVRConfig::instance()->screens[0].viewportXMin);
                    sysize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sy * (coVRConfig::instance()->screens[0].viewportYMax - coVRConfig::instance()->screens[0].viewportYMin);
                }

                float xp, yp, zp;
                xp = coCoviseConfig::getFloat("x", "COVER.ViewerPosition", 0.0);
                yp = coCoviseConfig::getFloat("y", "COVER.ViewerPosition", -450.0);
                zp = coCoviseConfig::getFloat("z", "COVER.ViewerPosition", 30.0);
                viewPos.set(xp, yp, zp);

                cparam.mat[0][0] = -viewPos[1];
                cparam.mat[1][1] = -viewPos[1];
                cparam.mat[0][1] = 0;

                float separation = coCoviseConfig::getFloat("separation", "COVER.Stereo", 60);
                if (coVRConfig::instance()->stereoState())
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] - (separation / 2.0);
                }
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] - (separation / 2.0);
                }
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] + (separation / 2.0);
                }
                else
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0];
                }
                cparam.mat[1][2] = ysize / 2.0;
            }
            arInitCparam(&cparam);
            arParamDisp(&cparam);
            arVideoCapStart();
            fprintf(stderr, "\n Starting video capturing ... \n");
        }
    }
    if (coCoviseConfig::isOn("COVER.Plugin.NFT.TrackObjects", false))
    {
        char configName[100];
        std::string pattern;
        ARToolKitMarker *objMarker = NULL;
        do
        {
            sprintf(configName, "ObjectMarker%d", (int)objectMarkers.size());
            std::string entry = std::string("COVER.Plugin.NFT.Marker:") + configName + std::string(".Pattern");
            pattern = coCoviseConfig::getEntry(entry);
            if (!pattern.empty())
            {
                objMarker = new ARToolKitMarker(configName);
                objectMarkers.push_back(objMarker);
            }
        } while (!pattern.empty());

        if (objectMarkers.size() == 0)
        {
            pattern = coCoviseConfig::getEntry("COVER.Plugin.NFT.Marker:ObjectMarker.Pattern");
            if (!pattern.empty())
            {
                objMarker = new ARToolKitMarker("ObjectMarker");
                if (objMarker)
                {
                    objectMarkers.push_back(objMarker);
                }
            }
        }
    }

    // feature tracking plugin GUI
    featureTrackingTab = new coTUITab("FeatureTracking", coVRTui::instance()->mainFolder->getID());
    featureTrackingTab->setPos(0, 0);

    startButton = new coTUIButton("Start", featureTrackingTab->getID());
    startButton->setPos(0, 0);
    startButton->setEventListener(this);
    stopButton = new coTUIButton("Stop", featureTrackingTab->getID());
    stopButton->setPos(1, 0);
    stopButton->setEventListener(this);
    camCalibButton = new coTUIButton("Calibrate cam", featureTrackingTab->getID());
    camCalibButton->setPos(2, 0);
    camCalibButton->setEventListener(this);
    registerButton = new coTUIButton("Register object", featureTrackingTab->getID());
    registerButton->setPos(3, 0);
    registerButton->setEventListener(this);

    qualityThresholdLabel = new coTUILabel("Quality:", featureTrackingTab->getID());
    qualityThresholdLabel->setPos(0, 1);
    qualityThresholdSlider = new coTUIFloatSlider("qualSlider", featureTrackingTab->getID(), true);
    qualityThresholdSlider->setPos(1, 1);
    qualityThresholdSlider->setRange(0.01, 0.10);
    qualityThresholdSlider->setValue(0.05);
    qualityThresholdSlider->setEventListener(this);

    matchingThresholdLabel = new coTUILabel("Matching:", featureTrackingTab->getID());
    matchingThresholdLabel->setPos(0, 2);
    matchingThresholdSlider = new coTUIFloatSlider("matchSlider", featureTrackingTab->getID(), true);
    matchingThresholdSlider->setPos(1, 2);
    matchingThresholdSlider->setRange(0.01, 0.90);
    matchingThresholdSlider->setValue(0.4);
    matchingThresholdSlider->setEventListener(this);

    kdLeavesLabel = new coTUILabel("Kd-tree leaves:", featureTrackingTab->getID());
    kdLeavesLabel->setPos(0, 3);
    kdLeavesSlider = new coTUISlider("kdSlider", featureTrackingTab->getID(), true);
    kdLeavesSlider->setPos(1, 3);
    kdLeavesSlider->setRange(1, 200);
    kdLeavesSlider->setValue(50);
    kdLeavesSlider->setEventListener(this);

    artoolkitButton = new coTUIToggleButton("ARToolKit", featureTrackingTab->getID());
    artoolkitButton->setPos(0, 4);
    artoolkitButton->setEventListener(this);
    arThresholdButton = new coTUIToggleButton("Monochrome img", featureTrackingTab->getID());
    arThresholdButton->setPos(1, 4);
    arThresholdButton->setEventListener(this);
    arThresholdSlider = new coTUISlider("AR thSlider", featureTrackingTab->getID());
    arThresholdSlider->setValue(markerThresh);
    arThresholdSlider->setTicks(256);
    arThresholdSlider->setMin(0);
    arThresholdSlider->setMax(256);
    arThresholdSlider->setPos(2, 4);
    arThresholdSlider->setEventListener(this);

    spaceLabel = new coTUILabel("", featureTrackingTab->getID());
    spaceLabel->setPos(0, 5);

    debugButton = new coTUIToggleButton("Debug mode", featureTrackingTab->getID());
    debugButton->setPos(0, 6);
    debugButton->setEventListener(this);

    singleFrameButton = new coTUIToggleButton("Use images", featureTrackingTab->getID());
    singleFrameButton->setPos(1, 6);
    singleFrameButton->setEventListener(this);
    loadSingleFrameButton = new coTUIFileBrowserButton("Load image", featureTrackingTab->getID());
    loadSingleFrameButton->setMode(coTUIFileBrowserButton::OPEN);
    loadSingleFrameButton->setFilterList("*.bmp");
    loadSingleFrameButton->setPos(2, 6);
    loadSingleFrameButton->setEventListener(this);

    noOfMatchesLabel = new coTUILabel("Matches:", featureTrackingTab->getID());
    noOfMatchesLabel->setPos(0, 9);
    noOfMatchesValue = new coTUILabel("0", featureTrackingTab->getID());
    noOfMatchesValue->setPos(1, 9);
    noOfCorrectMatchesLabel = new coTUILabel("Correct matches:", featureTrackingTab->getID());
    noOfCorrectMatchesLabel->setPos(0, 10);
    noOfCorrectMatchesValue = new coTUILabel("0", featureTrackingTab->getID());
    noOfCorrectMatchesValue->setPos(1, 10);

    camLabel = new coTUILabel("Cam position:", featureTrackingTab->getID());
    camLabel->setPos(0, 11);
    camFeatureLabel = new coTUILabel("FEATURE tracking:", featureTrackingTab->getID());
    camFeatureLabel->setPos(1, 11);
    rotLabel[0] = new coTUILabel("rotation X:", featureTrackingTab->getID());
    rotLabel[0]->setPos(1, 12);
    rotValue[0] = new coTUILabel("0", featureTrackingTab->getID());
    rotValue[0]->setPos(2, 12);
    rotLabel[1] = new coTUILabel("rotation Y:", featureTrackingTab->getID());
    rotLabel[1]->setPos(1, 13);
    rotValue[1] = new coTUILabel("0", featureTrackingTab->getID());
    rotValue[1]->setPos(2, 13);
    rotLabel[2] = new coTUILabel("rotation Z:", featureTrackingTab->getID());
    rotLabel[2]->setPos(1, 14);
    rotValue[2] = new coTUILabel("0", featureTrackingTab->getID());
    rotValue[2]->setPos(2, 14);
    trlLabel[0] = new coTUILabel("translation X:", featureTrackingTab->getID());
    trlLabel[0]->setPos(1, 15);
    trlValue[0] = new coTUILabel("0", featureTrackingTab->getID());
    trlValue[0]->setPos(2, 15);
    trlLabel[1] = new coTUILabel("translation Y:", featureTrackingTab->getID());
    trlLabel[1]->setPos(1, 16);
    trlValue[1] = new coTUILabel("0", featureTrackingTab->getID());
    trlValue[1]->setPos(2, 16);
    trlLabel[2] = new coTUILabel("translation Z:", featureTrackingTab->getID());
    trlLabel[2]->setPos(1, 17);
    trlValue[2] = new coTUILabel("0", featureTrackingTab->getID());
    trlValue[2]->setPos(2, 17);

    camARTKLabel = new coTUILabel("ARTOOLKIT tracking:", featureTrackingTab->getID());
    camARTKLabel->setPos(3, 11);
    rotARTKLabel[0] = new coTUILabel("rotation X:", featureTrackingTab->getID());
    rotARTKLabel[0]->setPos(3, 12);
    rotARTKValue[0] = new coTUILabel("0", featureTrackingTab->getID());
    rotARTKValue[0]->setPos(4, 12);
    rotARTKLabel[1] = new coTUILabel("rotation Y:", featureTrackingTab->getID());
    rotARTKLabel[1]->setPos(3, 13);
    rotARTKValue[1] = new coTUILabel("0", featureTrackingTab->getID());
    rotARTKValue[1]->setPos(4, 13);
    rotARTKLabel[2] = new coTUILabel("rotation Z:", featureTrackingTab->getID());
    rotARTKLabel[2]->setPos(3, 14);
    rotARTKValue[2] = new coTUILabel("0", featureTrackingTab->getID());
    rotARTKValue[2]->setPos(4, 14);
    trlARTKLabel[0] = new coTUILabel("translation X:", featureTrackingTab->getID());
    trlARTKLabel[0]->setPos(3, 15);
    trlARTKValue[0] = new coTUILabel("0", featureTrackingTab->getID());
    trlARTKValue[0]->setPos(4, 15);
    trlARTKLabel[1] = new coTUILabel("translation Y:", featureTrackingTab->getID());
    trlARTKLabel[1]->setPos(3, 16);
    trlARTKValue[1] = new coTUILabel("0", featureTrackingTab->getID());
    trlARTKValue[1]->setPos(4, 16);
    trlARTKLabel[2] = new coTUILabel("translation Z:", featureTrackingTab->getID());
    trlARTKLabel[2]->setPos(3, 17);
    trlARTKValue[2] = new coTUILabel("0", featureTrackingTab->getID());
    trlARTKValue[2]->setPos(4, 17);

    return true;
}

void FeatureTrackingPlugin::tabletEvent(coTUIElement *tuiItem)
{
    // ARTOOLKIT tracking
    if (tuiItem == artoolkitButton)
    {
        if (artoolkitButton->getState())
        {
            useMarkers = true;
        }
        else if (!artoolkitButton->getState())
        {
            useMarkers = false;
        }
    }

    // ARTOOLKIT DEBUG monochrome image button
    if (tuiItem == arThresholdButton)
    {
        arDebug = arThresholdButton->getState();
    }

    // ARTOOLKIT DEBUG threshold slider
    if (tuiItem == arThresholdSlider)
    {
        markerThresh = arThresholdSlider->getValue();
    }

    // DEBUG mode
    if (tuiItem == debugButton)
    {
        if (debugButton->getState())
        {
            isDebugMode = true;
            siftApp.setDebugMode(true);
            epiGeo.setDebugMode(true);
        }
        else if (!debugButton->getState())
        {
            isDebugMode = false;
            useSingleFrames = false;
            singleFrameButton->setState(false);
            siftApp.setDebugMode(false);
            epiGeo.setDebugMode(false);
        }
    }

    // SINGLE FRAMES mode
    if (tuiItem == singleFrameButton)
    {
        if (singleFrameButton->getState())
        {
            useSingleFrames = true;
            isDebugMode = true;
            haveRefFrame = false;
            haveCapFrame = false;
            doInit = false;
            debugButton->setState(true);
            siftApp.setDebugMode(true);
            epiGeo.setDebugMode(true);
        }
        else if (!singleFrameButton->getState())
        {
            useSingleFrames = false;
            isDebugMode = false;
            haveRefFrame = false;
            haveCapFrame = false;
            useMarkers = false;
            doInit = false;
            debugButton->setState(false);
            siftApp.setDebugMode(false);
            epiGeo.setDebugMode(false);
            siftApp.resetSIFTApplication();
            epiGeo.resetEpipolarGeo();
        }
    }

    // load SINGLE FRAME image
    if (tuiItem == loadSingleFrameButton)
    {
        std::string selectPath, realPath;
        selectPath = loadSingleFrameButton->getSelectedPath();

        /* get a useful string */
        if (!selectPath.empty())
        {
            realPath = selectPath.substr(selectPath.find("://") + 3);
        }

        if (!haveRefFrame)
        {
            trackObj_R->initTrackingObject(0, 0);

            if (trackObj_R->getImage()->LoadFromFile(realPath.c_str()))
            {
                siftApp.setImageSize(trackObj_R->getImage()->width, trackObj_R->getImage()->height);
                haveRefFrame = true;
                // path for saving the reference image
                refPath = realPath.insert((realPath.find_last_of("/") + 1), "result_ref_");
            }
            else
            {
                fprintf(stderr, "\n Error: could not load reference image. ");
            }
        }
        else if (haveRefFrame && !haveCapFrame)
        {
            trackObj_C->initTrackingObject(0, 0);

            if (trackObj_C->getImage()->LoadFromFile(realPath.c_str()))
            {
                siftApp.setImageSize(trackObj_C->getImage()->width, trackObj_C->getImage()->height);
                haveCapFrame = true;
                // path for saving the capture image
                capPath = realPath.insert((realPath.find_last_of("/") + 1), "result_cap_");
            }
            else
            {
                fprintf(stderr, "\n Error: could not load capture image. ");
            }
        }
    } // endif (loadSingleFrameButton)
}

void FeatureTrackingPlugin::tabletPressEvent(coTUIElement *tuiItem)
{
    // START application
    if (tuiItem == startButton)
    {
        // use captured frames from VIDEO CAMERA
        if (!useSingleFrames)
        {
            trackObj_R->initTrackingObject(ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight);
            trackObj_R->setQualityThreshold(qualityThresholdSlider->getValue());
            trackObj_R->setMatchingThreshold(matchingThresholdSlider->getValue());
            trackObj_R->setKdLeaves(kdLeavesSlider->getValue());
            trackObj_C->initTrackingObject(ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight);
            trackObj_C->setQualityThreshold(qualityThresholdSlider->getValue());
            trackObj_C->setMatchingThreshold(matchingThresholdSlider->getValue());
            trackObj_C->setKdLeaves(kdLeavesSlider->getValue());
            trackObj_P->initTrackingObject(ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight);
            trackObj_P->setQualityThreshold(qualityThresholdSlider->getValue());
            trackObj_P->setMatchingThreshold(matchingThresholdSlider->getValue());
            trackObj_P->setKdLeaves(kdLeavesSlider->getValue());

            siftApp.setImageSize(trackObj_R->getImage()->width, trackObj_R->getImage()->height);
            epiGeo.setFocalLength(14.0);
            epiGeo.setImageCenter(trackObj_R->getImage()->width / 2.0, trackObj_R->getImage()->height / 2.0);
            runTracking = true;
            doInit = true;
        }
        // use SINGLE FRAMES
        else
        {
            // ARToolKit marker tracking enabled
            if (useMarkers)
            {
                getCameraPoseFromMarker();
            }

            if (haveRefFrame && haveCapFrame)
            {
                siftApp.convertImage(trackObj_R->getImage()->pixels, trackObj_R);
                siftApp.convertImage(trackObj_C->getImage()->pixels, trackObj_C);
                siftApp.findKeypoints(trackObj_R);
                siftApp.findKeypoints(trackObj_C);
                siftApp.findMatches(trackObj_R, trackObj_C);

                // 0 = identity, 1 = rotation, 2 = translation, 3 = rotation plus translation
                //if(epiGeo.testFundMat(epiGeo.do2DTrafo(3)))  // test implementation with example
                //if(epiGeo.findCameraTransformation(epiGeo.do2DTrafo(3),trackObj_R, trackObj_C)) // normal implementation with example
                if (!epiGeo.findCameraTransformation(siftApp.getMatchesVector(), trackObj_R, trackObj_C)) // normal implementation with SIFT features
                {
                    fprintf(stderr, "\n Could not estimate camera pose! ");
                }
                osg::Matrix tmpMat;
                tmpMat = trackObj_R->getCameraPose() * epiGeo.getCameraTransform();
                trackObj_C->setCameraPosition(tmpMat);
                updateTUI();

                if (!trackObj_R->getImage()->SaveToFile(refPath.c_str()))
                {
                    fprintf(stderr, "\n Error: could not save reference image with keypoints! ");
                }

                if (!trackObj_C->getImage()->SaveToFile(capPath.c_str()))
                {
                    fprintf(stderr, "\n Error: could not save captured image with keypoints! ");
                }

                writeResultsToFile(DATA_RESULT, DESC, frameNum,
                                   trackObj_C->getCameraPose().getRotate().x(), camMarkerMat.getRotate().x(),
                                   trackObj_C->getCameraPose().getRotate().y(), camMarkerMat.getRotate().y(),
                                   trackObj_C->getCameraPose().getRotate().z(), camMarkerMat.getRotate().z(),
                                   trackObj_C->getCameraPose().getTrans().x(), camMarkerMat.getTrans().x(),
                                   trackObj_C->getCameraPose().getTrans().y(), camMarkerMat.getTrans().y(),
                                   trackObj_C->getCameraPose().getTrans().z(), camMarkerMat.getTrans().z(),
                                   trackObj_R->getNumberOfKeypoints(), trackObj_C->getNumberOfKeypoints(),
                                   siftApp.getNumberOfMatches(), epiGeo.getRateOfCorrectMatches(),
                                   trackObj_R->getQualityThreshold(), trackObj_R->getMatchingThreshold(), trackObj_R->getKdLeaves());
            }
            else
            {
                fprintf(stderr, "\n No images or keypoints available - initialize first! ");
            }
        } // endif (useSingleFrames)
    }

    // STOP application
    if (tuiItem == stopButton)
    {
        fprintf(stderr, "\n Stopping feature tracking ... ");

        arVideoClose();
        runTracking = false;
        doInit = false;
        useMarkers = false;
        camMarkerMat.makeIdentity();
        haveRefFrame = false;
        haveCapFrame = false;
        dataPtr = NULL;
        frameMod = 0;
        frameNum = 0;
        debugButton->setState(false);
        isDebugMode = false;
        useSingleFrames = false;
        singleFrameButton->setState(false);
        siftApp.resetSIFTApplication();
        epiGeo.resetEpipolarGeo();
    }

    // CAMERA CALIBRATION
    if (tuiItem == camCalibButton)
    {
        fprintf(stderr, "\n Camera calibration - not implemented yet. ");

        //######################################
        // TODO: do camera calibration
        //######################################
    }

    // REGISTRATION of object
    if (tuiItem == registerButton)
    {
        fprintf(stderr, "\n Automatic registration - not implemented yet. ");

        //######################################
        // TODO: do automatic registration
        //######################################
    }
}

void FeatureTrackingPlugin::preFrame()
{
#ifndef _WIN32
    struct myMsgbuf message;
#endif

    if (ARToolKit::instance()->running)
    {
#ifndef _WIN32
        if (msgQueue > 0)
        {
            /* allow right capture process to continue */
            message.mtype = 1;
            msgsnd(msgQueue, &message, 1, 0);
        }
#endif
        // capture frames from video camera constantly
        arVideoCapNext();
        if ((dataPtr = (ARUint8 *)arVideoGetImage()) == NULL)
        {
            return;
        }
        ARToolKit::instance()->videoData = dataPtr;

        // run feature tracking
        if (runTracking)
        {
            frameNum++;
            if (!doInit)
            {
                // for camera position from alternative frame
                if ((frameNum + frameMod / 2) % frameMod == 0)
                {
                    trackObj_P->setID(frameNum);
                    if (siftApp.convertImage(dataPtr, trackObj_P))
                    {
                        siftApp.findKeypoints(trackObj_P);
                    }
                    else
                    {
                        trackObj_P->setFitValue(0);
                    }
                }
                // determination of camera position
                if (frameNum % frameMod == 0)
                {
                    fprintf(stderr, "\n*** Processing trackObj_C (frame no. %i) ***", frameNum);
                    trackObj_C->setID(frameNum);
                    //gettimeofday(&startTime, 0);
                    if (siftApp.convertImage(dataPtr, trackObj_C))
                    {
                        //gettimeofday(&endTime, 0);
                        //convImg = endTime.tv_usec - startTime.tv_usec)/1000;

                        //gettimeofday(&startTime, 0);
                        siftApp.findKeypoints(trackObj_C);
                        //gettimeofday(&endTime, 0);
                        //findKPs = (endTime.tv_usec - startTime.tv_usec)/1000;
                        //fprintf(stderr,"\n*** Time for keypoints [ms]: %f ***", dftDuration);

                        //gettimeofday(&startTime, 0);
                        siftApp.findMatches(trackObj_R, trackObj_C);
                        //gettimeofday(&endTime, 0);
                        //findMatches = (endTime.tv_usec - startTime.tv_usec)/1000;

                        //gettimeofday(&startTime, 0);
                        if (!epiGeo.findCameraTransformation(siftApp.getMatchesVector(), trackObj_R, trackObj_C))
                        {
                            trackObj_C->setFitValue(-1);
                        }
                        //gettimeofday(&endTime, 0);
                        //camPosTime = (endTime.tv_usec - startTime.tv_usec)/1000;

                        if (trackObj_C->getFitValue() > 0)
                        {
                            fprintf(stderr, "\n*** Fit of trackObj_C is good. \n***");
                            trackObj_C->setCameraPosition(trackObj_R->getCameraPose() * epiGeo.getCameraTransform());
                            trackObj_R = trackObj_C;
                        }
                        // camera position from alternative frame
                        else
                        {
                            fprintf(stderr, "\n*** Fit of trackObj_C is bad. Using trackObj_P instead. \n***");
                            if (trackObj_P->getFitValue() > 0)
                            {
                                siftApp.findMatches(trackObj_R, trackObj_P);
                                if (epiGeo.findCameraTransformation(siftApp.getMatchesVector(), trackObj_R, trackObj_C))
                                {
                                    trackObj_P->setCameraPosition(trackObj_R->getCameraPose() * epiGeo.getCameraTransform());
                                    trackObj_C = trackObj_P;
                                }
                                else
                                {
                                    trackObj_P->setFitValue(-2);
                                }
                            }
                            else
                            {
                                fprintf(stderr, "\n*** Re-Initializing... (not implemented yet) ***");
                            }
                        }
                        osg::Matrix leftCamTransform;
                        leftCamTransform = VRViewer::instance()->getViewerMat();

                        if (coVRConfig::instance()->stereoState())
                        {
                            leftCamTransform.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                        }
                        else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                        {
                            leftCamTransform.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                        }
                        else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                        {
                            leftCamTransform.preMult(osg::Matrix::translate((VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                        }
                        cover->getObjectsXform()->setMatrix(trackObj_C->getCameraPose() * leftCamTransform);
                        //coVRCollaboration::instance()->SyncXform();

                        // ARToolKit marker tracking enabled
                        if (useMarkers)
                        {
                            camMarkerMat = getCameraPoseFromMarker();
                        }
                    }
                    else
                    {
                        trackObj_C->setFitValue(0);
                    }
                }
            }
            // the initial frame
            else
            {
                siftApp.setImageSize(ARToolKit::instance()->videoWidth, ARToolKit::instance()->videoHeight);
                if (siftApp.convertImage(dataPtr, trackObj_R))
                {
                    frameNum = 0;
                    fprintf(stderr, "\n*** Processing trackObj_R (frame no. %i) ***", frameNum);
                    trackObj_R->setID(frameNum);
                    trackObj_R->setCameraPosition(getCameraPoseFromMarker());
                    siftApp.findKeypoints(trackObj_R);
                    doInit = false;
                }
                else
                {
                    doInit = true;
                    fprintf(stderr, "\n Could not create reference frame, trying again ... ");
                }
            }

            if (isDebugMode)
            {
                writeResultsToFile(DATA_RESULT, DESC, frameNum,
                                   trackObj_C->getCameraPose().getRotate().x(), camMarkerMat.getRotate().x(),
                                   trackObj_C->getCameraPose().getRotate().y(), camMarkerMat.getRotate().y(),
                                   trackObj_C->getCameraPose().getRotate().z(), camMarkerMat.getRotate().z(),
                                   trackObj_C->getCameraPose().getTrans().x(), camMarkerMat.getTrans().x(),
                                   trackObj_C->getCameraPose().getTrans().y(), camMarkerMat.getTrans().y(),
                                   trackObj_C->getCameraPose().getTrans().z(), camMarkerMat.getTrans().z(),
                                   trackObj_R->getNumberOfKeypoints(), trackObj_C->getNumberOfKeypoints(),
                                   siftApp.getNumberOfMatches(), epiGeo.getRateOfCorrectMatches(),
                                   trackObj_R->getQualityThreshold(), trackObj_R->getMatchingThreshold(), trackObj_R->getKdLeaves());
            }

            // update only when a new frame is captured
            if (cover->frameTime() > oldFrameTime + 1.0)
            {
                updateTUI();
                oldFrameTime = cover->frameTime();
            }
        }
    } // endif (ARToolKit::instance()->running)
    else
    {
        fprintf(stderr, "\n ARToolKit instance is not running. ");
    }
}

void FeatureTrackingPlugin::updateTUI()
{
    char buf[32];

    sprintf(buf, "%i", siftApp.getNumberOfMatches());
    noOfMatchesValue->setLabel(buf);
    sprintf(buf, "%i", epiGeo.getRateOfCorrectMatches());
    noOfCorrectMatchesValue->setLabel(buf);

    // feature tracking
    sprintf(buf, "%f", trackObj_C->getCameraPose().getRotate().x());
    rotValue[0]->setLabel(buf);
    sprintf(buf, "%f", trackObj_C->getCameraPose().getRotate().y());
    rotValue[1]->setLabel(buf);
    sprintf(buf, "%f", trackObj_C->getCameraPose().getRotate().z());
    rotValue[2]->setLabel(buf);
    sprintf(buf, "%f", trackObj_C->getCameraPose().getTrans().x());
    trlValue[0]->setLabel(buf);
    sprintf(buf, "%f", trackObj_C->getCameraPose().getTrans().y());
    trlValue[1]->setLabel(buf);
    sprintf(buf, "%f", trackObj_C->getCameraPose().getTrans().z());
    trlValue[2]->setLabel(buf);

    // ARToolKit
    sprintf(buf, "%f", camMarkerMat.getRotate().x());
    rotARTKValue[0]->setLabel(buf);
    sprintf(buf, "%f", camMarkerMat.getRotate().y());
    rotARTKValue[1]->setLabel(buf);
    sprintf(buf, "%f", camMarkerMat.getRotate().z());
    rotARTKValue[2]->setLabel(buf);
    sprintf(buf, "%f", camMarkerMat.getTrans().x());
    trlARTKValue[0]->setLabel(buf);
    sprintf(buf, "%f", camMarkerMat.getTrans().y());
    trlARTKValue[1]->setLabel(buf);
    sprintf(buf, "%f", camMarkerMat.getTrans().z());
    trlARTKValue[2]->setLabel(buf);
}

osg::Matrix FeatureTrackingPlugin::getCameraPoseFromMarker()
{
    osg::Matrix markerPos;
    osg::Matrix camPos, leftCameraTrans;

    // check if specified marker is available
    if (arDetectMarker(dataPtr, markerThresh, &markerInfo, &markerNum) >= 0)
    {
        for (list<ARToolKitMarker *>::const_iterator it = objectMarkers.begin(); it != objectMarkers.end(); it++)
        {
            if (*it)
            {
                if (isVisible(markerInfo->id))
                {
                    // marker position in camera coordinates
                    markerPos = (*it)->getMarkerTrans();
                    leftCameraTrans = VRViewer::instance()->getViewerMat();

                    if (coVRConfig::instance()->stereoState())
                    {
                        leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                    }
                    else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                    {
                        leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                    }
                    else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                    {
                        leftCameraTrans.preMult(osg::Matrix::translate((VRViewer::instance()->getSeparation() / 2.0), 0, 0));
                    }
                    camPos = markerPos * leftCameraTrans;
                    break;
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "\n Error: Could not detect marker! ");
    }
    return camPos;
}

osg::Matrix FeatureTrackingPlugin::getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4])
{
    osg::Matrix Ctrans;
    Ctrans.makeIdentity();
    int k, j;
    // check for object visibility
    k = -1;
    for (j = 0; j < markerNum; j++)
    {
        if (pattID == markerInfo[j].id)
        {
            if (k == -1)
            {
                k = j;
            }
            else if (markerInfo[k].cf < markerInfo[j].cf)
            {
                k = j;
            }
        }
    }
    if (k < 0)
    {
        return Ctrans;
    }

    // get the transformation between marker and real camera
    arGetTransMat(&markerInfo[k], pattCenter, pattSize, pattTrans);
    osg::Matrix OpenGLMatrix;
    OpenGLMatrix.makeIdentity();
    int u, v;

    for (u = 0; u < 3; u++)
    {
        for (v = 0; v < 4; v++)
        {
            OpenGLMatrix(v, u) = pattTrans[u][v];
        }
    }
    osg::Matrix switchMatrix;
    //OpenGLMatrix.print(1,1,"OpenGLMatrix: ", stderr);
    switchMatrix.makeIdentity();
    switchMatrix(0, 0) = -1;
    switchMatrix(1, 1) = 0;
    switchMatrix(2, 2) = 0;
    switchMatrix(1, 2) = 1;
    switchMatrix(2, 1) = -1;
    Ctrans = OpenGLMatrix * switchMatrix;
    return Ctrans;
}

bool FeatureTrackingPlugin::isVisible(int pattID)
{
    int k, j;
    // check if object marker is visible
    k = -1;
    for (j = 0; j < markerNum; j++)
    {
        if (pattID == markerInfo[j].id)
        {
            if (k == -1)
            {
                k = j;
            }
            else if (markerInfo[k].cf < markerInfo[j].cf)
            {
                k = j;
            }
        }
    }
    if (k < 0)
    {
        return false;
    }
    else
    {
        return true;
    }
}

int FeatureTrackingPlugin::loadPattern(const char *p)
{
    int i;
    for (i = 0; i < markerNum; i++)
    {
        if (strcmp(p, markerNames[i]) == 0)
        {
            return markerIDs[i];
        }
    }
    markerIDs[markerNum] = arLoadPatt((char *)p);
    markerNames[markerNum] = new char[strlen(p) + 1];
    strcpy(markerNames[markerNum], p);
    markerNum++;
    return markerIDs[markerNum - 1];
}

void FeatureTrackingPlugin::captureRightVideo()
{
    ARToolKit::instance()->stereoVideo = true;

#if !defined(_WIN32) && !defined(__APPLE__)
    msgQueue = msgget(IPC_PRIVATE, 0666);
    if (msgQueue < 0)
    {
        perror("\n Could not initialize Message queue! ");
    }
    int ret = fork();
    if (ret == -1)
    {
        cout << "fork failed" << endl;
    }
    else if (ret == 0) // child process
    {
        cout << "Stereo Video server forked" << endl;

        ARParam wparam;
        fprintf(stderr, "\n Video Right init.\n");
        /* open the video path */
        if (arVideoOpen(vconf2) < 0)
        {
            ARToolKit::instance()->running = false;
            fprintf(stderr, "\n Error: Video Right init failed! \n");
        }
        else
        {

            ARToolKit::instance()->running = true;
            /* find the size of the window */
            if (arVideoInqSize(&xsize, &ysize) < 0)
            {
                exit(0);
            }
            printf("\n Right ARToolKitImage size (x,y) = (%d,%d) \n", xsize, ysize);
            ARToolKit::instance()->videoWidth = xsize;
            ARToolKit::instance()->videoHeight = ysize;
            int mode;
            arVideoInqMode(&mode);
            int driver = arVideoInqDriver();
            if (driver == VIDEO_LINUX_1394CAM && mode == MODE_1280x960_MONO)
            {
                //cover->videoDataRight=new unsigned char[xsize*ysize];
            }
            else
            {
                //cover->videoDataRight = new unsigned char[xsize*ysize*3];
            }

            /* set the initial camera parameters */
            if (arParamLoad(cconf, 1, &wparam) < 0)
            {
                printf("\n Error: Right Camera parameter load erro! \n");
                ARToolKit::instance()->running = false;
            }
            arParamChangeSize(&wparam, xsize, ysize, &cparam);

            if (coCoviseConfig::isOn("COVER.Plugin.NFTracking.AdjustCameraParameter", false))
            {
                osg::Vec3 viewPos;
                float sxsize;
                float sysize;

                if ((coVRConfig::instance()->screens[0].viewportXMax - coVRConfig::instance()->screens[0].viewportXMin) == 0)
                {
                    sxsize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sx;
                    sysize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sy;
                }
                else
                {
                    sxsize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sx * (coVRConfig::instance()->screens[0].viewportXMax - coVRConfig::instance()->screens[0].viewportXMin);
                    sysize = coVRConfig::instance()->windows[coVRConfig::instance()->screens[0].window].sy * (coVRConfig::instance()->screens[0].viewportYMax - coVRConfig::instance()->screens[0].viewportYMin);
                }
                /* initial view position */
                float xp, yp, zp;
                xp = coCoviseConfig::getFloat("x", "COVER.ViewerPosition", 0.0);
                yp = coCoviseConfig::getFloat("y", "COVER.ViewerPosition", -450.0);
                zp = coCoviseConfig::getFloat("z", "COVER.ViewerPosition", 30.0);
                viewPos.set(xp, yp, zp);

                cparam.mat[0][0] = -viewPos[1];
                cparam.mat[1][1] = -viewPos[1];
                cparam.mat[0][1] = 0;
                float separation = coCoviseConfig::getFloat("separation", "COVER.Stereo", 60);

                if (coVRConfig::instance()->stereoState())
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] + (separation / 2.0);
                }
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] - (separation / 2.0);
                }
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] + (separation / 2.0);
                }
                else
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0];
                }
                cparam.mat[1][2] = ysize / 2.0;
            }
            arInitCparam(&cparam);
            arParamDisp(&cparam);
            arVideoCapStart();
        }

#ifndef __linux__
        prctl(PR_TERMCHILD); // Exit when parent does
#endif
        sigset(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
        struct myMsgbuf message;
        while (ARToolKit::instance()->running)
        {
            if (msgQueue > 0)
            {
                // wait for left capture process to allow me to grab an image
                msgrcv(msgQueue, &message, 100, 1, 0);
            }
            /* grab a vide frame */
            if ((dataPtr = (ARUint8 *)arVideoGetImage()) == NULL)
            {
                ARToolKit::instance()->running = false;
                break;
            }
            //memcpy(cover->videoDataRight,dataPtr,ARToolKit::instance()->videoWidth*ARToolKit::instance()->videoHeight*ARToolKit::instance()->videoDepth);
            ARToolKit::instance()->videoDataRight = dataPtr;
            //cerr << "right" << endl;
            ///* detect the markers in the video frame */
            //if (arDetectMarker(dataPtr, markerThresh, &markerInfo, &markerNum) < 0)
            //{
            //    running = false;
            //}
            //fprintf(stderr,"\n markerNum: %d\n",markerNum);

            arVideoCapNext();
            //fprintf(stderr, "%2x\n",buttonData[0]);
            if (getppid() == 1)
            {
                //fprintf(stderr, "SERVER: exit\n");
                exit(1);
            }
        }
        exit(1);
    }
#endif
}

void FeatureTrackingPlugin::writeResultsToFile(const char *inPath, const char *inAlgoDesc, const int inFrameNum,
                                               const int inRotX_1, const int inRotX_2,
                                               const int inRotY_1, const int inRotY_2,
                                               const int inRotZ_1, const int inRotZ_2,
                                               const int inTrlX_1, const int inTrlX_2,
                                               const int inTrlY_1, const int inTrlY_2,
                                               const int inTrlZ_1, const int inTrlZ_2,
                                               const int inRefPoints, const int inCapPoints, const int inMatches, const int inCorrectMatches,
                                               const float inQualityTh, const float inMatchingTh, const int inKdLeaves)
{
    char descChar[30];
    strcpy(descChar, inAlgoDesc);
    std::string algoDesc = descChar;

    int correctMatchesPerc = 0;
    if (inCorrectMatches >= 0)
    {
        if (inMatches > 0)
        {
            correctMatchesPerc = inCorrectMatches / inMatches * 100;
        }
        else
        {
            correctMatchesPerc = 0;
        }
    }
    else
    {
        correctMatchesPerc = -1;
    }

    FILE *infoFile = fopen(inPath, "a");
    if (infoFile != NULL)
    {
        fprintf(infoFile, "%s;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%i;%f;%f;%i;\n",
                algoDesc.c_str(), inFrameNum,
                inRotX_1, inRotY_1, inRotZ_1, inTrlX_1, inTrlY_1, inTrlZ_1,
                inRotX_2, inRotY_2, inRotZ_2, inTrlX_2, inTrlY_2, inTrlZ_2,
                inRefPoints, inCapPoints, inMatches, correctMatchesPerc,
                inQualityTh, inMatchingTh, inKdLeaves);
    }
    else
    {
        fprintf(stderr, "\n Error: could not write to %c. ", inPath);
    }
    fclose(infoFile);
}

COVERPLUGIN(FeatureTrackingPlugin)
#endif

// EOF
