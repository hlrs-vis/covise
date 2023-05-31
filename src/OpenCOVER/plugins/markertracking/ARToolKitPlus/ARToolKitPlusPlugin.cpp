/***************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: ARToolKitPlus Plugin                                            **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-05  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#endif
#include "ARToolKitPlusPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRCollaboration.h>
#include <cover/RenderObject.h>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include "RemoteARP.h"
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include "DataBuffer.h"
#include "ARCaptureThread.h"

using std::cout;
using std::endl;
#include <signal.h>
#include <osg/MatrixTransform>

#ifdef __MINGW32__
#include <GL/glext.h>
#endif

#ifdef HAVE_AR
//#include <AR/gsub.h>
//#include <AR/video.h>
#define MODE_1280x960_MONO 130

#ifdef __linux__
#include <asm/ioctls.h>
#define sigset signal
#endif
#ifndef _WIN32
#include <sys/ipc.h>
#include <sys/msg.h>
#endif
struct myMsgbuf
{
    long mtype;
    char mtext[100];
};

class DataBuffer;

//extern int      arDebug;

int ARToolKitPlusPlugin::loadPattern(const char *p)
{
    // Return defined ID as integer, no loading required as
    // ARTK+ has the markers defined internally as numbers

    return atoi(p);
}

bool ARToolKitPlusPlugin::isVisible(int pattID)
{
    return DataBuffer::getInstance()->isVisible(pattID);
}

double pattSize;
double pattCenter[2];
double pattTrans[3][4];
osg::Matrix ARToolKitPlusPlugin::getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4])
{
    cerr << "ARToolKitPlusPlugin::getMat(): Entered!" << endl;
    osg::Matrix camera_transform;
    camera_transform.makeIdentity();
    //int marker_index = -1;

    //Check if the marker we want a transformation for is really visible
    if (!this->isVisible(pattID))
    {
        return camera_transform;
    }

    /* get the transformation between the marker and the real camera */
    //arGetTransMat(&marker_info[k], pattCenter, pattSize, pattTrans);

    // For ARToolKitPlus we have to do marker calculation
    // here as we need the pattern id to detect a sepcific pattern

    // As the marker is visible (detected) we retrieve the AR transformation
    // matrix

    osg::Matrix *bufferCameraTrans = DataBuffer::getInstance()->getMarkerMatrix(pattID);
    memcpy(&camera_transform, bufferCameraTrans, sizeof(osg::Matrix));

    return camera_transform;
}

ARToolKitPlusPlugin::ARToolKitPlusPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    m_marker_num = 0;

    dataPtr = NULL;

    m_arCapture = new ARCaptureThread();

    vconf = 0;
    vconf2 = 0;
    cconf = 0;
    pconf = 0;
}

void ARToolKitPlusPlugin::lock()
{
    DataBuffer::getInstance()->lockFront();
}

void ARToolKitPlusPlugin::unlock()
{
    DataBuffer::getInstance()->unlockFront();
}

bool ARToolKitPlusPlugin::init()
{
    // Initialise values of the ARToolKit Interface providing
    // the bridging to OpenCOVER
    MarkerTracking::instance()->arInterface = this;
    MarkerTracking::instance()->remoteAR = NULL;

    MarkerTracking::instance()->artTab->setLabel("ARToolKitPlus");

    fprintf(stderr, "ARToolKitPlusPlugin::ARToolKitPlusPlugin\n");

    MarkerTracking::instance()->m_artoolkitVariant = "ARToolKitPlus";

    //Loading all configuration settings from config file
    if (coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.Capture", false))
    {
        m_arData.setARCaptureEnabled(true);
        if (coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.MirrorRight", false))
        {
            MarkerTracking::instance()->videoMirrorRight = true;
        }
        if (coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.MirrorLeft", false))
        {
            MarkerTracking::instance()->videoMirrorLeft = true;
        }
        bool useBCH = coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.BCHMarkers", true);
        m_arData.setBCHEnabled(useBCH);
        if (coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.RemoteAR.Transmit", true))
        {
            bitrateSlider = new coTUISlider("Bitrate", MarkerTracking::instance()->artTab->getID());
            bitrateSlider->setValue(300);
            bitrateSlider->setTicks(4950);
            bitrateSlider->setMin(50);
            bitrateSlider->setMax(5000);
            bitrateSlider->setPos(4, 0);
            bitrateSlider->setEventListener(this);
            coTUILabel *arBitrateLabel = new coTUILabel("RemoteAR Bitrate", MarkerTracking::instance()->artTab->getID());
            arBitrateLabel->setPos(3, 0);
        }
        MarkerTracking::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.ARTooARToolKitPlus.FlipBufferV", true);
        std::string vconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKitPlus.VideoConfig", "-mode=PAL");
        m_arData.setVideoConfig(vconfs);
        std::string cconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKitPlus.Camera", "/mnt/raid/data/ARToolKit/camera_para.dat");
        m_arData.setCameraConfig(cconfs);
        std::string pconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKitPlus.ViewpointMarker", "/mnt/raid/data/ARToolKit/patt.hiro");
        std::string vconf2s = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKitPlus.VideoConfigRight", "-mode=PAL");
        thresh = coCoviseConfig::getInt("COVER.Plugin.ARToolKitPlus.Threshold", 100);
        m_arData.setThreshold(thresh);
        m_arData.setAutoThresholdEnabled(false);
        m_arData.setAutoThresholdRetries(20);
        msgQueue = -1;

        //Create TabletUI GUI elements
        arDebugButton = new coTUIToggleButton("Debug", MarkerTracking::instance()->artTab->getID());
        arDebugButton->setPos(0, 1);
        arDebugButton->setEventListener(this);
        arSettingsButton = new coTUIButton("Settings", MarkerTracking::instance()->artTab->getID());
        arSettingsButton->setPos(3, 1);
        arSettingsButton->setEventListener(this);

        thresholdEdit = new coTUISlider("Threshold", MarkerTracking::instance()->artTab->getID());
        thresholdEdit->setValue(thresh);
        thresholdEdit->setTicks(256);
        thresholdEdit->setMin(0);
        thresholdEdit->setMax(256);
        thresholdEdit->setPos(1, 0);
        thresholdEdit->setEventListener(this);

        coTUILabel *arThresLabel = new coTUILabel("Image Threshold", MarkerTracking::instance()->artTab->getID());
        arThresLabel->setPos(0, 0);

        arUseBCHButton = new coTUIToggleButton("Use BCH Markers", MarkerTracking::instance()->artTab->getID(), useBCH);
        arUseBCHButton->setPos(1, 1);
        arUseBCHButton->setEventListener(this);

        arDumpImage = new coTUIButton("Dump Camera Image", MarkerTracking::instance()->artTab->getID());
        arDumpImage->setPos(4, 1);
        arDumpImage->setEventListener(this);

        coTUILabel *lblBorderSelect = new coTUILabel("Marker border", MarkerTracking::instance()->artTab->getID());
        lblBorderSelect->setPos(0, 2);

        arBorderSelectBox = new coTUIComboBox("Marker Border", MarkerTracking::instance()->artTab->getID());
        arBorderSelectBox->setPos(1, 2);
        arBorderSelectBox->setEventListener(this);
        arBorderSelectBox->addEntry("Thin");
        arBorderSelectBox->addEntry("Standard");

        arBorderSelectBox->setSelectedText("Thin");
        this->m_arData.setMarkerBorderWidth(0.125f);

        coTUILabel *lblAutoThreshold = new coTUILabel("AutoThreshold Settings", MarkerTracking::instance()->artTab->getID());
        lblAutoThreshold->setPos(0, 3);

        arAutoThresholdButton = new coTUIToggleButton("Use AutoThresholding", MarkerTracking::instance()->artTab->getID());
        arAutoThresholdButton->setPos(2, 1);
        arAutoThresholdButton->setEventListener(this);

        arAutoThresholdValue = new coTUISlider("Auto-Threshold Retries", MarkerTracking::instance()->artTab->getID());
        arAutoThresholdValue->setValue(thresh);
        arAutoThresholdValue->setTicks(100);
        arAutoThresholdValue->setMin(0);
        arAutoThresholdValue->setMax(100);
        arAutoThresholdValue->setPos(1, 3);
        arAutoThresholdValue->setEventListener(this);

        // If stereo is enabled setup everything for the right view camera
        // Currently stereo mode is not implemented for Windows and thus
        // left out for now
        /*if(coCoviseConfig::isOn("COVER.Plugin.ARToolKitPlus.Stereo", false))
      {
         captureRightVideo();
      }*/

        //ARToolKitPlus::ARParam  wparam;

        //Create the VideoInput class and initialise the main camera
        // Start setting up ARToolKitPlus from here
        // Need to adapt old code to new ARToolKitPlus
        // interface

        // Setup ARToolkitPlus Tracker
        m_arCapture->setupVideoAndAR(m_arData);

        Size vidSize = m_arData.getImageSize();

        MarkerTracking::instance()->videoMode = GL_BGR;
        MarkerTracking::instance()->videoDepth = 3;
        MarkerTracking::instance()->videoWidth = vidSize.x;
        MarkerTracking::instance()->videoHeight = vidSize.y;
        MarkerTracking::instance()->flipH = flipBufferH;
        MarkerTracking::instance()->running = true;
    }
    m_arCapture->setARInstance(MarkerTracking::instance());
    DataBuffer::getInstance()->addListener(this);
    m_arCapture->start();
    MarkerTracking::instance()->remoteAR = new RemoteAR();
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
ARToolKitPlusPlugin::~ARToolKitPlusPlugin()
{
    m_arCapture->stop();
    while (m_arCapture->isRunning())
    {
        Sleep(1);
    }
    delete m_arCapture;

    /*DataBuffer* db = DataBufffer::getInstance();
   delete db;*/

    delete MarkerTracking::instance()->remoteAR;
    MarkerTracking::instance()->remoteAR = 0;
    MarkerTracking::instance()->arInterface = NULL;

    MarkerTracking::instance()->running = false;
    cerr << "ARToolKitPlugin::~ARToolKitPlugin" << endl;

    //delete[] vconf2;
    delete[] cconf;
    delete[] vconf;
    //delete[] pconf;
}

void ARToolKitPlusPlugin::update()
{
    MarkerTracking::instance()->videoData = (unsigned char *)DataBuffer::getInstance()->getImagePointer();
}

void ARToolKitPlusPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == arDebugButton)
    {
        //arDebug = arDebugButton->getState();
    }
    if (tUIItem == arSettingsButton)
    {
        //#ifdef WIN32
        //	arVideoShowDialog(1);
        //#endif
    }
    if (tUIItem == thresholdEdit)
    {
        thresh = thresholdEdit->getValue();
        m_arData.setThreshold(thresh);
    }
    else if (tUIItem == bitrateSlider)
    {
        MarkerTracking::instance()->remoteAR->updateBitrate(bitrateSlider->getValue());
    }
    else if (tUIItem == arAutoThresholdButton)
    {
        //m_tracker->activateAutoThreshold(arAutoThresholdButton->getState());
        m_arData.setAutoThresholdEnabled(arAutoThresholdButton->getState());
    }
    else if (tUIItem == this->arAutoThresholdValue)
    {
        //m_tracker->setNumAutoThresholdRetries(arAutoThresholdValue->getValue());
        m_arData.setAutoThresholdRetries(arAutoThresholdValue->getValue());
    }
    else if (tUIItem == this->arUseBCHButton)
    {
        //m_tracker->setMarkerMode(arUseBCHButton->getState() ? ARToolKitPlus::MARKER_ID_BCH : ARToolKitPlus::MARKER_ID_SIMPLE);
        m_arData.setBCHEnabled(arUseBCHButton->getState());
    }
    else if (tUIItem == this->arDumpImage)
    {
        m_arCapture->dumpImage();
    }
    else if (tUIItem == this->arBorderSelectBox)
    {
        if (strcmp(arBorderSelectBox->getSelectedText().c_str(), "Thin"))
        {
            //m_tracker->setBorderWidth(0.125f);
            m_arData.setMarkerBorderWidth(0.125f);
        }
        else
        {
            //m_tracker->setBorderWidth(0.250f);
            m_arData.setMarkerBorderWidth(0.250f);
        }
    }
    m_arCapture->updateData(m_arData);
}
void ARToolKitPlusPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void
ARToolKitPlusPlugin::preFrame()
{
    if (MarkerTracking::instance()->running)
    {

        //MarkerTracking::instance()->videoData=(unsigned char*)DataBuffer::getInstance()->getImagePointer();
    }
}

void ARToolKitPlusPlugin::captureRightVideo()
{
    MarkerTracking::instance()->stereoVideo = true;

#if !defined(_WIN32) && !defined(__APPLE__)
    msgQueue = msgget(IPC_PRIVATE, 0666);
    if (msgQueue < 0)
    {
        perror("Could not initialize Message queue");
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

        fprintf(stderr, "Video Right init\n");
        /* open the video path */
        if (arVideoOpen(vconf2) < 0)
        {
            MarkerTracking::instance()->running = false;
            fprintf(stderr, "Video Right init failed\n");
        }
        else
        {

            MarkerTracking::instance()->running = true;
            /* find the size of the window */
            if (arVideoInqSize(&xsize, &ysize) < 0)
                exit(0);
            printf("Right ARToolKitImage size (x,y) = (%d,%d)\n", xsize, ysize);
            MarkerTracking::instance()->videoWidth = xsize;
            MarkerTracking::instance()->videoHeight = ysize;
            int mode;
            arVideoInqMode(&mode);
            int driver = arVideoInqDriver();
            if (driver == VIDEO_LINUX_1394CAM && mode == MODE_1280x960_MONO)
            {
                //cover->videoDataRight=new unsigned char[xsize*ysize];
            }
            else
            {
                //cover->videoDataRight=new unsigned char[xsize*ysize*3];
            }

            /* set the initial camera parameters */
            if (arParamLoad(cconf, 1, &wparam) < 0)
            {
                printf("Right Camera parameter load error !!\n");
                MarkerTracking::instance()->running = false;
            }
            arParamChangeSize(&wparam, xsize, ysize, &cparam);

            if (coCoviseConfig::isOn("COVER.Plugin.MarkerTracking.AdjustCameraParameter", false))
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
                //initial view position
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
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] + (separation / 2.0);
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] - (separation / 2.0);
                }
                else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
                {
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] + (separation / 2.0);
                }
                else
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0];
                cparam.mat[1][2] = ysize / 2.0;
            }

            arInitCparam(&cparam);
            printf("*** Right Camera Parameter ***\n");
            arParamDisp(&cparam);

            arVideoCapStart();
        }

#ifndef __linux__
        prctl(PR_TERMCHILD); // Exit when parent does
#endif
        sigset(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
        struct myMsgbuf message;
        while (MarkerTracking::instance()->running)
        {

            if (msgQueue > 0)
            {
                // wait for left capture process to allow me to grab an image
                msgrcv(msgQueue, &message, 100, 1, 0);
            }
            /* grab a vide frame */
            if ((dataPtr = (ARUint8 *)arVideoGetImage()) == NULL)
            {
                MarkerTracking::instance()->running = false;
                break;
            }
            //memcpy(cover->videoDataRight,dataPtr,MarkerTracking::instance()->videoWidth*MarkerTracking::instance()->videoHeight*MarkerTracking::instance()->videoDepth);
            MarkerTracking::instance()->videoDataRight = dataPtr;
            //cerr << "right" << endl;
            ///* detect the markers in the video frame */
            //if( arDetectMarker(dataPtr, thresh, &marker_info, &marker_num) < 0 )
            //{
            //    running = false;
            //}
            //fprintf(stderr,"marker_num %d\n",marker_num);

            arVideoCapNext();
            //fprintf(stderr,"%2x\n",buttonData[0]);
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

COVERPLUGIN(ARToolKitPlusPlugin)
#endif
