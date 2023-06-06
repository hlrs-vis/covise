/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: ARToolKit Plugin                                            **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-05  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#ifndef GLUT_NO_LIB_PRAGMA
#define GLUT_NO_LIB_PRAGMA
#endif

#undef HAVE_CUDA
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#endif
#include "ARToolKitPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/RenderObject.h>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include "../common/RemoteAR.h"
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>

using std::cout;
using std::endl;
#include <signal.h>
#include <osg/MatrixTransform>

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
struct myMsgbuf
{
    long mtype;
    char mtext[100];
};

extern int arDebug;

int ARToolKitPlugin::loadPattern(const char *p)
{
    int i;
    for (i = 0; i < numNames; i++)
    {
        if (strcmp(p, names[i]) == 0)
            return ids[i];
    }
    ids[numNames] = arLoadPatt((char *)p);
    names[numNames] = new char[strlen(p) + 1];
    strcpy(names[numNames], p);
    numNames++;
    return ids[numNames - 1];
}

bool ARToolKitPlugin::isVisible(int pattID)
{
    int k, j;
    /* check for object visibility */
    k = -1;
    for (j = 0; j < marker_num; j++)
    {
        if (pattID == marker_info[j].id)
        {
            if (k == -1)
                k = j;
            else if (marker_info[k].cf < marker_info[j].cf)
                k = j;
        }
    }
    if (k < 0)
        return false;
    return true;
}

double pattSize;
double pattCenter[2];
double pattTrans[3][4];
osg::Matrix ARToolKitPlugin::getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4])
{
    osg::Matrix Ctrans;
    Ctrans.makeIdentity();
    int k, j;
    /* check for object visibility */
    k = -1;
    for (j = 0; j < marker_num; j++)
    {
        if (pattID == marker_info[j].id)
        {
            if (k == -1)
                k = j;
            else if (marker_info[k].cf < marker_info[j].cf)
                k = j;
        }
    }
    if (k < 0)
        return Ctrans;
    /* get the transformation between the marker and the real camera */
    arGetTransMat(&marker_info[k], pattCenter, pattSize, pattTrans);
    osg::Matrix OpenGLMatrix;
    OpenGLMatrix.makeIdentity();
    int u, v;
    for (u = 0; u < 3; u++)
        for (v = 0; v < 4; v++)
            OpenGLMatrix(v, u) = pattTrans[u][v];

    osg::Matrix switchMatrix;
    //OpenGLMatrix.print(1,1,"OpenGLMatrix: ",stderr);
    switchMatrix.makeIdentity();
    switchMatrix(0, 0) = -1;
    switchMatrix(1, 1) = 0;
    switchMatrix(2, 2) = 0;
    switchMatrix(1, 2) = 1;
    switchMatrix(2, 1) = -1;
    Ctrans = OpenGLMatrix * switchMatrix;
    return Ctrans;
}

ARToolKitPlugin::ARToolKitPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    marker_num = 0;
    marker_info = NULL;

    dataPtr = NULL;
    numNames = 0;

    vconf = 0;
    vconf2 = 0;
    cconf = 0;
    pconf = 0;
}

bool ARToolKitPlugin::init()
{
    //sleep(6);
    MarkerTracking::instance()->arInterface = this;
    MarkerTracking::instance()->remoteAR = NULL;

    fprintf(stderr, "ARToolKitPlugin::ARToolKitPlugin\n");

    if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.Capture", false))
    {
        if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.MirrorRight", false))
            MarkerTracking::instance()->videoMirrorRight = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.MirrorLeft", false))
            MarkerTracking::instance()->videoMirrorLeft = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.RemoteAR.Transmit", true))
        {
            bitrateSlider = new coTUISlider("Bitrate", MarkerTracking::instance()->artTab->getID());
            bitrateSlider->setValue(300);
            bitrateSlider->setTicks(4950);
            bitrateSlider->setMin(50);
            bitrateSlider->setMax(5000);
            bitrateSlider->setPos(3, 0);
            bitrateSlider->setEventListener(this);
        }
        MarkerTracking::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.ARToolKit.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.ARToolKit.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.ARToolKit.FlipBufferV", true);
        std::string vconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKit.VideoConfig", "-mode=PAL");
        std::string cconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKit.Camera", "/data/ARToolKit/camera_para.dat");
        std::string pconfs = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKit.ViewpointMarker", "/data/ARToolKit/patt.hiro");
        std::string vconf2s = coCoviseConfig::getEntry("value", "COVER.Plugin.ARToolKit.VideoConfigRight", "-mode=PAL");

        thresh = coCoviseConfig::getInt("COVER.Plugin.ARToolKit.Threshold", 100);
        msgQueue = -1;

        arDebugButton = new coTUIToggleButton("Debug", MarkerTracking::instance()->artTab->getID());
        arDebugButton->setPos(0, 0);
        arDebugButton->setEventListener(this);
        arSettingsButton = new coTUIButton("Settings", MarkerTracking::instance()->artTab->getID());
        arSettingsButton->setPos(1, 1);
        arSettingsButton->setEventListener(this);

        thresholdEdit = new coTUISlider("Threshold", MarkerTracking::instance()->artTab->getID());
        thresholdEdit->setValue(thresh);
        thresholdEdit->setTicks(256);
        thresholdEdit->setMin(0);
        thresholdEdit->setMax(256);
        thresholdEdit->setPos(1, 0);
        thresholdEdit->setEventListener(this);

        cconf = new char[cconfs.length() + 1];
        pconf = new char[pconfs.length() + 1];
        vconf = new char[vconfs.length() + 1];
        vconf2 = new char[vconf2s.length() + 1];

        strcpy(cconf, cconfs.c_str());
        strcpy(pconf, pconfs.c_str());
        strcpy(vconf, vconfs.c_str());
        strcpy(vconf2, vconf2s.c_str());

        if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.Stereo", false))
            captureRightVideo();

        ARParam wparam;

        fprintf(stderr, "test1\n");
        /* open the video path */

        if (arVideoOpen(vconf) < 0)
        {
            MarkerTracking::instance()->running = false;
            fprintf(stderr, "test2\n");
        }
        else
        {

            MarkerTracking::instance()->running = true;
/* find the size of the window */
#ifdef WIN32
            arVideoSetFlip(flipBufferH, flipBufferV);
#endif
            if (arVideoInqSize(&xsize, &ysize) < 0)
                exit(0);
            printf("ARToolKitImage size (x,y) = (%d,%d)\n", xsize, ysize);
#ifdef __APPLE__
            MarkerTracking::instance()->videoMode = GL_RGB;
            MarkerTracking::instance()->videoDepth = 3;
            MarkerTracking::instance()->videoData = new unsigned char[xsize * ysize * 3];
#else
            int mode;
            arVideoInqMode(&mode);
            int driver = arVideoInqDriver();
            MarkerTracking::instance()->videoDepth = 3;
            printf("ARToolKitImage node driver = (%d,%d)\n", mode, driver);
            if (driver == VIDEO_LINUX_DV)
            {
                fprintf(stderr, "\n\n\ncolorRGB MODE\n");
                MarkerTracking::instance()->videoMode = GL_RGB;
                MarkerTracking::instance()->videoDepth = 3;
                MarkerTracking::instance()->videoData = new unsigned char[xsize * ysize * 3];
            }
#if !defined(_WIN32)
            else if (driver == VIDEO_LINUX_V4L)
            {
                fprintf(stderr, "\n\n\ncolorBGR MODE\n");
                MarkerTracking::instance()->videoMode = GL_BGR;
                MarkerTracking::instance()->videoDepth = 3;
                //MarkerTracking::instance()->videoData=new unsigned char[xsize*ysize*3];
            }
#else

            // bitte drin lassen!!!! wird unter win32 benoetigt!!! danke uwe
            else if (driver == VIDEO_LINUX_FF)
            {
                fprintf(stderr, "\n\n\ncolorBGRA MODE\n");
                if (mode == AR_FORMAT_GRAY)
                {
                    MarkerTracking::instance()->videoMode = GL_LUMINANCE;
                    MarkerTracking::instance()->videoDepth = 1;
                }
                else if (mode == AR_FORMAT_BGR)
                {
                    MarkerTracking::instance()->videoMode = GL_BGR_EXT;
                    MarkerTracking::instance()->videoDepth = 3;
                }
                else
                {
                    MarkerTracking::instance()->videoMode = GL_RGB;
                    MarkerTracking::instance()->videoDepth = 3;
                }
            }
            else if (driver == VIDEO_LINUX_V4L)
            {
                fprintf(stderr, "\n\n\ncolorBGRA MODE\n");
                MarkerTracking::instance()->videoMode = GL_BGRA_EXT;
                MarkerTracking::instance()->videoDepth = 4;
            }
            // doesnot exist in win32 it does wird benoetigt!! Uwe Bitte aktuelles ARToolkit besorgen falls noetig.

            else if (driver == VIDEO_LINUX_DS)
            {
                fprintf(stderr, "\n\n\nDSVideoLib MODE\n");
                MarkerTracking::instance()->videoMode = GL_BGR_EXT;
                MarkerTracking::instance()->videoDepth = 3;
            }
            else if (driver == VIDEO_LINUX_VI)
            {
                fprintf(stderr, "\n\n\nVideoInput MODE\n");
                MarkerTracking::instance()->videoMode = GL_BGR_EXT;
                MarkerTracking::instance()->videoDepth = 3;
            }
/*    else if(driver==VIDEO_LINUX_V4W)
         {
         fprintf(stderr,"\n\n\ncolorBGR MODE\n");
         MarkerTracking::instance()->videoMode = GL_BGR_EXT;
         MarkerTracking::instance()->videoDepth = 3;
         }*/
#endif
            else if (driver == VIDEO_LINUX_1394CAM && mode == MODE_1280x960_MONO)
            {
                fprintf(stderr, "\n\n\ngrayscale MODE\n");
                MarkerTracking::instance()->videoMode = GL_LUMINANCE;
                //MarkerTracking::instance()->videoData=new unsigned char[xsize*ysize];
                MarkerTracking::instance()->videoDepth = 1;
            }
            else
            {
                fprintf(stderr, "\n\n\ncolor MODE\n");
                MarkerTracking::instance()->videoMode = GL_RGB;
                //MarkerTracking::instance()->videoData=new unsigned char[xsize*ysize*3];
                MarkerTracking::instance()->videoDepth = 3;
            }
#endif
            MarkerTracking::instance()->videoWidth = xsize;
            MarkerTracking::instance()->videoHeight = ysize;

            /* set the initial camera parameters */
            if (arParamLoad(cconf, 1, &wparam) < 0)
            {
                printf("Camera parameter load error !!\n");
                MarkerTracking::instance()->running = false;
            }
            arParamChangeSize(&wparam, xsize, ysize, &cparam);
            printf("*** 1 Camera Parameter ***\n");
            arParamDisp(&cparam);

            if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.AdjustCameraParameter", false))
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
                    cparam.mat[0][2] = xsize / 2.0 + viewPos[0] - (separation / 2.0);
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
                /*cparam.mat[2][0] = viewPos[0];
			cparam.mat[2][1] = viewPos[1];
			cparam.mat[2][2] = viewPos[2];*/
            }

            arInitCparam(&cparam);
            printf("*** 2 Camera Parameter ***\n");
            arParamDisp(&cparam);

            /*double m[16];
		 argConvGlpara(cparam.mat, m);

		 double* mp = (double*) &m;

		 osg::Matrixf mat(mp);
		
	     cerr << "/" << mat(0,0) << "," << mat(0,1) << "," << mat(0,2) << "," << mat(0,3) << "\\" << endl;
		 cerr << "|" << mat(1,0) << "," << mat(1,1) << "," << mat(1,2) << "," << mat(1,3) << "|" << endl;
		 cerr << "|" << mat(2,0) << "," << mat(2,1) << "," << mat(2,2) << "," << mat(2,3) << "|" << endl;
		 cerr << "\\" << mat(3,0) << "," << mat(3,1) << "," << mat(3,2) << "," << mat(3,3) << "/" << endl;*/

            arVideoCapStart();
        }
    }
    else
    {
        bool isDesktopMode = coCoviseConfig::isOn("COVER.Plugin.MarkerTracking.RemoteAR.DesktopMode", false);
        arDesktopMode = new coTUIToggleButton("DesktopMode", MarkerTracking::instance()->artTab->getID());
        arDesktopMode->setPos(1, 0);
        arDesktopMode->setEventListener(this);
        arDesktopMode->setState(isDesktopMode);
    }

    MarkerTracking::instance()->remoteAR = new RemoteAR();
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
ARToolKitPlugin::~ARToolKitPlugin()
{
    delete MarkerTracking::instance()->remoteAR;
    MarkerTracking::instance()->remoteAR = 0;
    MarkerTracking::instance()->arInterface = NULL;

    MarkerTracking::instance()->running = false;
    fprintf(stderr, "ARToolKitPlugin::~ARToolKitPlugin\n");
    arVideoCapStop();
    arVideoClose();

    delete[] vconf2;
    delete[] cconf;
    delete[] vconf;
    delete[] pconf;

    if (msgQueue >= 0)
    {
#ifndef _WIN32
        msgctl(msgQueue, IPC_RMID, NULL);
#endif
    }
}

void ARToolKitPlugin::updateViewerPos(const osg::Vec3f &vp)
{
    if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.AdjustCameraParameter", false))
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

        viewPos.set(vp);

        cparam.mat[0][0] = -viewPos[1];
        cparam.mat[1][1] = -viewPos[1];
        cparam.mat[0][1] = 0;
        float separation = coCoviseConfig::getFloat("separation", "COVER.Stereo", 60);
        if (coVRConfig::instance()->stereoState())
            cparam.mat[0][2] = xsize / 2.0 + viewPos[0] - (separation / 2.0);
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

        arVideoCapStop();
        arInitCparam(&cparam);
        printf("*** 2 Camera Parameter ***\n");
        arParamDisp(&cparam);
        arVideoCapStart();
    }
}

void ARToolKitPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == arDebugButton)
    {
        arDebug = arDebugButton->getState();
    }
    if (tUIItem == arSettingsButton)
    {
#ifdef WIN32
        arVideoShowDialog(1);
#endif
    }
    if (tUIItem == thresholdEdit)
    {
        thresh = thresholdEdit->getValue();
    }
    else if (tUIItem == bitrateSlider)
    {
        MarkerTracking::instance()->remoteAR->updateBitrate(bitrateSlider->getValue());
    }
    else if (tUIItem == arDesktopMode)
    {
        RemoteAR *remARInstance = dynamic_cast<RemoteAR *>(MarkerTracking::instance()->remoteAR);
        covise::DLinkList<RemoteVideo *> *list = remARInstance->getRemoteVideoList();

        RemoteVideo *remvid = NULL;

        list->reset();
        while ((remvid = list->current()) != NULL)
        {
            remvid->setDesktopMode(arDesktopMode->getState());
            list->next();
        }

        //Set Master/Slave in DesktopMode otherwise restore previous setting
        if (arDesktopMode->getState())
        {
            syncmode = coVRCollaboration::instance()->getSyncMode();
            coVRCollaboration::instance()->setSyncMode("MS");
        }
        else
        {
            string mode = "";

            switch (syncmode)
            {
            case coVRCollaboration::LooseCoupling:
            {
                mode = "LOOSE";
            }
            break;
            case coVRCollaboration::TightCoupling:
            {
                mode = "TIGHT";
            }
            break;
            case coVRCollaboration::MasterSlaveCoupling:
            {
                mode = "MS";
            }
            }
            coVRCollaboration::instance()->setSyncMode(mode.c_str());
        }
    }
}

void ARToolKitPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void ARToolKitPlugin::captureRightVideo()
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

            if (coCoviseConfig::isOn("COVER.Plugin.ARToolKit.AdjustCameraParameter", false))
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

void
ARToolKitPlugin::preFrame()
{
#ifndef _WIN32
    struct myMsgbuf message;
#endif
    if (MarkerTracking::instance()->running)
    {
#ifndef _WIN32
        if (msgQueue > 0)
        {
            // allow right capture process to continue
            message.mtype = 1;
            msgsnd(msgQueue, &message, 1, 0);
        }
#endif
        //if(dataPtr)
        //cerr << "1 " << (long long)dataPtr << " content: " << (int)dataPtr[100] << endl;
        arVideoCapNext();
        //if(dataPtr)
        //cerr << "2 " << (long long)dataPtr << " content: " << (int)dataPtr[100] << endl;
        /* grab a vide frame */
        if ((dataPtr = (ARUint8 *)arVideoGetImage()) == NULL)
        {
            return;
        }
        /*
      printf("frame data: ");
      for (int index = 0; index < 16; index ++)
	printf("%d ", dataPtr[index]);
      printf("\ntabletEv
      */

        //if(dataPtr)
        //cerr << "3 " << (long long)dataPtr << " content: " << (int)dataPtr[100] << endl;
        // memcpy(cover->videoData,dataPtr,cover->videoWidth*cover->videoHeight*cover->videoDepth);
        MarkerTracking::instance()->videoData = dataPtr;
        //cerr << "left" << endl;
        /* detect the markers in the video frame */
        if (arDetectMarker(dataPtr, thresh, &marker_info, &marker_num) < 0)
        {
            MarkerTracking::instance()->running = false;
        }
        //if(dataPtr)
        //cerr << "4 " << (long long)dataPtr << " content: " << (int)dataPtr[100] << endl;
        //cerr << " mn " << marker_num << endl;
    }
}

COVERPLUGIN(ARToolKitPlugin)
#endif
