/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: ARUCO Plugin                                                **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Mar-16  v1	    				       		                            **
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
#include "ARUCOTrackerPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/RenderObject.h>
#include <cover/ARToolKit.h>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include "RemoteAR.h"
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <opencv2\calib3d\calib3d.hpp>

#include <vector>
#include <string>

using std::cout;
using std::endl;
#include <signal.h>
#include <osg/MatrixTransform>

#ifdef __MINGW32__
#include <GL/glext.h>
#endif

#define MODE_1280x960_MONO 130

#ifdef __linux__
#include <asm/ioctls.h>
#define sigset signal
#endif
#ifndef _WIN32
#include <sys/ipc.h>
#include <sys/msg.h>
#endif

using namespace cv;


struct myMsgbuf
{
    long mtype;
    char mtext[100];
};

int ARUCOPlugin::loadPattern(const char *p)
{
    int pattID = atoi(p);
    if (pattID <= 0)
    {
        fprintf(stderr, "pattern load error !!\n");
        pattID = 0;
    }
    if (pattID > 1000)
    {
        fprintf(stderr, "Pattern ID out of range !!\n");
        pattID = 0;
    }
    return pattID;
}

bool ARUCOPlugin::isVisible(int pattID)
{
    for (size_t i = 0; i < ids.size(); i++)
    {
        if (ids[i] == pattID)
        {
            return true;
        }
    }
    return false;
}

double pattSize;
double pattCenter[2];
double pattTrans[3][4];
osg::Matrix ARUCOPlugin::getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4])
{

    osg::Matrix markerTrans;
    markerTrans.makeIdentity();
    
    for (size_t i = 0; i < ids.size(); i++)
    {
        if (ids[i] == pattID)
        {
            
            double markerPosed[16];
            cv::Mat markerPoseMat(4, 4, CV_64F, markerPosed);
            cv::Rodrigues(rvecs[i], markerPoseMat);
            markerPoseMat.at<double>(0, 3) = tvecs[i][0];
            markerPoseMat.at<double>(1, 3) = tvecs[i][1];
            markerPoseMat.at<double>(2, 3) = tvecs[i][2];

            int u, v;
            for (u = 0; u < 4; u++)
                for (v = 0; v < 4; v++)
                    markerTrans(v, u) = markerPosed[(u * 4) + v];

            /*	if(pattID==(*ARToolKit::instance()->markers.begin())->getPattern())
            {
            return OpenGLToOSGMatrix*markerTrans*OpenGLToOSGMatrix*(*ARToolKit::instance()->markers.begin())->getOffset();
            }
            else*/
            {
                return OpenGLToOSGMatrix * markerTrans * OpenGLToOSGMatrix;
            }
        }
    }

    return OpenGLToOSGMatrix * markerTrans * OpenGLToOSGMatrix;
}

ARUCOPlugin::ARUCOPlugin()
{
    marker_num = 0;

    OpenGLToOSGMatrix.makeRotate(M_PI / -2.0, 1, 0, 0);
    OSGToOpenGLMatrix.makeRotate(M_PI / 2.0, 1, 0, 0);
    //marker_info = NULL;

    dataPtr = NULL;
}

bool ARUCOPlugin::init()
{
    //sleep(6);
    ARToolKit::instance()->arInterface = this;
    ARToolKit::instance()->remoteAR = NULL;

    doCalibrate = false;
    calibrated = false;
    calibCount = 0;

    fprintf(stderr, "ARUCOPlugin::ARUCOPlugin\n");

    if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.Capture", false))
    {

        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorRight", false))
            ARToolKit::instance()->videoMirrorRight = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorLeft", false))
            ARToolKit::instance()->videoMirrorLeft = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.RemoteAR.Transmit", true))
        {
            bitrateSlider = new coTUISlider("Bitrate", ARToolKit::instance()->artTab->getID());
            bitrateSlider->setValue(300);
            bitrateSlider->setTicks(4950);
            bitrateSlider->setMin(50);
            bitrateSlider->setMax(5000);
            bitrateSlider->setPos(3, 0);
            bitrateSlider->setEventListener(this);
        }
        ARToolKit::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferV", true);
        std::string VideoDevice = coCoviseConfig::getEntry("value", "COVER.Plugin.ARUCO.VideoDevice", "0");
        calibrationFilename = coCoviseConfig::getEntry("value", "COVER.Plugin.ARUCO.CameraCalibrationFile", "/data/ARToolKit/defaultCalib.xml");
        xsize = coCoviseConfig::getInt("width", "COVER.Plugin.ARUCO.VideoDevice", 640);
        ysize = coCoviseConfig::getInt("height", "COVER.Plugin.ARUCO.VideoDevice", 480);
        int dictionaryId = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.dictionaryId", 1);
        thresh = coCoviseConfig::getInt("COVER.Plugin.ARUCO.Threshold", 100);
        
        detectorParams = aruco::DetectorParameters::create();
        detectorParams->doCornerRefinement = true; // do corner refinement in markers
        
        dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
        msgQueue = -1;

        arDebugButton = new coTUIToggleButton("Debug", ARToolKit::instance()->artTab->getID());
        arDebugButton->setPos(0, 0);
        arDebugButton->setEventListener(this);
        arSettingsButton = new coTUIButton("Settings", ARToolKit::instance()->artTab->getID());
        arSettingsButton->setPos(1, 1);
        arSettingsButton->setEventListener(this);

        calibrateButton = new coTUIToggleButton("CalibrateCamera", ARToolKit::instance()->artTab->getID());
        calibrateButton->setPos(2, 1);
        calibrateButton->setEventListener(this);
        calibrateLabel = new coTUILabel("notCalibrated", ARToolKit::instance()->artTab->getID());
        calibrateLabel->setPos(3, 1);

        visualizeButton = new coTUIToggleButton("VisualizeMarkers", ARToolKit::instance()->artTab->getID());
        visualizeButton->setPos(4, 1);
        visualizeButton->setEventListener(this);

        detectAdditional = new coTUIToggleButton("DetectAdditional", ARToolKit::instance()->artTab->getID());
        detectAdditional->setPos(5, 1);
        detectAdditional->setEventListener(this);
        detectAdditional->setState(false);

        useSFM = new coTUIToggleButton("useSFM", ARToolKit::instance()->artTab->getID());
        useSFM->setPos(6, 1);
        useSFM->setEventListener(this);
        useSFM->setState(false);

        //if(coCoviseConfig::isOn("COVER.Plugin.ARUCO.Stereo", false))
        //captureRightVideo();
        
        int selectedDevice;
        selectedDevice = atoi(VideoDevice.c_str());
        inputVideo.open(selectedDevice);
        if(inputVideo.isOpened())
        {
            inputVideo.set(CV_CAP_PROP_FRAME_WIDTH,xsize);
            inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT,ysize);

            
            cv::Mat image;
            inputVideo.retrieve(image);
                ARToolKit::instance()->running = true;

                ARToolKit::instance()->videoMode = GL_BGR;
                //ARToolKit::instance()->videoData=new unsigned char[xsize*ysize*3];
                ARToolKit::instance()->videoDepth = 3;
                ARToolKit::instance()->videoWidth = image.cols;
                ARToolKit::instance()->videoHeight = image.rows;

                adjustScreen();
            }
            else
            {
                std::cout << "Could not open selected camera &d\n" << selectedDevice << std::endl;
            }
    }

    ARToolKit::instance()->remoteAR = new RemoteAR();
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
ARUCOPlugin::~ARUCOPlugin()
{
    delete ARToolKit::instance()->remoteAR;
    ARToolKit::instance()->remoteAR = 0;
    ARToolKit::instance()->arInterface = NULL;

    ARToolKit::instance()->running = false;
    fprintf(stderr, "ARUCOPlugin::~ARUCOPlugin\n");
    if(inputVideo.isOpened())
    {
        inputVideo.release();
        if (msgQueue >= 0)
        {
#ifndef _WIN32
            msgctl(msgQueue, IPC_RMID, NULL);
#endif
        }
    }
}

void ARUCOPlugin::updateMarkerParams()
{
    //delete multiMarkerInitializer;
 /*   delete multiMarkerBundle;
    vector<int> ids;
    std::list<ARToolKitMarker *>::iterator it;
    for (it = ARToolKit::instance()->markers.begin(); it != ARToolKit::instance()->markers.end(); it++)
    {
        marker_detector.SetMarkerSizeForId((*it)->getPattern(), (*it)->getSize());
        ids.push_back((*it)->getPattern());
    }

    multiMarkerInitializer = new alvar::MultiMarkerInitializer(ids);
    multiMarkerBundle = new alvar::MultiMarkerBundle(ids);*/
    /*	for(it=ARToolKit::instance()->markers.begin();it!=ARToolKit::instance()->markers.end();it++)
	{
        alvar::Pose pose;
		pose.Reset();
		osg::Matrix markerTrans= OSGToOpenGLMatrix*(*it)->getOffset()*OpenGLToOSGMatrix;
		CvMat *posMat = cvCreateMat(4, 4, CV_64F);

		int u,v;
		for(u=0;u<4;u++)
			for(v=0;v<4;v++)
				posMat->data.db[(u*4)+v] = markerTrans(v,u);
		pose.SetMatrix(posMat);
		multiMarkerBundle->PointCloudAdd((*it)->getPattern(), (*it)->getSize(), pose);
	}*/
};

void ARUCOPlugin::adjustScreen()
{
    if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.AdjustScreenParameters", true))
    {
  /*      osg::Vec3 viewPos;

        float sxsize = xsize;
        float sysize = ysize;

        float d;

        d = cam.calib_K_data[0][0];
        sysize = ((double)ysize / cam.calib_K_data[1][1]) * d;

        coVRConfig::instance()->screens[0].hsize = sxsize;
        coVRConfig::instance()->screens[0].vsize = sysize;

        viewPos.set(cam.calib_K_data[0][2] - ((double)xsize / 2.0), -d, ((double)ysize / 2.0) - cam.calib_K_data[1][2]);

        VRViewer::instance()->setInitialViewerPos(viewPos);
        osg::Matrix viewMat;
        viewMat.makeIdentity();
        viewMat.setTrans(viewPos);
        VRViewer::instance()->setViewerMat(viewMat);*/
    }
}

void ARUCOPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == useSFM)
    {
        if (useSFM->getState())
        {
        }
        else
        {
        }
    }
    if (tUIItem == arDebugButton)
    {
        //arDebug = arDebugButton->getState();
    }
    if (tUIItem == arSettingsButton)
    {
#ifdef WIN32
//arVideoShowDialog(1);
#endif
    }
    if (tUIItem == bitrateSlider)
    {
        ARToolKit::instance()->remoteAR->updateBitrate(bitrateSlider->getValue());
    }
    else if (tUIItem == calibrateButton)
    {
        doCalibrate = calibrateButton->getState();
    }
}

void ARUCOPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
static void _getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    _objPoints.create(4, 1, CV_32FC3);
    Mat objPoints = _objPoints.getMat();
    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.ptr< Vec3f >(0)[0] = Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[1] = Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[2] = Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[3] = Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
}


/**
  * ParallelLoopBody class for the parallelization of the single markers pose estimation
  * Called from function estimatePoseSingleMarkers()
  */
class SinglePoseEstimationParallel : public ParallelLoopBody {
    public:
    SinglePoseEstimationParallel(std::vector<int> &_IDs, InputArrayOfArrays _corners,
                                 InputArray _cameraMatrix, InputArray _distCoeffs,
                                 Mat& _rvecs, Mat& _tvecs)
        : IDs(_IDs), corners(_corners), cameraMatrix(_cameraMatrix),
          distCoeffs(_distCoeffs), rvecs(_rvecs), tvecs(_tvecs) {}

    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;

        for(int i = begin; i < end; i++) {
            
            Mat markerObjPoints;
            float size=-1;
            std::list<ARToolKitMarker *>::iterator it;
            for (it = ARToolKit::instance()->markers.begin(); it != ARToolKit::instance()->markers.end(); it++)
            {
                if((*it)->getPattern() == IDs[i])
                {
                    size = (*it)->getSize();
                }
            }
            if(size < 0) // marker not configured
            {
                char sizeName[100];
                sprintf(sizeName,"%d",IDs[i]);
                ARToolKitMarker *objMarker = new ARToolKitMarker(sizeName);
                objMarker->setObjectMarker(true);
                ARToolKit::instance()->addMarker(objMarker);
                size = objMarker->getSize();
            }
            if(size > 0) //only if Marker is configured
            {
                _getSingleMarkerObjectPoints(size, markerObjPoints);
                cv::solvePnP(markerObjPoints, corners.getMat(i), cameraMatrix, distCoeffs,
                    rvecs.at<Vec3d>(0, i), tvecs.at<Vec3d>(0, i));
            }
        }
    }

    private:
    SinglePoseEstimationParallel &operator=(const SinglePoseEstimationParallel &); // to quiet MSVC

    InputArrayOfArrays corners;
    InputArray cameraMatrix, distCoeffs;
    std::vector<int> &IDs;
    Mat& rvecs, tvecs;
};

/**
  */
void ARUCOPlugin::estimatePoseSingleMarker(InputArrayOfArrays _corners,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs) {

    CV_Assert(markerLength > 0);

    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);

    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();

    parallel_for_(Range(0, nMarkers),
                  SinglePoseEstimationParallel(ids, _corners, _cameraMatrix,
                                               _distCoeffs, rvecs, tvecs));
}
void
ARUCOPlugin::preFrame()
{
#ifndef _WIN32
    struct myMsgbuf message;
#endif
    if (ARToolKit::instance()->running)
    {
#ifndef _WIN32
        if (msgQueue > 0)
        {
            // allow right capture process to continue
            message.mtype = 1;
            msgsnd(msgQueue, &message, 1, 0);
        }
#endif
        if (inputVideo.isOpened())
        {
            
        inputVideo.retrieve(image);
        
                ARToolKit::instance()->videoData = (unsigned char *)image.ptr();

        double tick = (double)getTickCount();
        
        ids.clear();
        corners.clear();
        rejected.clear();
        rvecs.clear();
        tvecs.clear();

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if(ids.size() > 0)
        {
            estimatePoseSingleMarker(corners, camMatrix, distCoeffs, rvecs,
                                             tvecs);
            
        }

      
        // draw results
        image.copyTo(imageCopy);
        if(ids.size() > 0 && visualizeButton->getState()) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);
                for(unsigned int i = 0; i < ids.size(); i++)
                    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                                    markerLength * 0.5f);
        }

        if(visualizeButton->getState() && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        
                ARToolKit::instance()->videoData = (unsigned char *)image.ptr();
                ARToolKit::instance()->videoData = (unsigned char *)imageCopy.ptr();



                if (doCalibrate)
                {

                 /*   const int calibCountMax = 50;
                    const int calibRows = 6;
                    const int calibColumns = 8;
                    if (!calibrated)
                    {
                        // If we have already collected enough data to make the calibration
                        // - We are ready to end the capture loop
                        // - Calibrate
                        // - Save the calibration file
                        if (calibCount >= calibCountMax)
                        {
                            std::cout << "Calibrating..." << endl;
                            calibCount = 0;
                            cam.Calibrate(projPoints);
                            projPoints.Reset();
                            cam.SaveCalib(calibrationFilename.c_str());
                            std::cout << "Saving calibration: " << calibrationFilename << endl;
                            adjustScreen();
                            calibrated = true;
                        }
                        // If we are still collecting calibration data
                        // - For every 1.5s add calibration data from detected 7*9 chessboard (and visualize it if true)
                        else
                        {
                            static double lastTime = 0;
                            double currentTime = cover->frameTime();
                            if (currentTime > (lastTime + 0.5))
                            {
                                if (projPoints.AddPointsUsingChessboard(frame, 2.42, calibRows, calibColumns, true))
                                {
                                    lastTime = currentTime;
                                    calibCount++;
                                    //cout<<calibCount<<"/"<<calibCountMax<<endl;
                                    char tmpText[100];
                                    sprintf(tmpText, "%d%%", (int)(((float)calibCount / (float)calibCountMax)) * 100.0);
                                    calibrateLabel->setLabel(tmpText);
                                }
                            }
                        }
                    }
                    else
                    {
                        if (projPoints.AddPointsUsingChessboard(frame, 2.5, calibRows, calibColumns, true))
                        {
                            alvar::Pose pose;
                            cam.CalcExteriorOrientation(projPoints.object_points, projPoints.image_points, &pose);
                            cam.ProjectPoints(projPoints.object_points, &pose, projPoints.image_points);
                            for (size_t i = 0; i < projPoints.image_points.size(); i++)
                            {
                                cvCircle(frame, cvPoint((int)projPoints.image_points[i].x, (int)projPoints.image_points[i].y), 6, CV_RGB(0, 0, 255));
                            }
                            projPoints.Reset();
                        }
                    }
                }*/
            }
        }
    }
}


COVERPLUGIN(ARUCOPlugin)
