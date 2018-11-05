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
#include "../common/RemoteAR.h"
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <opencv2/calib3d/calib3d.hpp>


#include <cover/coTabletUI.h>
#include <cover/coVRPlugin.h>
#include <cover/coInteractor.h>

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

// ----------------------------------------------------------------------------

using namespace cv;

struct myMsgbuf
{
    long mtype;
    char mtext[100];
};

// ----------------------------------------------------------------------------
// ARUCOPlugin::Plugin()
// ----------------------------------------------------------------------------
ARUCOPlugin::ARUCOPlugin()
    : ui::Owner("ARUCO", cover->ui)
{
    //std::cerr << "ARUCOPlugin::ARUCOPlugin()" << std::endl;
    
    marker_num = 0;

    OpenGLToOSGMatrix.makeRotate(M_PI / -2.0, 1, 0, 0);
    OSGToOpenGLMatrix.makeRotate(M_PI / 2.0, 1, 0, 0);
    //marker_info = NULL;

    dataPtr = NULL;
}

// ----------------------------------------------------------------------------
// ARUCOPlugin::~ARUCOPlugin()
// this is called if the plugin is removed at runtime or destroyed
// ----------------------------------------------------------------------------
ARUCOPlugin::~ARUCOPlugin()
{
    //std::cerr << "ARUCOPlugin::~ARUCOPlugin()" << std::endl;

    delete ARToolKit::instance()->remoteAR;
    ARToolKit::instance()->remoteAR = 0;
    ARToolKit::instance()->arInterface = NULL;
    ARToolKit::instance()->running = false;
   
    if(inputVideo.isOpened())
    {
        inputVideo.release();

#ifndef _WIN32
        if (msgQueue >= 0)
        {
            msgctl(msgQueue, IPC_RMID, NULL);

        }
#endif
    }
}

// ----------------------------------------------------------------------------
// ARUCOPlugin::init()
// ----------------------------------------------------------------------------
bool ARUCOPlugin::init()
{
    //std::cerr << "ARUCOPlugin::init()" << std::endl;

    // check for opencv version
    
    std::cerr << "using opencv version " << CV_VERSION << std::endl;

    if (CV_MAJOR_VERSION < 3 || (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION < 1))
    {
        std::cerr << "error: ARUCOPlugin requires opencv version >= 3.1" << std::endl;
        return false;
    }

    // class init
    
    bDrawDetMarker = true;
    bDrawRejMarker = false;
    
    // ui init

    uiMenu = new ui::Menu("uiMenu", this);
    uiMenu->setText("ARUCO");

    uiBtnDrawDetMarker = new ui::Button(uiMenu, "uiBtnDrawDetMarker");
    uiBtnDrawDetMarker->setText("draw detected markers");
    uiBtnDrawDetMarker->setEnabled(true);
    uiBtnDrawDetMarker->setState(bDrawDetMarker);
    uiBtnDrawDetMarker->setCallback([this](bool state)
    {
        bDrawDetMarker = state;
    });
    
    uiBtnDrawRejMarker = new ui::Button(uiMenu, "uiBtnDrawRejMarker");
    uiBtnDrawRejMarker->setText("draw rejected markers");
    uiBtnDrawRejMarker->setEnabled(true);
    uiBtnDrawRejMarker->setState(bDrawRejMarker);
    uiBtnDrawRejMarker->setCallback([this](bool state)
    {
        bDrawRejMarker = state;
    });

    // AR init
    
    ARToolKit::instance()->arInterface = this;
    ARToolKit::instance()->remoteAR = NULL;

    doCalibrate = false;
    calibrated = false;
    calibCount = 0;

    if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.Capture", false))
    {

        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorRight", false))
            ARToolKit::instance()->videoMirrorRight = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorLeft", false))
            ARToolKit::instance()->videoMirrorLeft = true;

        ARToolKit::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferV", true);
        std::string VideoDevice = coCoviseConfig::getEntry("value", "COVER.Plugin.ARUCO.VideoDevice", "0");

        calibrationFilename = coCoviseConfig::getEntry("value", "COVER.Plugin.ARUCO.CameraCalibrationFile", "/data/aruco/cameras/default.yaml");

        xsize = coCoviseConfig::getInt("width", "COVER.Plugin.ARUCO.VideoDevice", 640);
        ysize = coCoviseConfig::getInt("height", "COVER.Plugin.ARUCO.VideoDevice", 480);

        int dictionaryId = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.DictionaryID", 7);

        thresh = coCoviseConfig::getInt("COVER.Plugin.ARUCO.Threshold", 100);
        
        detectorParams = aruco::DetectorParameters::create();
        
#if (CV_VERSION_MAJOR < 3 || (CV_VERSION_MAJOR == 3 && CV_VERSION_MINOR < 3))
        detectorParams->doCornerRefinement = true; // do corner refinement in markers
#else
        detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
#endif
        
        dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
        msgQueue = -1;
       
        int selectedDevice = atoi(VideoDevice.c_str());
        
        if (inputVideo.open(selectedDevice))
        {
            cout << "capture device: device " << selectedDevice << " is open" << endl;

            inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, xsize);
            inputVideo.set(cv::CAP_PROP_FRAME_HEIGHT, ysize);
            
            std::cout << "                width  = " << inputVideo.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
            std::cout << "                height = " << inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

            cv::Mat image;

            inputVideo >> image;

            ARToolKit::instance()->running = true;
            ARToolKit::instance()->videoMode = GL_BGR;
            //ARToolKit::instance()->videoData=new unsigned char[xsize*ysize*3];
            ARToolKit::instance()->videoDepth = 3;
            ARToolKit::instance()->videoWidth = image.cols;
            ARToolKit::instance()->videoHeight = image.rows;

            adjustScreen();

            // load calib data from file

            std::cout << "loading calibration data from file " << calibrationFilename << std::endl;

            cv::FileStorage fs;

            fs.open(calibrationFilename, cv::FileStorage::READ);
            if (fs.isOpened())
            {
                fs["camera_matrix"] >> matCameraMatrix;
                fs["dist_coefs"] >> matDistCoefs;

                std::cout << "camera matrix: " << std::endl;
                std::cout << matCameraMatrix << std::endl;
                std::cout << "dist coefs: " << std::endl;
                std::cout << matDistCoefs << std::endl;
            }
            else
            {
                std::cerr << "failed to open file " << calibrationFilename << std::endl;
                return false;
            }
        }
        else
        {
            std::cout << "capture device: failed to open " << selectedDevice << std::endl;
            return false;
        }
    }

    ARToolKit::instance()->remoteAR = new RemoteAR();
    return true;
}

// ----------------------------------------------------------------------------
// ARUCOPlugin::destroy()
// ----------------------------------------------------------------------------
bool ARUCOPlugin::destroy()
{
    delete uiMenu;
    return true;
}

// ----------------------------------------------------------------------------
// ARUCOPlugin::preFrame()
// ----------------------------------------------------------------------------
void ARUCOPlugin::preFrame()
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
            inputVideo >> image;
            
            ids.clear();
            corners.clear();
            rejected.clear();
            rvecs.clear();
            tvecs.clear();

            // detect markers and estimate pose
            
            cv::aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

            if(ids.size() > 0)
            {
                //cout << "#marker detected " << ids.size() << endl;
                //cout << "#marker rejected " << rejected.size() << endl;

                cv::aruco::estimatePoseSingleMarkers(corners, 0.1, matCameraMatrix,
                                                     matDistCoefs, rvecs, tvecs);
            }

            // draw results
            
            if (bDrawDetMarker && ids.size() > 0)
            {
                //std::cerr << "draw detected" << std::endl;
                
                aruco::drawDetectedMarkers(image, corners, ids);

                for(unsigned int i = 0; i < ids.size(); ++i)
                {
                    cv::aruco::drawAxis(image, matCameraMatrix, matDistCoefs,
                                        rvecs[i], tvecs[i],
                                        0.1); //markerLength * 0.5f);
                }
            }

            if (bDrawRejMarker && rejected.size() > 0)
            {
                //std::cerr << "draw rejected" << std::endl;
                
                aruco::drawDetectedMarkers(image, rejected, noArray(), Scalar(100, 0, 255));
            }
            
            ARToolKit::instance()->videoData = (unsigned char *)image.ptr();
        }
    }
}

// ----------------------------------------------------------------------------
// ARUCOPlugin::update()
// ----------------------------------------------------------------------------
bool ARUCOPlugin::update()
{
    return ARToolKit::instance()->running;
}

// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
void ARUCOPlugin::adjustScreen()
{
    //std::cerr << "ARUCOPlugin::adjustScreen()" << std::endl;
    
    if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.AdjustScreenParameters", true))
    {
/*
        osg::Vec3 viewPos;

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
        VRViewer::instance()->setViewerMat(viewMat);
*/
    }
}

// ----------------------------------------------------------------------------
void ARUCOPlugin::tabletEvent(coTUIElement *tUIItem)
{
//     if (tUIItem == useSFM)
//     {
//         if (useSFM->getState())
//         {
//         }
//         else
//         {
//         }
//     }
//     if (tUIItem == arDebugButton)
//     {
//         //arDebug = arDebugButton->getState();
//     }
//     if (tUIItem == arSettingsButton)
//     {
// #ifdef WIN32
// //arVideoShowDialog(1);
// #endif
//     }
//     if (tUIItem == bitrateSlider)
//     {
//         ARToolKit::instance()->remoteAR->updateBitrate(bitrateSlider->getValue());
//     }
//     else if (tUIItem == calibrateButton)
//     {
//         doCalibrate = calibrateButton->getState();
//     }
}

// ----------------------------------------------------------------------------
void ARUCOPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

// ----------------------------------------------------------------------------
// @brief Return object points for the system centered in a single marker, given the marker length
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// ParallelLoopBody class for the parallelization of the single markers pose estimation
// Called from function estimatePoseSingleMarkers()
// ----------------------------------------------------------------------------
class SinglePoseEstimationParallel : public ParallelLoopBody
{
public:
    SinglePoseEstimationParallel(std::vector<int> &_IDs, InputArrayOfArrays _corners,
                                 InputArray _cameraMatrix, InputArray _distCoeffs,
                                 Mat& _rvecs, Mat& _tvecs)
        : IDs(_IDs), corners(_corners), cameraMatrix(_cameraMatrix),
          distCoeffs(_distCoeffs), rvecs(_rvecs), tvecs(_tvecs) {}
    
    void operator()(const Range &range) const
        {
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

// ----------------------------------------------------------------------------
void ARUCOPlugin::estimatePoseSingleMarker(InputArrayOfArrays _corners,
                                           InputArray _cameraMatrix, InputArray _distCoeffs,
                                           OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs)
{
    CV_Assert(markerLength > 0);
    
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);
    
    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();
    
    parallel_for_(Range(0, nMarkers),
                  SinglePoseEstimationParallel(ids, _corners, _cameraMatrix,
                                               _distCoeffs, rvecs, tvecs));
}
    
// ----------------------------------------------------------------------------
COVERPLUGIN(ARUCOPlugin)
