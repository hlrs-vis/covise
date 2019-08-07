/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include <cover/coVRFileManager.h>


#include <cover/coTabletUI.h>
#include <cover/coVRPlugin.h>
#include <cover/coInteractor.h>
#include <util/unixcompat.h>

#include <vector>
#include <string>

using std::cout;
using std::endl;

#include <signal.h>
#include <osg/MatrixTransform>
#include <osg/io_utils>

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

//#define ARUCO_DEBUG

// ----------------------------------------------------------------------------

using namespace cv;

struct myMsgbuf
{
    long mtype;
    char mtext[100];
};

// ----------------------------------------------------------------------------
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
//! this is called if the plugin is removed at runtime or destroyed
// ----------------------------------------------------------------------------
ARUCOPlugin::~ARUCOPlugin()
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::~ARUCOPlugin()" << std::endl;
#endif

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
// ----------------------------------------------------------------------------
bool ARUCOPlugin::init()
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::init()" << std::endl;
#endif
    
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
        
        markerSize = 150; // set default marker size
        updateMarkerParams(); // update marker sizes from config
        
        dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
        msgQueue = -1;
       
        int selectedDevice = atoi(VideoDevice.c_str());
        
        if (inputVideo.open(selectedDevice))
        {
            cout << "capture device: device " << selectedDevice << " is open" << endl;

            int width = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
            int height =  inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
            std::cout << "   current size  = " << width << "x" << height << std::endl;

            xsize = coCoviseConfig::getInt("width", "COVER.Plugin.ARUCO.VideoDevice", width);
            ysize = coCoviseConfig::getInt("height", "COVER.Plugin.ARUCO.VideoDevice", height);

            if ((xsize != width) || (ysize != height))
            {
                inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, xsize);
                inputVideo.set(cv::CAP_PROP_FRAME_HEIGHT, ysize);

                width = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
                height =  inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

                if ((xsize != width) || (ysize != height))
                {
                    std::cout << "WARNING: could not set capture frame size" << std::endl;
                }
                else
                {
                    std::cout << "   new size  = " << width << "x" << height << std::endl;
                }
            }

            cv::Mat image;

            std::cout << "capture first frame after resetting camera" << std::endl;
            try
            {
                inputVideo >> image;
            }
            catch (cv::Exception &ex)
            {
                std::cerr << "OpenCV exception: " << ex.what() << std::endl;
            }

            ARToolKit::instance()->running = true;
            ARToolKit::instance()->videoMode = GL_BGR;
            //ARToolKit::instance()->videoData=new unsigned char[xsize*ysize*3];
            ARToolKit::instance()->videoDepth = 3;
            ARToolKit::instance()->videoWidth = image.cols;
            ARToolKit::instance()->videoHeight = image.rows;

            adjustScreen();

            // load calib data from file

            // calibrationFilename = coVRFileManager::instance()->getName(calibrationFilename.c_str());

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
                std::cout << "failed to open camera calibration file " << calibrationFilename << std::endl;
                std::cout << "trying to guess a calibration ... " << std::endl;

                double matCameraData[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
                // approximately normal focal length
                matCameraData[0] = 2 * width / 3;
                matCameraData[4] = 2 * width / 3;
                matCameraData[2] = width / 2;
                matCameraData[5] = height / 2;
                cv::Mat matC = cv::Mat(3, 3, CV_64F, matCameraData);
                matCameraMatrix = matC.clone();

                double matDistData[5] = {0, 0, 0, 0, 0};
                cv::Mat matD = cv::Mat(1, 5, CV_64F, matDistData);
                matDistCoefs = matD.clone();

                std::cout << "camera matrix: " << std::endl;
                std::cout << matCameraMatrix << std::endl;
                std::cout << "dist coefs: " << std::endl;
                std::cout << matDistCoefs << std::endl;
            }
        }
        else
        {
            std::cout << "capture device: failed to open " << selectedDevice << std::endl;
            return false;
        }
    }
    ARToolKit::instance()->remoteAR = new RemoteAR();

    opencvRunning = true;
    opencvThread = std::thread([this](){
        opencvLoop();
    });

    return true;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
bool ARUCOPlugin::destroy()
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::destroy()" << std::endl;
#endif

    std::unique_lock<std::mutex> guard(opencvMutex);
    opencvRunning = false;
    guard.unlock();
    if (opencvThread.joinable())
        opencvThread.join();

    delete uiMenu;
    return true;
}

// ----------------------------------------------------------------------------
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

        std::lock_guard<std::mutex> guard(opencvMutex);
        displayIdx = readyIdx;

        ARToolKit::instance()->videoData = (unsigned char *)image[displayIdx].ptr();
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void ARUCOPlugin::opencvLoop()
{
    for (;;)
    {
        std::unique_lock<std::mutex> guard(opencvMutex);
        if (!opencvRunning)
            return;
        guard.unlock();

        if (inputVideo.isOpened())
        {
            guard.lock();
            while (captureIdx == displayIdx || captureIdx == readyIdx) {
                captureIdx = (captureIdx+1)%3;
            }

            guard.unlock();

            inputVideo >> image[captureIdx];
            
            ids[captureIdx].clear();
            corners.clear();
            rejected.clear();
            rvecs[captureIdx].clear();
            tvecs[captureIdx].clear();

            // detect markers and estimate pose
            
            cv::aruco::detectMarkers(image[captureIdx], dictionary, corners, ids[captureIdx], detectorParams, rejected);

            if(ids[captureIdx].size() > 0)
            {
                //cout << "#marker detected " << ids[captureIdx].size() << endl;
                //cout << "#marker rejected " << rejected.size() << endl;

                try
                {
                    // todo: uses default marker size only
                    cv::aruco::estimatePoseSingleMarkers(corners, markerSize / 1000.0, matCameraMatrix,
                                                         matDistCoefs, rvecs[captureIdx], tvecs[captureIdx]);
                    // estimatePoseSingleMarker(corners, markerSize / 1000.0, matCameraMatrix,
                    //                          matDistCoefs, rvecs[captureIdx], tvecs[captureIdx]);
                }
                catch (cv::Exception &ex)
                {
                    std::cerr << "OpenCV exception: " << ex.what() << std::endl;
                    ids[captureIdx].clear();
                    rvecs[captureIdx].clear();
                    tvecs[captureIdx].clear();
                }
            }

            // draw results
            
            if (bDrawDetMarker && ids[captureIdx].size() > 0)
            {
                aruco::drawDetectedMarkers(image[captureIdx], corners, ids[captureIdx]);

                for(unsigned int i = 0; i < ids[captureIdx].size(); ++i)
                {
                    cv::aruco::drawAxis(image[captureIdx], matCameraMatrix, matDistCoefs,
                                        rvecs[captureIdx][i], tvecs[captureIdx][i],
                                        0.1); //markerLength * 0.5f);
                }
            }

            if (bDrawRejMarker && rejected.size() > 0)
            {
                aruco::drawDetectedMarkers(image[captureIdx], rejected, noArray(), Scalar(100, 0, 255));
            }

            guard.lock();
            readyIdx = captureIdx;
            guard.unlock();
        }

        guard.lock();
        if (!opencvRunning)
            return;
        guard.unlock();

        usleep(5000);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
bool ARUCOPlugin::update()
{
    return ARToolKit::instance()->running;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
bool ARUCOPlugin::isVisible(int pattID)
{
// #ifdef ARUCO_DEBUG
//     std::cout << "ARUCOPlugin::isVisible(" << pattID << ")" << std::endl;
// #endif

    for (size_t i = 0; i < ids[displayIdx].size(); i++)
    {
        if (ids[displayIdx][i] == pattID)
        {
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
osg::Matrix ARUCOPlugin::getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4])
{
// #ifdef ARUCO_DEBUG
//     std::cout << "ARUCOPlugin::getMat()" << std::endl;
// #endif

    osg::Matrix markerTrans;
    markerTrans.makeIdentity();

    for (size_t i = 0; i < ids[displayIdx].size(); i++)
    {
        if (ids[displayIdx][i] == pattID)
        {
            // get rotation matrix
            cv::Mat markerRotMat(3, 3, CV_64F);
            cv::setIdentity(markerRotMat);
            cv::Rodrigues(rvecs[displayIdx][i], markerRotMat);
            
            // transform matrix
            double markerTransformData[16];
            cv::Mat markerTransformMat(4, 4, CV_64F, markerTransformData);
            cv::setIdentity(markerTransformMat);
                     
            // copy rot matrix to transform matrix
            markerTransformMat.at<double>(0, 0) = markerRotMat.at<double>(0, 0);
            markerTransformMat.at<double>(1, 0) = markerRotMat.at<double>(1, 0);
            markerTransformMat.at<double>(2, 0) = markerRotMat.at<double>(2, 0);

            markerTransformMat.at<double>(0, 1) = markerRotMat.at<double>(0, 1);
            markerTransformMat.at<double>(1, 1) = markerRotMat.at<double>(1, 1);
            markerTransformMat.at<double>(2, 1) = markerRotMat.at<double>(2, 1);

            markerTransformMat.at<double>(0, 2) = markerRotMat.at<double>(0, 2);
            markerTransformMat.at<double>(1, 2) = markerRotMat.at<double>(1, 2);
            markerTransformMat.at<double>(2, 2) = markerRotMat.at<double>(2, 2);

            // copy trans vector to transform matrix
            markerTransformMat.at<double>(0, 3) = tvecs[displayIdx][i][0] * 1000;
            markerTransformMat.at<double>(1, 3) = tvecs[displayIdx][i][1] * 1000;
            markerTransformMat.at<double>(2, 3) = tvecs[displayIdx][i][2] * 1000;

            // cout << "---------------------------------------" << endl;
            // cout << markerTransformMat << endl;
            // cout << "---------------------------------------" << endl;
            
            int u, v;
            for (u = 0; u < 4; u++)
                for (v = 0; v < 4; v++)
                    markerTrans(v, u) = markerTransformData[(u * 4) + v];

            return OpenGLToOSGMatrix * markerTrans * OpenGLToOSGMatrix;
        }
    }

    return OpenGLToOSGMatrix * markerTrans * OpenGLToOSGMatrix;
}

// ----------------------------------------------------------------------------
//! set new marker sizes according to tabletUI
// ----------------------------------------------------------------------------
void ARUCOPlugin::updateMarkerParams()
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::updateMarkerParams()" << std::endl;
#endif
    
    std::list<ARToolKitMarker *>::iterator it;
    for (it = ARToolKit::instance()->markers.begin();
         it != ARToolKit::instance()->markers.end();
         it++)
    {
        // cout << "--------------------- getsize: " << (*it)->getSize() << endl;
        // cout << "--------------------- getsize: " << (*it)->getPattern() << endl;

        // todo: set individual markers sizes

        // ALVAR: marker_detector.SetMarkerSizeForId((*it)->getPattern(), (*it)->getSize());
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void ARUCOPlugin::adjustScreen()
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::adjustScreen()" << std::endl;
#endif

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
// ----------------------------------------------------------------------------
void ARUCOPlugin::tabletEvent(coTUIElement *tUIItem)
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::tabletEvent()" << std::endl;
#endif

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
// ----------------------------------------------------------------------------
void ARUCOPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::tabletPressEvent()" << std::endl;
#endif

}

// ----------------------------------------------------------------------------
//! return object points for the system centered in a single marker, 
//! given the marker length
// ----------------------------------------------------------------------------
static void _getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints)
{
#ifdef ARUCO_DEBUG
    std::cout << "static _getSingleMarkerObjectPoints" << std::endl;
#endif

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
//! ParallelLoopBody class for the parallelization of the single markers pose estimation
//! Called from function estimatePoseSingleMarkers()
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
// ----------------------------------------------------------------------------
void ARUCOPlugin::estimatePoseSingleMarker(InputArrayOfArrays _corners,
                                           InputArray _cameraMatrix,
                                           InputArray _distCoeffs,
                                           OutputArrayOfArrays _rvecs,
                                           OutputArrayOfArrays _tvecs)
{
#ifdef ARUCO_DEBUG
    std::cout << "ARUCOPlugin::estimatePoseSingleMarker()" << std::endl;
#endif

    CV_Assert(markerLength > 0);
    
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);
    
    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();
    
    parallel_for_(Range(0, nMarkers),
                  SinglePoseEstimationParallel(ids[captureIdx], _corners, _cameraMatrix,
                                               _distCoeffs, rvecs, tvecs));
}


    
// ----------------------------------------------------------------------------
COVERPLUGIN(ARUCOPlugin)
