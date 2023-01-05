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

#ifndef _WIN32
struct myMsgbuf
{
    long mtype;
    char mtext[100];
};
#endif

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

    uiBtnCalib = new ui::Action(uiMenu, "calibrate");
    uiBtnCalib->setText("calibrate");
    uiBtnCalib->setEnabled(true);
    uiBtnCalib->setCallback([this]()
        {
            startCallibration();
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

        int dictionaryId = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.DictionaryID", 7); // 16 = ARUCO_DEFAULT

#if( CV_VERSION_MAJOR < 4)
        detectorParams = aruco::DetectorParameters::create();
#else
        detectorParams = new aruco::DetectorParameters();
#endif
        
#if (CV_VERSION_MAJOR < 3 || (CV_VERSION_MAJOR == 3 && CV_VERSION_MINOR < 3))
        detectorParams->doCornerRefinement = true; // do corner refinement in markers
#else
        detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
        detectorParams->useAruco3Detection = true;
        detectorParams->minSideLengthCanonicalImg = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.minSideLengthCanonicalImg", 50);
        detectorParams->markerBorderBits = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.markerBorderBits", 1);
#endif
        
        markerSize = 150; // set default marker size
        updateMarkerParams(); // update marker sizes from config

#if( CV_VERSION_MAJOR < 4)
        dictionary = aruco::getPredefinedDictionary(dictionaryId);
#else
        dictionary = aruco::getPredefinedDictionary(dictionaryId);
#endif

#if( CV_VERSION_MAJOR >= 4)
        detector = new cv::aruco::ArucoDetector(dictionary,*detectorParams);
#endif
        msgQueue = -1;
       
        int selectedDevice = atoi(VideoDevice.c_str());

#if CV_VERSION_MAJOR > 3 || (CV_VERSION_MAJOR==3 && CV_VERSION_MINOR>1)
        for (int cap: {CAP_V4L2, CAP_ANY})
        {
            if (inputVideo.open(selectedDevice, cap))
                break;
        }
#else
        inputVideo.open(selectedDevice);
#endif
        
        if (inputVideo.isOpened())
        {
            cout << "capture device: device " << selectedDevice << " is open" << endl;

            bool exists = false;
            std::string fourcc = coCoviseConfig::getEntry("fourcc", "COVER.Plugin.ARUCO.VideoDevice", "", &exists);
            if (fourcc.length() == 4)
            {
                std::cerr << "Setting FOURCC to " << fourcc << std::endl;
                inputVideo.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]));
                std::cerr << "FOURCC: " << inputVideo.get(cv::CAP_PROP_FOURCC) << std::endl;
            }

            float fps = coCoviseConfig::getFloat("fps", "COVER.Plugin.ARUCO.VideoDevice", 0.f, &exists);
            if (fps > 0.f)
            {
                std::cerr << "Setting FPS to " << fps << std::endl;
                inputVideo.set(cv::CAP_PROP_FPS, fps);
                std::cerr << "Frame rate: " << inputVideo.get(cv::CAP_PROP_FPS) << std::endl;
            }

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
                    xsize = width;
                    ysize = height;
                    std::cout << "WARNING: could not set capture frame size" << std::endl;
                    std::cout << "   new size  = " << width << "x" << height << std::endl;
                }
            }

            //inputVideo.set(cv::CAP_PROP_CONVERT_RGB, false);

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

            std::cout << "Capturing " << image.cols << "x" << image.rows << " pixels" << std::endl;


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
                std::cout << matCameraMatrix << std::endl;
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
        adjustScreen();
        /* 
        how to generate callibration patterns:
         cd src/opencv_contrib/modules/aruco/misc/pattern_generator/
          python MarkerPrinter.py --charuco --file "./charuco.pdf" --dictionary DICT_5X5_1000 --size_x 16 --size_y 9 --square_length 0.09 --marker_length 0.07 --border_b
its 1
        */
        int squaresX = coCoviseConfig::getInt("xSize", "COVER.Plugin.ARUCO.Callibration", 8);
        int squaresY = coCoviseConfig::getInt("ysize", "COVER.Plugin.ARUCO.Callibration", 5);
        float squareLength = coCoviseConfig::getInt("squareSize", "COVER.Plugin.ARUCO.Callibration", 18);
        float markerLength = coCoviseConfig::getInt("markerSize", "COVER.Plugin.ARUCO.Callibration", 14);
        charucoboard = new aruco::CharucoBoard(Size(squaresX, squaresY), squareLength, markerLength, dictionary);
        cv::aruco::CharucoParameters cp;
        cv::aruco::RefineParameters rp;
        
        charucoDetector = new cv::aruco::CharucoDetector(*charucoboard, cp, *detectorParams,rp);
        //board = charucoboard.staticCast<cv::aruco::Board>();
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
    ARToolKit::instance()->videoData = nullptr;
    return true;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void ARUCOPlugin::preFrame()
{
    if (ARToolKit::instance()->running)
    {

#ifndef _WIN32
        if (msgQueue > 0)
        {
            struct myMsgbuf message;
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

inline static bool saveCameraParams(const std::string& filename, cv::Size imageSize, float aspectRatio, int flags,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double totalAvgErr) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return false;

    time_t tt;
    time(&tt);
    struct tm* t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if (flags & cv::CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

    if (flags != 0) {
        sprintf(buf, "flags: %s%s%s%s",
            flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    }
    fs << "flags" << flags;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "avg_reprojection_error" << totalAvgErr;
    return true;
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


#if( CV_VERSION_MAJOR >= 4)
            detector->detectMarkers(image[captureIdx], corners, ids[captureIdx],  rejected);
#else
            cv::aruco::detectMarkers(image[captureIdx], dictionary, corners, ids[captureIdx], detectorParams, rejected);
#endif

            if(ids[captureIdx].size() > 0)
            {
                //cout << "#marker detected " << ids[captureIdx].size() << endl;
                //cout << "#marker rejected " << rejected.size() << endl;

                try
                {
                    // todo: uses default marker size only
                    //cv::aruco::estimatePoseSingleMarkers(corners, 49, matCameraMatrix,
                    //                                     matDistCoefs, rvecs[captureIdx], tvecs[captureIdx]);
                     estimatePoseSingleMarker(corners,  matCameraMatrix,
                                              matDistCoefs, rvecs[captureIdx], tvecs[captureIdx]);
                }
                catch (cv::Exception &ex)
                {
                    std::cerr << "OpenCV exception: " << ex.what() << std::endl;
                    std::cerr << "Camera might need callibration: "  << std::endl;
                }
            }
            if (doCalibrate)
            {
                if (ids[captureIdx].size() > 25)
                {

                    Mat currentCharucoCorners, currentCharucoIds;
                    charucoDetector->detectBoard(image[captureIdx], currentCharucoCorners, currentCharucoIds, corners, ids[captureIdx]);

                    if (currentCharucoCorners.total() > 0)
                        aruco::drawDetectedCornersCharuco(image[captureIdx], currentCharucoCorners, currentCharucoIds);

                    allCorners.push_back(corners);
                    allIds.push_back(ids[captureIdx]);
                    allImgs.push_back(image[captureIdx]);
                    imgSize = image[captureIdx].size();
                }
                if (allIds.size() > 20)
                {
                    cv::aruco::CharucoParameters cp;
                    cv::aruco::RefineParameters rp;
                    Mat cameraMatrix, distCoeffs;
                    vector< Mat > rvecs, tvecs;
                    double repError;

                    /*if (calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
                        cameraMatrix = Mat::eye(3, 3, CV_64F);
                        cameraMatrix.at< double >(0, 0) = aspectRatio;
                    }*/

                    // prepare data for calibration
                    vector< vector< Point2f > > allCornersConcatenated;
                    vector< int > allIdsConcatenated;
                    vector< int > markerCounterPerFrame;
                    markerCounterPerFrame.reserve(allCorners.size());
                    for (unsigned int i = 0; i < allCorners.size(); i++) {
                        markerCounterPerFrame.push_back((int)allCorners[i].size());
                        for (unsigned int j = 0; j < allCorners[i].size(); j++) {
                            allCornersConcatenated.push_back(allCorners[i][j]);
                            allIdsConcatenated.push_back(allIds[i][j]);
                        }
                    }
                    int calibrationFlags = 0;
                    float aspectRatio = 1;
                   // if (parser.has("a")) {
                   //     calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
                   //     aspectRatio = parser.get<float>("a");
                   // }
                   // if (parser.get<bool>("zt")) calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
                   // if (parser.get<bool>("pc")) calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;

                    // calibrate camera using aruco markers
                    double arucoRepErr;
                    arucoRepErr = aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
                        markerCounterPerFrame, charucoboard, imgSize, cameraMatrix,
                        distCoeffs, noArray(), noArray(), calibrationFlags);

                    // prepare data for charuco calibration
                    int nFrames = (int)allCorners.size();
                    vector< Mat > allCharucoCorners;
                    vector< Mat > allCharucoIds;
                    vector< Mat > filteredImages;
                    allCharucoCorners.reserve(nFrames);
                    allCharucoIds.reserve(nFrames);

                    for (int i = 0; i < nFrames; i++) {
                        // interpolate using camera parameters
                        Mat currentCharucoCorners, currentCharucoIds;
                        cp.cameraMatrix = cameraMatrix;
                        cp.distCoeffs = distCoeffs;
                        charucoDetector->detectBoard(allImgs[i], currentCharucoCorners, currentCharucoIds, allCorners[i], allIds[i]);

                        allCharucoCorners.push_back(currentCharucoCorners);
                        allCharucoIds.push_back(currentCharucoIds);
                        filteredImages.push_back(allImgs[i]);
                    }

                    if (allCharucoCorners.size() < 4) {
                        cerr << "Not enough corners for calibration" << endl;
                    }
                    else
                    {

                        // calibrate camera using charuco
                        repError =
                            aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard, imgSize,
                                cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

                        bool saveOk = saveCameraParams(calibrationFilename, imgSize, aspectRatio, calibrationFlags,
                            cameraMatrix, distCoeffs, repError);
                        if (!saveOk) {
                            cerr << "Cannot save output file" << endl;
                        }
                        else
                        {

                            cout << "Rep Error: " << repError << endl;
                            cout << "Rep Error Aruco: " << arucoRepErr << endl;
                            cout << "Calibration saved to " << calibrationFilename << endl;
                        }
                    }

                    doCalibrate = false;
                    allImgs.clear();
                    allIds.clear();
                    allCorners.clear();
                }

            }

            // draw results
            
            if (bDrawDetMarker && ids[captureIdx].size() > 0)
            {
                aruco::drawDetectedMarkers(image[captureIdx], corners, ids[captureIdx]);

                for(unsigned int i = 0; i < ids[captureIdx].size(); ++i)
                {
#if CV_VERSION_MAJOR < 4
                    cv::aruco::drawAxis(image[captureIdx], matCameraMatrix, matDistCoefs,
                                        rvecs[captureIdx][i], tvecs[captureIdx][i],
                                        0.1); //markerLength * 0.5f);
#endif
                }
            }

            if (bDrawRejMarker && rejected.size() > 0)
            {
                aruco::drawDetectedMarkers(image[captureIdx], rejected, noArray(), Scalar(100, 0, 255));
            }
        /*    if (doCalibrate)
            {

                const int calibCountMax = 50;
                const int calibRows = 6;
                const int calibColumns = 9;
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
                                sprintf(tmpText, "%d%%", (int)(((float)calibCount / (float)calibCountMax) * 100.0));
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
            }
            */
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

void ARUCOPlugin::startCallibration()
{
    allCorners.clear();
    allIds.clear();
    allImgs.clear();
    doCalibrate = true;
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
            markerTransformMat.at<double>(0, 3) = tvecs[displayIdx][i][0];
            markerTransformMat.at<double>(1, 3) = tvecs[displayIdx][i][1];
            markerTransformMat.at<double>(2, 3) = tvecs[displayIdx][i][2];

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

        osg::Vec3 viewPos;

        float sxsize = xsize;
        float sysize = ysize;

        float d;

        d = matCameraMatrix.at<double>(0, 0);
        sysize = ((double)ysize / matCameraMatrix.at<double>(1, 1)) * d;

        coVRConfig::instance()->screens[0].hsize = sxsize;
        coVRConfig::instance()->screens[0].vsize = sysize;

        viewPos.set(matCameraMatrix.at<double>(0, 2) - ((double)xsize / 2.0), -d, ((double)ysize / 2.0) - matCameraMatrix.at<double>(1, 2));

        VRViewer::instance()->setInitialViewerPos(viewPos);
        osg::Matrix viewMat;
        viewMat.makeIdentity();
        viewMat.setTrans(viewPos);
        VRViewer::instance()->setViewerMat(viewMat);

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

OpenThreads::Mutex mutex;
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
                    OpenThreads::ScopedLock<OpenThreads::Mutex> sl(mutex);
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

    
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);
    
    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();
    
    parallel_for_(Range(0, nMarkers),
                  SinglePoseEstimationParallel(ids[captureIdx], _corners, _cameraMatrix,
                                               _distCoeffs, rvecs, tvecs));
}


int ARUCOPlugin::loadPattern(const char* p)
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
    
// ----------------------------------------------------------------------------
COVERPLUGIN(ARUCOPlugin)
