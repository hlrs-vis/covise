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
#include "MatrixUtil.h"

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/RenderObject.h>
#include <cover/MarkerTracking.h>
#include <config/CoviseConfig.h>
#include <cover/coVRConfig.h>
#include "../common/RemoteAR.h"
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cover/coVRFileManager.h>


#include <cover/coTabletUI.h>
#include <cover/coVRPlugin.h>
#include <cover/coInteractor.h>
#include <util/unixcompat.h>
#include <util/environment.h>

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

#if (CV_MAJOR_VERSION < 3 || (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION < 1))
#error "At least OpenCV version 3.1 is required"
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
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ARUCO", cover->ui)
{
    OpenGLToOSGMatrix.makeRotate(M_PI / -2.0, 1, 0, 0);
    OSGToOpenGLMatrix.makeRotate(M_PI / 2.0, 1, 0, 0);
}

ARUCOPlugin::~ARUCOPlugin()
{
    delete MarkerTracking::instance()->remoteAR;
    MarkerTracking::instance()->remoteAR = nullptr;
    MarkerTracking::instance()->arInterface = nullptr;
    MarkerTracking::instance()->running = false;
   
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

void ARUCOPlugin::initCamera(int selectedDevice, bool &exists)
{
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
    std::cerr << "   current size  = " << width << "x" << height << std::endl;
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
            std::cerr << "WARNING: could not set capture frame size" << std::endl;
            std::cerr << "   new size  = " << width << "x" << height << std::endl;
        }
    }
    
    cv::Mat image;
    std::cerr << "capture first frame after resetting camera" << std::endl;
    try
    {
        inputVideo >> image;
    }
    catch (cv::Exception &ex)
    {
        std::cerr << "OpenCV exception: " << ex.what() << std::endl;
    }
    
    MarkerTracking::instance()->running = true;
    MarkerTracking::instance()->videoMode = GL_BGR;
    MarkerTracking::instance()->videoDepth = 3;
    MarkerTracking::instance()->videoWidth = image.cols;
    MarkerTracking::instance()->videoHeight = image.rows;

    std::cerr << "Capturing " << image.cols << "x" << image.rows << " pixels" << std::endl;

    // load calib data from file
    std::cerr << "loading calibration data from file " << calibrationFilename << std::endl;

    cv::FileStorage fs;
    try
    {
        fs.open(calibrationFilename, cv::FileStorage::READ);
    }
    catch (cv::Exception e)
    {
    }
    if (fs.isOpened())
    {
        fs["camera_matrix"] >> matCameraMatrix;
        fs["dist_coefs"] >> matDistCoefs;

        std::cerr << "camera matrix: " << std::endl;
        std::cerr << matCameraMatrix << std::endl;
        std::cerr << "dist coefs: " << std::endl;
        std::cerr << matCameraMatrix << std::endl;
    }
    else
    {
        std::cerr << "failed to open camera calibration file " << calibrationFilename << std::endl;
        std::cerr << "trying to guess a calibration ... " << std::endl;

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

        std::cerr << "camera matrix: " << std::endl;
        std::cerr << matCameraMatrix << std::endl;
        std::cerr << "dist coefs: " << std::endl;
        std::cerr << matDistCoefs << std::endl;
    }
}

bool ARUCOPlugin::initAR()
{
    MarkerTracking::instance()->arInterface = this;
    MarkerTracking::instance()->remoteAR = NULL;

    doCalibrate = false;
    calibrated = false;
    calibCount = 0;

    if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.Capture", false))
    {

        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorRight", false))
            MarkerTracking::instance()->videoMirrorRight = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorLeft", false))
            MarkerTracking::instance()->videoMirrorLeft = true;

        MarkerTracking::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferV", true);
        std::string VideoDevice = coCoviseConfig::getEntry("value", "COVER.Plugin.ARUCO.VideoDevice", "0");

        calibrationFilename = coCoviseConfig::getEntry("value", "COVER.Plugin.ARUCO.CameraCalibrationFile", "/data/aruco/cameras/default.yaml");

        xsize = coCoviseConfig::getInt("width", "COVER.Plugin.ARUCO.VideoDevice", 640);
        ysize = coCoviseConfig::getInt("height", "COVER.Plugin.ARUCO.VideoDevice", 480);

        int dictionaryId = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.DictionaryID", 7); // 16 = ARUCO_DEFAULT

        if (!detectorParams)
        {
#if (CV_VERSION_MAJOR < 4)
            detectorParams = aruco::DetectorParameters::create();
#else
            detectorParams = new aruco::DetectorParameters();
#endif
        }

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
        if (detector)
        {
            detector.reset();
        }
        detector = new cv::aruco::ArucoDetector(dictionary,*detectorParams);
#endif
        msgQueue = -1;
       
        int selectedDevice = atoi(VideoDevice.c_str());
        bool exists = false;
        // FIXME: this leaks memory if plugin is reloaded
        if (coCoviseConfig::isOn("hw_transforms", "COVER.Plugin.ARUCO.VideoDevice", false, &exists))
            putenv(strdup("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=1"));
        else
            putenv(strdup("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0")); // this disables slow camera initialization, only enable if necessary

#if CV_VERSION_MAJOR > 3 || (CV_VERSION_MAJOR==3 && CV_VERSION_MINOR>1)
        for (int cap: {CAP_V4L2, CAP_ANY})
            if (inputVideo.open(selectedDevice, cap))
                break;
#else
        inputVideo.open(selectedDevice);
#endif
        
        if (inputVideo.isOpened())
        {
            std::cerr << "capture device: device " << selectedDevice << " is open" << endl;
            initCamera(selectedDevice, exists);
        }
        else
        {
            std::cerr << "capture device: failed to open " << selectedDevice << std::endl;
            return false;
        }
        adjustScreen();
        /* 
        how to generate callibration patterns:
         cd src/opencv_contrib/modules/aruco/misc/pattern_generator/
          python MarkerPrinter.py --charuco --file "./charuco.pdf" --dictionary DICT_5X5_1000 --size_x 16 --size_y 9 --square_length 0.09 --marker_length 0.07 --border_b
its 1
        */
        int squaresX = coCoviseConfig::getInt("xSize", "COVER.Plugin.ARUCO.Callibration", 16);
        int squaresY = coCoviseConfig::getInt("ysize", "COVER.Plugin.ARUCO.Callibration", 9);
        float squareLength = coCoviseConfig::getInt("squareSize", "COVER.Plugin.ARUCO.Callibration", 18);
        float markerLength = coCoviseConfig::getInt("markerSize", "COVER.Plugin.ARUCO.Callibration", 14);
        charucoboard = new aruco::CharucoBoard(Size(squaresX, squaresY), squareLength, markerLength, dictionary);
        cv::aruco::CharucoParameters cp;
        cv::aruco::RefineParameters rp;
        
        charucoDetector = new cv::aruco::CharucoDetector(*charucoboard, cp, *detectorParams,rp);
    }
    MarkerTracking::instance()->remoteAR = new RemoteAR();
    return true;
}

void ARUCOPlugin::initUI()
{
    uiMenu = new ui::Menu("uiMenu", this);
    uiMenu->setText("ARUCO");

    uiBtnDrawDetMarker = new ui::Button(uiMenu, "uiBtnDrawDetMarker");
    uiBtnDrawDetMarker->setText("Draw detected markers");
    uiBtnDrawDetMarker->setEnabled(true);
    uiBtnDrawDetMarker->setState(bDrawDetMarker);
    uiBtnDrawDetMarker->setCallback([this](bool state)
    {
        bDrawDetMarker = state;
    });
    
    uiBtnDrawRejMarker = new ui::Button(uiMenu, "uiBtnDrawRejMarker");
    uiBtnDrawRejMarker->setText("Draw rejected markers");
    uiBtnDrawRejMarker->setEnabled(true);
    uiBtnDrawRejMarker->setState(bDrawRejMarker);
    uiBtnDrawRejMarker->setCallback([this](bool state)
    {
        bDrawRejMarker = state;
    });

    uiBtnCalib = new ui::Action(uiMenu, "calibrate");
    uiBtnCalib->setText("Calibrate camera");
    uiBtnCalib->setEnabled(true);
    uiBtnCalib->setCallback([this]()
    {
        startCallibration();
    });
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
bool ARUCOPlugin::init()
{
#ifdef ARUCO_DEBUG
    std::cerr << "ARUCOPlugin::init()" << std::endl;
#endif

    // class init
    bDrawDetMarker = true;
    bDrawRejMarker = false;
    
    // ui init
    initUI();

    // ar init
    if (!initAR())
        return false;

    opencvRunning = true;
    opencvThread = std::thread(
        [this]()
        {
            for (;;)
            {
                try
                {
                    if (opencvRunning)
                    {
                        opencvLoop();
                        if (!opencvRunning)
                            return;
                    }
                    else
                    {
                        usleep(10000);
                    }
                }
                catch (const cv::Exception &ex)
                {
                    std::unique_lock<std::mutex> guard(opencvMutex);
                    opencvRunning = false;
                    //MarkerTracking::instance()->running = false;
                    guard.unlock();
                    std::cerr << "error: unhandled OpenCV exception " << ex.what()
                              << ", trying to reinitialize ARUCO plugin" << std::endl;
                }
            }
        });

    return true;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
bool ARUCOPlugin::destroy()
{
#ifdef ARUCO_DEBUG
    std::cerr << "ARUCOPlugin::destroy()" << std::endl;
#endif

    std::unique_lock<std::mutex> guard(opencvMutex);
    opencvRunning = false;
    guard.unlock();
    if (opencvThread.joinable())
        opencvThread.join();

    delete uiMenu;
    MarkerTracking::instance()->videoData = nullptr;
    return true;
}

void ARUCOPlugin::createUnconfiguredTrackedMarkers()
{
    for(auto id : ids[captureIdx] )
    {
        auto m = findMarker(m_markers, id);
        if(!m)
        {
            auto idString =  std::to_string(id);
            MarkerTracking::instance()->getOrCreateMarker(idString, idString, (double)50.0, osg::Matrix::identity(), false);
        }
    }
}

void ARUCOPlugin::preFrame()
{
    if (MarkerTracking::instance()->running)
    {
        std::unique_lock<std::mutex> guard(opencvMutex);
        if (!opencvRunning)
        {
            if (initAR())
            {
                opencvRunning = true;
            }
            return;
        }
        guard.unlock();

#ifndef _WIN32
        if (msgQueue > 0)
        {
            struct myMsgbuf message;
            // allow right capture process to continue
            message.mtype = 1;
            msgsnd(msgQueue, &message, 1, 0);
        }
#endif

        guard.lock();
        displayIdx = readyIdx;

        MarkerTracking::instance()->videoData = (unsigned char *)image[displayIdx].ptr();
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

void ARUCOPlugin::calibrate()
{
    std::string text = "Valid Frames: " + std::to_string(allIds.size());
    cv::putText(image[captureIdx], text, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX,1.0, cv::Scalar(255, 255, 255));

    if (ids[captureIdx].size() > 25 && (cover->frameTime() - lastCalibCapture)>1.0)
    {
        lastCalibCapture = cover->frameTime();
        Mat currentCharucoCorners, currentCharucoIds;
        charucoDetector->detectBoard(image[captureIdx], currentCharucoCorners, currentCharucoIds, corners, ids[captureIdx]);

        if (currentCharucoCorners.total() > 0)
            aruco::drawDetectedCornersCharuco(image[captureIdx], currentCharucoCorners, currentCharucoIds);

        allCorners.push_back(corners);
        allIds.push_back(ids[captureIdx]);
        allImgs.push_back(image[captureIdx]);
        imgSize = image[captureIdx].size();
    }
    if (allIds.size() > 15)
    {
        cv::aruco::CharucoParameters cp;
        cv::aruco::RefineParameters rp;
        Mat cameraMatrix, distCoeffs;
        vector< Mat > rvecs, tvecs;
        double repError;

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
                cerr << "Rep Error: " << repError << endl;
                cerr << "Rep Error Aruco: " << arucoRepErr << endl;
                cerr << "Calibration saved to " << calibrationFilename << endl;
            }
        }

        doCalibrate = false;
        allImgs.clear();
        allIds.clear();
        allCorners.clear();
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

            // detect markers and estimate pose
#if( CV_VERSION_MAJOR >= 4)
            detector->detectMarkers(image[captureIdx], corners, ids[captureIdx],  rejected);
            assert(corners.size() == ids[captureIdx].size());
#else
            cv::aruco::detectMarkers(image[captureIdx], dictionary, corners, ids[captureIdx], detectorParams, rejected);
#endif
            if(ids[captureIdx].size() > 0)
            {
                try
                {
                    std::lock_guard<std::mutex> g(markerMutex);
                    estimatePoseMarker(corners, matCameraMatrix,
                                            matDistCoefs);
                }
                catch (cv::Exception &ex)
                {
                    std::cerr << "OpenCV exception: " << ex.what() << std::endl;
                    std::cerr << "Camera might need callibration: "  << std::endl;
                }
            }
            if (doCalibrate)
                calibrate();

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
                aruco::drawDetectedMarkers(image[captureIdx], rejected, noArray(), Scalar(100, 0, 255));
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
    return MarkerTracking::instance()->running;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
bool ARUCOPlugin::isVisible(const MarkerTrackingMarker *marker)
{
    auto m = findMarker(m_markers, marker);
    if(m)
        return std::find(ids[displayIdx].begin(), ids[displayIdx].end(), m->markerId) !=  ids[displayIdx].end();
    return false;
}

osg::Matrix ARUCOPlugin::getMat(const MarkerTrackingMarker *marker)
{
    for(auto &multiMarker : m_markers)
    {
        for(auto &m : multiMarker)
        {
            if(m.markerTrackingMarker == marker && m.getCapturedAt(ids[displayIdx]) != -1)
            {
                auto mat = cvToOsgMat(m.cameraRot(displayIdx), m.cameraTrans(displayIdx));
                mat = marker->getOffset() * mat;
                return mat;
            }
        }
    }
    return osg::Matrix::identity();
}

// ----------------------------------------------------------------------------
//! set new marker sizes according to tabletUI
// ----------------------------------------------------------------------------
void ARUCOPlugin::updateMarkerParams()
{
    std::map<int, std::vector<ArucoMarker>> markerSets;
    for(const auto &m : MarkerTracking::instance()->markers)
    {
        const auto marker = m.second.get();
        markerSets[marker->getMarkerGroup()].emplace_back(ArucoMarker{marker});
    }
    std::lock_guard<std::mutex> g(markerMutex);
    m_markers.clear();
    for(auto &set : markerSets)
    {
        if(set.first == noMarkerGroup) //treat these markers separetely
        {
            for(auto &marker : set.second)
            {
                MultiMarker mm;
                mm.emplace_back(std::move(marker));
                m_markers.emplace_back(std::move(mm));
            }
        }
        else
            m_markers.emplace_back(std::move(set.second));
    }
}

void ARUCOPlugin::adjustScreen()
{
#ifdef ARUCO_DEBUG
    std::cerr << "ARUCOPlugin::adjustScreen()" << std::endl;
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

OpenThreads::Mutex mutex;
// ----------------------------------------------------------------------------
//! ParallelLoopBody class for the parallelization of the markers pose estimation
// ----------------------------------------------------------------------------
class PoseEstimationParallel : public ParallelLoopBody
{
public:
    PoseEstimationParallel(std::vector<MultiMarkerPtr> &&markers, const std::vector<std::vector<Point2f>> &corners,
                                 const Mat &cameraMatrix, const Mat &distCoeffs, int captureIdx)
        : captureIdx(captureIdx)
        , markers(markers)
        , corners(corners)
        , cameraMatrix(cameraMatrix)
        , distCoeffs(distCoeffs)
        {}

    void operator()(const Range &range) const
    {
        for(int i = range.start; i < range.end; i++)
        {
            std::vector<Point2f> imageCorners;
            std::vector<cv::Vec3d> worldCorners; //4 corners per marker
            for(const auto marker : markers[i])
            {
                imageCorners.insert(imageCorners.end(), corners[marker->capturedAt].begin(), corners[marker->capturedAt].end());
                worldCorners.insert(worldCorners.end(), marker->corners.begin(), marker->corners.end());
            }
            cv::Vec3d rot, trans;
            rot = markers[i][0]->cameraRot(markers[i][0]->lastCaptureIndex);
            trans = markers[i][0]->cameraTrans(markers[i][0]->lastCaptureIndex);
            solvePnP(worldCorners, imageCorners, cameraMatrix, distCoeffs,  rot, trans, false, SOLVEPNP_ITERATIVE);

            for (auto marker : markers[i])
            {
                marker->setCamera(rot, trans, captureIdx);
                marker->lastCaptureIndex = captureIdx;
            }
        }
    }

private:
    PoseEstimationParallel &operator=(const PoseEstimationParallel &); // to quiet MSVC
    int captureIdx;
    const std::vector<std::vector<Point2f>> &corners;
    const Mat &cameraMatrix, &distCoeffs;
    std::vector<MultiMarkerPtr> markers; //as long as every instance only writes at its MultiMarker mutable should work
};

std::vector<MultiMarkerPtr> getTrackedMarkers(std::vector<MultiMarker> &markers, const std::vector<int> &trackedIds)
{
    std::vector<MultiMarkerPtr> trackedMarkers;
    for(auto &multiMarker : markers)
    {
        MultiMarkerPtr trackedMarker;
        for(auto &marker : multiMarker)
        {
            if(marker.getCapturedAt(trackedIds) >= 0)
                trackedMarker.push_back(&marker);
        }
        if(!trackedMarker.empty())
            trackedMarkers.push_back(trackedMarker);
    }
    return trackedMarkers;
}

void ARUCOPlugin::estimatePoseMarker(const std::vector<std::vector<Point2f>> &corners, const Mat &cameraMatrix, const Mat &distCoeffs)
{
    auto trackedMarkers = getTrackedMarkers(m_markers, ids[captureIdx]);
    parallel_for_(Range(0, trackedMarkers.size()),
                  PoseEstimationParallel(std::move(trackedMarkers), corners, cameraMatrix, distCoeffs, captureIdx));
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
