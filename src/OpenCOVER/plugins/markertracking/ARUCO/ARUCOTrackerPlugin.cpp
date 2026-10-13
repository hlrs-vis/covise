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
#include <config/coConfig.h>
#include <cover/coVRConfig.h>
#include "../common/RemoteAR.h"
#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/calib3d.hpp>
#else
#include <opencv2/calib3d/calib3d.hpp>
#endif

#include <opencv2/imgproc.hpp>
#include <cover/coVRFileManager.h>

#include <cover/coVRPlugin.h>
#include <cover/coInteractor.h>
#include <cover/ui/View.h>
#include <util/unixcompat.h>
#include <util/environment.h>

#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

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

    m_requestedDevice = -1;
    captureIdx = 0;
    readyIdx = 0;
    displayIdx = 0;
    camera.setCalibrationFilename(coCoviseConfig::getEntry("value", "COVER.Plugin.ARUCO.CameraCalibrationFile", "~/cameras/default.yaml"));
}

ARUCOPlugin::~ARUCOPlugin()
{
    delete MarkerTracking::instance()->remoteAR;
    MarkerTracking::instance()->remoteAR = nullptr;
    MarkerTracking::instance()->arInterface = nullptr;
    MarkerTracking::instance()->running = false;
   
    if(camera.isOpened())
    {
        camera.close();

#ifndef _WIN32
        if (msgQueue >= 0)
        {
            msgctl(msgQueue, IPC_RMID, NULL);
        }
#endif
    }
}

bool ARUCOPlugin::initAR()
{
    std::cerr << "ARUCO build fingerprint: " << __FILE__ << " " << __DATE__ << " " << __TIME__ << std::endl;

    MarkerTracking::instance()->arInterface = this;
    MarkerTracking::instance()->remoteAR = NULL;

    if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.Capture", false))
    {

        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorRight", false))
            MarkerTracking::instance()->videoMirrorRight = true;
        if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.MirrorLeft", false))
            MarkerTracking::instance()->videoMirrorLeft = true;

        MarkerTracking::instance()->flipH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipHorizontal", false);
        flipBufferH = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferH", false);
        flipBufferV = coCoviseConfig::isOn("COVER.Plugin.ARUCO.FlipBufferV", true);
        camera.setCalibrationFilename(coCoviseConfig::getEntry(
            "value",
            "COVER.Plugin.ARUCO.CameraCalibrationFile",
            "/home/rosba/covise/cameras/default.yaml"));
    }

    // --- ADD: ARUCO detector setup ---
    try
    {
        // ChArUco config from XML
        const int dictId = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.DictionaryID", 7);
        const int squaresX = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.SquaresX", 5);
        const int squaresY = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.SquaresY", 4);
        float squareLength = coCoviseConfig::getFloat("value", "COVER.Plugin.ARUCO.SquareLength", 0.015f);
        float markerLength = coCoviseConfig::getFloat("value", "COVER.Plugin.ARUCO.MarkerSquare", 0.009f);

        std::cerr << "ARUCO cfg: dict=" << dictId
                  << " squares=" << squaresX << "x" << squaresY
                  << " squareLength=" << squareLength
                  << " markerLength=" << markerLength << std::endl;

        switch (dictId)
        {
        case 0: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50); break;
        case 1: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100); break;
        case 2: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250); break;
        case 3: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000); break;
        case 4: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50); break;   // default
        case 5: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100); break;
        case 6: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250); break;
        case 7: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000); break;
        case 8: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50); break;
        case 9: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100); break;
        case 10: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250); break;
        case 11: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000); break;
        case 12: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50); break;
        case 13: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_100); break;
        case 14: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250); break;
        case 15: dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000); break;
        default:
            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
            break;
        }

#if CV_VERSION_MAJOR >= 5 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
        detectorParams = cv::makePtr<cv::aruco::DetectorParameters>();
#else
        detectorParams = cv::aruco::DetectorParameters::create();
#endif

detectorParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

#if CV_VERSION_MAJOR >= 5 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
        detector = cv::makePtr<cv::aruco::ArucoDetector>(dictionary, *detectorParams);
#else
        detector = cv::makePtr<cv::aruco::ArucoDetector>(*dictionary, *detectorParams);
#endif

#if CV_VERSION_MAJOR >= 5 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
        charucoboard = cv::makePtr<cv::aruco::CharucoBoard>(
            cv::Size(squaresX, squaresY), squareLength, markerLength, dictionary);

        // for boards printed with older OpenCV charuco pattern
        const bool legacyPattern = coCoviseConfig::isOn("COVER.Plugin.ARUCO.LegacyPattern", true);
        std::cerr << "LegacyPattern=" << legacyPattern << std::endl;
        charucoboard->setLegacyPattern(legacyPattern);

        cv::aruco::CharucoParameters cparams;
        cparams.tryRefineMarkers = true;
        cparams.minMarkers = 2; // less strict
        charucoDetector = cv::makePtr<cv::aruco::CharucoDetector>(*charucoboard, cparams);
#else
        charucoboard = cv::aruco::CharucoBoard::create(
            squaresX, squaresY, squareLength, markerLength, dictionary);
        charucoDetector = cv::makePtr<cv::aruco::CharucoDetector>(*charucoboard);
#endif
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "ARUCO init failed: " << e.what() << std::endl;
        return false;
    }

    if (!detectorParams || !detector || !charucoboard || !charucoDetector)
    {
        std::cerr << "ARUCO init failed: detector/board objects are null" << std::endl;
        return false;
    }
    // --- END ADD ---

    MarkerTracking::instance()->remoteAR = new RemoteAR();
    return true;
}

void ARUCOPlugin::initUI()
{
    uiMenu = new ui::Menu("uiMenu", this);
    uiMenu->setText("ARUCO");
    uiMenu->setVisible(true, ui::View::Tablet);

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
        startCalibration();
    });

    // add camera controls
    uiBtnDetectCamera = new ui::Action(uiMenu, "detectCamera");
    uiBtnDetectCamera->setText("Detect Cameras");
    uiBtnDetectCamera->setCallback([this]() { detectCameras(); });

    uiCameraDevices = new ui::SelectionList(uiMenu, "cameraDevices");
    uiCameraDevices->setText("Camera Device");
    uiCameraDevices->setCallback([this](int idx) {
        if (idx >= 0 && idx < static_cast<int>(m_cameraDeviceIds.size()))
            requestCameraSwitch(m_cameraDeviceIds[idx]);
    });

    const int focus = coCoviseConfig::getInt("Focus", "COVER.Plugin.ARUCO", 35);
    uiManualFocus = new ui::EditField(uiMenu, "manualFocus");
    uiManualFocus->setText("Manual focus");
    uiManualFocus->setValue(focus);

    uiBtnSaveFocus = new ui::Action(uiMenu, "saveFocus");
    uiBtnSaveFocus->setText("Apply and save focus");
    uiBtnSaveFocus->setCallback([this]() { applyAndSaveManualFocus(); });
}

void ARUCOPlugin::applyAndSaveManualFocus()
{
    const std::string value = uiManualFocus->value();
    size_t parsedCharacters = 0;
    int requestedFocus;
    try
    {
        requestedFocus = std::stoi(value, &parsedCharacters);
    }
    catch (const std::exception &)
    {
        std::cerr << "ARUCO: invalid manual focus value '" << value << "'" << std::endl;
        return;
    }
    if (value.find_first_not_of(" \t", parsedCharacters) != std::string::npos || requestedFocus < 0)
    {
        std::cerr << "ARUCO: manual focus must be a non-negative integer" << std::endl;
        return;
    }

    auto *config = coConfig::getInstance();
    const bool wasAdminMode = config->isAdminMode();
    config->setAdminMode(true);
    if (!config->isAdminMode())
    {
        std::cerr << "ARUCO: cannot enable COCONFIG admin mode; the global config is not group-writable" << std::endl;
        config->setAdminMode(wasAdminMode);
        requestManualFocus(requestedFocus);
        return;
    }

    config->setValueInConfig("value", std::to_string(requestedFocus),
                         "COVER.Plugin.ARUCO.Focus", "global");
    const bool saved = config->save();
    config->setAdminMode(wasAdminMode);
    if (!saved)
    {
        std::cerr << "ARUCO: failed to save manual focus " << requestedFocus << std::endl;
    }
    else
    {
        std::cerr << "ARUCO: saved manual focus " << requestedFocus << " to the global COCONFIG file" << std::endl;
    }
    requestManualFocus(requestedFocus);
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

    // init AR first (sets calibration path and detectors)
    if (!initAR())
        return false;

    // detect cameras after AR init
    detectCameras();

    const int preferredDevice = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.VideoDevice", -1);
    if (std::find(m_cameraDeviceIds.begin(), m_cameraDeviceIds.end(), preferredDevice) != m_cameraDeviceIds.end())
        switchCamera(preferredDevice);
    else if (!m_cameraDeviceIds.empty())
    {
        std::cerr << "ARUCO: configured camera " << preferredDevice
                  << " is unavailable; opening the first detected camera" << std::endl;
        switchCamera(m_cameraDeviceIds.front());
    }

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
        MarkerTracking::instance()->videoData =
            displayImage[displayIdx].empty() ? nullptr : (unsigned char *)displayImage[displayIdx].ptr();
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void ARUCOPlugin::opencvLoop()
{
    static int zeroMarkerStreak = 0;

    // ADD THIS:
    const int squaresX = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.SquaresX", 8);
    const int squaresY = coCoviseConfig::getInt("value", "COVER.Plugin.ARUCO.SquaresY", 11);
    const float squareLength = coCoviseConfig::getFloat("value", "COVER.Plugin.ARUCO.SquareLength", 0.015f);
    const float markerLength = coCoviseConfig::getFloat("value", "COVER.Plugin.ARUCO.Marhttps://github.com/opencv/opencv/blob/5.x/samples/cpp/calibration.cppkerSquare", 0.011f);
    const bool legacyPattern = coCoviseConfig::isOn("COVER.Plugin.ARUCO.LegacyPattern", true);

    auto setDictionaryById = [this, squaresX, squaresY, squareLength, markerLength, legacyPattern](int dictId) -> bool
    {

        if (!detectorParams)
            return false;

#if CV_VERSION_MAJOR >= 5 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
        detector = cv::makePtr<cv::aruco::ArucoDetector>(dictionary, *detectorParams);

        charucoboard = cv::makePtr<cv::aruco::CharucoBoard>(
            cv::Size(squaresX, squaresY), squareLength, markerLength, dictionary);
        charucoboard->setLegacyPattern(legacyPattern);

        cv::aruco::CharucoParameters cparams;
        cparams.tryRefineMarkers = true;
        cparams.minMarkers = 2;
        charucoDetector = cv::makePtr<cv::aruco::CharucoDetector>(*charucoboard, cparams);
#else
        detector = cv::makePtr<cv::aruco::ArucoDetector>(*dictionary, *detectorParams);
        charucoboard = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
        charucoDetector = cv::makePtr<cv::aruco::CharucoDetector>(*charucoboard);
#endif
        return true;
    };

    for (;;)
    {
        int req = -1;
        {
            std::lock_guard<std::mutex> g(opencvMutex);
            if (m_requestedDevice >= 0)
            {
                req = m_requestedDevice;
                m_requestedDevice = -1;
            }
        }
        if (req >= 0)
            switchCamera(req);

        int requestedFocus = 0;
        bool applyFocus = false;
        {
            std::lock_guard<std::mutex> g(opencvMutex);
            if (camera.isOpened() && m_focusChangeRequested)
            {
                requestedFocus = m_requestedFocus;
                m_focusChangeRequested = false;
                applyFocus = true;
            }
        }
        if (applyFocus)
            camera.setManualFocus(requestedFocus);

        std::unique_lock<std::mutex> guard(opencvMutex);
        if (!opencvRunning)
            return;
        guard.unlock();

        if (camera.isOpened())
        {
            guard.lock();
            while (captureIdx == displayIdx || captureIdx == readyIdx)
                captureIdx = (captureIdx + 1) % 3;
            guard.unlock();

            camera.read(image[captureIdx]);
            if (image[captureIdx].empty())
            {
                usleep(5000);
                continue;
            }

            // prevent null deref crash
            if (!detector)
            {
                std::cerr << "ARUCO: detector is null, skipping frame" << std::endl;
                usleep(5000);
                continue;
            }

            if (charucoDetector && charucoboard && camera.consumeDeferredCalibrationRequest())
            {
                startCalibration();
            }


            ids[captureIdx].clear();
            corners.clear();
            rejected.clear();

            cv::Mat gray;
            if (image[captureIdx].channels() == 3)
                cv::cvtColor(image[captureIdx], gray, cv::COLOR_BGR2GRAY);
            else
                gray = image[captureIdx];

            ids[captureIdx].clear();
            corners.clear();
            rejected.clear();

#if (CV_VERSION_MAJOR >= 4)
            detector->detectMarkers(gray, corners, ids[captureIdx], rejected);
#else
            cv::aruco::detectMarkers(gray, dictionary, corners, ids[captureIdx], detectorParams, rejected);
#endif

            if (ids[captureIdx].empty())
            {
                ++zeroMarkerStreak;
            
                if (zeroMarkerStreak >= 3) // temporary lower threshold from 10 to 3
                {

                    #if (CV_VERSION_MAJOR >= 4)
                                        detector->detectMarkers(gray, corners, ids[captureIdx], rejected);
                    #else
                                        cv::aruco::detectMarkers(gray, dictionary, corners, ids[captureIdx], detectorParams, rejected);
                    #endif
                    zeroMarkerStreak = 0;
                }
            }
            else{
                zeroMarkerStreak = 0;
            }

            //std::cerr << "ARUCO: detected markers number = " << ids[captureIdx].size() << std::endl;
            
            // Copy image otherwise it changes the original image when drawing markers
            // (this would be bad for calibration, as the corners would be drawn on the image and then used for calibration)
            displayImage[captureIdx] = image[captureIdx].clone();
            // draw detected/rejected markers on the camera image
             if (bDrawDetMarker && !displayImage[captureIdx].empty())
            {
                cv::aruco::drawDetectedMarkers(displayImage[captureIdx], corners, ids[captureIdx]);
            }
            if (bDrawRejMarker && !rejected.empty())
            {
                cv::aruco::drawDetectedMarkers(displayImage[captureIdx], rejected, cv::noArray(), cv::Scalar(100, 0, 255));
            } 

            //std::cerr << "ARUCO: detected corners = " << corners.size() << std::endl;

            // estimate pose for all markers in image
            if (ids[captureIdx].size() > 0)
            {
                const bool intrinsicsOK = camera.hasIntrinsics();
                if (intrinsicsOK)
                {
                    try
                    {
                        std::lock_guard<std::mutex> g(markerMutex);
                        estimatePoseMarker(corners, camera.cameraMatrix(), camera.distortionCoefficients());
                    }
                    catch (const cv::Exception &ex)
                    {
                        std::cerr << "OpenCV exception: " << ex.what() << std::endl;
                        std::cerr << "Camera might need calibration" << std::endl;
                    }
                }
}

            if (camera.isCalibrating())
                camera.calibrateFrame(image[captureIdx], corners, ids[captureIdx], charucoDetector, charucoboard);

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

void ARUCOPlugin::startCalibration()
{
    camera.startCalibration();
    std::cerr << "ARUCO: calibration started. Please show/move ChArUco board in front of camera." << std::endl;
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

    if (!camera.hasIntrinsics())
        return;

    if (coCoviseConfig::isOn("COVER.Plugin.ARUCO.AdjustScreenParameters", true))
    {

        osg::Vec3 viewPos;

        float sxsize = camera.width();
        float sysize = camera.height();

        float d;

        d = camera.cameraMatrix().at<double>(0, 0);
        sysize = ((double)camera.height() / camera.cameraMatrix().at<double>(1, 1)) * d;

        coVRConfig::instance()->screens[0].hsize = sxsize;
        coVRConfig::instance()->screens[0].vsize = sysize;

        viewPos.set(camera.cameraMatrix().at<double>(0, 2) - ((double)camera.width() / 2.0), -d, ((double)camera.height() / 2.0) - camera.cameraMatrix().at<double>(1, 2));

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

void ARUCOPlugin::detectCameras()
{
    const auto devices = ARCamera::availableDevices();
    m_cameraDeviceIds.clear();
    std::vector<std::string> labels;
    for (const auto &device : devices)
    {
        m_cameraDeviceIds.push_back(device.id);
        labels.push_back(device.label);
    }

    if (uiCameraDevices)
    {
        uiCameraDevices->setList(labels);
        uiCameraDevices->setEnabled(!labels.empty());

        // do not auto-trigger a switch through UI callback here
        // uiCameraDevices->select(0, false);
    }
}

void ARUCOPlugin::requestCameraSwitch(int deviceId)
{
    if (std::find(m_cameraDeviceIds.begin(), m_cameraDeviceIds.end(), deviceId) == m_cameraDeviceIds.end())
        return;

    std::lock_guard<std::mutex> g(opencvMutex);
    m_requestedDevice = deviceId;
}

void ARUCOPlugin::requestManualFocus(int focus)
{
    std::lock_guard<std::mutex> g(opencvMutex);
    m_requestedFocus = focus;
    m_focusChangeRequested = true;
}

void ARUCOPlugin::switchCamera(int deviceId)
{
    // reject garbage ids
    if (deviceId < 0 || deviceId > 255)
    {
        std::cerr << "ARUCO: invalid camera id " << deviceId << std::endl;
        return;
    }

    if (camera.open(deviceId))
    {
        std::cerr << "ARUCO: switched to camera device " << deviceId << std::endl;
        auto *config = coConfig::getInstance();
        config->setValue("value", std::to_string(deviceId), "COVER.Plugin.ARUCO.VideoDevice");
        if (!config->save())
            std::cerr << "ARUCO: failed to save default camera device " << deviceId << std::endl;
        MarkerTracking::instance()->running = true;
        MarkerTracking::instance()->videoMode = GL_BGR;
        MarkerTracking::instance()->videoDepth = 3;
        MarkerTracking::instance()->videoWidth = camera.width();
        MarkerTracking::instance()->videoHeight = camera.height();
    }
    else
    {
        std::cerr << "ARUCO: failed to open camera device " << deviceId << std::endl;
        MarkerTracking::instance()->running = false;
    }
}

// ----------------------------------------------------------------------------
COVERPLUGIN(ARUCOPlugin)