/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ARUCO_PLUGIN_H
#define _ARUCO_PLUGIN_H

#include "Marker.h"
#include "../common/ARCamera.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRCollaboration.h>

#include <cover/MarkerTracking.h>

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>

#include <cover/coVRPlugin.h>

#include <opencv2/core/version.hpp>

#if (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7) || (CV_VERSION_MAJOR >= 5)
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#endif

#include <OpenVRUI/coMenu.h>

#include <util/coTabletUIMessages.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/Action.h>
#include <cover/ui/EditField.h>
#include <cover/ui/SelectionList.h>

using namespace covise;
using namespace opencover;

namespace opencover { namespace ui {
class Action;
class SelectionList;
}}

class ARUCOPlugin : public opencover::coVRPlugin,
                    public opencover::MarkerTrackingInterface,
                    public ui::Owner,
                    public opencover::coTUIListener
{
public:
    ARUCOPlugin();
    virtual ~ARUCOPlugin();

    bool init() override;
    void preFrame() override;
    bool update() override;
    bool destroy() override;
    int loadPattern(const char* p);

protected:
    ARCamera camera;
    cv::Mat image[3]; // for triple buffering
    cv::Mat displayImage[3]; // rendered marker frames published to OpenCOVER
    int displayIdx = 0, readyIdx = 1, captureIdx = 2;

    std::vector<int> ids[3];
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;

    cv::aruco::Dictionary dictionary;
    cv::Ptr<cv::aruco::ArucoDetector> detector;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

private:

    bool bDrawDetMarker;
    bool bDrawRejMarker;

    ui::Menu* uiMenu = nullptr;
    ui::Button* uiBtnDrawDetMarker = nullptr;
    ui::Button* uiBtnDrawRejMarker = nullptr;
    ui::Action* uiBtnCalib = nullptr;
    ui::EditField* uiManualFocus = nullptr;
    ui::Action* uiBtnSaveFocus = nullptr;
    opencover::ui::Action *uiBtnDetectCamera = nullptr;
    opencover::ui::SelectionList *uiCameraDevices = nullptr;

    int markerSize; // default marker size

    coTUISlider *bitrateSlider;

    //void captureRightVideo();
    int msgQueue;
    unsigned char *dataPtr = nullptr;
    int marker_num = 0;
    bool flipBufferH;
    bool flipBufferV;

    void adjustScreen();

    osg::Matrix getMat(const MarkerTrackingMarker *marker) override;
    bool isVisible(const MarkerTrackingMarker *marker) override;
    void updateMarkerParams() override;
    void createUnconfiguredTrackedMarkers() override;


    cv::Mat imageCopy;
    float markerLength;

private:
    coVRCollaboration::SyncMode syncmode;
    std::vector<MultiMarker> m_markers;
    std::mutex markerMutex;

    void estimatePoseMarker(const std::vector<std::vector<cv::Point2f>> &corners, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs);

    void initUI();
    bool initAR();

    std::mutex opencvMutex;
    std::thread opencvThread;
    bool opencvRunning = false;
    int m_requestedFocus = 35;
    bool m_focusChangeRequested = false;

    void opencvLoop();
    void requestManualFocus(int focus);
    void applyAndSaveManualFocus();

    // charuco board callibration
    // create charuco board object
    cv::Ptr<cv::aruco::CharucoBoard> charucoboard;
    cv::Ptr<cv::aruco::CharucoDetector>  charucoDetector;
    //Ptr<aruco::Board> board;

    void startCalibration();

    opencover::coTUITab *arucoTab = nullptr;
    opencover::coTUIButton *detectCameraButton = nullptr;
    opencover::coTUIComboBox *cameraDeviceCombo = nullptr;

    std::vector<int> m_cameraDeviceIds;
    int m_requestedDevice = -1;

    void detectCameras();
    void requestCameraSwitch(int deviceId);
    void switchCamera(int deviceId);
};
#endif
