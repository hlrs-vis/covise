/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ARUCO_PLUGIN_H
#define _ARUCO_PLUGIN_H

#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRCollaboration.h>

#include <cover/ARToolKit.h>

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>

#include <cover/coVRPlugin.h>

#include <opencv2/videoio/videoio.hpp>
#if( CV_VERSION_MAJOR == 4)
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/aruco.hpp>
#else
#include <opencv2/aruco.hpp>
#endif

#include <OpenVRUI/coMenu.h>

#include <util/coTabletUIMessages.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/Action.h>

using namespace covise;
using namespace opencover;

class ARUCOPlugin : public opencover::coVRPlugin,
                    public opencover::ARToolKitInterface,
                    public ui::Owner
{
public:
    ARUCOPlugin();
    virtual ~ARUCOPlugin();

    virtual bool init();
    virtual void preFrame();
    virtual bool update();
    virtual bool destroy();
    int loadPattern(const char* p);
    
protected:
    cv::VideoCapture inputVideo;
    cv::Mat image[3]; // for triple buffering
    int displayIdx = 0, readyIdx = 1, captureIdx = 2;
    
    cv::Mat matCameraMatrix;
    cv::Mat matDistCoefs;

    std::vector<int> ids[3];
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;
    std::vector<cv::Vec3d> rvecs[3];
    std::vector<cv::Vec3d> tvecs[3];
    
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
    
    int markerSize; // default marker size



    

    coTUISlider *bitrateSlider;

    //void captureRightVideo();
    int msgQueue;
    unsigned char *dataPtr;
    int xsize, ysize;
    int thresh;
    int marker_num;
    bool flipBufferH;
    bool flipBufferV;
    //ARMarkerInfo    *markerInfo;
    osg::Matrix OpenGLToOSGMatrix;
    osg::Matrix OSGToOpenGLMatrix;
    

    void adjustScreen();
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    virtual osg::Matrix getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4]);
    virtual bool isVisible(int);
    virtual void updateMarkerParams();
    std::string calibrationFilename;


    cv::Mat imageCopy;
    float markerLength;


private:
    coVRCollaboration::SyncMode syncmode;
    
    void estimatePoseSingleMarker(cv::InputArrayOfArrays _corners,
                                  cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
                                  cv::OutputArrayOfArrays _rvecs, cv::OutputArrayOfArrays _tvecs);

    std::mutex opencvMutex;
    std::thread opencvThread;
    bool opencvRunning = false;

    void opencvLoop();

    // charuco board callibration
    // create charuco board object
    cv::Ptr<cv::aruco::CharucoBoard> charucoboard;
    cv::Ptr<cv::aruco::CharucoDetector>  charucoDetector;
    //Ptr<aruco::Board> board;

    // collect data from each frame
    vector< vector< vector< cv::Point2f > > > allCorners;
    vector< vector< int > > allIds;
    vector< cv::Mat > allImgs;
    cv::Size imgSize;


    bool doCalibrate;
    bool calibrated;
    int calibCount;
    void startCallibration();
};
#endif
