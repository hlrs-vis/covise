/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ARUCO_PLUGIN_H
#define _ARUCO_PLUGIN_H
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
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRCollaboration.h>

#include <cover/ARToolKit.h>

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>

#include <cover/coVRPlugin.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/aruco.hpp>

#include <OpenVRUI/coMenu.h>

#include <util/coTabletUIMessages.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>

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
    
protected:
    cv::VideoCapture inputVideo;
    cv::Mat image;
    
    cv::Mat matCameraMatrix;
    cv::Mat matDistCoefs;

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
    
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams;

    vrui::coSubMenuItem *arMenuEntry;
    
private:

    bool bDrawDetMarker;
    bool bDrawRejMarker;

    ui::Menu* uiMenu = nullptr;
    ui::Button* uiBtnDrawDetMarker = nullptr;
    ui::Button* uiBtnDrawRejMarker = nullptr;
    
   
    


    bool doCalibrate;
    bool calibrated;
    int calibCount;

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
};
#endif
