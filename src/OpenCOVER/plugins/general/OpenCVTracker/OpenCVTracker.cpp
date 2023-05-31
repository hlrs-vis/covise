/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coInteractor.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRShader.h>
#include <cover/ARToolKit.h>

#include <osg/Group>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Switch>
#include <osg/TexGenNode>
#include <osg/Geode>
#include <osg/Point>
#include <osg/ShapeDrawable>

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coMouseButtonInteraction.h>
#include <cover/coBillboard.h>
#include <cover/VRVruiRenderInterface.h>

#include <PluginUtil/PluginMessageTypes.h>

#include "OpenCVTracker.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
string face_cascade_name = "haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

/**
* @function detectAndDisplay
*/
void OpenCVTracker::detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);

        Mat faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 0, 0), 3, 8, 0);
        }
    }
    //-- Show what you got

    fprintf(stderr, "width %d  height %d\n", frame.cols, frame.rows);

    //cout << "frame (csv) = " << format(frame,"csv") << ";" << endl << endl;
    ARToolKit::instance()->videoWidth = frame.cols;
    ARToolKit::instance()->videoHeight = frame.rows;
    ARToolKit::instance()->videoData = frame.data;
    //imshow( window_name, frame );
}

using namespace osg;

OpenCVTracker *OpenCVTracker::plugin = NULL;

OpenCVTracker::OpenCVTracker()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

OpenCVTracker::~OpenCVTracker()
{
}

bool OpenCVTracker::destroy()
{
    if (capture)
    {
        cvReleaseCapture(&capture);
    }
    return true;
}

void OpenCVTracker::preFrame()
{
    if (capture)
    {
#if (CV_VERSION_MAJOR < 3)
        frame = cvQueryFrame(capture);
#else
        IplImage* _img = cvQueryFrame(capture);
        if( !_img )
        {
            frame.release();
        }
        else
        {
            if(_img->origin == IPL_ORIGIN_TL)
                cv::cvarrToMat(_img).copyTo(frame);
            else
            {
                Mat temp = cv::cvarrToMat(_img);
                flip(temp, frame, 0);
            }
        }
#endif

        //-- 3. Apply the classifier to the frame
        if (!frame.empty())
        {
            detectAndDisplay(frame);
        }
        else
        {
            printf(" --(!) No captured frame -- Break!");
        }
    }
}

bool OpenCVTracker::init()
{
    capture = NULL;

    if (OpenCVTracker::plugin != NULL)
        return false;

    OpenCVTracker::plugin = this;

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading\n");
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        printf("--(!)Error loading\n");
        return -1;
    };
    //-- 2. Read the video stream
    capture = cvCaptureFromCAM(-1);

    double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

    return true;
}

COVERPLUGIN(OpenCVTracker)
