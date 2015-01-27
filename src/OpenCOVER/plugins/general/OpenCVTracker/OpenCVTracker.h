/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _OPENCVTRACKER_H
#define _OPENCVTRACKER_H
#include <util/common.h>

#include <osg/Vec3>
#include <osg/Vec4>

#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coPopupHandle.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace opencover
{
class SystenCover;
}

using namespace covise;
using namespace opencover;

class OpenCVTracker : public coVRPlugin
{
private:
protected:
    /** Function Headers */
    void detectAndDisplay(cv::Mat frame);
    CvCapture *capture;
    cv::Mat frame;

public:
    static OpenCVTracker *plugin;

    OpenCVTracker();
    virtual ~OpenCVTracker();
    bool init();

    // this will be called in PreFrame
    void preFrame();

    virtual bool destroy();
};

#endif
