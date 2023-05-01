#ifndef _ARTOOLKITPLUSPLUGIN_H
#define _ARTOOLKITPLUSPLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: ARToolKitPlus Plugin                                            **
 **                                                                          **
 **                                                                          **
 ** Author: M.Braitmaier		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-10  v1	    				       		                             **
 **                                                                          **
 * This Plugin is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This plugin is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library (see license.txt); if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRCollaboration.h>
#include <cover/MarkerTracking.h>
#include <cover/coVRPlugin.h>
#include "ARCaptureThread.h"

using namespace covise;
using namespace opencover;

class ARToolKitPlusPlugin : public coVRPlugin, public MarkerTrackingInterface, public coTUIListener, public DataBufferListener
{
public:
    ARToolKitPlusPlugin();
    virtual ~ARToolKitPlusPlugin();

    // this will be called in PreFrame
    virtual bool init();
    virtual void preFrame();

    // TabletUI events
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    /**
    * Has no real function for ARToolKitPlus, as pattern loading is no longer
    * required. However it returns the integer ID of a pattern specified in the 
    * config file. Example: <Pattern value="034"/> for pattern with id value 34
    * will result in loadPattern returning 34.
    */
    virtual int loadPattern(const char *);

    /**
    * Checks for a given pattern Id value whether this pattern is in the list
    * of detected markers after a call to arDetectMarker
    */
    virtual bool isVisible(int);

    virtual void lock();

    virtual void unlock();

    void update();

protected:
    //ARToolKitPlus TabeltUI Elements
    coTUIToggleButton *arDebugButton;
    coTUIButton *arSettingsButton;
    coTUISlider *thresholdEdit;
    coTUISlider *bitrateSlider;
    coTUIToggleButton *arUseBCHButton;
    coTUIToggleButton *arAutoThresholdButton;
    coTUISlider *arAutoThresholdValue;
    coTUIComboBox *arBorderSelectBox;
    coTUIButton *arDumpImage;

    void captureRightVideo();

    //Configuration parameters for ARToolKitPlus from covis-config
    // Really all required??

    /**
    * Configuration string of camera settings as retrieved from
    * the config file.
    */
    char *vconf;

    /**
    * Configuration string of camera settings for stereoscopic AR as retrieved 
    * from the config file.
    */
    char *vconf2;

    /**
    * Camera parameters as retrieved from the camera *.dat file specified in 
    * the config file.
    */
    char *cconf;

    /**
    * DEPRECATED
    */
    char *pconf;

    //ARToolKitPlus variables

    /**
    * Message queue for stereoscopic AR, currently not used until stereoscopic
    * AR is implemented
    */
    int msgQueue;

    /**
    * ARToolKit structure that contains the information of all detected
    * markers after a call of arDetectMarker in ARToolkitPlus::preFrame()
    * Is updated with every call to arDetectMarker
    */
    //ARToolKitPlus::ARMarkerInfo    *m_marker_info;

    //ARToolKitPlus::ARParam  cparam;

    //Matrix Retrieval for OpenCOVER
    virtual osg::Matrix getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4]);

    /**
    * Data array for image data
    */
    unsigned char *dataPtr;

    /**
    * Size of captured image
    */
    int xsize, ysize;

    /**
    * Threshold for image recognition
    */
    int thresh;

    /**
    * Contains the number of detected AR markers in a image.
    * The value is updated with every call of arDetectMarker in
    * ARToolKitPlus::preFrame()
    */
    int m_marker_num;

    /**
    * Indicates whether horizontal or vertical fliping of the imagebuffer
    * should be applied
    */
    bool flipBufferH;
    bool flipBufferV;

private:
    //ARToolKitPlus objects
    //ARToolKitPlus::TrackerSingleMarker* m_tracker;
    ARCaptureThread *m_arCapture;
    AugmentedRealityData m_arData;

    /**
    * Reference to the video capture class instance
    * that delivers the video frames
    */
    VICapture m_vi;
};
#endif
