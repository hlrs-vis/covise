/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ALVAR_PLUGIN_H
#define _ALVAR_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: ALVAR Plugin                                            **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** Mar-05  v1	    				       		                             **
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
#include <cover/coTabletUI.h>
#include <cover/coVRPlugin.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <platform/CaptureFactory.h>
#include <MarkerDetector.h>
#include <Camera.h>
#include <MultiMarker.h>
#include <MultiMarkerInitializer.h>
#include <MultiMarkerBundle.h>
#include <SfM.h>

using namespace covise;
using namespace opencover;

class ALVARPlugin : public coVRPlugin, public MarkerTrackingInterface, public coTUIListener
{
public:
    ALVARPlugin();
    virtual ~ALVARPlugin();

    // this will be called in PreFrame
    virtual bool init();
    virtual bool update();
    virtual void preFrame();

private:
    coTUIToggleButton *arDebugButton;
    coTUIButton *arSettingsButton;
    coTUIToggleButton *calibrateButton;
    coTUIToggleButton *visualizeButton;
    coTUIToggleButton *detectAdditional;
    coTUIToggleButton *useSFM;

    coTUILabel *calibrateLabel;
    bool doCalibrate;
    bool calibrated;
    int calibCount;
    alvar::ProjPoints projPoints;

    alvar::SimpleSfM *sfm;

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
    virtual int loadPattern(const char *);
    virtual osg::Matrix getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4]);
    virtual bool isVisible(int);
    virtual void updateMarkerParams();
    int ids[500];
    char *names[500];
    int numNames;
    std::string calibrationFilename;
    alvar::Capture *cap;
    alvar::MarkerDetector<alvar::MarkerData> marker_detector;
    alvar::Camera cam;
    alvar::MultiMarkerInitializer *multiMarkerInitializer;
    /*std::vector<alvar::MultiMarkerBundle *>multiMarkerBundles;
	std::vector<alvar::Pose *>bundlePoses;*/

    alvar::MultiMarkerBundle *multiMarkerBundle;
    alvar::Pose bundlePose;

private:
    coVRCollaboration::SyncMode syncmode;
    void outputEnumeratedPlugins(alvar::CaptureFactory::CapturePluginVector &plugins);

    void outputEnumeratedDevices(alvar::CaptureFactory::CaptureDeviceVector &devices, int selectedDevice);

    int defaultDevice(alvar::CaptureFactory::CaptureDeviceVector &devices);
};
#endif
