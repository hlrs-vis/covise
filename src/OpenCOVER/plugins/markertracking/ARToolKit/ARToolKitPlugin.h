#ifndef _ARToolKit_PLUGIN_H
#define _ARToolKit_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: ARToolKit Plugin                                            **
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
#include <cover/coVRPlugin.h>

using namespace covise;
using namespace opencover;

#ifdef HAVE_AR
#include <AR/param.h>
#include <AR/ar.h>

class ARToolKitPlugin : public coVRPlugin, public MarkerTrackingInterface, public coTUIListener
{
public:
    ARToolKitPlugin();
    virtual ~ARToolKitPlugin();

    // this will be called in PreFrame
    virtual bool init();
    virtual void preFrame();
    virtual void updateViewerPos(const osg::Vec3f &vp);

private:
    coTUIToggleButton *arDebugButton;
    coTUIButton *arSettingsButton;
    coTUIToggleButton *arDesktopMode;

    coTUISlider *thresholdEdit;
    coTUISlider *bitrateSlider;

    void captureRightVideo();
    char *vconf;
    char *vconf2;
    char *cconf;
    char *pconf;
    int msgQueue;
    ARParam cparam;
    unsigned char *dataPtr;
    int xsize, ysize;
    int thresh;
    int marker_num;
    bool flipBufferH;
    bool flipBufferV;
    ARMarkerInfo *marker_info;
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual int loadPattern(const char *);
    virtual osg::Matrix getMat(int pattID, double pattCenter[2], double pattSize, double pattTrans[3][4]);
    virtual bool isVisible(int);
    virtual bool isARToolKit()
    {
        return true;
    };
    int ids[500];
    char *names[500];
    int numNames;

private:
    coVRCollaboration::SyncMode syncmode;
};
#endif
#endif
