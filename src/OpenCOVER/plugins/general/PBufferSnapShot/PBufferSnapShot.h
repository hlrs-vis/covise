/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef PBUFFER_SNAPSHOT_PLUGIN_H
#define PBUFFER_SNAPSHOT_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2004 HLRS  **
 **                                                                          **
 ** Description: Snapshot plugin using pbuffers                              **
 **                                                                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>
#include <osg/Camera>
#include <OpenVRUI/coMenu.h>
#include <cover/coVRPlugin.h>

#include <string>

namespace vrui
{
class coButtonMenuItem;
}
namespace opencover
{
class coVRSceneHandler;
class coVRSceneView;
}

using namespace vrui;
using namespace opencover;

#define NUM_RENDER_TARGET_IMPLEMENTATIONS 5

class PBufferSnapShot : public coVRPlugin,
                        public coMenuListener,
                        public coTUIListener
{
    friend class DrawCallback;

public:
    PBufferSnapShot();
    virtual ~PBufferSnapShot();
    bool init();

    void preFrame();
    void preSwapBuffers(int windowNumber);

    void guiToRenderMsg(const char *msg);
    void message(int toWhom, int type, int len, const void *buf);

private:
    osg::ref_ptr<osg::Camera::DrawCallback> drawCallback;
    osg::ref_ptr<osg::Camera> pBufferCameras[NUM_RENDER_TARGET_IMPLEMENTATIONS];
    osg::ref_ptr<osg::Camera> pBufferCamera;
    osg::ref_ptr<osg::Camera> pBufferCamerasR[NUM_RENDER_TARGET_IMPLEMENTATIONS];
    osg::ref_ptr<osg::Camera> pBufferCameraR;
    osg::ref_ptr<osg::Image> image;
    osg::ref_ptr<osg::Image> imageR;

    mutable bool removeCamera;
    mutable bool doSnap;
    bool myDoSnap;

    mutable int counter;
    bool stereo;
    bool doInit;

    std::string filename;
    std::string lastSavedFile;

    int snapID;

    struct Resolution
    {
        const std::string description;
        int x;
        int y;
    };

    static Resolution resolutions[];
    int NumResolutions;

    coButtonMenuItem *snapButton;

    coTUITab *tuiSnapTab;
    coTUIButton *tuiSnapButton;
    coTUIComboBox *tuiResolution;
    coTUILabel *tuiResolutionLabel;
    coTUIEditIntField *tuiResolutionX;
    coTUIEditIntField *tuiResolutionY;
    coTUILabel *tuiFileNameLabel;
    coTUIEditField *tuiFileName;
    coTUILabel *tuiSavedFileLabel;
    coTUILabel *tuiSavedFile;
    coTUIComboBox *tuiRenderingMethod;
    coTUIToggleButton *tuiStereoCheckbox;
    coTUIToggleButton *tuiSnapOnSlaves;
    coTUIToggleButton *tuiTransparentBackground;

    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);

    void prepareSnapshot();

    virtual void menuEvent(coMenuItem *);

    inline void cameraCallbackExit() const;

    void initUI();
    void deleteUI();

    std::string suggestFileName(std::string suggestedFilename);
    bool endsWith(std::string main, std::string end);
};
#endif
