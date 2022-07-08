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
    ~PBufferSnapShot() override;
    bool init() override;

    void preFrame() override;
    void preSwapBuffers(int windowNumber) override;

    void guiToRenderMsg(const grmsg::coGRMsg &msg)  override;
    void message(int toWhom, int type, int len, const void *buf) override;

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
    int NumResolutions = 0;

    coButtonMenuItem *snapButton = nullptr;

    coTUITab *tuiSnapTab = nullptr;
    coTUIButton *tuiSnapButton = nullptr;
    coTUIComboBox *tuiResolution = nullptr;
    coTUILabel *tuiResolutionLabel = nullptr;
    coTUIEditIntField *tuiResolutionX = nullptr;
    coTUIEditIntField *tuiResolutionY = nullptr;
    coTUILabel *tuiFileNameLabel = nullptr;
    coTUIEditField *tuiFileName = nullptr;
    coTUILabel *tuiSavedFileLabel = nullptr;
    coTUILabel *tuiSavedFile = nullptr;
    coTUIComboBox *tuiRenderingMethod = nullptr;
    coTUIToggleButton *tuiStereoCheckbox = nullptr;
    coTUIToggleButton *tuiSnapOnSlaves = nullptr;
    coTUIToggleButton *tuiTransparentBackground = nullptr;

    void tabletEvent(coTUIElement *tUIItem) override;
    void tabletPressEvent(coTUIElement *tUIItem) override;
    void tabletReleaseEvent(coTUIElement *tUIItem) override;

    void prepareSnapshot();

    void menuEvent(coMenuItem *) override;

    inline void cameraCallbackExit() const;

    void initUI();
    void deleteUI();

    std::string suggestFileName(std::string suggestedFilename);
    bool endsWith(std::string main, std::string end);
};
#endif
