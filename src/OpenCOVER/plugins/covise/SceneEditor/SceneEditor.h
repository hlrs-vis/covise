/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SCENE_EDITOR_H
#define SCENE_EDITOR_H

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRNavigationManager.h>

#include <osg/Camera>

#include "SceneObject.h"
#include "SceneObjectManager.h"

class SceneEditor : public opencover::coVRPlugin
{
public:
    // create pinboard button viewpoints and submenu
    SceneEditor();
    virtual ~SceneEditor();

    static SceneEditor *plugin;

    virtual bool init();
    virtual void preFrame();
    virtual void key(int type, int keySym, int mod);
    virtual void userEvent(int mod);
    //    virtual void param(const char *paramName, bool inMapLoading );
    virtual void guiToRenderMsg(const char *msg);
    //    virtual void message(int toWhom, int type, int length, const void *data);

    static int loadCoxml(const char *filename, osg::Group *group, const char *ck = "");
    static int replaceCoxml(const char *filename, osg::Group *group, const char *ck = "");
    static int unloadCoxml(const char *filename, const char *ck = "");

    void deselect(SceneObject *);

private:
    SceneObjectManager *_sceneObjectManager;

    vrui::coNavInteraction *interactionA;
    vrui::coNavInteraction *interactionC;

    osg::Node *_mouseOverNode;
    SceneObject *_mouseOverSceneObject;
    SceneObject *_selectedSceneObject;

    osg::Matrix _oldSceneGraphTransform;
};

#endif
