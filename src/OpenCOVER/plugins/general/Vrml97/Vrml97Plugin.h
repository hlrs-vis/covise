/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRML97_PLUGIN_H
#define _VRML97_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Vrml97 Plugin (does nothing)                                **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner                                                       **
 **                                                                          **
 ** History:                                                                 **
 ** Nov-01  v1                                                               **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <cover/ui/Owner.h>
#include <PluginUtil/coSensor.h>

class ViewerOsg;

namespace vrml
{
class VrmlScene;
}

namespace opencover {
namespace ui {
class Menu;
class Element;
}
}

using namespace vrml;
using namespace opencover;

class ListenerCover;
class SystemCover;

class VRML97PLUGINEXPORT Vrml97Plugin : public coVRPlugin, public ui::Owner
{
    friend class ListenerCover;
    friend class SystemCover;
    friend class ViewerOsg;
    friend class coSensor;

public:
    Vrml97Plugin();
    ~Vrml97Plugin();

    static Vrml97Plugin *plugin;

    static osg::Node *getRegistrationRoot();
    static int loadUrl(const Url &url, osg::Group *group, const char *ck = "");
    static int loadVrml(const char *filename, osg::Group *group, const char *ck = "");
    static int replaceVrml(const char *filename, osg::Group *group, const char *ck = "");
    static int unloadVrml(const char *filename, const char *ck = "");

    static void worldChangedCB(int reason);

    bool init() override;

    void key(int type, int keySym, int mod) override;

    void message(int toWhom, int type, int len, const void *buf) override;
    void guiToRenderMsg(const grmsg::coGRMsg &msg)  override;

    void addNode(osg::Node *, const RenderObject *) override;

    Player *getPlayer() const
    {
        return player;
    }
    ViewerOsg *getViewer()
    {
        return viewer;
    };
    VrmlScene *getVrmlScene()
    {
        return vrmlScene;
    };
    SystemCover *getSystemCover()
    {
        return system;
    }

    void activateTouchSensor(int id);
    ui::Element *getMenuButton(const std::string &buttonName);

    bool update() override;
    // this will be called in PreFrame
    void preFrame() override;

	//! this function is called from the draw thread before drawing the scenegraph (after drawing the AR background)
	void preDraw(osg::RenderInfo &) override;
    bool isNewVRML;

protected:
private:
    std::string vrmlFilename;

    SystemCover *system;
    ListenerCover *listener;
    ViewerOsg *viewer;
    VrmlScene *vrmlScene;
    Player *player;

    bool raw;

    //void menuEvent(coMenuItem *);
};
#endif
