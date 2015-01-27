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
#include <OpenVRUI/coMenu.h>

class ViewerOsg;

namespace vrml
{
class VrmlScene;
}

using namespace vrml;
using namespace vrui;
using namespace opencover;

class ListenerCover;
class SystemCover;

class VRML97PLUGINEXPORT Vrml97Plugin : public coVRPlugin, coMenuListener
{
    friend class ListenerCover;
    friend class SystemCover;
    friend class ViewerOsg;

public:
    Vrml97Plugin();
    ~Vrml97Plugin();

    static Vrml97Plugin *plugin;

    static osg::Node *getRegistrationRoot();
    static int loadVrml(const char *filename, osg::Group *group, const char *ck = "");
    static int replaceVrml(const char *filename, osg::Group *group, const char *ck = "");
    static int unloadVrml(const char *filename, const char *ck = "");

    static void worldChangedCB(int reason);

    bool init();

    void key(int type, int keySym, int mod);

    void message(int type, int len, const void *buf);
    void guiToRenderMsg(const char *msg);

    virtual void addNode(osg::Node *, RenderObject *);

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
    coMenuItem *getMenuButton(const std::string &buttonName);

    // this will be called in PreFrame
    void preFrame();
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

    void menuEvent(coMenuItem *);
};
#endif
