/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TREADMILL_PLUGIN_H
#define _TREADMILL_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <osg/Camera>

#include "TexturePopup.h"
#include "MessageReceiver.h"

class coRowMenu;
class coMovableBackgroundMenuItem;
using namespace covise;
using namespace opencover;

class Treadmill : public coVRPlugin
{
public:
    // create pinboard button viewpoints and submenu
    Treadmill();
    virtual ~Treadmill();

    static Treadmill *plugin;

    virtual bool init();
    virtual void preFrame();
    virtual void key(int type, int keySym, int mod);
    //    virtual void param(const char *paramName, bool inMapLoading );
    //    virtual void guiToRenderMsg(const char *msg);
    //    virtual void message(int type, int length, const void *data);

private:
    MessageReceiver *_messageReceiver;

    int _port;
    int _timeout;

    double _forwardSpeed;
    double _yawSpeed;
    double _sideSpeed;
    double _forwardSpeedScale;
    double _yawSpeedScale;
    double _sideSpeedScale;
    double _floorOffset;
    int _turnDecision;
    double frameFactor(double delta);

    void _handleTokens(const std::vector<std::string> &tokens);
    void _handleMouseButtons();

    std::string _levelsDirectory;

    TexturePopup *_arrowLeftPopup;
    TexturePopup *_arrowRightPopup;
    TexturePopup *_animalPopup;
    //    virtual void menuEvent (coMenuItem *menuItem);

    std::string _dirSeparator;
};

#endif
