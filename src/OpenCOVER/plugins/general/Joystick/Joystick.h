/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _JoystickPlugin_H
#define _JoystickPlugin_H
#include <util/common.h>

#include <cover/input/dev/Joystick/Joystick.h>

#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

class PLUGINEXPORT VrmlNodeJoystick : public vrml::VrmlNodeChild
{
public:
    // Define the fields of SteeringWheel nodes
    static vrml::VrmlNodeType* defineType(vrml::VrmlNodeType* t = 0);
    virtual vrml::VrmlNodeType* nodeType() const;

    VrmlNodeJoystick(vrml::VrmlScene* scene = 0);
    VrmlNodeJoystick(const VrmlNodeJoystick& n);
    virtual ~VrmlNodeJoystick();

    virtual VrmlNode* cloneMe() const;

    virtual VrmlNodeJoystick* toSteeringWheel() const;

    virtual ostream& printFields(ostream& os, int indent);

    virtual void setField(const char* fieldName, const vrml::VrmlField& fieldValue);
    const vrml::VrmlField* getField(const char* fieldName);

    void eventIn(double timeStamp, const char* eventName,
        const vrml::VrmlField* fieldValue);

    virtual void render(vrml::Viewer*);

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    // Fields
    vrml::VrmlSFBool d_enabled;
    vrml::VrmlSFInt d_joystickNumber;

    // State
    vrml::VrmlMFFloat d_axes;
    vrml::VrmlMFInt d_buttons;
};
namespace vrui
{
class coButtonMenuItem;
class coCheckboxMenuItem;
class coTrackerButtonInteraction;
class coNavInteraction;
class coMouseButtonInteraction;
class coPotiMenuItem;
class coSubMenuItem;
class coRowMenu;
class coFrame;
class coPanel;
}

using namespace vrui;
using namespace opencover;


class JoystickPlugin : public coVRPlugin
{

private:


protected:

public:
    static JoystickPlugin *plugin;
    Joystick* dev;
    JoystickPlugin();
    virtual ~JoystickPlugin();
    bool init();

    // this will be called in PreFrame
    bool update();
};
#endif
