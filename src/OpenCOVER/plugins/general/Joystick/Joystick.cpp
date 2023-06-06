/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "Joystick.h"

#include <cover/input/dev/Joystick/Joystick.h>

#include <cover/input/input.h>
using namespace vrml;
using namespace opencover;



static VrmlNode* creator(VrmlScene* scene)
{
    return new VrmlNodeJoystick(scene);
}

// Define the built in VrmlNodeType:: "SteeringWheel" fields

VrmlNodeType* VrmlNodeJoystick::defineType(VrmlNodeType* t)
{
    static VrmlNodeType* st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Joystick", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addEventIn("set_time", VrmlField::SFTIME);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("joystickNumber", VrmlField::SFINT32);
    t->addEventOut("buttons_changed", VrmlField::MFINT32);
    t->addEventOut("axes_changed", VrmlField::MFFLOAT);
    t->addEventOut("sliders_changed", VrmlField::MFFLOAT);
    t->addEventOut("POVs_changed", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType* VrmlNodeJoystick::nodeType() const
{
    return defineType(0);
}

VrmlNodeJoystick::VrmlNodeJoystick(VrmlScene* scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
    , d_joystickNumber(-1)
{
    setModified();
}

VrmlNodeJoystick::VrmlNodeJoystick(const VrmlNodeJoystick& n)
    : VrmlNodeChild(n.d_scene)
    , d_enabled(n.d_enabled)
    , d_joystickNumber(n.d_joystickNumber)
{

    setModified();
}

VrmlNodeJoystick::~VrmlNodeJoystick()
{
}

VrmlNode* VrmlNodeJoystick::cloneMe() const
{
    return new VrmlNodeJoystick(*this);
}

VrmlNodeJoystick* VrmlNodeJoystick::toSteeringWheel() const
{
    return (VrmlNodeJoystick*)this;
}

ostream& VrmlNodeJoystick::printFields(ostream& os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!d_joystickNumber.get())
        PRINT_FIELD(joystickNumber);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeJoystick::setField(const char* fieldName,
    const VrmlField& fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(joystickNumber, SFInt)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField* VrmlNodeJoystick::getField(const char* fieldName)
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "joystickNumber") == 0)
        return &d_joystickNumber;
    else if (strcmp(fieldName, "axes_changed") == 0)
        return &d_axes;
    else if (strcmp(fieldName, "sliders_changed") == 0)
        return &d_sliders;
    else if (strcmp(fieldName, "POVs_changed") == 0)
        return &d_POVs;
    else if (strcmp(fieldName, "buttons_changed") == 0)
        return &d_buttons;
    else
        cout << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeJoystick::eventIn(double timeStamp,
    const char* eventName,
    const VrmlField* fieldValue)
{
    if (strcmp(eventName, "set_time") == 0)
    {
    }
    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeJoystick::render(Viewer*)
{
    if (!d_enabled.get())
        return;

    int joystickNumber = d_joystickNumber.get();
    if ((joystickNumber >= JoystickPlugin::plugin->dev->numLocalJoysticks) || (joystickNumber >= JoystickPlugin::plugin->dev->numLocalJoysticks + 1))
        return;
    double timeStamp = System::the->time();
    //if (eventType & JOYSTICK_AXES_EVENTS)
    {
        if (JoystickPlugin::plugin->dev->number_axes[joystickNumber] && JoystickPlugin::plugin->dev->axes[joystickNumber])
        {
            d_axes.set(JoystickPlugin::plugin->dev->number_axes[joystickNumber], JoystickPlugin::plugin->dev->axes[joystickNumber]);
            // Send the new value
            eventOut(timeStamp, "axes_changed", d_axes);
        }
        if (JoystickPlugin::plugin->dev->number_sliders[joystickNumber] && JoystickPlugin::plugin->dev->sliders[joystickNumber])
        {
            d_sliders.set(JoystickPlugin::plugin->dev->number_sliders[joystickNumber], JoystickPlugin::plugin->dev->sliders[joystickNumber]);
            // Send the new value
            eventOut(timeStamp, "sliders_changed", d_sliders);
        }
        if (JoystickPlugin::plugin->dev->number_POVs[joystickNumber] && JoystickPlugin::plugin->dev->POVs[joystickNumber])
        {
            d_POVs.set(JoystickPlugin::plugin->dev->number_POVs[joystickNumber], JoystickPlugin::plugin->dev->POVs[joystickNumber]);
            // Send the new value
            eventOut(timeStamp, "POVs_changed", d_POVs);
        }
    }

    //if (eventType & JOYSTICK_BUTTON_EVENTS)
    {
        if (JoystickPlugin::plugin->dev->number_buttons[joystickNumber] && JoystickPlugin::plugin->dev->buttons[joystickNumber])
        {
            d_buttons.set(JoystickPlugin::plugin->dev->number_buttons[joystickNumber],
                JoystickPlugin::plugin->dev->buttons[joystickNumber]);
            // Send the new value
            eventOut(timeStamp, "buttons_changed", d_buttons);
        }
    }
}


JoystickPlugin* JoystickPlugin::plugin=nullptr;

JoystickPlugin::JoystickPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool JoystickPlugin::init()
{

    if (JoystickPlugin::plugin != NULL)
        return false;

    JoystickPlugin::plugin = this;
    dev = (Joystick *)(Input::instance()->getDevice("joystick"));

    VrmlNamespace::addBuiltIn(VrmlNodeJoystick::defineType());
    return true;
}

// this is called if the plugin is removed at runtime
JoystickPlugin::~JoystickPlugin()
{
    
}

// hide all schweissbrenners (make transparent)
bool JoystickPlugin::update()
{
    return true;
}

COVERPLUGIN(JoystickPlugin)
