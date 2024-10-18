/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "Joystick.h"

#include <cover/input/dev/Joystick/Joystick.h>

#include <cover/input/input.h>
#include <cover/coVRMSController.h>
using namespace vrml;
using namespace opencover;

void VrmlNodeJoystick::initFields(VrmlNodeJoystick* node, VrmlNodeType* t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
        exposedField("enabled", node->d_enabled),
        exposedField("joystickNumber", node->d_joystickNumber));

    if (t)
    {
        t->addEventIn("set_time", VrmlField::SFTIME);
        t->addEventOut("buttons_changed", VrmlField::MFINT32);
        t->addEventOut("axes_changed", VrmlField::MFFLOAT);
        t->addEventOut("sliders_changed", VrmlField::MFFLOAT);
        t->addEventOut("POVs_changed", VrmlField::MFFLOAT);
    }
}

const char* VrmlNodeJoystick::name()
{
    return "Joystick";
}

VrmlNodeJoystick::VrmlNodeJoystick(VrmlScene* scene)
    : VrmlNodeChild(scene, name())
    , d_enabled(true)
    , d_joystickNumber(-1)
{
    setModified();
}

VrmlNodeJoystick::VrmlNodeJoystick(const VrmlNodeJoystick& n)
    : VrmlNodeChild(n)
    , d_enabled(n.d_enabled)
    , d_joystickNumber(n.d_joystickNumber)
{
    setModified();
}

VrmlNodeJoystick* VrmlNodeJoystick::toSteeringWheel() const
{
    return (VrmlNodeJoystick*)this;
}

const VrmlField* VrmlNodeJoystick::getField(const char* fieldName) const
{
    if (strcmp(fieldName, "axes_changed") == 0)
        return &d_axes;
    else if (strcmp(fieldName, "sliders_changed") == 0)
        return &d_sliders;
    else if (strcmp(fieldName, "POVs_changed") == 0)
        return &d_POVs;
    else if (strcmp(fieldName, "buttons_changed") == 0)
        return &d_buttons;
    return VrmlNodeChild::getField(fieldName);
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
    if ((joystickNumber >= JoystickPlugin::plugin->numLocalJoysticks) || (joystickNumber >= JoystickPlugin::plugin->numLocalJoysticks + 1))
        return;
    double timeStamp = System::the->time();
    
    //if (eventType & JOYSTICK_AXES_EVENTS)
    {
        if (JoystickPlugin::plugin->number_axes[joystickNumber] && JoystickPlugin::plugin->axes[joystickNumber])
        {
            if(coVRMSController::instance()->isMaster())
            {
                coVRMSController::instance()->sendSlaves(JoystickPlugin::plugin->dev->axes[joystickNumber], sizeof(float)*JoystickPlugin::plugin->number_axes[joystickNumber]);
                d_axes.set(JoystickPlugin::plugin->number_axes[joystickNumber], JoystickPlugin::plugin->dev->axes[joystickNumber]);
	    }
	    else
	    {
                coVRMSController::instance()->readMaster(JoystickPlugin::plugin->axes[joystickNumber], sizeof(float)*JoystickPlugin::plugin->number_axes[joystickNumber]);
                d_axes.set(JoystickPlugin::plugin->number_axes[joystickNumber], JoystickPlugin::plugin->axes[joystickNumber]);
	    }
            // Send the new value
            eventOut(timeStamp, "axes_changed", d_axes);
        }
        if (JoystickPlugin::plugin->number_sliders[joystickNumber] && JoystickPlugin::plugin->sliders[joystickNumber])
        {
            if(coVRMSController::instance()->isMaster())
            {
                coVRMSController::instance()->sendSlaves(JoystickPlugin::plugin->dev->sliders[joystickNumber], sizeof(float)*JoystickPlugin::plugin->number_sliders[joystickNumber]);
                d_sliders.set(JoystickPlugin::plugin->number_sliders[joystickNumber], JoystickPlugin::plugin->dev->sliders[joystickNumber]);
	    }
	    else
	    {
                coVRMSController::instance()->readMaster(JoystickPlugin::plugin->sliders[joystickNumber], sizeof(float)*JoystickPlugin::plugin->number_sliders[joystickNumber]);
                d_sliders.set(JoystickPlugin::plugin->number_sliders[joystickNumber], JoystickPlugin::plugin->sliders[joystickNumber]);
	    }
            // Send the new value
            eventOut(timeStamp, "sliders_changed", d_sliders);
        }
        if (JoystickPlugin::plugin->number_POVs[joystickNumber] && JoystickPlugin::plugin->POVs[joystickNumber])
        {
            if(coVRMSController::instance()->isMaster())
            {
                coVRMSController::instance()->sendSlaves(JoystickPlugin::plugin->dev->POVs[joystickNumber], sizeof(float)*JoystickPlugin::plugin->number_POVs[joystickNumber]);
                d_POVs.set(JoystickPlugin::plugin->number_POVs[joystickNumber], JoystickPlugin::plugin->dev->POVs[joystickNumber]);
	    }
	    else
	    {
                coVRMSController::instance()->readMaster(JoystickPlugin::plugin->POVs[joystickNumber], sizeof(float)*JoystickPlugin::plugin->number_POVs[joystickNumber]);
                d_POVs.set(JoystickPlugin::plugin->number_POVs[joystickNumber], JoystickPlugin::plugin->POVs[joystickNumber]);
	    }
            // Send the new value
            eventOut(timeStamp, "POVs_changed", d_POVs);
        }
    }

    //if (eventType & JOYSTICK_BUTTON_EVENTS)
    {
        if (JoystickPlugin::plugin->number_buttons[joystickNumber] && JoystickPlugin::plugin->buttons[joystickNumber])
        {
            if(coVRMSController::instance()->isMaster())
            {
                coVRMSController::instance()->sendSlaves(JoystickPlugin::plugin->dev->buttons[joystickNumber], sizeof(int)*JoystickPlugin::plugin->number_buttons[joystickNumber]);
                d_buttons.set(JoystickPlugin::plugin->number_buttons[joystickNumber], JoystickPlugin::plugin->dev->buttons[joystickNumber]);
	    }
	    else
	    {
                coVRMSController::instance()->readMaster(JoystickPlugin::plugin->buttons[joystickNumber], sizeof(int)*JoystickPlugin::plugin->number_buttons[joystickNumber]);
                d_buttons.set(JoystickPlugin::plugin->number_buttons[joystickNumber], JoystickPlugin::plugin->buttons[joystickNumber]);
	    }
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
    dev = nullptr;

    if (JoystickPlugin::plugin != NULL)
        return false;
   
    JoystickPlugin::plugin = this;
    dev = (Joystick *)(Input::instance()->getDevice("joystick"));
    if(coVRMSController::instance()->isMaster())
    {
    numLocalJoysticks = dev->numLocalJoysticks;
    for(int i=0;i<numLocalJoysticks;i++)
    {
        number_buttons[i] = dev->number_buttons[i];
        number_axes[i] = dev->number_axes[i];
        number_sliders[i] = dev->number_sliders[i];
        number_POVs[i] = dev->number_POVs[i];
    }
       coVRMSController::instance()->sendSlaves(&numLocalJoysticks, sizeof(int));
        coVRMSController::instance()->sendSlaves(&number_buttons, sizeof(unsigned char)*numLocalJoysticks);
        coVRMSController::instance()->sendSlaves(&number_axes, sizeof(unsigned char)*numLocalJoysticks);
        coVRMSController::instance()->sendSlaves(&number_sliders, sizeof(unsigned char)*numLocalJoysticks);
        coVRMSController::instance()->sendSlaves(&number_POVs, sizeof(unsigned char)*numLocalJoysticks);
    
    }
    else
    {
        dev = nullptr;
        coVRMSController::instance()->readMaster(&numLocalJoysticks, sizeof(int));
        coVRMSController::instance()->readMaster(&number_buttons, sizeof(unsigned char)*numLocalJoysticks);
        coVRMSController::instance()->readMaster(&number_axes, sizeof(unsigned char)*numLocalJoysticks);
        coVRMSController::instance()->readMaster(&number_sliders, sizeof(unsigned char)*numLocalJoysticks);
        coVRMSController::instance()->readMaster(&number_POVs, sizeof(unsigned char)*numLocalJoysticks);
    }

    for(int i=0;i<numLocalJoysticks;i++)
    {
		if (number_buttons[i] > 0)
		{
			buttons[i] = new int[number_buttons[i]];
			for (int n = 0; n < number_buttons[i]; n++)
				buttons[i][n] = 0;
		}
		if (number_sliders[i] > 0)
		{
			sliders[i] = new float[number_sliders[i]];
			for (int n = 0; n < number_sliders[i]; n++)
				sliders[i][n] = 0;
		}
		if (number_axes[i] > 0)
		{
			axes[i] = new float[number_axes[i]];
			for (int n = 0; n < number_axes[i]; n++)
				axes[i][n] = 0;
		}
		if (number_POVs[i] > 0)
		{
			POVs[i] = new float[number_POVs[i]];
			for (int n = 0; n < number_POVs[i]; n++)
				POVs[i][n] = 0;
		}
    }
    VrmlNamespace::addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeJoystick>());
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
