/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "buttondevice.h"
#include "inputdevice.h"
#include "input.h"

#include <config/CoviseConfig.h>

#include <OpenVRUI/sginterface/vruiButtons.h>
#include <iostream>
#include <cstring>
#include <util/unixcompat.h>

using namespace covise;
using namespace std;

namespace opencover
{

/**
 * @brief mapButton map raw buttons to VRUI constants
 * @param raw
 * @param buttonMap
 * @return
 */
unsigned ButtonDevice::mapButton(unsigned raw) const
{
    unsigned mapped = 0;
    for (unsigned bit = 1; bit; bit <<= 1)
    {
        if ((raw & bit) == 0)
            continue;

        ButtonMap::const_iterator it = buttonMap.find(bit);
        if (it == buttonMap.end())
            mapped |= bit;
        else
            mapped |= it->second;
    }
    return mapped;
}

/**
 * @brief ButtonDevice::createButtonMap
 *
 */
void ButtonDevice::createButtonMap(const std::string &confbase)
{
    std::vector<std::string> buttonIndices = coCoviseConfig::getScopeNames(confbase, "Map");
    for (size_t i = 0; i < buttonIndices.size(); ++i)
    {

        const std::string mapped = coCoviseConfig::getEntry(confbase + ".Map:" + buttonIndices[i]);
        unsigned int number = atoi(buttonIndices[i].c_str());
        unsigned bit = 1U << number;

#define MB(name)                                                                         \
    if (strcasecmp(mapped.c_str(), #name) == 0)                                          \
    {                                                                                    \
        handled = true;                                                                  \
        buttonMap[bit] = vrui::vruiButtons::name;                                        \
        /* cerr << "map: " << number << " (bit: " << bit << ") -> " << #name << endl; */ \
    }

        bool handled = false;

        MB(NO_BUTTON);
        MB(ACTION_BUTTON);
        MB(DRIVE_BUTTON);
        MB(XFORM_BUTTON);
        MB(USER1_BUTTON);
        MB(USER4_BUTTON);
        MB(TOGGLE_DOCUMENTS);
        MB(INTER_PREV);
        MB(INTER_NEXT);
        MB(MENU_BUTTON);
        MB(FORWARD_BUTTON);
        MB(BACKWARD_BUTTON);
        MB(ZOOM_BUTTON);
        MB(QUIT_BUTTON);
        MB(DRAG_BUTTON);
        MB(WHEEL_UP);
        MB(WHEEL_DOWN);
        MB(JOYSTICK_RIGHT);
        MB(JOYSTICK_DOWN);
        MB(JOYSTICK_LEFT);
        MB(JOYSTICK_UP);

        if (!handled)
            cerr << "Input: ButtonDevice: unknown button name \"" << mapped << "\" in " << confbase << endl;

#undef MB
    }
}

ButtonDevice::ButtonDevice(const string &name)
    : m_dev(NULL)
    , btnstatus(0)
{
    const std::string conf = "COVER.Input.Buttons." + name;
    const std::string driver = coCoviseConfig::getEntry("device", conf, "default");

    if (name != "Mouse")
    {
        m_dev = Input::instance()->getDevice(driver);
        if (!m_dev)
            m_dev = Input::instance()->getDevice("const");
    }

    createButtonMap(conf);
}

/**
 * @brief ButtonDevice::update Must be called from Input::update(). Saves the buttonstate into btnstatus member
 * @return 0
 */
void ButtonDevice::update()
{
    if (m_dev)
    {
        unsigned raw = 0;
        for (size_t i = 0; i < m_dev->numButtons(); ++i)
        {
            if (m_dev->getButtonState(i))
                raw |= (1U << i);
        }
        btnstatus = mapButton(raw);
    }
}

/**
 * @brief ButtonDevice::getButtonState
 * @return state of the button device
 */
unsigned int ButtonDevice::getButtonState() const
{
    return btnstatus;
}

void ButtonDevice::setButtonState(unsigned int state, bool isRaw)
{

    if (isRaw)
        btnstatus = mapButton(state);
    else
        btnstatus = state;
}
}
