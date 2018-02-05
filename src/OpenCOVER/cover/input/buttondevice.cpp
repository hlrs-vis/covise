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
#include <cstdlib>
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
    ButtonMap::const_iterator it = multiButtonMap.find(raw);
    if (it != multiButtonMap.end())
    {
        return it->second;
    }

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
        MB(FORWARD_BUTTON);
        MB(BACKWARD_BUTTON);
        MB(TOGGLE_DOCUMENTS);
        MB(INTER_PREV);
        MB(INTER_NEXT);
        MB(PERSON_PREV);
        MB(PERSON_NEXT);
        MB(MENU_BUTTON);
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
            cerr << "Input: ButtonDevice: unknown button name \"" << mapped << "\" in " << confbase << ".Map" << endl;

#undef MB
    }

    std::vector<std::string> multiIndices = coCoviseConfig::getScopeNames(confbase, "MultiMap");
    for (size_t i = 0; i < multiIndices.size(); ++i)
    {

        const std::string mapped = coCoviseConfig::getEntry(confbase + ".MultiMap:" + multiIndices[i]);
        unsigned int bits = atoi(multiIndices[i].c_str());

#define MB(name)                                                                         \
    if (strcasecmp(mapped.c_str(), #name) == 0)                                          \
    {                                                                                    \
        handled = true;                                                                  \
        multiButtonMap[bits] = vrui::vruiButtons::name;                                  \
        /* cerr << "map: " << number << " (bit: " << bit << ") -> " << #name << endl; */ \
    }

        bool handled = false;

        MB(NO_BUTTON);
        MB(ACTION_BUTTON);
        MB(DRIVE_BUTTON);
        MB(XFORM_BUTTON);
        MB(FORWARD_BUTTON);
        MB(BACKWARD_BUTTON);
        MB(TOGGLE_DOCUMENTS);
        MB(INTER_PREV);
        MB(INTER_NEXT);
        MB(PERSON_PREV);
        MB(PERSON_NEXT);
        MB(MENU_BUTTON);
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
            cerr << "Input: ButtonDevice: unknown button name \"" << mapped << "\" in " << confbase << ".MultiMap" << endl;

#undef MB
    }
}

ButtonDevice::ButtonDevice(const string &name)
    : InputSource(name, "Buttons")
    , m_raw(0)
    , m_oldRaw(m_raw)
    , m_btnstatus(0)
{
    createButtonMap(config());
}

/**
 * @brief ButtonDevice::update Must be called from Input::update(). Saves the buttonstate into m_btnstatus member
 * @return 0
 */
void ButtonDevice::update()
{
    InputSource::update();

    if (device())
    {
        m_raw = 0;
        for (size_t i = 0; i < device()->numButtons(); ++i)
        {
            if (device()->getButtonState(i))
                m_raw |= (1U << i);
        }
    }

    if (Input::debug(Input::Buttons) && Input::debug(Input::Raw) && m_oldRaw!=m_raw)
    {
        std::cerr << "Input: " << name() << " buttons: raw=0x" << std::hex << m_raw << std::dec << std::endl;
    }
    m_oldRaw = m_raw;

    m_btnstatus = mapButton(m_raw);
}

/**
 * @brief ButtonDevice::getButtonState
 * @return state of the button device
 */
unsigned int ButtonDevice::getButtonState() const
{
    return m_btnstatus;
}

void ButtonDevice::setButtonState(unsigned int state, bool isRaw)
{

    if (isRaw)
    {
        m_raw = state;
    }
    else
    {
        m_btnstatus = state;
    }
}
}
