/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputhdw.cpp
 *
 *  Created on: Dec 9, 2014
 *      Author: svnvlad
 */
#include "windowsHID.h"

#include <config/CoviseConfig.h>

#include <iostream>
//#include <chrono>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace std;
using namespace covise;


windowsHID::windowsHID(const std::string &configBase)
    : InputDevice(configBase)
{
    rawMouseManager = coRawDeviceManager::instance();
    //mouse related stuff
    rawMouse = NULL;
    btnstatus = 0;
    oldbtnstatus = 0;

    string devstring = coCoviseConfig::getEntry("device", configPath(), "HID#VID_046D&amp;PID_C52D&amp;MI_00"); // Logitech Presenter R400
    cout << devstring << endl;
    rawMouse = new coRawDevice(devstring.c_str());

    m_buttonStates.resize(MAX_RAW_MOUSE_BUTTONS);
}

windowsHID::~windowsHID()
{
    delete rawMouse;
}

void windowsHID::update() //< called by Input::update()
{
    rawMouseManager->update();
    if(rawMouse)
    {
        for (int i = 0; i < MAX_RAW_MOUSE_BUTTONS; ++i)
        {
            m_buttonStates[i] = rawMouse->getButton(i);
        }

        for (int n = 0; n < rawMouse->getNumValues(); n++)
        {
            while (n >= m_valuatorValues.size())
            {
                m_valuatorValues.push_back(0);
                m_valuatorRanges.push_back(std::pair<double, double>(-1.0, 1.0));
            }
            m_valuatorValues[n] = rawMouse->getValue(n);
        }
    }
    InputDevice::update();
}

INPUT_PLUGIN(windowsHID)
