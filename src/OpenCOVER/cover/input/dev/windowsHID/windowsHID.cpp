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

    string devstring = coCoviseConfig::getEntry("device", configPath(), "/dev/input/mouse0"); //UNIX device string
    cout << devstring << endl;
    rawMouse = new coRawDevice(devstring.c_str());

    m_buttonStates.resize(8);
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
        for (int i = 0; i < 8; ++i)
        {
            m_buttonStates[i] = rawMouse->getButton(i);
        }
    }
    InputDevice::update();
}

INPUT_PLUGIN(windowsHID)
