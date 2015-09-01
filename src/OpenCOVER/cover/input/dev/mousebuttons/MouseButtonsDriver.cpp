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
#include "MouseButtonsDriver.h"

#include <config/CoviseConfig.h>

#include <iostream>
//#include <chrono>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace std;
using namespace covise;

MouseButtonsDriver::MouseButtonsDriver(const std::string &configBase)
    : InputDevice(configBase)
{
    //mouse related stuff
    mousebuttons = NULL;
    btnstatus = 0;
    oldbtnstatus = 0;

    string devstring = coCoviseConfig::getEntry("device", configPath(), "/dev/input/mouse0"); //UNIX device string
    cout << devstring << endl;
    mousebuttons = new MouseButtons(devstring.c_str());

    m_buttonStates.resize(4);
}

MouseButtonsDriver::~MouseButtonsDriver()
{
    stopLoop();
    cout << "MouseButtonsDriver destr" << endl;
}

/**
 * @brief MouseButtonsDriver::run ImputHdw main loop
 *
 * Gets the status of the input devices
 */
bool MouseButtonsDriver::poll()
{
    // Getting mouse button status and events
    oldbtnstatus = btnstatus;
    if (mousebuttons != NULL)
        mousebuttons->getButtons(0, &btnstatus);
    //events
    if (oldbtnstatus != btnstatus)
    {
        m_mutex.lock();
        for (size_t i = 0; i < numButtons(); ++i)
        {
            unsigned int btn_mask = 0x1 << i; // mask for button #n
            //event checks

            //unsigned long timestamp = chrono::high_resolution_clock::now().time_since_epoch().count(); //nanoseconds?
            //cout << timestamp/1000000 <<endl;
            m_buttonStates[i] = (btnstatus & btn_mask)!=0;
        }
        m_mutex.unlock();
    }
    return true;
}

INPUT_PLUGIN(MouseButtonsDriver)
