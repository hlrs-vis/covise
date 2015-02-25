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
#include "MikeDriver.h"
#include "../legacy/serialcom.h"

#include <config/CoviseConfig.h>

#include <iostream>
#include <chrono>
#include <osg/Matrix>


using namespace std;
using namespace covise;

MikeDriver::MikeDriver(const std::string &configBase)
    : InputDevice(configBase)
{
    //mouse related stuff
    btnstatus = 0;
    oldbtnstatus = 0;

    string devstring = coCoviseConfig::getEntry("device", configPath(), "/dev/ttyUSB0"); //UNIX device string
    cout << devstring << endl;
      fprintf(stderr, "connecting to %s !!\n", devstring.c_str());
    if (!::Init((char *)devstring.c_str(), 9600))
        {
            fprintf(stderr, "error connecting to %s !!\n", devstring.c_str());
        }

    m_buttonStates.resize(4);
}

MikeDriver::~MikeDriver()
{
    stopLoop();
    close_port();
    cout << "MouseButtonsDriver destr" << endl;
}

/**
 * @brief MouseButtonsDriver::run ImputHdw main loop
 *
 * Gets the status of the input devices
 */
bool MikeDriver::poll()
{
    // Getting mouse button status and eventsnswer(1, buttonData);
    oldbtnstatus = btnstatus;
    get_answer(1, (unsigned char *)&btnstatus);
    //events
    if (oldbtnstatus != btnstatus)
    {
        m_mutex.lock();
        for (int i = 0; i < numButtons(); ++i)
        {
            unsigned int btn_mask = 0x1 << i; // mask for button #n
            //event checks

            unsigned long timestamp = chrono::high_resolution_clock::now().time_since_epoch().count(); //nanoseconds?
            //cout << timestamp/1000000 <<endl;
            m_buttonStates[i] = btnstatus & btn_mask;
        }
        m_mutex.unlock();
    }
    return true;
}

INPUT_PLUGIN(MikeDriver)
