/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputhdw.h
 *
 *  Created on: Dec 9, 2014
 *      Author: svnvlad
 */

#ifndef MOUSEBUTTONSDRIVER_H
#define MOUSEBUTTONSDRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>

#include <cover/input/inputdevice.h>
#include "MouseButtons.h"

using namespace opencover;

/**
 * @brief The InputHdw class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */
class MouseButtonsDriver : public InputDevice
{
    //---------------------Mouse related stuff
    MouseButtons *mousebuttons; /// HLRS mouse buttons hardware interaction class
    unsigned int btnstatus, oldbtnstatus; /// Mouse Button status bit masks

    //=========End of hardware related stuff======================================
    bool init();

public:
    MouseButtonsDriver(const std::string &configPath);
    virtual ~MouseButtonsDriver();
    bool poll();
};

#endif
