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

#ifndef windowsHID_H
#define windowsHID_H

#include <OpenThreads/Thread>
#include <osg/Matrix>

#include <cover/input/inputdevice.h>
#include "coRawDevice.h"

using namespace opencover;

/**
 * @brief The InputDevice class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 */
class windowsHID : public InputDevice
{
    //---------------------Mouse related stuff
    coRawDeviceManager *rawMouseManager; /// Windows HID Devices class
    coRawDevice *rawMouse; // the inputDevice

    unsigned int btnstatus, oldbtnstatus; /// Mouse Button status bit masks

    //=========End of hardware related stuff======================================
    bool init();

public:
    windowsHID(const std::string &configPath);
    virtual ~windowsHID();
    virtual bool needsThread() const {return false; } ; //< whether a thread should be spawned - reimplement if not necessary   we don't need a thread
    
    virtual void update(); //< called by Input::update()
};

#endif
