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

#ifndef LEGACYDRIVER_H
#define LEGACYDRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>

#include <cover/input/inputdevice.h>
#include "VRTracker.h"

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
class LegacyDriver : public InputDevice
{
    //=========End of hardware related stuff======================================
    bool init();

public:
    LegacyDriver(const std::string &configPath);
    virtual ~LegacyDriver();
    bool poll();
};

#endif
