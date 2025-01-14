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
#include "LegacyDriver.h"

#include <config/CoviseConfig.h>

#include <iostream>
//#include <chrono>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace std;
using namespace covise;

LegacyDriver::LegacyDriver(const std::string &configBase)
    : InputDevice(configBase)
{
    m_bodyMatricesValid.resize(VRTracker::instance()->getNumStation(), true);
    m_bodyMatrices.resize(VRTracker::instance()->getNumStation());
    m_buttonStates.resize(4);
}

LegacyDriver::~LegacyDriver()
{
    stopLoop();
    delete VRTracker::instance();
}

/**
 * @brief LegacyDriver::run ImputHdw main loop
 *
 * Gets the status of the input devices
 */
bool LegacyDriver::poll()
{
    VRTracker::instance()->update();

    m_mutex.lock();

    if (ssize_t(m_bodyMatrices.size()) < VRTracker::instance()->getNumStation())
        m_bodyMatrices.resize(VRTracker::instance()->getNumStation());
    for (int i = 0; i < VRTracker::instance()->getNumStation(); ++i)
    {
        m_bodyMatrices[i] = VRTracker::instance()->getStationMat(i);
    }
    m_mutex.unlock();

    return true;
}

INPUT_PLUGIN(LegacyDriver)
