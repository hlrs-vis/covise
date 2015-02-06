/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * RIFTDriver.cpp
 *
 *  Created on: Feb 5, 2014
 *      Author: hpcwoess
 */

#include "RIFTDriver.h"

#include <config/CoviseConfig.h>

#include <iostream>
#include <chrono>
#include <osg/Matrix>
#include <util/unixcompat.h>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace std;
using namespace covise;

RIFTDriver::RIFTDriver(const std::string &config)
    : InputDevice(config)
{
    
   
    stopLoop();// we don't need this, we set the matrix from the rift plugin

}

//====================END of init section============================

RIFTDriver::~RIFTDriver()
{
}

//==========================main loop =================

/**
 * @brief RIFTDriver::run ImputHdw main loop
 *
 * Gets the orientation of the HMD
 */
bool RIFTDriver::poll()
{
        usleep(10000);
    return true;
}
void RIFTDriver::setMatrix(osg::Matrix &m)
{
    if (m_bodyMatrices.size() < 1)
    {
        m_mutex.lock();
        m_bodyMatrices.resize(1);
        m_bodyMatricesValid.resize(1);
        m_mutex.unlock();
    }
    m_mutex.lock();
    m_bodyMatricesValid[0] = true;
    m_bodyMatrices[0] = m;
    m_mutex.unlock();
}

INPUT_PLUGIN(RIFTDriver)
