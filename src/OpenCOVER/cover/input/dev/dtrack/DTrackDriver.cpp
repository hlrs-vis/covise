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
#include "DTrackDriver.h"

#include <config/CoviseConfig.h>

#include <iostream>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace std;
using namespace covise;

DTrackDriver::DTrackDriver(const std::string &config)
    : InputDevice(config)
{
    m_isVarying = true;
    m_is6Dof = true;

    cout << "Initializing DTrack:" << configPath() << endl;
    int dtrack_port = coCoviseConfig::getInt("port", configPath(), 5000);
    dt = new DTrackSDK(dtrack_port);

    if (!dt->isLocalDataPortValid())
        cout << "Cannot initialize DTrack!" << endl;

    if (!dt->receive())
        cout << "Error receiving data!" << endl;

    m_numFlySticks = dt->getNumFlyStick();
    m_valuatorBase.push_back(0);
    m_buttonBase.push_back(0);
    for (size_t i = 0; i < m_numFlySticks; ++i)
    {
        DTrack_FlyStick_Type_d *f = dt->getFlyStick(i);
        m_valuatorBase.push_back(m_valuatorBase.back() + f->num_joystick);
        m_buttonBase.push_back(m_buttonBase.back() + f->num_button);
    }
    m_valuatorValues.resize(m_valuatorBase.back());
    m_valuatorRanges.resize(m_valuatorBase.back());
    for (size_t i = 0; i < m_valuatorRanges.size(); ++i)
    {
        m_valuatorRanges[i].first = -1.;
        m_valuatorRanges[i].second = 1.;
    }
    m_buttonStates.resize(m_buttonBase.back());

    m_numBodies = dt->getNumBody(); // Corect only if the bodies are tracked!
    cout << "Bodies: " << m_numBodies << ", Flysticks: " << m_numFlySticks << endl;

    m_bodyMatrices.resize(m_numFlySticks + m_numBodies);
    m_bodyMatricesValid.resize(m_numFlySticks + m_numBodies);
    m_bodyBase = m_numFlySticks;

    dt->startMeasurement();
}

//====================END of init section============================

//====================Hardware read methods============

/**
 * @brief DTrackDriver::getDTrackBodyMatrix Gets common DTrack body tracking data
 * @param mat   osg::Matrixd to store the data
 * @param devidx DTrack body index
 * @return 0 if it reads and tracks the body; otherwise returns -1
 */

template <class Data>
bool getDTrackMatrix(osg::Matrix &mat, const Data &d)
{

    if (d.quality < 0.)
        return false;

    mat.set(d.rot[0], d.rot[1], d.rot[2], 0,
            d.rot[3], d.rot[4], d.rot[5], 0,
            d.rot[6], d.rot[7], d.rot[8], 0,
            0, 0, 0, 1);

    osg::Vec3d pos(d.loc[0], d.loc[1], d.loc[2]);
    mat.setTrans(pos);

    return true;
}

bool DTrackDriver::updateBodyMatrix(size_t idx)
{
    // cout<<"!!!!!!!INputHDW getDTrackBodyMatrix STARTS!!!!!!!"<<devidx<<endl;
    //mat.makeIdentity();

    if (ssize_t(idx) >= dt->getNumBody())
    {
        std::cout << "!!!body id out of range " << idx << std::endl;
        return false;
    }
    /* DTrack API issue:
    * If a configured and calibrated tracking body won't be tracked durung DTrack init,
    * dt->getBody() will crash, and dt->getNumBody() won't return the right number.
    * DTrack API will think that this device doesn't exist until it's tracked.
    */
    DTrack_Body_Type_d *b = dt->getBody(idx);
    return getDTrackMatrix(m_bodyMatrices[m_bodyBase + idx], *b);
}

bool DTrackDriver::updateFlyStick(size_t idx)
{
    if (ssize_t(idx) >= dt->getNumFlyStick())
    {
        std::cout << "!!!flystick id out of range " << idx << std::endl;
        return false;
    }

    /* DTrack API issue:
    * If a configured and calibrated tracking body won't be tracked durung DTrack init,
    * dt->getBody() will crash, and dt->getNumBody() won't return the right number.
    * DTrack API will think that this device doesn't exist until it's tracked.
    */
    DTrack_FlyStick_Type_d *f = dt->getFlyStick(idx);

    for (int i = 0; i < f->num_button; ++i)
    {
        m_buttonStates[i + m_buttonBase[idx]] = f->button[i] != 0 ? true : false;
    }

    for (int i = 0; i < f->num_joystick; ++i)
    {
        m_valuatorValues[i + m_valuatorBase[idx]] = f->joystick[i];
    }

    return getDTrackMatrix(m_bodyMatrices[idx], *f);
}

DTrackDriver::~DTrackDriver()
{
    stopLoop();
    dt->stopMeasurement();
    delete dt;
}

//==========================main loop =================

/**
 * @brief DTrackDriver::run ImputHdw main loop
 *
 * Gets the status of the input devices
 */
bool DTrackDriver::poll()
{
    if (!dt)
        return false;

    if (!dt->receive())
    {
        // error messages from example
        // cout<<"DTrack input error!"<<endl;

        if (dt->getLastDataError() == DTrackSDK::ERR_TIMEOUT)
        {
            cout << "--- timeout while waiting for tracking data" << endl;
            //return -1;
        }

        if (dt->getLastDataError() == DTrackSDK::ERR_NET)
        {
            cout << "--- error while receiving tracking data" << endl;
            //return -1;
        }

        if (dt->getLastDataError() == DTrackSDK::ERR_PARSE)
        {
            cout << "--- error while parsing tracking data" << endl;
            //return -1;
        }

        return true;
    }

    size_t numMat = dt->getNumBody() + dt->getNumFlyStick();
    if (m_bodyMatrices.size() < numMat)
    {
        m_mutex.lock();
        m_bodyMatrices.resize(numMat);
        m_bodyMatricesValid.resize(numMat);
        m_mutex.unlock();
    }

    m_mutex.lock();
    for (int i = 0; i < dt->getNumFlyStick(); ++i)
    {
        m_bodyMatricesValid[i] = updateFlyStick(i);
    }
    m_mutex.unlock();

    m_mutex.lock();
    for (int i = 0; i < dt->getNumBody(); ++i)
    {
        m_bodyMatricesValid[m_bodyBase+i] = updateBodyMatrix(i);
    }
    m_mutex.unlock();

    return true;
}

INPUT_PLUGIN(DTrackDriver)
