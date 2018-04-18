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
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "DTrackDriver.h"

#include <config/CoviseConfig.h>

#include <iostream>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT
#include <algorithm> // for min/max

using namespace std;
using namespace covise;
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

void DTrackDriver::initArrays()
{
    m_numFlySticks = dt->getNumFlyStick();
    m_valuatorBase.push_back(0);
    m_buttonBase.push_back(0);
    for (size_t i = 0; i < m_numFlySticks; ++i)
    {
        DTrack_FlyStick_Type_d *f = dt->getFlyStick(i);
        m_valuatorBase.push_back(m_valuatorBase.back() + f->num_joystick);
        m_buttonBase.push_back(m_buttonBase.back() + f->num_button);
    }
    m_handButtonBase.push_back(m_buttonBase.back()); // Hand "buttons" bases start from the end of flstk bases

    m_valuatorValues.resize(m_valuatorBase.back());
    m_valuatorRanges.resize(m_valuatorBase.back());
    for (size_t i = 0; i < m_valuatorRanges.size(); ++i)
    {
        m_valuatorRanges[i].first = -1.;
        m_valuatorRanges[i].second = 1.;
    }
    m_buttonStates.resize(m_buttonBase.back());

    //
    m_numHands = dt->getNumHand();
    cout<<"DTrack hands calibrated:"<<m_numHands<<endl;
    for (size_t i=0;i<m_numHands;++i)
    {
    	DTrack_Hand_Type_d *h = dt->getHand(i);
    	m_handButtonBase.push_back(m_handButtonBase.back()+2); //each hand will have 2 virtual "buttons"
    }

    m_buttonStates.resize(m_handButtonBase.back()); //adding values for hand "buttons"
}

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

    initArrays();
    //


    m_numBodies = 0;
    m_numFlySticks = 0;
    m_flystickBase = 0;
    m_bodyBase = m_flystickBase+m_numFlySticks;

    m_numHands = 0;
    m_handBase = m_bodyBase+m_numFlySticks;

    dt->startMeasurement();

    poll(); // try to retrieve number of bodies/flysticks
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
bool DTrackDriver::updateHand(size_t idx)
{
	if (ssize_t(idx)>= dt->getNumHand())
	{
		std::cout<<"!!! Hand id is out of range: "<<idx<<std::endl;
	}
	DTrack_Hand_Type_d *h=dt->getHand(idx);

	osg::Vec3d fingerpos[3]; //using 3 fingers now

	for(int n=0;n<3;++n)
		fingerpos[n]=osg::Vec3d(h->finger[n].loc[0],h->finger[n].loc[1],h->finger[n].loc[2]);

	// grab with fingers no. 1&2 -> first button, with 1&3 -> second button
	double dist12=(fingerpos[0]-fingerpos[1]).length();
	double dist13=(fingerpos[0]-fingerpos[2]).length();


	//std::cout<<"ft 1st:"  <<dist12<<" 2nd:"<< dist13<<endl;

	// 50mm for very bad precision, 23mm -- for good one
	// 17..19 mm is a minimal distance between fingers
	m_buttonStates[0 + m_handButtonBase[idx]] = (dist12>0)&&(dist12<23.0);
	m_buttonStates[1 + m_handButtonBase[idx]] = (dist12>0)&&(dist13<23.0);


	return getDTrackMatrix(m_bodyMatrices[m_handBase+idx],*h);
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

    if (dt->getNumBody() != m_numBodies || dt->getNumFlyStick() != m_numFlySticks ||
    		dt->getNumHand() != m_numHands )
    {
        m_mutex.lock();
        m_numFlySticks = dt->getNumFlyStick();
        m_numBodies = dt->getNumBody();
        m_bodyBase = m_numFlySticks;
        initArrays();

        m_numHands = dt->getNumHand();
        m_handBase = m_bodyBase + m_numBodies;

        m_bodyBase = coCoviseConfig::getInt("bodyBase", configPath(), m_bodyBase);
        m_flystickBase = coCoviseConfig::getInt("flystickBase", configPath(), m_flystickBase);
        m_handBase = coCoviseConfig::getInt("handBase", configPath(), m_handBase);

        if (m_bodyBase < 0)
        {
            std::cerr << "bodyBase " << m_bodyBase << " below zero" << std::endl;
            m_bodyBase = 0;
        }
        if (m_handBase < 0)
        {
            std::cerr << "handBase " << m_handBase << " below zero" << std::endl;
            m_handBase = 0;
        }
        if (m_flystickBase < 0)
        {
            std::cerr << "flystickBase " << m_flystickBase << " below zero" << std::endl;
            m_flystickBase = 0;
        }
        // FIXME: check that indices don't overlap

        const size_t numMat = std::max(std::max(m_bodyBase+m_numBodies, m_flystickBase+m_numFlySticks), m_handBase+m_numHands);
        m_bodyMatrices.resize(numMat);
        m_bodyMatricesValid.resize(numMat);
        m_mutex.unlock();
    }

    bool valid = false;
    m_mutex.lock();
    for (int i = 0; i < dt->getNumFlyStick(); ++i)
    {
        m_bodyMatricesValid[i] = updateFlyStick(i);
        if (m_bodyMatricesValid[i])
            valid = true;
    }
    m_mutex.unlock();

    m_mutex.lock();
    for (int i = 0; i < dt->getNumBody(); ++i)
    {
        m_bodyMatricesValid[m_bodyBase+i] = updateBodyMatrix(i);
        if (m_bodyMatricesValid[m_bodyBase+i])
            valid = true;
    }
    m_mutex.unlock();

    //hands matrices update

    m_mutex.lock();
	for (int i = 0; i < dt->getNumHand(); ++i)
	{
		m_bodyMatricesValid[m_handBase+i] = updateHand(i);
        if (m_bodyMatricesValid[m_handBase+i])
            valid = true;
    }
    m_valid = valid;
    m_mutex.unlock();

    return true;
}

INPUT_PLUGIN(DTrackDriver)
