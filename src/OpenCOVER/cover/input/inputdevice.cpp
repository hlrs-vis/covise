/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputdevice.cpp
 *
 *  Created on: Dec 9, 2014
 *      Author: svnvlad
 */
#include "inputdevice.h"

#include <config/CoviseConfig.h>

#include <iostream>
#include <sstream>
#include <cassert>
#include <limits>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

#ifdef __linux
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#endif


using namespace std;
using namespace covise;

namespace opencover
{

osg::Matrix InputDevice::s_identity = osg::Matrix::identity();

InputDevice::InputDevice(const std::string &config)
    : loop_is_running(true)
    , m_config(config)
    , m_valid(false)
    , m_offsetMatrix(osg::Matrix::identity())
    , m_isVarying(true)
    , m_is6Dof(false)
    , m_validFrame(m_valid)
{
    if(m_config.compare(0,19,"COVER.Input.Device.")==0)
        m_name = m_config.substr(19,std::string::npos);
    else
        m_name = m_config;
    // system offset and orientation
    float trans[3];
    trans[0] = coCoviseConfig::getFloat("x", configPath("Offset"), 0);
    trans[1] = coCoviseConfig::getFloat("y", configPath("Offset"), 0);
    trans[2] = coCoviseConfig::getFloat("z", configPath("Offset"), 0);

    float rot[3];
    rot[0] = coCoviseConfig::getFloat("h", configPath("Orientation"), 0);
    rot[1] = coCoviseConfig::getFloat("p", configPath("Orientation"), 0);
    rot[2] = coCoviseConfig::getFloat("r", configPath("Orientation"), 0);

	for (int i = 0; i < 3; i++)
	{
        std::stringstream str;
        str << "CallibrationPoint." << i;
		std::string cPath = configPath(str.str());
		m_calibrationPoints[i].x() = coCoviseConfig::getFloat("x", cPath, 0);
		m_calibrationPoints[i].y() = coCoviseConfig::getFloat("y", cPath, 0);
		m_calibrationPoints[i].z() = coCoviseConfig::getFloat("z", cPath, 0);
		m_calibrationPointNames[i] = coCoviseConfig::getEntry("info", cPath, "noName");
	}

    //cout<<" Offset=("<<trans[0]<<" "<<trans[1]<<" "<<trans[2]<<") " <<" Orientation=("<<rot[0]<<" "<<rot[1]<<" "<<rot[2]<<") "<<endl;

    MAKE_EULER_MAT(m_offsetMatrix, rot[0], rot[1], rot[2]);
    //fprintf(stderr, "offset from device('%d) %f %f %f\n", device_ID, deviceOffsets[device_ID].trans[0], deviceOffsets[device_ID].trans[1], deviceOffsets[device_ID].trans[2]);

    osg::Matrix translationMat;
    translationMat.makeTranslate(trans[0], trans[1], trans[2]);
    m_offsetMatrix.postMult(translationMat);
}

void InputDevice::setOffsetMat(const osg::Matrix &m)
{
    m_offsetMatrix = m;
}

string &InputDevice::getCalibrationPointName(int i)
{
    return m_calibrationPointNames[i];
}

osg::Vec3 &InputDevice::getCalibrationPoint(int i)
{
    return m_calibrationPoints[i];
}

InputDevice::~InputDevice()
{
    stopLoop();
}

const osg::Matrix &InputDevice::getOffsetMat() const
{
    return m_offsetMatrix;
}

bool InputDevice::needsThread() const
{

    return true;
}

bool InputDevice::isVarying() const
{

    return m_isVarying;
}

bool InputDevice::is6Dof() const
{

    return m_is6Dof;
}

const string &InputDevice::getName() const
{
    return m_name;
}

std::string InputDevice::configPath(const std::string &ent) const
{

    if (ent.empty())
        return m_config;

    return m_config + "." + ent;
}

//==========================main loop =================

/**
 * @brief InputDevice::stopLoop Stops the main loop
 */
void InputDevice::stopLoop()
{
    m_mutex.lock();
    loop_is_running = false; // stop the main loop
    m_mutex.unlock();

    while (isRunning())
    {

        OpenThreads::Thread::microSleep(1000); //wait for the main loop stop
        //cout<<"stopping....."<<endl;
    }
}

bool InputDevice::isValid() const
{
    return m_validFrame;
}

/**
 * @brief InputDevice::run InputDevice main loop
 *
 * Gets the status of the input devices
 */
void InputDevice::run()
{
#ifdef __linux
#if __GLIBC__>=2 && __GLIBC_MINOR__>=12
    pthread_setname_np(pthread_self(), m_name.c_str());
#endif
#endif

    bool again = true;
    while (again)
    {
        {
            m_mutex.lock();
            again = loop_is_running;
            m_mutex.unlock();
        }
        if (!poll())
            again = false;
        else
            OpenThreads::Thread::microSleep(5000);
    }
}

bool InputDevice::poll()
{

    return true;
}

void InputDevice::update()
{
    m_mutex.lock();

    m_validFrame = m_valid;

    if (m_bodyMatricesRelativeFrame.size() != m_bodyMatricesRelative.size())
        m_bodyMatricesRelativeFrame.resize(m_bodyMatricesRelative.size());
    std::copy(m_bodyMatricesRelative.begin(), m_bodyMatricesRelative.end(), m_bodyMatricesRelativeFrame.begin());

    if (m_bodyMatricesValidFrame.size() != m_bodyMatricesValid.size())
        m_bodyMatricesValidFrame.resize(m_bodyMatricesValid.size());
    std::copy(m_bodyMatricesValid.begin(), m_bodyMatricesValid.end(), m_bodyMatricesValidFrame.begin());
    if (m_bodyMatricesFrame.size() != m_bodyMatrices.size())
        m_bodyMatricesFrame.resize(m_bodyMatrices.size());
    for (size_t i = 0; i < m_bodyMatrices.size(); ++i)
    {
        m_bodyMatricesFrame[i] = m_bodyMatrices[i] * m_offsetMatrix;
    }

    if (m_buttonStatesFrame.size() != m_buttonStates.size())
        m_buttonStatesFrame.resize(m_buttonStates.size());
    std::copy(m_buttonStates.begin(), m_buttonStates.end(), m_buttonStatesFrame.begin());

    if (m_valuatorValuesFrame.size() != m_valuatorValues.size())
        m_valuatorValuesFrame.resize(m_valuatorValues.size());
    std::copy(m_valuatorValues.begin(), m_valuatorValues.end(), m_valuatorValuesFrame.begin());

    if (m_valuatorRangesFrame.size() != m_valuatorRanges.size())
        m_valuatorRangesFrame.resize(m_valuatorRanges.size());
    std::copy(m_valuatorRanges.begin(), m_valuatorRanges.end(), m_valuatorRangesFrame.begin());

    assert(m_valuatorValues.size() == m_valuatorRanges.size());

    m_mutex.unlock();
}

bool InputDevice::getButtonState(size_t num) const
{

    if (num >= m_buttonStatesFrame.size())
        return false;

    return m_buttonStatesFrame[num];
}

double InputDevice::getValuatorValue(size_t num) const
{

    if (num >= m_valuatorValuesFrame.size())
        return 0.;

    return m_valuatorValuesFrame[num];
}

std::pair<double, double> InputDevice::getValuatorRange(size_t num) const
{

    if (num >= m_valuatorRangesFrame.size())
        return std::make_pair(-std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max());

    return m_valuatorRangesFrame[num];
}

bool InputDevice::isBodyMatrixValid(size_t idx) const
{
    if (idx >= m_bodyMatricesValidFrame.size())
        return false;

    return m_bodyMatricesValidFrame[idx];
}

bool InputDevice::isBodyMatrixRelative(size_t idx) const
{
    if (idx >= m_bodyMatricesRelativeFrame.size())
        return false;

    return m_bodyMatricesRelativeFrame[idx];
}

const osg::Matrix &InputDevice::getBodyMatrix(size_t idx) const
{

    if (idx >= m_bodyMatricesFrame.size())
        return s_identity;

    return m_bodyMatricesFrame[idx];
}

DriverFactoryBase::DriverFactoryBase(const std::string &name)
    : m_name(name)
    , m_handle(0)
{
    //std::cerr << "Input: new driver factory \"" << m_name << "\"" << std::endl;
}

DriverFactoryBase::~DriverFactoryBase()
{
}

void DriverFactoryBase::setLibHandle(CO_SHLIB_HANDLE handle)
{

    m_handle = handle;
}

const std::string &DriverFactoryBase::name() const
{

    return m_name;
}

CO_SHLIB_HANDLE DriverFactoryBase::getLibHandle() const
{
    return m_handle;
}
}
