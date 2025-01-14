/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputsource.cpp
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#include "input.h"
#include "inputsource.h"
#include "inputdevice.h"
#include <config/CoviseConfig.h>

namespace vive
{

InputSource::InputSource(const std::string &name, const std::string &kind)
: m_name(name)
, m_conf("COVER.Input." + kind + "." + name)
{
    const std::string driver = covise::coCoviseConfig::getEntry("device", config(), "default");

    if (name == "Mouse")
    {
        m_valid = true;
    }
    else
    {
        m_dev.push_back(Input::instance()->getDevice(driver));
        if (!m_dev[0])
            m_dev[0] = Input::instance()->getDevice("const");
    }
    for(int i=1;i<10;i++)
    {
        const std::string driver = covise::coCoviseConfig::getEntry("device"+std::to_string(i), config(), "");
        if(driver.length()>0)
        {
            if (Input::debug(Input::Config))
            {
                std::cerr << "Input: device" << i << " configured as " << driver << std::endl;
            }
            InputDevice *dev = Input::instance()->getDevice(driver);
            if(dev)
                m_dev.push_back(dev);
        }
        else
            break;
    }
}

InputSource::~InputSource()
{
}

const std::string &InputSource::name() const
{
    return m_name;
}

const std::string &InputSource::config() const
{
    return m_conf;
}

InputDevice *InputSource::device() const
{
    if(m_validDevice>=0 && m_validDevice<m_dev.size())
        return m_dev[m_validDevice];
    
    return nullptr;
}
InputDevice *InputSource::device(int i) const
{
    if(i<m_dev.size())
        return m_dev[i];
    
    return nullptr;
}

void InputSource::setValid(bool valid)
{
    m_valid = valid;
}

bool InputSource::isValid() const
{
    return m_valid;
}

void InputSource::update()
{
    if (device())
    {
        m_valid = false;
        for(int i=0;i<m_dev.size();i++)
	{
	    if(m_dev[i]->isValid())
	    {
	        m_valid = true;
		m_validDevice = i;
		break;
	    }
	}
    }
    if (Input::debug(Input::Raw) && m_valid != m_oldValid)
    {
        std::cerr << "Input: raw " << name() << " valid=" << m_valid << std::endl;
    }
    m_oldValid = m_valid;
}


}
