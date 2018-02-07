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

namespace opencover
{

InputSource::InputSource(const std::string &name, const std::string &kind)
: m_name(name)
, m_conf("COVER.Input." + kind + "." + name)
, m_dev(NULL)
{
    const std::string driver = covise::coCoviseConfig::getEntry("device", config(), "default");

    if (name == "Mouse")
    {
        m_valid = true;
    }
    else
    {
        m_dev = Input::instance()->getDevice(driver);
        if (!m_dev)
            m_dev = Input::instance()->getDevice("const");
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
    return m_dev;
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
        m_valid = device()->isValid();
    if (Input::debug(Input::Raw) && m_valid != m_oldValid)
    {
        std::cerr << "Input: raw " << name() << " valid=" << m_valid << std::endl;
    }
    m_oldValid = m_valid;
}


}
