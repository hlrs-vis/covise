/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "valuator.h"
#include "inputdevice.h"
#include "input.h"
#include <config/CoviseConfig.h>
#include <limits>

namespace opencover
{

Valuator::Valuator(const std::string &name)
    : m_value(0.)
    , m_min(-std::numeric_limits<double>::max())
    , m_max(std::numeric_limits<double>::max())
{
    const std::string conf = "COVER.Input.Valuator." + name;
    const std::string device = covise::coCoviseConfig::getEntry("device", conf, "default");

    m_dev = Input::instance()->getDevice(device);
    if (!m_dev)
        m_dev = Input::instance()->getDevice("const");
    m_idx = covise::coCoviseConfig::getInt("valuatorIndex", conf, 0);
    std::cerr << "new valuator: conf=" << conf << ", dev=" << device << ", idx=" << m_idx << std::endl;
    if (m_idx >= m_dev->numValuators())
    {
        std::cerr << "Valuator: valuator index " << m_idx << " out of range - " << m_dev->numValuators() << " valuators" << std::endl;
        m_idx = 0;
    }
}

double Valuator::getValue() const
{

    return m_value;
}

std::pair<double, double> Valuator::getRange() const
{

    return std::make_pair(m_min, m_max);
}

void Valuator::setValue(const double val)
{

    m_value = val;
}

void Valuator::setRange(const double min, const double max)
{

    m_min = min;
    m_max = max;
}

void Valuator::update()
{

    m_value = m_dev->getValuatorValue(m_idx);
    std::pair<double, double> range = m_dev->getValuatorRange(m_idx);
    m_min = range.first;
    m_max = range.second;
}
}
