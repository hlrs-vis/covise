/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * valuator.h
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifndef VALUATOR_H
#define VALUATOR_H

#include <string>

namespace opencover
{

class InputDevice;

class Valuator
{
    friend class Input;

public:
    double getValue() const;
    std::pair<double, double> getRange() const;

private:
    Valuator(const std::string &name);

    void update();
    void setValue(double value);
    void setRange(double min, double max);

    InputDevice *m_dev;
    size_t m_idx;
    double m_value;
    double m_min, m_max;
};
}
#endif
