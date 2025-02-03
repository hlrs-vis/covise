/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once


#include <string>
#include <util/coExport.h>
#include "inputsource.h"

namespace vive
{

class InputDevice;

class VVCORE_EXPORT Valuator: public InputSource
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

    size_t m_idx;
    double m_value=0., m_oldValue = 0.;
    double m_min=-1., m_max=1.;
};
}
