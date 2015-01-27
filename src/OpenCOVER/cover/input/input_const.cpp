/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "input_const.h"

namespace opencover
{

ConstInputDevice::ConstInputDevice(const std::string &name)
    : InputDevice(name)
{
    osg::Matrix mat = osg::Matrix::identity();
    m_bodyMatrices.push_back(mat);
    m_bodyMatrices.push_back(mat);

    m_isVarying = false;
    m_is6Dof = false;
}

bool ConstInputDevice::needsThread() const
{

    return false;
}
}
