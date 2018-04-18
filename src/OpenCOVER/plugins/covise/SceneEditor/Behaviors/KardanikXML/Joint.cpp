/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Joint.h"

#include <stdexcept>

#include "BodyJointDesc.h"
#include "Body.h"

using namespace std;

namespace KardanikXML
{

Joint::Joint()
    : m_Axis(NO_AXIS)
{
}

Joint::Joint(std::shared_ptr<BodyJointDesc> bodyA, std::shared_ptr<BodyJointDesc> bodyB)
    : m_BodyA(bodyA)
    , m_BodyB(bodyB)
{
}

void Joint::SetUpperLimit(float upperLimit)
{
    m_UpperLimit = upperLimit;
}

boost::optional<float> Joint::GetUpperLimit() const
{
    return m_UpperLimit;
}

void Joint::SetLowerLimit(float lowerLimit)
{
    m_LowerLimit = lowerLimit;
}

boost::optional<float> Joint::GetLowerLimit() const
{
    return m_LowerLimit;
}

void Joint::SetInitialAngle(float angle)
{
    m_InitialAngle = angle;
}

boost::optional<float> Joint::GetInitialAngle() const
{
    return m_InitialAngle;
}

void Joint::SetBodyJointDescA(std::shared_ptr<BodyJointDesc> bodyA)
{
    m_BodyA = bodyA;
    m_BodyA->GetBody()->AddConnectedJoint(shared_from_this());
}

std::shared_ptr<BodyJointDesc> Joint::GetBodyA() const
{
    return m_BodyA;
}

void Joint::SetBodyJointDescB(std::shared_ptr<BodyJointDesc> bodyB)
{
    m_BodyB = bodyB;
    m_BodyB->GetBody()->AddConnectedJoint(shared_from_this());
}

void Joint::SetAxis(Axis axis)
{
    m_Axis = axis;
}

void Joint::SetAxis(const string &axis)
{
    if (axis == "NO_AXIS")
    {
        m_Axis = NO_AXIS;
    }
    else if (axis == "X_AXIS")
    {
        m_Axis = X_AXIS;
    }
    else if (axis == "Y_AXIS")
    {
        m_Axis = Y_AXIS;
    }
    else if (axis == "Z_AXIS")
    {
        m_Axis = Z_AXIS;
    }
    else
    {
        throw invalid_argument("Wrong axis specification.");
    }
}

Joint::Axis Joint::GetAxis() const
{
    return m_Axis;
}

std::shared_ptr<BodyJointDesc> Joint::GetBodyB() const
{
    return m_BodyB;
}
}
