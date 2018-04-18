/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Line.h
 *
 *  Created on: Jan 2, 2012
 *      Author: jw_te
 */

#pragma once

#include <string>
#include <memory>
#include <boost/optional.hpp>

namespace KardanikXML
{

class BodyJointDesc;

class Joint : public std::enable_shared_from_this<Joint>
{
private:
public:
    Joint();
    Joint(std::shared_ptr<BodyJointDesc> bodyA, std::shared_ptr<BodyJointDesc> bodyB);

    enum Axis
    {
        NO_AXIS,
        X_AXIS,
        Y_AXIS,
        Z_AXIS
    };

    void SetAxis(Axis axis);
    void SetAxis(const std::string &axis);
    Axis GetAxis() const;

    void SetUpperLimit(float upperLimit);
    boost::optional<float> GetUpperLimit() const;

    void SetLowerLimit(float lowerLimit);
    boost::optional<float> GetLowerLimit() const;

    void SetInitialAngle(float angle);
    boost::optional<float> GetInitialAngle() const;

    void SetBodyJointDescA(std::shared_ptr<BodyJointDesc> bodyA);
    std::shared_ptr<BodyJointDesc> GetBodyA() const;

    void SetBodyJointDescB(std::shared_ptr<BodyJointDesc> bodyB);
    std::shared_ptr<BodyJointDesc> GetBodyB() const;

private:
    std::shared_ptr<BodyJointDesc> m_BodyA;
    std::shared_ptr<BodyJointDesc> m_BodyB;

    boost::optional<float> m_UpperLimit;
    boost::optional<float> m_LowerLimit;
    boost::optional<float> m_InitialAngle;

    Axis m_Axis;
};
}
