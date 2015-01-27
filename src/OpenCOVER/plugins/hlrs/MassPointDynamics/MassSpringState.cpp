/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MassSpringState.h"

#include <cmath>
#include <iostream>

MassSpringState::MassSpringState(const Vec3d &rN, const Vec3d &vN, const MassSpringParameters &pSet)
    : r_(rN)
    , v_(vN)
    , params_(pSet)
{
}

const Vec3d &MassSpringState::r() const
{
    return r_;
}

const Vec3d &MassSpringState::v() const
{
    return v_;
}

const MassSpringParameters &MassSpringState::params() const
{
    return params_;
}

MassSpringState MassSpringState::operator+(const MassSpringState &addState) const
{
    return MassSpringState(r_ + addState.r(), v_ + addState.v(), params_);
}

const MassSpringState &MassSpringState::operator+=(const MassSpringState &addState)
{
    r_ += addState.r();
    v_ += addState.v();
    return *this;
}

MassSpringState MassSpringState::operator*(double h) const
{
    return MassSpringState(r_ * h, v_ * h, params_);
}

MassSpringState MassSpringState::dstate(double) const
{
    Vec3d forceVec(0.0, 0.0, 0.0);

    for (int i = 0; i < linkVector.size(); ++i)
    {
        Vec3d distVec = linkVector[i]->r() - r_;
        double absDist = distVec.norm();

        forceVec += distVec * (pow(absDist - params_.l, 3) * params_.k * (1 / absDist));

        Vec3d relVelVec = linkVector[i]->v() - v_;
        forceVec += distVec * ((relVelVec * distVec) * params_.d / pow(absDist, 2));
    }

    Vec3d accVec = forceVec * (1 / params_.m);
    accVec += Vec3d(0.0, 0.0, -9.81);

    if (r_.z < 0.0)
    {
        accVec += Vec3d(1000 * r_.z * tanh(v_.x), 1000 * r_.z * tanh(v_.y), -(params_.k * pow(r_.z, 3) + params_.d * v_.z) / params_.m);
    }

    return MassSpringState(v_, accVec, params_);
}

void MassSpringState::addLink(MassSpringState &linkState)
{
    linkVector.push_back(&linkState);
}
