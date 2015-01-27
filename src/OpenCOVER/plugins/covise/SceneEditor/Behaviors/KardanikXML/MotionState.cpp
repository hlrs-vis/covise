/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * MotionState.cpp
 *
 *  Created on: 11.10.2012
 *      Author: jogo
 */

#include "MotionState.h"

#include "../KinematicsBehavior.h"

#include <osg/MatrixTransform>
#include <iostream>

using namespace std;

namespace
{
ostream &operator<<(ostream &os, const btVector3 &vec)
{
    os << "[Vector3: " << vec[0] << ", " << vec[1] << ", " << vec[2] << "]";
    return os;
}

ostream &operator<<(ostream &os, const btMatrix3x3 &matrix)
{
    os << "[Matrix3x3: " << endl;
    os << matrix.getRow(0) << endl;
    os << matrix.getRow(1) << endl;
    os << matrix.getRow(2) << endl;
    os << "]" << endl;
    return os;
}

ostream &operator<<(ostream &os, const btTransform &transform)
{
    os << transform.getBasis() << endl;
    os << transform.getOrigin() << endl;
    return os;
}
}

namespace KardanikXML
{

MotionState::MotionState(osg::MatrixTransform *anchorNode, KinematicsBehavior *kinematicsBehavior, const btTransform &centerOfMass,
                         const btTransform &centerOfMassOffset, const std::string &bodyName)
    : m_KinematicsBehavior(kinematicsBehavior)
    , m_AnchorNode(anchorNode)
    , m_CenterOfMassOffset(centerOfMassOffset)
    , m_CenterOfMass(centerOfMass)
    , m_BodyName(bodyName)
{
    setWorldTransform(centerOfMass);
}

void MotionState::getWorldTransform(btTransform &centerOfMassWorldTrans) const
{
    centerOfMassWorldTrans = m_CenterOfMass;
}

void MotionState::setWorldTransform(const btTransform &centerOfMassWorldTrans)
{
    // btTransform bulletGraphicsWorldTrans = btTransform(btQuaternion(btVector3(1,0,0),  M_PI/2.0)) * centerOfMassWorldTrans * m_CenterOfMassOffset;
    btTransform bulletGraphicsWorldTrans = centerOfMassWorldTrans
                                           * m_CenterOfMassOffset;

    static float oglMatrix[16];
    bulletGraphicsWorldTrans.getOpenGLMatrix(oglMatrix);
    osg::Matrixf anchorWorldTransform;
    anchorWorldTransform.set(oglMatrix);

    // m to mm
    osg::Vec3f trans = anchorWorldTransform.getTrans();
    trans *= 1000.0f;
    anchorWorldTransform.setTrans(trans);
    if (m_AnchorNode)
    {
        m_KinematicsBehavior->SetWorldTransformOfAnchor(m_AnchorNode,
                                                        anchorWorldTransform);
    }
}
}
