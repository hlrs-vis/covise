/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2005 Robert Osfield 
 *
 * This library is open source and may be redistributed and/or modified under  
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or 
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * OpenSceneGraph Public License for more details.
*/
#include "coBillboard.h"
#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include <osg/CullStack>
#include <iostream>

using namespace osg;
using namespace covise;
using namespace opencover;

coBillboard::coBillboard()
    : _axis(0., 0., 1.)
    , _normal(0., -1., 0.)
    , _mode(AXIAL_ROT)
{
    //    setNumChildrenRequiringUpdateTraversal(1);
}

coBillboard::coBillboard(const coBillboard &pat, const CopyOp &copyop)
    : Transform(pat, copyop)
    , _axis(pat._axis)
    , _normal(pat._normal)
    , _mode(pat._mode)
{
    //    setNumChildrenRequiringUpdateTraversal(getNumChildrenRequiringUpdateTraversal()+1);
}

bool coBillboard::computeLocalToWorldMatrix(Matrix &matrix, NodeVisitor *) const
{
    if (_referenceFrame == RELATIVE_RF)
    {
        matrix.preMult(_cachedMatrix);
    }
    else // absolute
    {
        matrix = _cachedMatrix;
    }
    return true;
}

bool coBillboard::computeWorldToLocalMatrix(Matrix &matrix, NodeVisitor *) const
{
    if (_referenceFrame == RELATIVE_RF)
    {
        matrix.postMult(_cachedInvMatrix);
    }
    else // absolute
    {
        matrix = _cachedInvMatrix;
    }
    return true;
}

void coBillboard::setMode(coBillboard::RotationMode mode)
{
    _mode = mode;
}

coBillboard::RotationMode coBillboard::getMode() const
{
    return _mode;
}

void coBillboard::setAxis(const osg::Vec3 &axis)
{
    _axis = axis;
    checkAndAdjustNormal();
}

const osg::Vec3 &coBillboard::getAxis() const
{
    return _axis;
}

void coBillboard::setNormal(const osg::Vec3 &normal)
{
    _normal = normal;
    checkAndAdjustNormal();
}

const osg::Vec3 &coBillboard::getNormal() const
{
    return _normal;
}

void coBillboard::checkAndAdjustNormal()
{
    _axis.normalize();
    _normal.normalize();

    if (fabs(_normal * _axis) < 1e-6)
        return;

    _normal -= _axis * (_normal * (_axis));
    if (_normal.length() > 1e-3)
    {
        _normal.normalize();
    }
    else
    {
        if (fabs(_axis[0]) > 0.)
        {
            _normal[0] = 0.;
            _normal[1] = 1.;
        }
        else
        {
            _normal[0] = 1.;
            _normal[1] = 0.;
        }
        _normal.normalize();
        _normal -= _axis * (_normal * (_axis));
    }

    if (fabs(_normal * (_axis)) > 1e-6)
    {
        std::cerr << "coBillboard::checkAndAdjustNormal(): uups" << std::endl;
    }
}

void coBillboard::accept(NodeVisitor &nv)
{
    // if app traversal update the frame count.
    if (nv.getVisitorType() == NodeVisitor::UPDATE_VISITOR)
    {
    }
    else if (nv.getVisitorType() == NodeVisitor::CULL_VISITOR)
    {

        CullStack *cs = dynamic_cast<CullStack *>(&nv);
        if (cs)
        {
            osg::Vec3 eyePoint = cs->getEyeLocal();
            //osg::Vec3 localUp = cs->getUpLocal();

            //const osg::Matrix& projection = cs->getProjectionMatrix();

            if (_mode == AXIAL_ROT)
            {
                osg::Vec3 PosToEye = eyePoint;
                float t = -((PosToEye * _axis) / (_axis * _axis));
                osg::Vec3 vPro = PosToEye + _axis * t;
                vPro.normalize();

                float angle = acos(vPro * _normal);

                osg::Vec3 tmp = (vPro ^ _normal);

                if ((tmp * _axis) > 0)
                    angle = -angle;

                _cachedMatrix.makeRotate(angle, _axis[0], _axis[1], _axis[2]);
                //           _cachedInvMatrix.invert(_cachedMatrix);
            }
            else if (_mode == POINT_ROT_WORLD)
            {
                //osg::Quat rotation;
                osg::Matrix tmpMat;
                if(coVRConfig::instance()->getEnvMapMode() == coVRConfig::NONE)
                {
                    tmpMat = *cs->getModelViewMatrix();
                }
                else
                    tmpMat = (*cs->getModelViewMatrix() * cover->envCorrectMat);
                //tmpMat.get(rotation);

                Matrix axisNormal;
                Matrix rotMat;
                axisNormal.makeIdentity();
                Vec3 perp = _axis ^ _normal;
                perp.normalize();
                axisNormal(0, 0) = perp[0];
                axisNormal(0, 1) = perp[1];
                axisNormal(0, 2) = perp[2];
                axisNormal(1, 0) = -_normal[0];
                axisNormal(1, 1) = -_normal[1];
                axisNormal(1, 2) = -_normal[2];
                axisNormal(2, 0) = _axis[0];
                axisNormal(2, 1) = _axis[1];
                axisNormal(2, 2) = _axis[2];

                rotMat.makeIdentity();
                Vec3 tmpVec1;
                Vec3 tmpVec2;
                Vec3 tmpVec3;
                tmpVec1.set(tmpMat(0, 0), tmpMat(0, 1), tmpMat(0, 2));
                tmpVec1.set(tmpMat(0, 0), tmpMat(0, 1), tmpMat(0, 2));
                tmpVec1.set(tmpMat(0, 0), tmpMat(0, 1), tmpMat(0, 2));
                tmpVec2.set(tmpMat(1, 0), tmpMat(1, 1), tmpMat(1, 2));
                tmpVec2.set(tmpMat(1, 0), tmpMat(1, 1), tmpMat(1, 2));
                tmpVec2.set(tmpMat(1, 0), tmpMat(1, 1), tmpMat(1, 2));
                tmpVec3.set(tmpMat(2, 0), tmpMat(2, 1), tmpMat(2, 2));
                tmpVec3.set(tmpMat(2, 0), tmpMat(2, 1), tmpMat(2, 2));
                tmpVec3.set(tmpMat(2, 0), tmpMat(2, 1), tmpMat(2, 2));

                tmpVec1.normalize();
                tmpVec2.normalize();
                tmpVec3.normalize();

                rotMat(0, 0) = tmpVec1[0];
                rotMat(0, 1) = tmpVec1[1];
                rotMat(0, 2) = tmpVec1[2];
                rotMat(1, 0) = tmpVec2[0];
                rotMat(1, 1) = tmpVec2[1];
                rotMat(1, 2) = tmpVec2[2];
                rotMat(2, 0) = tmpVec3[0];
                rotMat(2, 1) = tmpVec3[1];
                rotMat(2, 2) = tmpVec3[2];

                //   rotMat.set(rotation);
                _cachedInvMatrix.set(rotMat * axisNormal);
                _cachedMatrix.set(Matrix::inverse(_cachedInvMatrix));
            }
            else if (_mode == POINT_ROT_EYE)
            {

                osg::Vec3 PosToEye = eyePoint;
                PosToEye.normalize();
                osg::Matrix matrix;
                osg::Matrix modelview;
                if(coVRConfig::instance()->getEnvMapMode() == coVRConfig::NONE)
                {
                    modelview = Matrix::inverse(*cs->getModelViewMatrix());
                }
                else
                    modelview = Matrix::inverse((*cs->getModelViewMatrix() * cover->envCorrectMat));
                Vec3 up(modelview(2, 0), modelview(2, 1), modelview(2, 2));
                Vec3 right(up ^ PosToEye);
                right.normalize();
                up = PosToEye ^ right;
                up.normalize();

                matrix(0, 0) = right.x();
                matrix(0, 1) = right.y();
                matrix(0, 2) = right.z();
                matrix(1, 0) = -PosToEye.x();
                matrix(1, 1) = -PosToEye.y();
                matrix(1, 2) = -PosToEye.z();
                matrix(2, 0) = up.x();
                matrix(2, 1) = up.y();
                matrix(2, 2) = up.z();

                Matrix axisNormal;
                axisNormal.makeIdentity();
                Vec3 perp = _axis ^ _normal;
                perp.normalize();
                axisNormal(0, 0) = perp[0];
                axisNormal(0, 1) = perp[1];
                axisNormal(0, 2) = perp[2];
                axisNormal(1, 0) = -_normal[0];
                axisNormal(1, 1) = -_normal[1];
                axisNormal(1, 2) = -_normal[2];
                axisNormal(2, 0) = _axis[0];
                axisNormal(2, 1) = _axis[1];
                axisNormal(2, 2) = _axis[2];

                _cachedInvMatrix.set(Matrix::inverse(matrix) * axisNormal);
                _cachedMatrix.set(Matrix::inverse(_cachedInvMatrix));
            }
        }
    }

    // now do the proper accept
    Transform::accept(nv);
}
