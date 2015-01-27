/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BILLBOARD_H
#define CO_BILLBOARD_H

/*! \file
 \brief  osg billboard group node

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   2004
 */

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

/*
     Billboards can either rotate about an axis or a point.

     Axial billboards rotate about the axis specified by coBillboard::setAxis.
     The rotation is about the origin (0,0,0).  In all cases,
     if the geometry is modeled in the XY plane, with -Z forward, then
     the billboard is rotated so that the normal(Z by default) axis points
     back to the eye point.
     The +Y axis is the default axis of rotation.  An axial rotate
     billboard is specified by setting the rotation mode of the billboard to
     the value coBillboard::AXIAL_ROT using coBillboard::setMode.  The axis of
rotation (x, y, z) is specified using coBillboard::setAxis.
coBillboard::getAxis returns the axis of the coBillboard.

If the rotation mode of the billboard is set to coBillboard::POINT_ROT_EYE,
the billboard is rotated so that the +Z axis of the osg::Geometry aligns
as closely as possible with the axis specified with
pfBillboard::setAxis, in eye space.  In particular, if the axis is
+Y (as it is by default) and you display a text, then, the text always
stays readable.

If the rotation mode of the billboard is set to
coBillboard::POINT_ROT_WORLD, the billboard is rotated so that the angle
between the +Y axis of the osg::Geometry and axis specified with
pfBillboard::setAxis (in world space) is minimized.

Both coBillboard::AXIAL_ROT and coBillboard::POINT_ROT_WORLD billboards
may "spin" about the rotation or alignment axis when viewed along that
axis.
*/

#include <osg/Transform>
#include <osg/Quat>
#include <util/coExport.h>

namespace opencover
{
/** coBillboard is a derived form of Transform that automatically
  * scales or rotates to keep its children aligned with screen coordinates.
*/
class COVEREXPORT coBillboard : public osg::Transform
{
public:
    coBillboard();

    coBillboard(const coBillboard &pat, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);

    virtual osg::Object *cloneType() const
    {
        return new coBillboard();
    }
    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new coBillboard(*this, copyop);
    }
    virtual bool isSameKindAs(const osg::Object *obj) const
    {
        return dynamic_cast<const coBillboard *>(obj) != NULL;
    }
    virtual const char *className() const
    {
        return "coBillboard";
    }
    virtual const char *libraryname() const
    {
        return "OpenCOVER";
    }

    virtual void accept(osg::NodeVisitor &nv);

    enum RotationMode
    {
        STOP_ROT = 0,
        AXIAL_ROT,
        POINT_ROT_EYE,
        POINT_ROT_WORLD
    };

    // rotation mode
    void setMode(RotationMode mode);
    RotationMode getMode() const;

    // rotation axis for AXIAL_ROT billboard,
    // "up" vector for POINT_ROT billboards
    // default ist +Z (0 0 1)
    void setAxis(const osg::Vec3 &axis);
    const osg::Vec3 &getAxis() const;

    // the normal of the geometry
    // default is -Y (0 -1 0)
    void setNormal(const osg::Vec3 &normal);
    const osg::Vec3 &getNormal() const;

    virtual bool computeLocalToWorldMatrix(osg::Matrix &matrix, osg::NodeVisitor *nv) const;

    virtual bool computeWorldToLocalMatrix(osg::Matrix &matrix, osg::NodeVisitor *nv) const;

protected:
    virtual ~coBillboard()
    {
    }

    osg::Vec3 _axis;
    osg::Vec3 _normal;

    RotationMode _mode;

    mutable osg::Quat _rotation;

    mutable osg::Matrix _cachedMatrix;
    mutable osg::Matrix _cachedInvMatrix;

private:
    void checkAndAdjustNormal();
};
}
#endif
