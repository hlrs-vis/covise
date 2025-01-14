/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

/*
     Billboards can either rotate about an axis or a point.

     Axial billboards rotate about the axis specified by vvBillboard::setAxis.
     The rotation is about the origin (0,0,0).  In all cases,
     if the geometry is modeled in the XY plane, with -Z forward, then
     the billboard is rotated so that the normal(Z by default) axis points
     back to the eye point.
     The +Y axis is the default axis of rotation.  An axial rotate
     billboard is specified by setting the rotation mode of the billboard to
     the value vvBillboard::AXIAL_ROT using vvBillboard::setMode.  The axis of
rotation (x, y, z) is specified using vvBillboard::setAxis.
vvBillboard::getAxis returns the axis of the vvBillboard.

If the rotation mode of the billboard is set to vvBillboard::POINT_ROT_EYE,
the billboard is rotated so that the +Z axis of the vsg::Node aligns
as closely as possible with the axis specified with
pfBillboard::setAxis, in eye space.  In particular, if the axis is
+Y (as it is by default) and you display a text, then, the text always
stays readable.

If the rotation mode of the billboard is set to
vvBillboard::POINT_ROT_WORLD, the billboard is rotated so that the angle
between the +Y axis of the vsg::Node and axis specified with
pfBillboard::setAxis (in world space) is minimized.

Both vvBillboard::AXIAL_ROT and vvBillboard::POINT_ROT_WORLD billboards
may "spin" about the rotation or alignment axis when viewed along that
axis.
*/
#include <vsg/nodes/MatrixTransform.h>
#include <util/coExport.h>

namespace vive
{
/** vvBillboard is a derived form of Transform that automatically
  * scales or rotates to keep its children aligned with screen coordinates.
*/
class VVCORE_EXPORT vvBillboard : public vsg::Inherit<vsg::MatrixTransform, vvBillboard>
{
public:
    vvBillboard();



    virtual void accept(vsg::Visitor &nv);

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
    void setAxis(const vsg::vec3 &axis);
    const vsg::vec3 &getAxis() const;

    // the normal of the geometry
    // default is -Y (0 -1 0)
    void setNormal(const vsg::vec3 &normal);
    const vsg::vec3 &getNormal() const;

protected:
    virtual ~vvBillboard()
    {
    }

    vsg::vec3 _axis;
    vsg::vec3 _normal;

    RotationMode _mode;

    mutable vsg::quat _rotation;

    mutable vsg::dmat4 _cachedMatrix;
    mutable vsg::dmat4 _cachedInvMatrix;

private:
    void checkAndAdjustNormal();
};
}

EVSG_type_name(vive::vvBillboard);