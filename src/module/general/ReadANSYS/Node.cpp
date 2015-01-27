/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Node.h"
#include <util/coviseCompat.h>

Node::Node()
{
    Rotation_[Rotation::XX] = 1.0;
    Rotation_[Rotation::XY] = 0.0;
    Rotation_[Rotation::XZ] = 0.0;
    Rotation_[Rotation::YX] = 0.0;
    Rotation_[Rotation::YY] = 1.0;
    Rotation_[Rotation::YZ] = 0.0;
    Rotation_[Rotation::ZX] = 0.0;
    Rotation_[Rotation::ZY] = 0.0;
    Rotation_[Rotation::ZZ] = 1.0;
}

Rotation &
    Rotation::
    operator*=(const Rotation &rhs)
{
    Rotation tmp;
    tmp[XX] = rotation_[XX];
    tmp[XY] = rotation_[XY];
    tmp[XZ] = rotation_[XZ];
    tmp[YX] = rotation_[YX];
    tmp[YY] = rotation_[YY];
    tmp[YZ] = rotation_[YZ];
    tmp[ZX] = rotation_[ZX];
    tmp[ZY] = rotation_[ZY];
    tmp[ZZ] = rotation_[ZZ];

    rotation_[XX] = tmp[XX] * rhs[XX] + tmp[XY] * rhs[YX] + tmp[XZ] * rhs[ZX];
    rotation_[XY] = tmp[XX] * rhs[XY] + tmp[XY] * rhs[YY] + tmp[XZ] * rhs[ZY];
    rotation_[XZ] = tmp[XX] * rhs[XZ] + tmp[XY] * rhs[YZ] + tmp[XZ] * rhs[ZZ];
    rotation_[YX] = tmp[YX] * rhs[XX] + tmp[YY] * rhs[YX] + tmp[YZ] * rhs[ZX];
    rotation_[YY] = tmp[YX] * rhs[XY] + tmp[YY] * rhs[YY] + tmp[YZ] * rhs[ZY];
    rotation_[YZ] = tmp[YX] * rhs[XZ] + tmp[YY] * rhs[YZ] + tmp[YZ] * rhs[ZZ];
    rotation_[ZX] = tmp[ZX] * rhs[XX] + tmp[ZY] * rhs[YX] + tmp[ZZ] * rhs[ZX];
    rotation_[ZY] = tmp[ZX] * rhs[XY] + tmp[ZY] * rhs[YY] + tmp[ZZ] * rhs[ZY];
    rotation_[ZZ] = tmp[ZX] * rhs[XZ] + tmp[ZY] * rhs[YZ] + tmp[ZZ] * rhs[ZZ];

    return *this;
}

// I am assuming that the order of
// the Rotation_s is Z, X, Y !!!!!
void
Node::MakeRotation()
{
    float cxy = (float)cos(thxy_);
    float sxy = (float)sin(thxy_);
    float cyz = (float)cos(thyz_);
    float syz = (float)sin(thyz_);
    float czx = (float)cos(thzx_);
    float szx = (float)sin(thzx_);

    Rotation_[Rotation::XX] = cxy;
    Rotation_[Rotation::YY] = cxy;
    Rotation_[Rotation::XY] = -sxy;
    Rotation_[Rotation::YX] = sxy;

    Rotation Rot_yz;
    Rot_yz[Rotation::XX] = 1.0;
    Rot_yz[Rotation::XY] = 0.0;
    Rot_yz[Rotation::XZ] = 0.0;
    Rot_yz[Rotation::YX] = 0.0;
    Rot_yz[Rotation::YY] = cyz;
    Rot_yz[Rotation::YZ] = -syz;
    Rot_yz[Rotation::ZX] = 0.0;
    Rot_yz[Rotation::ZY] = syz;
    Rot_yz[Rotation::ZZ] = cyz;

    Rotation_ *= Rot_yz;

    Rotation Rot_zx;
    Rot_zx[Rotation::XX] = czx;
    Rot_zx[Rotation::XY] = 0.0;
    Rot_zx[Rotation::XZ] = szx;
    Rot_zx[Rotation::YX] = 0.0;
    Rot_zx[Rotation::YY] = 1.0;
    Rot_zx[Rotation::YZ] = 0.0;
    Rot_zx[Rotation::ZX] = -szx;
    Rot_zx[Rotation::ZY] = 0.0;
    Rot_zx[Rotation::ZZ] = czx;

    Rotation_ *= Rot_zx;
}
