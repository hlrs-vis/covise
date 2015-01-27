/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// class stripped down for tracking by Marcel

// **********************************************************************
// * \author Sven Havemann, Institute for Computer Graphics,            *
// *         TU Braunschweig, Muehlenpfordtstrasse 23,                  *
// *         38106 Braunschweig, Germany                                *
// *         http://www.cg.cs.tu-bs.de      mailto:s.havemann@tu-bs.de  *
// * \date   01.10.2000                                                 *
// **********************************************************************

#ifndef CGVOTQUATERNIONTR_HPP
#define CGVOTQUATERNIONTR_HPP

#include "vec3_basetr.h"

/** Quaternion class
  
   Class Quaternion implements the abstract data type quaternion
   and all needed operations on it. 
   Reference for rotate(), toRotMatrix(), fromRotMatrix: 
   Watt/Watt, Advanced Animation and rendering techniques, pp. 360-368. 
   Note: no destructor, which is ok (no virtual functions) */

class Quaternion
{
public:
    Quaternion();
    Quaternion(float s, const CGVVec3 &v);

    void assign(float s, const CGVVec3 &v);
    void assign(float s, float x, float y, float z);

    friend std::ostream &operator<<(std::ostream &os, const Quaternion &m);
    friend std::istream &operator>>(std::istream &is, Quaternion &m);

    //private:
    float v_s;
    CGVVec3 v_v;
};

/** Rotation Quaternion class
  
    This class is for the convenient description of a rotation by 
    rotation axis and angle. 

    q.rotate(v)       is cheaper only when used ONCE:

    q.rotate(v)       costs 24 mults and 14 adds, 
    q.toRotMatrix()v  costs 26 mults and 23 adds.  */

class QuaternionRot : public Quaternion
{
public:
    QuaternionRot();

    /** Normal constructor: Give rotation axis and angle, the quaternion is
       initialized as [cos(angle/2),sin(angle/2)(axis.x,axis.y,axis.z)].  The axis
       must be normalized (which is NOT checked, for efficiency) and the
       rotation angle is given in radian. This is a rotational quaternion, for
       use with rotate() or toRotMatrix(). */
    QuaternionRot(float angle, const CGVVec3 &axis);

    void assignRot(float angle, const CGVVec3 &axis);

    /** Rotation: For a 3D vector v=(0,x,y,z) and a normalized quaternion 
       q=(s,ax,ay,az), the conjugate is q^=(s,-ax,-ay,-az) and 
       the rotation of v about the axis (ax,ay,ay) with the angle w 
       so that cos(w/2)=s can be expressed as v'=qvq^. 
       Cost: 24 mults and 14 adds  */
    CGVVec3 rotate(const CGVVec3 &v) const;

    /** Make a rotation matrix from a unit quaternion 
       The unity of q (this) is not checked.
       Cost: 17 mults, 17 adds  */
    void toRotMatrix(float *matrix) const; // 4x4 matrix

    /** Make a unit quaternion from an - orthogonal! - rotation matrix
       The orthogonality of rotmat is not checked. 
       Cost: 1 sqrt, 5 mults, 6 adds or 1 sqrt, 6 mults, 4 adds/ */
    Quaternion &fromRotMatrix(const float *rotmat); // 4x4 matrix

    /** Spherical interpolation of Quaternions, taken from A. Watt, M. Watt:
       Advanced Animation and Rendering Techniques, p.360-368 */
    QuaternionRot &slerp(const QuaternionRot &p,
                         const QuaternionRot &q, double t);

    // uses a glRotatef() call to modify the current OpenGL matrix
    void applyToCurrentGLMatrix();
};

#endif
