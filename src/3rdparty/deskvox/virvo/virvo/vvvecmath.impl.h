// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

//============================================================================
// vvBaseVector3 Template Methods
//============================================================================

//----------------------------------------------------------------------------
/// Constructor for an empty vector
template <typename T>
vvBaseVector3<T>::vvBaseVector3()
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] = 0;
  }
}

//----------------------------------------------------------------------------
/// Constructor with one value for all 3 elements
template <typename T>
vvBaseVector3<T>::vvBaseVector3(const T val)
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] = val;
  }
}

//----------------------------------------------------------------------------
/// Constructor for a preset vector
template <typename T>
vvBaseVector3<T>::vvBaseVector3(const T x, const T y, const T z)
{
  e[0] = x;
  e[1] = y;
  e[2] = z;
}

//----------------------------------------------------------------------------
/// Copy constructor for vvVector4
template <typename T>
vvBaseVector3<T>::vvBaseVector3(const vvBaseVector4<T>& v)
{
  T w = v[3];
  if (w != T(0))
  {
    e[0] = v[0] / w;
    e[1] = v[1] / w;
    e[2] = v[2] / w;
  }
  else
  {
    e[0] = v[0];
    e[1] = v[1];
    e[2] = v[2];
  }
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
template <typename T>
T &vvBaseVector3<T>::operator[](size_t index)
{
  assert(index < 3);
  return e[index];
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
template <typename T>
T const& vvBaseVector3<T>::operator[](size_t index) const
{
  assert(index < 3);
  return e[index];
}

//----------------------------------------------------------------------------
/// Set all elements of a vector.
template <typename T>
void vvBaseVector3<T>::set(const T x, const T y, const T z)
{
  e[0] = x;
  e[1] = y;
  e[2] = z;
}

//----------------------------------------------------------------------------
/// Get all elements of a vector.
template <typename T>
void vvBaseVector3<T>::get(T* x, T* y, T* z) const
{
  *x = e[0];
  *y = e[1];
  *z = e[2];
}

//----------------------------------------------------------------------------
/// Add two vectors
template <typename T>
void vvBaseVector3<T>::add(const vvBaseVector3<T>& v)
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] += v.e[i];
  }
}

//----------------------------------------------------------------------------
/// Add the same scalar value to each component.
template <typename T>
void vvBaseVector3<T>::add(const T val)
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] += val;
  }
}

//----------------------------------------------------------------------------
/// Add a different scalar value to each component.
template <typename T>
void vvBaseVector3<T>::add(const T x, const T y, const T z)
{
  e[0] += x;
  e[1] += y;
  e[2] += z;
}

//----------------------------------------------------------------------------
/// Subtract a vector from another
template <typename T>
void vvBaseVector3<T>::sub(const vvBaseVector3<T>& v)
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] -= v.e[i];
  }
}

//----------------------------------------------------------------------------
/// Subtract a scalar value
template <typename T>
void vvBaseVector3<T>::sub(const T val)
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] -= val;
  }
}

//----------------------------------------------------------------------------
/// Scale a vector by a scalar
template <typename T>
void vvBaseVector3<T>::scale(const T s)
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] *= s;
  }
}

//----------------------------------------------------------------------------
/// Scale a vector by the elements of another vector
template <typename T>
void vvBaseVector3<T>::scale(const vvBaseVector3<T>& v)
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] *= v.e[i];
  }
}

//----------------------------------------------------------------------------
/// Scale a vector by different scalars for each component
template <typename T>
void vvBaseVector3<T>::scale(const T x, const T y, const T z)
{
  e[0] *= x;
  e[1] *= y;
  e[2] *= z;
}

//----------------------------------------------------------------------------
/// Return the dot product of two vectors
template <typename T>
T vvBaseVector3<T>::dot(const vvBaseVector3<T>& v) const
{
  return e[0] * v.e[0] + e[1] * v.e[1] + e[2] * v.e[2];
}

//----------------------------------------------------------------------------
/// Return the angle (in radians) between two vectors
template <typename T>
T vvBaseVector3<T>::angle(const vvBaseVector3<T>& v) const
{
  const T multLength = this->length() * v.length();
  if (multLength == 0.0) return 0.0;

  const T div = this->dot(v) / multLength;

  if (div < -1.0) return (T)(VV_PI * 0.5);
  if (div > 1.0)  return 0.0f;

  return acosf(div);
}

//----------------------------------------------------------------------------
/// Create the cross product of two vectors
template <typename T>
void vvBaseVector3<T>::cross(const vvBaseVector3<T>& v)
{
  const vvBaseVector3<T> bak(*this);

  e[0] = bak.e[1] * v.e[2] - bak.e[2] * v.e[1];
  e[1] = bak.e[2] * v.e[0] - bak.e[0] * v.e[2];
  e[2] = bak.e[0] * v.e[1] - bak.e[1] * v.e[0];
}

//----------------------------------------------------------------------------
/// Multiplies a vector with a matrix (V' = M x V)
template <typename T>
void vvBaseVector3<T>::multiply(const vvMatrix& m)
{
  const T v1[4] = { e[0], e[1], e[2], 1.0f };

  for (size_t row=0; row<3; ++row)
  {
    e[row] = 0.0f;
    for (size_t col=0; col<4; ++col)
      e[row] += m(row, col) * v1[col];
  }

  const T w = m(3, 0) * v1[0]
            + m(3, 1) * v1[1]
            + m(3, 2) * v1[2]
            + m(3, 3) * v1[3];

  if (w != 1.0f)
  {
    const T wInv = 1.0f / w;
    for (size_t row=0; row<3; ++row)
      e[row] *= wInv;
  }
}

//----------------------------------------------------------------------------
/// Compute the distance between two points
template <typename T>
T vvBaseVector3<T>::distance(const vvBaseVector3<T>& v) const
{
  return (v - *this).length();
}

//----------------------------------------------------------------------------
/// Compute the length of a vector
template <typename T>
T vvBaseVector3<T>::length() const
{
  return static_cast<T>( std::sqrt(static_cast<double>(dot(*this))) );
}

//----------------------------------------------------------------------------
/** Compute the normal of a plane which is defined by two points and a
    vector on the plane.
  @param point1,point2 two points defining the plane
  @param dir directional vector on the plane
*/
template <typename T>
void vvBaseVector3<T>::planeNormalPPV(const vvBaseVector3<T>& point1,
                                      const vvBaseVector3<T>& point2,
                                      const vvBaseVector3<T>& dir)
{
  vvBaseVector3<T> diff;    // difference vector between point1 and point2

  diff = vvBaseVector3<T>(point2);
  diff.sub(point1);
  for (int i = 0; i < 3; ++i)
  {
    e[i] = dir[i];
  }
  this->cross(diff);
}

//----------------------------------------------------------------------------
/** Computes the distance of a point from a plane.
  @param this  point to compute distance for
  @param n,p   normal and point defining a plane
  @return distance of point to plane, sign determines side of the plane on which the
          point lies: positive=side into which normal points, negative=other side
*/
template <typename T>
T vvBaseVector3<T>::distPointPlane(const vvBaseVector3<T>& n,
                                   const vvBaseVector3<T>& p) const
{
  vvBaseVector3<T> normal = vvBaseVector3<T>(n);                // normalized plane normal
  T d;                                   // scalar component of hessian form of plane

  normal.normalize();
  d = -normal.dot(p);
  return (normal.dot(*this) + d);
}

//----------------------------------------------------------------------------
/// Normalize a vector
template <typename T>
void vvBaseVector3<T>::normalize()
{
  T len = length();
  if (len != T(0)) {
    e[0] /= len;
    e[1] /= len;
    e[2] /= len;
  }
}

//----------------------------------------------------------------------------
/// Negate a vector
template <typename T>
void vvBaseVector3<T>::negate()
{
  for (int i = 0; i < 3; ++i)
  {
    e[i] = -e[i];
  }
}

//----------------------------------------------------------------------------
/// Compares two vectors. Returns true if equal, otherwise false
template <typename T>
bool vvBaseVector3<T>::equal(const vvBaseVector3<T>& v)
{
  for (int i = 0; i < 3; ++i)
  {
    if (e[i] != v.e[i])
    {
      return false;
    }
  }
  return true;
}

//----------------------------------------------------------------------------
/// Creates a vector with random integer numbers in range from..to
template <typename T>
void vvBaseVector3<T>::random(int from, int to)
{
  int row;

  for (row=0; row<3; ++row)
    e[row] = (T)(from + (rand() % (to - from + 1)));
}

//----------------------------------------------------------------------------
/// Creates a vector with random float numbers in range from..to
template <typename T>
void vvBaseVector3<T>::random(float from, float to)
{
  int row;

  for (row=0; row<3; ++row)
    e[row] = (T)from + ((float)rand()/(float)RAND_MAX * (float)(to - from));
}

//----------------------------------------------------------------------------
/** Creates a vector with random float numbers in range from..to.
 This method is there to allow calls like random(1.0, 5.0) instead of
 random(1.0f, 5.0f)
*/
template <typename T>
void vvBaseVector3<T>::random(double from, double to)
{
  random((float)from, (float)to);
}

/// Print a vector
template <typename T>
void vvBaseVector3<T>::print(const char* text) const
{
  std::cerr.setf(std::ios::fixed, std::ios::floatfield);
  std::cerr.precision(3);

  if (text != 0)
  {
    std::cerr << text << "\t" << e[0] << ", " << e[1] << ", " << e[2] << std::endl;
  }
  else
  {
    std::cerr << e[0] << ", " << e[1] << ", " << e[2] << std::endl;
  }
}

//----------------------------------------------------------------------------
/** Get a row vector from a matrix
  @param m    matrix
  @param row  row number [0..2]
*/
template <typename T>
void vvBaseVector3<T>::getRow(const vvMatrix& m, const int row)
{
  e[0] = m(row, 0);
  e[1] = m(row, 1);
  e[2] = m(row, 2);
}

//----------------------------------------------------------------------------
/** Get a column vector from a matrix
  @param m    matrix
  @param col  column number [0..2]
*/
template <typename T>
void vvBaseVector3<T>::getColumn(const vvMatrix& m, const int col)
{
  e[0] = m(0, col);
  e[1] = m(1, col);
  e[2] = m(2, col);
}

//----------------------------------------------------------------------------
/// Swap two vector element matrices
template <typename T>
void vvBaseVector3<T>::swap(vvBaseVector3<T>& v)
{
  std::swap(v.e[0], e[0]);
  std::swap(v.e[1], e[1]);
  std::swap(v.e[2], e[2]);
}

//----------------------------------------------------------------------------
/** Intersect a plane and a straight line.
  @param  n       normal of plane (doesn't need to be normalized)
  @param  p       an arbitrary point on the plane
  @param  v1,v2   arbitrary points on the line
  @return true if an intersection occurred, result in this pointer.
          If no intersection occurred, false is returned and all
          vector components are set to 0.0f
*/
template <typename T>
bool vvBaseVector3<T>::isectPlaneLine(const vvBaseVector3<T>& n,
                                      const vvBaseVector3<T>& p,
                                      const vvBaseVector3<T>& v1,
                                      const vvBaseVector3<T>& v2)
{
  vvBaseVector3 normal = vvBaseVector3(n);                           // normalized normal

  normal.normalize();
  vvBaseVector3 diff1 = v1 - v2;                  // diff1 = v1 - v2
  const T denom = diff1.dot(normal);              // denom = diff1 . n
  if (denom==0.0f)                                // are ray and plane parallel?
  {
    e[0] = e[1] = e[2] = 0.0f;                    // no intersection
    return false;
  }
  const vvBaseVector3 diff2 = p - v1;             // diff2 = p - v1
  const T numer = diff2.dot(normal);              // number = diff2 . n
  diff1.scale(numer / denom);                     // diff1 = diff1 * numer / denom
  for (size_t i = 0; i < 3; ++i)
  {
    e[i] = v1[i];
  }
  this->add(diff1);                               // this = v1 + diff1
  return true;
}

//----------------------------------------------------------------------------
/** Intersect a plane and a ray (line starting at a point).
  @param  n       normal of plane (must be normalized!)
  @param  p       an arbitrary point on the plane
  @param  v1      starting point of ray
  @param  v2      arbitrary point on ray
  @return true if an intersection occurred, result at this pointer.
          If no intersection occurred, false is returned and all vector
          components are set to 0.0f
*/
template <typename T>
bool vvBaseVector3<T>::isectPlaneRay(const vvBaseVector3<T>& n,
                                     const vvBaseVector3<T>& p,
                                     const vvBaseVector3<T>& v1,
                                     const vvBaseVector3<T>& v2)
{
  vvBaseVector3<T> diff1;                         // difference vector between v1 and v2
  vvBaseVector3<T> diff2;                         // difference vector between this and v1
  T factor;                                       // distance factor
  size_t i;

  // Check for intersection with straight line:
  if (this->isectPlaneLine(n, p, v1, v2) == false) return false;

  diff1 = vvBaseVector3<T>(v2);
  diff1.sub(v1);                                  // diff1 = v2 - v1

  diff2 = vvBaseVector3<T>(*this);
  diff2.sub(v1);                                  // diff2 = this - v1

  // Find out how to represent diff2 by diff1 times a factor:
  factor = 0.0f;
  for (i=0; i<3; ++i)
  {
    if (diff1.e[i] != 0.0f)
      factor = diff2.e[i] / diff1.e[i];
  }

  if (factor < 0)                                 // intersection in opposite direction than ray?
  {
    this->zero();
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------
/** Compute all intersections of a plane and a cuboid.<P>
  Example call:<BR><TT>
  vvVector3 isect[6];<BR>
  isect[0].isectPlaneCuboid(...);</TT><P>
  @param  normal    normal of plane (must be normalized!)
  @param  point     an arbitrary point on the plane
  @param  minv,maxv Minimum and maximum vertices of cuboid.<BR>
                    All minimum vertex's coordinates must be smaller than
                    the maximum vertex's coordinates.<BR>
                    The cuboid's edges are assumed to be parallel to the coordinate axes.
  @return number of intersection points. The intersection points are returned
in an array, of which the first element must be the caller. There must
be <I>at least</I> 6 elements defined in the calling vector!<P>
If less than six intersections occurred, the remaining vector
components of this[] are undefined.
*/
template <typename T>
size_t vvBaseVector3<T>::isectPlaneCuboid(const vvBaseVector3<T>& normal,
                                         const vvBaseVector3<T>& point,
                                         const vvBaseVector3<T>& minv,
                                         const vvBaseVector3<T>& maxv)
{
  vvBaseVector3 p[2];                             // two cuboid vertices defining a cuboid edge
  vvBaseVector3 corner[2];                        // cuboid corners (copied from parameters)
  size_t key[4][3] =                              // cuboid edge components
  {
    {
      0, 0, 0
    }
    ,
    {
      1, 0, 1
    }
    ,
    {
      1, 1, 0
    }
    ,
    {
      0, 1, 1
    }
  };
  size_t isectCnt = 0;                            // intersection counter
  size_t i,j;

  // Copy parameters for easier access:
  corner[0] = minv;
  corner[1] = maxv;

  // Search for intersections between texture plane (defined by texPoint and
  // normal) and texture object (0..1):
  for (i=0; i<4 && isectCnt<6; ++i)               // loop thru volume key vertices
  {
    for (j=0; j<3 && isectCnt<6; ++j)             // loop thru secondary vertices
    {
      // Compute vertices of one cuboid edge:
      p[0].set(corner[key[i][0]].e[0], corner[key[i][1]].e[1], corner[key[i][2]].e[2]);
      p[1].set((j==0) ? (corner[1-key[i][0]].e[0]) : corner[key[i][0]].e[0],
        (j==1) ? (corner[1-key[i][1]].e[1]) : corner[key[i][1]].e[1],
        (j==2) ? (corner[1-key[i][2]].e[2]) : corner[key[i][2]].e[2]);

      // Compute intersections of one cuboid edge with the plane:
      if (this[isectCnt].isectPlaneLine(normal, point, p[0], p[1]))
      {
        if (this[isectCnt].e[0] >= minv.e[0] && this[isectCnt].e[0] <= maxv.e[0] &&
          this[isectCnt].e[1] >= minv.e[1] && this[isectCnt].e[1] <= maxv.e[1] &&
          this[isectCnt].e[2] >= minv.e[2] && this[isectCnt].e[2] <= maxv.e[2])
        {
          ++isectCnt;                             // accept this entry
        }
      }
    }
  }
  assert(isectCnt<7);                             // there can be at most 6 intersections
  return isectCnt;
}

//----------------------------------------------------------------------------
/** Intersect a cylinder (infinitely long) with a ray (a line
  that emerges from a point and is infinitely long in the other direction).

  @param cylBase      arbitrary point on axis of symmetry for cylinder
  @param cylAxis      direction into which cylinder extends (does not need to be normalized)
  @param cylRadius    radius of cylinder
  @param rayBase      starting point of ray
  @param rayDir       direction into which ray extends (does not need to be normalized)
  @return The return value is the number of intersection points (0..2).
          If this value is greater than 0, the closest intersection point to
          the ray starting point is returned in the vector on which the function
was executed.
*/
template <typename T>
int vvBaseVector3<T>::isectRayCylinder(const vvBaseVector3<T>& cylBase,
                                       const vvBaseVector3<T>& cylAxis,
                                       T cylRadius,
                                       const vvBaseVector3<T>& rayBase,
                                       const vvBaseVector3<T>& rayDir)
{
  vvBaseVector3 ortho;                            // vector orthogonal to cylAxis and rayDir
  vvBaseVector3 diff;                             // difference vector between base points
  vvBaseVector3 temp;                             // temporary vector
  vvBaseVector3 closest;                          // closest point on ray-line to cylinder axis
  vvBaseVector3 cylAxisN;                         // normalized cylinder axis
  vvBaseVector3 rayDirN;                          // normalized ray direction vector
  T     dist;                                     // distance between cylinder axis and closest point on ray
  T     t;                                        // distance of closest from rayBase
  T     s;                                        // distance of intersections from closest
  T     len;                                      // length of ortho before normalization
  T     i1, i2;                                   // distances of intersections from rayBase
  T     first;                                    // first intersection = intersection closest to rayBase
  int       i;                                    // counter

  // Compute normalized direction vectors:
  cylAxisN = vvBaseVector3<T>(cylAxis);
  cylAxisN.normalize();
  rayDirN = vvBaseVector3<T>(rayDir);
  rayDirN.normalize();

  // Compute distance between closest point on line and cylinder axis:
  ortho = vvBaseVector3<T>(rayDir);
  ortho.cross(cylAxis);
  len = ortho.length();
  ortho.normalize();
  if (ortho.isZero())                             // is the ray parallel to the cylinder?
  {
    this->zero();
    return 0;
  }
  diff = vvBaseVector3<T>(rayBase);
  diff.sub(cylBase);
  dist = (T)fabs(static_cast<double>(diff.dot(ortho)));
  if (dist > cylRadius)                           // do the ray and the cylinder intersect at all?
  {
    this->zero();
    return 0;
  }

  // Find point on line closest to cylinder axis:
  temp = vvBaseVector3<T>(diff);
  temp.cross(cylAxisN);
  t = -1.0f * temp.dot(ortho) / len;
  closest = vvBaseVector3<T>(rayDirN);
  closest.scale(t);
  closest.add(rayBase);

  // Find intersections of ray-line and cylinder:
  s = std::sqrt(static_cast<double>(cylRadius * cylRadius - dist * dist));
  i1 = t + s;                                     // closest intersection
  i2 = t - s;                                     // second intersection

  // Only the positive values of i1 and i2 are located on the ray:
  if (i1<0.0f && i2<0.0f)
  {
    this->zero();
    return 0;
  }
  if (i1 < 0.0f) first = i2;
  else if (i2 < 0.0f) first = i1;
  else first = (i1 < i2) ? i1 : i2;
  for (i=0; i<3; ++i)
    this->e[i] = rayBase.e[i] + first * rayDirN.e[i];
  if (i1<0.0f || i2<0.0f) return 1;
  else return 2;
}

//----------------------------------------------------------------------------
/** Intersect a ray with a triangle.
  @param rayPt  starting point of ray
  @param rayDir direction of ray
  @param tri1   first triangle vertex
  @param tri2   second triangle vertex
  @param tri3   third triangle vertex
  @return true if ray intersects triangle, this pointer contains intersection point.
          false if no intersection, this pointer contains zero vector.
*/
template <typename T>
bool vvBaseVector3<T>::isectRayTriangle(const vvBaseVector3<T>& rayPt,
                                        const vvBaseVector3<T>& rayDir,
                                        const vvBaseVector3<T>& tri1,
                                        const vvBaseVector3<T>& tri2,
                                        const vvBaseVector3<T>& tri3)
{
  vvBaseVector3 rayPt2;                           // arbitrary point on ray
  vvBaseVector3 normal;                           // normal vector of triangle (normalized)
  vvBaseVector3 diff1;                            // vector from tri1 to tri2
  vvBaseVector3 diff2;                            // vector from tri1 to tri3
  vvBaseVector3 sideNormal;                       // normal vector made from one triangle side and the triangle normal

  // Compute second point on ray:
  rayPt2 = vvBaseVector3<T>(rayPt);
  rayPt2.add(rayDir);

  // Compute triangle normal:
  diff1 = vvBaseVector3<T>(tri2);
  diff1.sub(tri1);
  diff2 = vvBaseVector3<T>(tri3);
  diff2.sub(tri1);
  normal = vvBaseVector3<T>(diff1);
  normal.cross(diff2);
  normal.normalize();

  // Compute intersection of ray and triangle plane:
  if (this->isectPlaneRay(normal, tri1, rayPt, rayPt2) == false) return false;

  // Compute three triangle side normals and check if intersection point lies on same
  // side as third triangle point, respectively:
  sideNormal.planeNormalPPV(tri1, tri2, normal);
  if (vvVecmath::sgn(tri3->distPointPlane(sideNormal, tri1)) ==
    vvVecmath::sgn(this->distPointPlane(sideNormal, tri1)))
  {
    sideNormal.planeNormalPPV(tri1, tri3, normal);
    if (vvVecmath::sgn(tri2->distPointPlane(sideNormal, tri1)) ==
      vvVecmath::sgn(this->distPointPlane(sideNormal, tri1)))
    {
      sideNormal.planeNormalPPV(tri2, tri3, normal);
      if (vvVecmath::sgn(tri1->distPointPlane(sideNormal, tri2)) ==
        vvVecmath::sgn(this->distPointPlane(sideNormal, tri2)))
      {
        return true;
      }
    }
  }
  this->zero();
  return false;
}

//----------------------------------------------------------------------------
/** Intersect two lines.
  Source: http://astronomy.swin.edu.au/~pbourke/geometry/lineline3d/
   Calculate the line segment PaPb that is the shortest route between
   two lines P1P2 and P3P4. Calculate also the values of mua and mub where
      Pa = P1 + mua (P2 - P1)
      Pb = P3 + mub (P4 - P3)
   Return FALSE if no solution exists.
  @param  p1,p2  points on each line
  @param  v1,v2  each line's direction vectors
  @return shortest distance between lines (0.0f if lines intersect).
          If lines don't intersect, the this vector will be the point
on line 1 that is closest to line 2.
*/
template <typename T>
T vvBaseVector3<T>::isectLineLine(const vvBaseVector3<T>& pt1,
                                  const vvBaseVector3<T>& v1,
                                  const vvBaseVector3<T>& pt2,
                                  const vvBaseVector3<T>& v2)
{
  const T EPS = 0.0001;
  vvBaseVector3 p1, p2, p3, p4, p13, p43, p21, pa, pb;
  T d1343, d4321, d1321, d4343, d2121;
  T numer, denom, mua, mub;

  /*
    float dist;

    // Create plane from line 1 and direction of line 2:
    vvPlane plane(p1, v1, v2);

    // Compute shortest distance between lines:
    for (int i = 0; i < 3; ++i)
    {
      e[i] = p2[i];
    }
    dist = this->distPointPlane(plane.normal, plane.point);

    return dist;
  */
  p1 = vvBaseVector3<T>(pt1);
  p2 = vvBaseVector3<T>(pt1);
  p2.add(v1);
  p3 = vvBaseVector3<T>(pt2);
  p4 = vvBaseVector3<T>(pt2);
  p4.add(v2);

  p13.e[0] = p1.e[0] - p3.e[0];
  p13.e[1] = p1.e[1] - p3.e[1];
  p13.e[2] = p1.e[2] - p3.e[2];
  p43.e[0] = p4.e[0] - p3.e[0];
  p43.e[1] = p4.e[1] - p3.e[1];
  p43.e[2] = p4.e[2] - p3.e[2];
  if (fabs(static_cast<double>(p43.e[0])) < EPS
	  && fabs(static_cast<double>(p43.e[1])) < EPS
	  && fabs(static_cast<double>(p43.e[2]))  < EPS)
    return(0.0f);
  p21.e[0] = p2.e[0] - p1.e[0];
  p21.e[1] = p2.e[1] - p1.e[1];
  p21.e[2] = p2.e[2] - p1.e[2];
  if (fabs(static_cast<double>(p21.e[0]))  < EPS
	  && fabs(static_cast<double>(p21.e[1]))  < EPS
	  && fabs(static_cast<double>(p21.e[2]))  < EPS)
    return(0.0f);

  d1343 = p13.e[0] * p43.e[0] + p13.e[1] * p43.e[1] + p13.e[2] * p43.e[2];
  d4321 = p43.e[0] * p21.e[0] + p43.e[1] * p21.e[1] + p43.e[2] * p21.e[2];
  d1321 = p13.e[0] * p21.e[0] + p13.e[1] * p21.e[1] + p13.e[2] * p21.e[2];
  d4343 = p43.e[0] * p43.e[0] + p43.e[1] * p43.e[1] + p43.e[2] * p43.e[2];
  d2121 = p21.e[0] * p21.e[0] + p21.e[1] * p21.e[1] + p21.e[2] * p21.e[2];

  denom = d2121 * d4343 - d4321 * d4321;
  if (fabs(static_cast<double>(denom)) < EPS) return(0.0);
  numer = d1343 * d4321 - d1321 * d4343;

  mua = numer / denom;
  mub = (d1343 + d4321 * (mua)) / d4343;

  pa.e[0] = p1.e[0] + mua * p21.e[0];
  pa.e[1] = p1.e[1] + mua * p21.e[1];
  pa.e[2] = p1.e[2] + mua * p21.e[2];
  pb.e[0] = p3.e[0] + mub * p43.e[0];
  pb.e[1] = p3.e[1] + mub * p43.e[1];
  pb.e[2] = p3.e[2] + mub * p43.e[2];

  for (int i = 0; i < 3; ++i)
  {
    e[i] = pa[i];
  }

  return(pa.distance(pb));
}

//----------------------------------------------------------------------------
/** Check if two points are on the same side relative to a line formed by
  two points. The third coordinate of all points must be zero, as this
  is a 2D algorithm!
  Algorithm:<p>
  cp1 = CrossProduct(b-a, p1-a)<br>
  cp2 = CrossProduct(b-a, p2-a)<br>
  if DotProduct(cp1, cp2) >= 0 then return true else return false<p>
  @param p1,p2 points to check
  @param a,b points forming the line
*/
template <typename T>
bool vvBaseVector3<T>::isSameSideLine2D(const vvBaseVector3<T>& p1,
                                        const vvBaseVector3<T>& p2,
                                        const vvBaseVector3<T>& a,
                                        const vvBaseVector3<T>& b)
{
  vvBaseVector3 diff1, diff2;
  vvBaseVector3 cp1, cp2;
  vvBaseVector3 dp;

  assert(p1.e[2]==0.0f && p2.e[2]==0.0f && a.e[2]==0.0f && b.e[2]==0.0f);
  diff1.e[0] = p1.e[0] - a.e[0];
  diff1.e[1] = p1.e[1] - a.e[1];
  diff2.e[0] = p2.e[0] - a.e[0];
  diff2.e[1] = p2.e[1] - a.e[1];
  cp1.e[0] = b.e[0] - a.e[0];
  cp1.e[1] = b.e[1] - a.e[1];
  cp2 = vvBaseVector3<T>(cp1);
  cp1.cross(diff1);
  cp2.cross(diff2);
  dp = vvBaseVector3<T>(cp1);
  if (dp.dot(cp2) >= 0.0f) return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Check if a point is in a triangle (so far third coordinate must be zero!)
  @param this point to check
  @param v1,v2,v3 triangle vertices
*/
template <typename T>
bool vvBaseVector3<T>::isInTriangle(const vvBaseVector3<T>& v1,
                                    const vvBaseVector3<T>& v2,
                                    const vvBaseVector3<T>& v3)
{
  assert(this->e[2]==0.0f && v1.e[2]==0.0f && v2.e[2]==0.0f && v3.e[2]==0.0f);

  if (isSameSideLine2D(this, v1, v2, v3) && isSameSideLine2D(this, v2, v1, v3) &&
    isSameSideLine2D(this, v3, v1, v2)) return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Cyclically sort a number of vectors in an array starting with this vector.
  Example call:<BR><TT>
  vvVector3 vecArray[6];<BR>
  vecArray[0].cyclicSort(6);</TT><P>
  @param numVectors number of vectors in 'this' array to be sorted
  @param axis     normal vector defining cycle direction
*/
template <typename T>
void vvBaseVector3<T>::cyclicSort(const int numVectors,
                                  const vvBaseVector3<T>& axis)
{
  vvBaseVector3* diff;                            // difference vectors between pairs of points
  vvBaseVector3 normal;                           // normal vector
  bool swapped;                                   // true = vectors were swapped

  // Allocate array of difference vectors:
  diff = new vvBaseVector3[numVectors-1];

  // Compute difference vectors:
  for (int i=0; i<numVectors-1; ++i)
  {
    diff[i] = vvBaseVector3<T>(this[i+1]);
    diff[i].sub(this[0]);
  }

  // Sort vectors:
  swapped = true;
  while (swapped)
  {
    swapped = false;
    for (int i=0; i<numVectors-2 && swapped==false; ++i)
    {
      for (int j=i+1; j<numVectors-1 && swapped==false; ++j)
      {
        normal = vvBaseVector3<T>(diff[i]);
        normal.cross(diff[i+1]);
        normal.normalize();
        if (normal.dot(axis) < 0.0f)              // do normals point into opposite directions?
        {
          this[i+1].swap(this[j+1]);             // swap points
          diff[i].swap(diff[j]);                  // swap difference vectors
          swapped = true;
        }
      }
    }
  }
  delete[] diff;
}

//----------------------------------------------------------------------------
/// Set all components to zero.
template <typename T>
void vvBaseVector3<T>::zero()
{
  e[0] = e[1] = e[2] = 0;
}

//----------------------------------------------------------------------------
/// @return true if all vector elements are zero
template <typename T>
bool vvBaseVector3<T>::isZero() const
{
  return (e[0] == 0.0 && e[1] == 0.0 && e[2] == 0.0);
}

//----------------------------------------------------------------------------
/** Convert cartesian coordinates to spherical coordinates
  The sphere coordinate equations are taken from Bronstein page 154f.<BR>
  The used coordinate system is:<PRE>
      y
      |
      |___x
     /
    z
  </PRE>

  @param p      point to convert
@param r      distance from given point to center of sphere
@param phi    angle in the x/z axis, starting at (0|0|1), passing
(1|0|0), (0|0|-1), and (-1|0|0) in this order.
@param theta  angle from the x/y/z position vector to the vector (0|1|0).
@return spherical coordinates in r, phi, and theta
*/
template <typename T>
void vvBaseVector3<T>::getSpherical(T* r, T* phi, T* theta)
{
  vvBaseVector3 upvec;                            // up vector (points into negative y direction [voxel space]

  // Compute sphere coordinates of current destination voxel:
  upvec.set(0.0f, 1.0f, 0.0f);
  *r = length();
  *theta = angle(upvec);

  // Phi must be computed differently for each quadrant:
  if (e[0] >= 0.0f)
  {
    if (e[2] >= 0.0f)
      *phi = (T)atan(static_cast<double>(e[0] / e[2]));            // first quadrant
    else
    {
      *phi = (T)atan(static_cast<double>(e[0] / -e[2]));           // second quadrant
      *phi = VV_PI - *phi;
    }
  }
  else
  {
    if (e[2] <= 0.0f)
    {
      *phi = (T)atan(static_cast<double>(e[0] / e[2]));            // third quadrant
      *phi += VV_PI;
    }
    else
    {
      *phi = (T)atan(static_cast<double>(-e[0] / e[2]));           // fourth quadrant
      *phi = 2.0f * VV_PI - *phi;
    }
  }
}

/** Compute the direction cosines of a vector.
  @param src vector from which to compute the direction cosines
  @return direction cosines in 'this' vector
*/
template <typename T>
void vvBaseVector3<T>::directionCosines(const vvBaseVector3<T>& src)
{
  for (int i=0; i<3; ++i)
    e[i] = cosf(atanf(src.e[i]));
}


//============================================================================
// vvVector4 Class Methods
//============================================================================

//----------------------------------------------------------------------------
/// Copy a vector
template <typename T>
vvBaseVector4<T>::vvBaseVector4()
{
  e[0] = e[1] = e[2] = e[3] = 0;
}

//----------------------------------------------------------------------------
/// Constructor with one value for x, y, z and w
template <typename T>
vvBaseVector4<T>::vvBaseVector4(const T val)
{
  e[0] = e[1] = e[2] = e[3] = val;
}

//----------------------------------------------------------------------------
/// xyzw constructor
template <typename T>
vvBaseVector4<T>::vvBaseVector4(const T x, const T y, const T z, const T w)
{
  e[0] = x;
  e[1] = y;
  e[2] = z;
  e[3] = w;
}

//----------------------------------------------------------------------------
/// array constructor
template <typename T>
vvBaseVector4<T>::vvBaseVector4(T const v[4])
{
  e[0] = v[0];
  e[1] = v[1];
  e[2] = v[2];
  e[3] = v[3];
}

//----------------------------------------------------------------------------
/// constructor for vector3 + w
template <typename T>
vvBaseVector4<T>::vvBaseVector4(const vvBaseVector3<T>& v, const T w)
{
  e[0] = v[0];
  e[1] = v[1];
  e[2] = v[2];
  e[3] = w;
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
template <typename T>
T &vvBaseVector4<T>::operator[](size_t index)
{
  assert(index < 4);
  return e[index];
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
template <typename T>
T const& vvBaseVector4<T>::operator[](size_t index) const
{
  assert(index < 4);
  return e[index];
}

//----------------------------------------------------------------------------
/// Set elements of a 4 vector
template <typename T>
void vvBaseVector4<T>::set(const T a, const T b, const T c, const T d)
{
  e[0] = a;
  e[1] = b;
  e[2] = c;
  e[3] = d;
}

//----------------------------------------------------------------------------
/// Add two vectors
template <typename T>
void vvBaseVector4<T>::add(const vvBaseVector4<T>& v)
{
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  e[3] += v.e[3];
}

//----------------------------------------------------------------------------
/// Subtract two vectors
template <typename T>
void vvBaseVector4<T>::sub(const vvBaseVector4<T>& v)
{
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  e[3] -= v.e[3];
}

//----------------------------------------------------------------------------
/// Multiplies a vector with a matrix (V' = M x V)
template <typename T>
void vvBaseVector4<T>::multiply(const vvMatrix& m)
{
  size_t row, col;
  vvBaseVector4<T> bak(*this);

  for (row=0; row<4; ++row)
  {
    e[row] = 0.0f;
    for(col=0; col<4; ++col)
      e[row] += m(row, col) * bak.e[col];
  }
}

//----------------------------------------------------------------------------
/// Print a 4-vector
template <typename T>
void vvBaseVector4<T>::print(const char* text) const
{
  std::cerr.setf(std::ios::fixed, std::ios::floatfield);
  std::cerr.precision(3);

  if (text == 0)
  {
    std::cerr << e[0] << ", " << e[1] << ", " << e[2] << ", " << e[3] << std::endl;
  }
  else
  {
    std::cerr << text << "\t" << e[0] << ", " << e[1] << ", " << e[2] << ", " << e[3] << std::endl;
  }
  std::cerr << "   Normalized: " << e[0]/e[3] << ", " << e[1]/e[3] << ", " <<
    e[2]/e[3] << std::endl;
}

//----------------------------------------------------------------------------
/// Do a perspective divide
template <typename T>
void vvBaseVector4<T>::perspectiveDivide()
{
  e[0] = e[0] / e[3];
  e[1] = e[1] / e[3];
  e[2] = e[2] / e[3];
  e[3] = 1.0;
}

template <typename T>
vvBaseVector2<T>::vvBaseVector2()
{
  for (size_t i = 0; i < 2; ++i)
  {
    e[i] = 0;
  }
}

//----------------------------------------------------------------------------
/// Constructor with one value for all 2 elements
template <typename T>
vvBaseVector2<T>::vvBaseVector2(const T val)
{
  for (size_t i = 0; i < 2; ++i)
  {
    e[i] = val;
  }
}

//----------------------------------------------------------------------------
/// Constructor for a preset vector
template <typename T>
vvBaseVector2<T>::vvBaseVector2(const T x, const T y)
{
  e[0] = x;
  e[1] = y;
}


//----------------------------------------------------------------------------
/// Overload subscription operator.
template <typename T>
T &vvBaseVector2<T>::operator[](size_t index)
{
  assert(index < 2);
  return e[index];
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
template <typename T>
T const& vvBaseVector2<T>::operator[](size_t index) const
{
  assert(index < 2);
  return e[index];
}


