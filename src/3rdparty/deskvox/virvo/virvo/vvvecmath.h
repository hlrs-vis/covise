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

#ifndef VV_VECMATH_H
#define VV_VECMATH_H

#include <float.h>
#include <iostream>

#include "math/forward.h"

#include "vvexport.h"
#include "vvinttypes.h"

//============================================================================
// Constant Definitions
//============================================================================

                                                  ///< compiler independent definition for pi
const float VV_PI = 3.1415926535897932384626433832795028841971693993751058f;
const float VV_FLT_MAX = FLT_MAX;                 ///< maximum float value

//============================================================================
// Forward Declarations
//============================================================================

class vvMatrix;
template <typename T>
class vvBaseVector2;
template <typename T>
class vvBaseVector3;
template <typename T>
class vvBaseVector4;

typedef vvBaseVector3<size_t>               vvsize3;
typedef vvBaseVector3<ssize_t>              vvssize3;

typedef vvBaseVector2<int>                  vvVector2i;
typedef vvBaseVector2<unsigned int>         vvVector2ui;
typedef vvBaseVector2<short>                vvVector2s;
typedef vvBaseVector2<unsigned short>       vvVector2us;
typedef vvBaseVector2<long>                 vvVector2l;
typedef vvBaseVector2<unsigned long>        vvVector2ul;
typedef vvBaseVector2<long long>            vvVector2ll;
typedef vvBaseVector2<unsigned long long>   vvVector2ull;
typedef vvBaseVector2<float>                vvVector2f;
typedef vvBaseVector2<double>               vvVector2d;
typedef vvVector2f                          vvVector2;

typedef vvBaseVector3<int>                  vvVector3i;
typedef vvBaseVector3<unsigned int>         vvVector3ui;
typedef vvBaseVector3<short>                vvVector3s;
typedef vvBaseVector3<unsigned short>       vvVector3us;
typedef vvBaseVector3<long>                 vvVector3l;
typedef vvBaseVector3<unsigned long>        vvVector3ul;
typedef vvBaseVector3<long long>            vvVector3ll;
typedef vvBaseVector3<unsigned long long>   vvVector3ull;
typedef vvBaseVector3<float>                vvVector3f;
typedef vvBaseVector3<double>               vvVector3d;
typedef vvVector3f                          vvVector3;

typedef vvBaseVector4<int>                  vvVector4i;
typedef vvBaseVector4<unsigned int>         vvVector4ui;
typedef vvBaseVector4<short>                vvVector4s;
typedef vvBaseVector4<unsigned short>       vvVector4us;
typedef vvBaseVector4<long>                 vvVector4l;
typedef vvBaseVector4<unsigned long>        vvVector4ul;
typedef vvBaseVector4<long long>            vvVector4ll;
typedef vvBaseVector4<unsigned long long>   vvVector4ull;
typedef vvBaseVector4<float>                vvVector4f;
typedef vvBaseVector4<double>               vvVector4d;
typedef vvVector4f                          vvVector4;

namespace virvo
{
typedef vvMatrix                            Matrix;

typedef vvsize3                             size3;
typedef vvssize3                            ssize3;

typedef vvVector2i                          Vec2i;
typedef vvVector2ui                         Vec2ui;
typedef vvVector2s                          Vec2s;
typedef vvVector2us                         Vec2us;
typedef vvVector2l                          Vec2l;
typedef vvVector2ul                         Vec2ul;
typedef vvVector2ll                         Vec2ll;
typedef vvVector2ull                        Vec2ull;
typedef vvVector2f                          Vec2f;
typedef vvVector2d                          Vec2d;
typedef vvVector2                           Vec2;

typedef vvVector3i                          Vec3i;
typedef vvVector3ui                         Vec3ui;
typedef vvVector3s                          Vec3s;
typedef vvVector3us                         Vec3us;
typedef vvVector3l                          Vec3l;
typedef vvVector3ul                         Vec3ul;
typedef vvVector3ll                         Vec3ll;
typedef vvVector3ull                        Vec3ull;
typedef vvVector3f                          Vec3f;
typedef vvVector3d                          Vec3d;
typedef vvVector3                           Vec3;

typedef vvVector4i                          Vec4i;
typedef vvVector4ui                         Vec4ui;
typedef vvVector4s                          Vec4s;
typedef vvVector4us                         Vec4us;
typedef vvVector4l                          Vec4l;
typedef vvVector4ul                         Vec4ul;
typedef vvVector4ll                         Vec4ll;
typedef vvVector4ull                        Vec4ull;
typedef vvVector4f                          Vec4f;
typedef vvVector4d                          Vec4d;
typedef vvVector4                           Vec4;
}

//============================================================================
// Class Definitions
//============================================================================

class VIRVOEXPORT vvVecmath
{
  public:
    enum AxisType                                 ///<  names for coordinate axes
    { X_AXIS = 0, Y_AXIS = 1, Z_AXIS = 2 };
    static float sgn(float);
};

/** 4x4 matrix type.
 Matrix elements are: e[row][column]
 @author Jurgen P. Schulze (jschulze@ucsd.edu)
*/
class VIRVOEXPORT vvMatrix
{
  private:
    void LUDecomposition(int index[4], float &d);
    void LUBackSubstitution(int index[4], float b[4]);

    float e[4][4];                                ///< matrix elements: [row][column]

  public:
    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          a & e[i][j];
    }

  public:
    vvMatrix();
    template < typename U >
    vvMatrix(virvo::matrix< 4, 4, U > const& rhs)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                e[i][j] = rhs(i, j);
            }
        }
    }
    template < typename U >
    operator virvo::matrix< 4, 4, U >() const
    {
        return virvo::matrix< 4, 4, U >
        (
            U(e[0][0]), U(e[1][0]), U(e[2][0]), U(e[3][0]),
            U(e[0][1]), U(e[1][1]), U(e[2][1]), U(e[3][1]),
            U(e[0][2]), U(e[1][2]), U(e[2][2]), U(e[3][2]),
            U(e[0][3]), U(e[1][3]), U(e[2][3]), U(e[3][3])
        );
    }
    // Constructs a diagonal matrix
    vvMatrix(float d0, float d1, float d2, float d3);
    vvMatrix(float* glf);
    float& operator()(size_t row, size_t col);
    float const& operator()(size_t row, size_t col) const;
    vvMatrix operator+(const vvMatrix& operand) const;
    vvMatrix operator-(const vvMatrix& operand) const;
    vvMatrix operator*(const vvMatrix& operand) const;
    void print(const char*) const;
    void identity();
    void zero();

    // Returns a pointer to the matrix elements
    float* data() { return &e[0][0]; }

    // Returns a pointer to the matrix elements
    const float* data() const { return &e[0][0]; }

    // Multiplies this matrix from the left with a translation matix
    // Note: assumes the 4th row of this matrix equals (0,0,0,1)
    vvMatrix& translate(float x, float y, float z);

    // Multiplies this matrix from the left with a translation matix
    // Note: assumes the 4th row of this matrix equals (0,0,0,1)
    vvMatrix& translate(const vvVector3& t);

    // Multiplies this matrix from the right with a scaling matrix
    vvMatrix& scaleLocal(float x, float y, float z);

    // Multiplies this matrix from the right with a uniform scaling matrix
    vvMatrix& scaleLocal(float s);

    vvMatrix rotate(float, float, float, float);
    vvMatrix rotate(float, const vvVector3& vec);

    // Multiplies this matrix from the left with the given matrix: this = LHS * this
    vvMatrix& multiplyLeft(const vvMatrix& LHS);

    // Multiplies this matrix from the right with the given matrix: this = this * RHS
    vvMatrix& multiplyRight(const vvMatrix& RHS);

    void transpose();
    float diagonal();
    void invertOrtho();
    void invert2D();
    void copyTrans(const vvMatrix& m);
    void copyRot(const vvMatrix& m);
    void transOnly();
    void rotOnly();
    void killTrans();
    void killRot();
    bool equal(const vvMatrix& m) const;
    void getGL(float*) const;
    void setGL(const float*);
    void setGL(const double*);
    void get(float*) const;
    void set(const float*);
    void get(double*) const;
    void set(const double*);
    void setRow(int, float, float, float, float);
    void setRow(int, const vvVector3& vec);
    void setColumn(int, float, float, float, float);
    void setColumn(int, const vvVector3& vec);
    void getRow(int, float*, float*, float*, float*) const;
    void getRow(int, vvVector3*) const;
    void getColumn(int, float*, float*, float*, float*);
    void getColumn(int, vvVector3& vec);
    void random(int, int);
    void random(float, float);
    void invert();
    void swapRows(int, int);
    void swapColumns(int, int);
    void setProjOrtho(float, float, float, float, float, float);
    void getProjOrtho(float*, float*, float*, float*, float*, float*);
    void setProjPersp(float, float, float, float, float, float);
    void getProjPersp(float*, float*, float*, float*, float*, float*);
    bool isProjOrtho() const;
    void makeLookAt(float, float, float, float, float, float, float, float, float);
    float getNearPlaneZ() const;
    vvMatrix trackballRotation(int, int, int, int, int, int);
    void computeEulerAngles(float*, float*, float*);
};

/** 3D vector primitive.
 @author Juergen Schulze-Doebold (schulze@hlrs.de)
*/
template <typename T>
class vvBaseVector4
{
    T e[4];                                   ///< vector elements (x|y|z|w)

  public:
    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & e[0];
      a & e[1];
      a & e[2];
      a & e[3];
    }

  public:
    vvBaseVector4();
    explicit vvBaseVector4(T val);
    // for transition
    /* implicit */ vvBaseVector4(virvo::vector< 4, T > const& v)
    {
        e[0] = v.x;
        e[1] = v.y;
        e[2] = v.z;
        e[3] = v.w;
    }
    template < typename U >
    /* implicit */ vvBaseVector4(virvo::vector< 4, U > const& v)
    {
        e[0] = T(v.x);
        e[1] = T(v.y);
        e[2] = T(v.z);
        e[3] = T(v.w);
    }
    operator virvo::vector< 4, T >() const { return virvo::vector< 4, T >( e[0], e[1], e[2], e[3] ); }
    template < typename U >
    operator virvo::vector< 4, U >() const { return virvo::vector< 4, U >( U(e[0]), U(e[1]), U(e[2]), U(e[3]) ); }
    vvBaseVector4(T x, T y, T z, T w);
    vvBaseVector4(T const v[4]);
    vvBaseVector4(const vvBaseVector3<T>& v, const T w);
    T &operator[](size_t index);
    T const& operator[](size_t index) const;
    void set(T x, T y, T z, T w);
    void multiply(const vvMatrix& m);
    void add(const vvBaseVector4& rhs);
    void sub(const vvBaseVector4& rhs);
    void print(const char* text = 0) const;
    void perspectiveDivide();
};

/** base vector primitive
 @author Jurgen P. Schulze (jschulze@ucsd.edu)
*/
template <typename T>
class vvBaseVector3
{
    T e[3];                                   ///< vector elements (x|y|z)

  public:

    typedef T value_type;

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & e[0];
      a & e[1];
      a & e[2];
    }

  public:
    vvBaseVector3();
    explicit vvBaseVector3(T);

    // for transition
    /* implicit */ vvBaseVector3(virvo::vector< 3, T > const& v)
    {
        e[0] = v.x;
        e[1] = v.y;
        e[2] = v.z;
    }
    template < typename U >
    /* implicit */ vvBaseVector3(virvo::vector< 3, U > const& v)
    {
        e[0] = T(v.x);
        e[1] = T(v.y);
        e[2] = T(v.z);
    }
    operator virvo::vector< 3, T >() const { return virvo::vector< 3, T >( e[0], e[1], e[2] ); }
    template < typename U >
    operator virvo::vector< 3, U >() const { return virvo::vector< 3, U >( U(e[0]), U(e[1]), U(e[2]) ); }
    vvBaseVector3(T x, T y, T z);
    vvBaseVector3(const vvBaseVector4<T>& v);

    T &operator[](size_t index);
    T const& operator[](size_t index) const;
    void  set(T x, T y, T z);
    void  get(T* x, T* y, T* z) const;
    void  add(const vvBaseVector3& rhs);
    void  add(T val);
    void  add(T x, T y, T z);
    void  sub(const vvBaseVector3& rhs);
    void  sub(T val);
    void  scale(T s);
    void  scale(const vvBaseVector3& rhs);
    void  scale(T x, T y, T z);
    T dot(const vvBaseVector3& v) const;
    T angle(const vvBaseVector3& v) const;
    void  cross(const vvBaseVector3& rhs);
    void  multiply(const vvMatrix& m);
    T distance(const vvBaseVector3& v) const;
    T length() const;
    void  planeNormalPPV(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    T distPointPlane(const vvBaseVector3&, const vvBaseVector3&) const;
    void  normalize();
    void  negate();
    bool  equal(const vvBaseVector3& rhs);
    void  random(int, int);
    void  random(float, float);
    void  random(double, double);
    void  print(const char* text = 0) const;
    void  getRow(const vvMatrix& m, const int);
    void  getColumn(const vvMatrix& m, const int);
    void  swap(vvBaseVector3<T>& v);
    bool  isectPlaneLine(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    bool  isectPlaneRay(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    size_t isectPlaneCuboid(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    int   isectRayCylinder(const vvBaseVector3&, const vvBaseVector3&, T, const vvBaseVector3&, const vvBaseVector3&);
    bool  isectRayTriangle(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    T isectLineLine(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    bool  isSameSideLine2D(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    bool  isInTriangle(const vvBaseVector3&, const vvBaseVector3&, const vvBaseVector3&);
    void  cyclicSort(const int, const vvBaseVector3& axis);
    void  zero();
    bool  isZero() const;
    void  getSpherical(T*, T*, T*);
    void  directionCosines(const vvBaseVector3&);
};

template <typename T>
class vvBaseVector2
{
  public:
    vvBaseVector2();
    explicit vvBaseVector2(T);
    vvBaseVector2(T x, T y);

    T &operator[](size_t index);
    T const& operator[](size_t index) const;
  private:
    T e[2];
  public:
    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & e[0];
      a & e[1];
    }
};

/** 3D plane primitive.
 @author Jurgen Schulze (jschulze@ucsd.edu)
*/
class vvPlane
{
  public:
    vvVector3 _point;
    vvVector3 _normal;

  public:
    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & _point;
      a & _normal;
    }

    vvPlane();
    vvPlane(const vvVector3& p, const vvVector3& n);
    vvPlane(const vvVector3& p, const vvVector3& dir1, const vvVector3& dir2);
    bool isSameSide(const vvVector3&, const vvVector3&) const;
    float dist(const vvVector3&) const;
};

inline std::ostream& operator<<(std::ostream& out, const vvMatrix& m)
{
  for (size_t i = 0; i < 4; ++i)
  {
    for (size_t j = 0; j < 4; ++j)
    {
      out << " " << m(i, j);
    }
    out << "\n";
  }
  return out;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const vvBaseVector2<T>& v)
{
  out << v[0] << " " << v[1];
  return out;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const vvBaseVector3<T>& v)
{
  out << v[0] << " " << v[1] << " " << v[2];
  return out;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const vvBaseVector4<T>& v)
{
  out << v[0] << " " << v[1] << " " << v[2] << " " << v[3];
  return out;
}


//------------------------------------------------------------------------------
// vvBaseVector2 operators
//------------------------------------------------------------------------------


template<typename T>
bool operator ==(vvBaseVector2<T> const& u, vvBaseVector2<T> const& v)
{
  return u[0] == v[0] && u[1] == v[1];
}


template<typename T>
bool operator !=(vvBaseVector2<T> const& u, vvBaseVector2<T> const& v)
{
  return !(u == v);
}


template<typename T>
vvBaseVector2<T>& operator +=(vvBaseVector2<T>& u, vvBaseVector2<T> const& v)
{
  u[0] += v[0];
  u[1] += v[1];

  return u;
}


template<typename T>
vvBaseVector2<T>& operator -=(vvBaseVector2<T>& u, vvBaseVector2<T> const& v)
{
  u[0] -= v[0];
  u[1] -= v[1];

  return u;
}


template<typename T>
vvBaseVector2<T>& operator *=(vvBaseVector2<T>& u, vvBaseVector2<T> const& v)
{
  u[0] *= v[0];
  u[1] *= v[1];

  return u;
}


template<typename T>
vvBaseVector2<T>& operator /=(vvBaseVector2<T>& u, vvBaseVector2<T> const& v)
{
  u[0] /= v[0];
  u[1] /= v[1];

  return u;
}


template<typename T>
vvBaseVector2<T>& operator +=(vvBaseVector2<T>& u, T const& v)
{
  u[0] += v;
  u[1] += v;

  return u;
}


template<typename T>
vvBaseVector2<T>& operator -=(vvBaseVector2<T>& u, T const& v)
{
  u[0] -= v;
  u[1] -= v;

  return u;
}


template<typename T>
vvBaseVector2<T>& operator *=(vvBaseVector2<T>& u, T const& v)
{
  u[0] *= v;
  u[1] *= v;

  return u;
}


template<typename T>
vvBaseVector2<T>& operator /=(vvBaseVector2<T>& u, T const& v)
{
  u[0] /= v;
  u[1] /= v;

  return u;
}


template<typename T>
vvBaseVector2<T> operator -(vvBaseVector2<T> const& u)
{
  return vvBaseVector2<T>(-u[0], -u[1]);
}


template<typename T>
vvBaseVector2<T> operator +(vvBaseVector2<T> const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u[0] + v[0], u[1] + v[1]);
}


template<typename T>
vvBaseVector2<T> operator -(vvBaseVector2<T> const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u[0] - v[0], u[1] - v[1]);
}


template<typename T>
vvBaseVector2<T> operator *(vvBaseVector2<T> const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u[0] * v[0], u[1] * v[1]);
}


template<typename T>
vvBaseVector2<T> operator /(vvBaseVector2<T> const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u[0] / v[0], u[1] / v[1]);
}


template<typename T>
vvBaseVector2<T> operator +(vvBaseVector2<T> const& u, T const& v)
{
  return vvBaseVector2<T>(u[0] + v, u[1] + v);
}


template<typename T>
vvBaseVector2<T> operator -(vvBaseVector2<T> const& u, T const& v)
{
  return vvBaseVector2<T>(u[0] - v, u[1] - v);
}


template<typename T>
vvBaseVector2<T> operator *(vvBaseVector2<T> const& u, T const& v)
{
  return vvBaseVector2<T>(u[0] * v, u[1] * v);
}


template<typename T>
vvBaseVector2<T> operator /(vvBaseVector2<T> const& u, T const& v)
{
  return vvBaseVector2<T>(u[0] / v, u[1] / v);
}


template<typename T>
vvBaseVector2<T> operator +(T const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u + v[0], u + v[1]);
}


template<typename T>
vvBaseVector2<T> operator -(T const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u - v[0], u - v[1]);
}


template<typename T>
vvBaseVector2<T> operator *(T const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u * v[0], u * v[1]);
}


template<typename T>
vvBaseVector2<T> operator /(T const& u, vvBaseVector2<T> const& v)
{
  return vvBaseVector2<T>(u / v[0], u / v[1]);
}


//------------------------------------------------------------------------------
// vvBaseVector3 operators
//------------------------------------------------------------------------------


template<typename T>
bool operator ==(vvBaseVector3<T> const& u, vvBaseVector3<T> const& v)
{
  return u[0] == v[0] && u[1] == v[1] && u[2] == v[2];
}


template<typename T>
bool operator !=(vvBaseVector3<T> const& u, vvBaseVector3<T> const& v)
{
  return !(u == v);
}


template<typename T>
vvBaseVector3<T>& operator +=(vvBaseVector3<T>& u, vvBaseVector3<T> const& v)
{
  u[0] += v[0];
  u[1] += v[1];
  u[2] += v[2];

  return u;
}


template<typename T>
vvBaseVector3<T>& operator -=(vvBaseVector3<T>& u, vvBaseVector3<T> const& v)
{
  u[0] -= v[0];
  u[1] -= v[1];
  u[2] -= v[2];

  return u;
}


template<typename T>
vvBaseVector3<T>& operator *=(vvBaseVector3<T>& u, vvBaseVector3<T> const& v)
{
  u[0] *= v[0];
  u[1] *= v[1];
  u[2] *= v[2];

  return u;
}


template<typename T>
vvBaseVector3<T>& operator /=(vvBaseVector3<T>& u, vvBaseVector3<T> const& v)
{
  u[0] /= v[0];
  u[1] /= v[1];
  u[2] /= v[2];

  return u;
}


template<typename T>
vvBaseVector3<T>& operator +=(vvBaseVector3<T>& u, T const& v)
{
  u[0] += v;
  u[1] += v;
  u[2] += v;

  return u;
}


template<typename T>
vvBaseVector3<T>& operator -=(vvBaseVector3<T>& u, T const& v)
{
  u[0] -= v;
  u[1] -= v;
  u[2] -= v;

  return u;
}


template<typename T>
vvBaseVector3<T>& operator *=(vvBaseVector3<T>& u, T const& v)
{
  u[0] *= v;
  u[1] *= v;
  u[2] *= v;

  return u;
}


template<typename T>
vvBaseVector3<T>& operator /=(vvBaseVector3<T>& u, T const& v)
{
  u[0] /= v;
  u[1] /= v;
  u[2] /= v;

  return u;
}


template<typename T>
vvBaseVector3<T> operator -(vvBaseVector3<T> const& u)
{
  return vvBaseVector3<T>(-u[0], -u[1], -u[2]);
}


template<typename T>
vvBaseVector3<T> operator +(vvBaseVector3<T> const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}


template<typename T>
vvBaseVector3<T> operator -(vvBaseVector3<T> const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}


template<typename T>
vvBaseVector3<T> operator *(vvBaseVector3<T> const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}


template<typename T>
vvBaseVector3<T> operator /(vvBaseVector3<T> const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u[0] / v[0], u[1] / v[1], u[2] / v[2]);
}


template<typename T>
vvBaseVector3<T> operator +(vvBaseVector3<T> const& u, T const& v)
{
  return vvBaseVector3<T>(u[0] + v, u[1] + v, u[2] + v);
}


template<typename T>
vvBaseVector3<T> operator -(vvBaseVector3<T> const& u, T const& v)
{
  return vvBaseVector3<T>(u[0] - v, u[1] - v, u[2] - v);
}


template<typename T>
vvBaseVector3<T> operator *(vvBaseVector3<T> const& u, T const& v)
{
  return vvBaseVector3<T>(u[0] * v, u[1] * v, u[2] * v);
}


template<typename T>
vvBaseVector3<T> operator /(vvBaseVector3<T> const& u, T const& v)
{
  return vvBaseVector3<T>(u[0] / v, u[1] / v, u[2] / v);
}


template<typename T>
vvBaseVector3<T> operator +(T const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u + v[0], u + v[1], u + v[2]);
}


template<typename T>
vvBaseVector3<T> operator -(T const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u - v[0], u - v[1], u - v[2]);
}


template<typename T>
vvBaseVector3<T> operator *(T const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u * v[0], u * v[1], u * v[2]);
}


template<typename T>
vvBaseVector3<T> operator /(T const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(u / v[0], u / v[1], u / v[2]);
}


template<typename T>
vvBaseVector3<T> operator *(vvMatrix const& m, vvBaseVector3<T> const& v)
{
  vvBaseVector3<T> result = v;
  result.multiply(m);
  return result;
}

// Returns the cross product of u and v
template<typename T>
vvBaseVector3<T> operator ^(vvBaseVector3<T> const& u, vvBaseVector3<T> const& v)
{
  return vvBaseVector3<T>(
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]
    );
}


//------------------------------------------------------------------------------
// vvBaseVector4 operators
//------------------------------------------------------------------------------


template<typename T>
bool operator ==(vvBaseVector4<T> const& u, vvBaseVector4<T> const& v)
{
  return u[0] == v[0] && u[1] == v[1] && u[2] == v[2] && u[3] == v[3];
}


template<typename T>
bool operator !=(vvBaseVector4<T> const& u, vvBaseVector4<T> const& v)
{
  return !(u == v);
}


template<typename T>
vvBaseVector4<T>& operator +=(vvBaseVector4<T>& u, vvBaseVector4<T> const& v)
{
  u[0] += v[0];
  u[1] += v[1];
  u[2] += v[2];
  u[3] += v[3];

  return u;
}


template<typename T>
vvBaseVector4<T>& operator -=(vvBaseVector4<T>& u, vvBaseVector4<T> const& v)
{
  u[0] -= v[0];
  u[1] -= v[1];
  u[2] -= v[2];
  u[3] -= v[3];

  return u;
}


template<typename T>
vvBaseVector4<T>& operator *=(vvBaseVector4<T>& u, vvBaseVector4<T> const& v)
{
  u[0] *= v[0];
  u[1] *= v[1];
  u[2] *= v[2];
  u[3] *= v[3];

  return u;
}


template<typename T>
vvBaseVector4<T>& operator /=(vvBaseVector4<T>& u, vvBaseVector4<T> const& v)
{
  u[0] /= v[0];
  u[1] /= v[1];
  u[2] /= v[2];
  u[3] /= v[3];

  return u;
}


template<typename T>
vvBaseVector4<T>& operator +=(vvBaseVector4<T>& u, T const& v)
{
  u[0] += v;
  u[1] += v;
  u[2] += v;
  u[3] += v;

  return u;
}


template<typename T>
vvBaseVector4<T>& operator -=(vvBaseVector4<T>& u, T const& v)
{
  u[0] -= v;
  u[1] -= v;
  u[2] -= v;
  u[3] -= v;

  return u;
}


template<typename T>
vvBaseVector4<T>& operator *=(vvBaseVector4<T>& u, T const& v)
{
  u[0] *= v;
  u[1] *= v;
  u[2] *= v;
  u[3] *= v;

  return u;
}


template<typename T>
vvBaseVector4<T>& operator /=(vvBaseVector4<T>& u, T const& v)
{
  u[0] /= v;
  u[1] /= v;
  u[2] /= v;
  u[3] /= v;

  return u;
}


template<typename T>
vvBaseVector4<T> operator -(vvBaseVector4<T> const& u)
{
  return vvBaseVector4<T>(-u[0], -u[1], -u[2], -u[3]);
}


template<typename T>
vvBaseVector4<T> operator +(vvBaseVector4<T> const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u[0] + v[0], u[1] + v[1], u[2] + v[2], u[3] + v[3]);
}


template<typename T>
vvBaseVector4<T> operator -(vvBaseVector4<T> const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u[0] - v[0], u[1] - v[1], u[2] - v[2], u[3] - v[3]);
}


template<typename T>
vvBaseVector4<T> operator *(vvBaseVector4<T> const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u[0] * v[0], u[1] * v[1], u[2] * v[2], u[3] * v[3]);
}


template<typename T>
vvBaseVector4<T> operator /(vvBaseVector4<T> const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u[0] / v[0], u[1] / v[1], u[2] / v[2], u[3] / v[3]);
}


template<typename T>
vvBaseVector4<T> operator +(vvBaseVector4<T> const& u, T const& v)
{
  return vvBaseVector4<T>(u[0] + v, u[1] + v, u[2] + v, u[3] + v);
}


template<typename T>
vvBaseVector4<T> operator -(vvBaseVector4<T> const& u, T const& v)
{
  return vvBaseVector4<T>(u[0] - v, u[1] - v, u[2] - v, u[3] - v);
}


template<typename T>
vvBaseVector4<T> operator *(vvBaseVector4<T> const& u, T const& v)
{
  return vvBaseVector4<T>(u[0] * v, u[1] * v, u[2] * v, u[3] * v);
}


template<typename T>
vvBaseVector4<T> operator /(vvBaseVector4<T> const& u, T const& v)
{
  return vvBaseVector4<T>(u[0] / v, u[1] / v, u[2] / v, u[3] / v);
}


template<typename T>
vvBaseVector4<T> operator +(T const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u + v[0], u + v[1], u + v[2], u + v[3]);
}


template<typename T>
vvBaseVector4<T> operator -(T const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u - v[0], u - v[1], u - v[2], u - v[3]);
}


template<typename T>
vvBaseVector4<T> operator *(T const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u * v[0], u * v[1], u * v[2], u * v[3]);
}


template<typename T>
vvBaseVector4<T> operator /(T const& u, vvBaseVector4<T> const& v)
{
  return vvBaseVector4<T>(u / v[0], u / v[1], u / v[2], u / v[3]);
}


template<typename T>
vvBaseVector4<T> operator *(vvMatrix const& m, vvBaseVector4<T> const& v)
{
  vvBaseVector4<T> result = v;
  result.multiply(m);
  return result;
}


//------------------------------------------------------------------------------
// vvBaseVector3 functions
//------------------------------------------------------------------------------

namespace virvo
{
namespace vecmath
{
template <typename T>
T dot(vvBaseVector3<T> const& v)
{
  return v.dot();
}


template <typename T>
T length(vvBaseVector3<T> const& v)
{
  return v.length();
}


template <typename T>
vvBaseVector3<T> normalize(vvBaseVector3<T> const& v)
{
  vvBaseVector3<T> result = v;
  result.normalize();
  return result;
}

} // vecmath
} // virvo

#include "vvvecmath.impl.h"

#endif

// EOF
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
