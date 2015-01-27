/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VEC_H
#define VEC_H

#include <math.h>
#include "Matrix.h"

typedef double REAL;

///////////////////////////////// Vec2d /////////////////////////////////
class Vec2d
{
public:
    REAL vec[2];

    // constructors and destructors
    Vec2d(REAL _x, REAL _y)
    {
        set(_x, _y);
    }
    Vec2d(){};

    //sets and gets
    inline void set(REAL _x, REAL _y)
    {
        vec[0] = _x;
        vec[1] = _y;
    }

    //------- for debugging
    inline void output()
    {
        cout << vec[0] << "\t" << vec[1] << endl;
    }
    //------

    // operators
    inline REAL &operator[](int i)
    {
        return vec[i];
    }
    inline const REAL &operator[](int i) const
    {
        return vec[i];
    }

    // Vec2d operators
    inline Vec2d operator+(const Vec2d &_v) const
    {
        return Vec2d(vec[0] + _v[0], vec[1] + _v[1]);
    }

    friend inline Vec2d operator*(const Matrix_2x2 &_m, const Vec2d &_v);
    friend inline Vec2d operator*(REAL _s, const Vec2d &);
    friend inline Vec2d operator*(const Vec2d &_v, REAL _s);

    // Assignment Operators
    inline Vec2d &operator=(const Vec2d &_v)
    {
        vec[0] = _v[0];
        vec[1] = _v[1];
        return *this;
    }
};

// out of class
inline Vec2d operator*(const Matrix_2x2 &_m, const Vec2d &_v)
{

    return (Vec2d(_m[0][0] * _v[0] + _m[0][1] * _v[1],
                  _m[1][0] * _v[0] + _m[1][1] * _v[1]));
}

inline Vec2d operator*(REAL _s, const Vec2d &_v)
{
    return Vec2d(_v[0] * _s, _v[1] * _s);
}

inline Vec2d operator*(const Vec2d &_v, REAL _s)
{
    return Vec2d(_v[0] * _s, _v[1] * _s);
}

///////////////////////////////// Vec3d /////////////////////////////////
class Vec3d
{
public:
    REAL vec[3];

    // constructors
    Vec3d(REAL _x, REAL _y, REAL _z)
    {
        set(_x, _y, _z);
    }
    Vec3d(){};

    // sets and gets
    inline void set(REAL _x, REAL _y, REAL _z)
    {
        vec[0] = _x;
        vec[1] = _y;
        vec[2] = _z;
    }

    // other functions
    inline REAL dot(const Vec3d &v) const
    {
        return (vec[0] * v[0] + vec[1] * v[1] + vec[2] * v[2]);
    }

    inline REAL length() const
    {
        return sqrt(dot(*this));
    }

    inline void VecProd(const Vec3d &v1, const Vec3d &v2)
    {
        vec[0] = v1[1] * v2[2] - v1[2] * v2[1];
        vec[1] = v1[2] * v2[0] - v1[0] * v2[2];
        vec[2] = v1[0] * v2[1] - v1[1] * v2[0];
    }

    //------- for debugging
    inline void output()
    {
        cout << vec[0] << "\t" << vec[1] << "\t" << vec[2] << endl;
    }
    //------

    // Operators
    inline REAL &operator[](int i)
    {
        return vec[i];
    }
    inline const REAL &operator[](int i) const
    {
        return vec[i];
    }

    // Vec3d operators
    inline Vec3d operator+(const Vec3d &_v) const
    {
        return Vec3d(vec[0] + _v[0], vec[1] + _v[1], vec[2] + _v[2]);
    }

    inline Vec3d operator-(const Vec3d &_v) const
    {
        return Vec3d(vec[0] - _v[0], vec[1] - _v[1], vec[2] - _v[2]);
    }

    friend inline Vec3d operator*(REAL _s, const Vec3d &);
    friend inline Vec3d operator*(const Vec3d &_v, REAL _s);
    friend inline Vec3d operator/(REAL _s, const Vec3d &);
    friend inline Vec3d operator/(const Vec3d &_v, REAL _s);
    friend inline ostream &operator<<(ostream &, const Vec3d &);
    friend inline istream &operator>>(istream &, Vec3d &);

    // Assignment Operators
    inline Vec3d &operator=(const Vec3d &_v)
    {
        vec[0] = _v[0];
        vec[1] = _v[1];
        vec[2] = _v[2];
        return *this;
    }
};

// out of class
inline Vec3d operator*(REAL _s, const Vec3d &_v)
{
    return Vec3d(_v[0] * _s, _v[1] * _s, _v[2] * _s);
}

inline Vec3d operator*(const Vec3d &_v, REAL _s)
{
    return Vec3d(_v[0] * _s, _v[1] * _s, _v[2] * _s);
}

inline Vec3d operator/(REAL _s, const Vec3d &_v)
{
    return Vec3d(_v[0] / _s, _v[1] / _s, _v[2] / _s);
}

inline Vec3d operator/(const Vec3d &_v, REAL _s)
{
    return Vec3d(_v[0] / _s, _v[1] / _s, _v[2] / _s);
}

inline ostream &operator<<(ostream &OS, const Vec3d &_v)
{
    OS << "<" << _v[0] << "," << _v[1] << "," << _v[2] << ">";
    return OS;
}

inline istream &operator>>(istream &IS, Vec3d &_v)
{
    int c;

    if ((c = IS.get()) != '<')
    {
        IS.clear(ios::badbit | IS.rdstate());
        return IS;
    }

    while (IS && (c = IS.get()) != '<')
        ;
    IS >> _v[0];

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> _v[1];

    while (IS && (c = IS.get()) != ',')
        ;
    IS >> _v[2];

    while (IS && (c = IS.get()) != '>')
        ;

    return IS;
}

///////////////////////////////// Vec4d /////////////////////////////////
class Vec4d
{
public:
    REAL vec[4];

    // constructors and destructors
    Vec4d(REAL _x, REAL _y, REAL _z, REAL _w)
    {
        set(_x, _y, _z, _w);
    }
    Vec4d(const Vec3d &_v, REAL _w)
    {
        set(_v[0], _v[1], _v[2], _w);
    }
    Vec4d(){};

    // sets and gets
    inline void set(REAL _x, REAL _y, REAL _z, REAL _w)
    {
        vec[0] = _x;
        vec[1] = _y;
        vec[2] = _z;
        vec[3] = _w;
    }

    inline void set(const Vec3d &_v, REAL _w)
    {
        vec[0] = _v[0];
        vec[1] = _v[1];
        vec[2] = _v[2];
        vec[3] = _w;
    }

    // other functions
    inline REAL dot(const Vec4d &v) const
    {
        return (vec[0] * v[0] + vec[1] * v[1] + vec[2] * v[2] + vec[3] * v[3]);
    }

    inline REAL length() const
    {
        return sqrt(dot(*this));
    }

    //------- for debugging
    inline void output()
    {
        cout << vec[0] << "\t" << vec[1] << "\t"
             << vec[2] << "\t" << vec[3] << endl;
    }
    //------

    // Operators
    inline REAL &operator[](int i)
    {
        return vec[i];
    }
    inline const REAL &operator[](int i) const
    {
        return vec[i];
    }

    // Vec4d operators
    inline Vec4d operator+(const Vec4d &_v) const
    {
        return Vec4d(vec[0] + _v[0], vec[1] + _v[1],
                     vec[2] + _v[2], vec[3] + _v[3]);
    }

    inline Vec4d operator-(const Vec4d &_v) const
    {
        return Vec4d(vec[0] - _v[0], vec[1] - _v[1],
                     vec[2] - _v[2], vec[3] - _v[3]);
    }

    friend inline Vec4d operator*(REAL _s, const Vec4d &);
    friend inline Vec4d operator*(const Vec4d &_v, REAL _s);
    friend inline Vec4d operator/(REAL _s, const Vec4d &);
    friend inline Vec4d operator/(const Vec4d &_v, REAL _s);

    // Assignment Operators
    inline Vec4d &operator=(const Vec4d &_v)
    {
        vec[0] = _v[0];
        vec[1] = _v[1];
        vec[2] = _v[2];
        vec[3] = _v[3];
        return *this;
    }
};

// out of class
inline Vec4d operator*(REAL _s, const Vec4d &_v)
{
    return Vec4d(_v[0] * _s, _v[1] * _s, _v[2] * _s, _v[3] * _s);
}

inline Vec4d operator*(const Vec4d &_v, REAL _s)
{
    return Vec4d(_v[0] * _s, _v[1] * _s, _v[2] * _s, _v[3] * _s);
}

inline Vec4d operator/(REAL _s, const Vec4d &_v)
{
    return Vec4d(_v[0] / _s, _v[1] / _s, _v[2] / _s, _v[3] / _s);
}

inline Vec4d operator/(const Vec4d &_v, REAL _s)
{
    return Vec4d(_v[0] / _s, _v[1] / _s, _v[2] / _s, _v[3] / _s);
}
#endif // VEC_H
