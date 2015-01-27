/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Types_h
#define Types_h

#include <map>
#include <vector>
#include <osg/Matrix>

class Vector2D
{
public:
    Vector2D(const double &x, const double &y);

    void set(const double &x, const double &y);

    double &x();
    double x() const;
    double &y();
    double y() const;

    double &operator[](const int &i);
    double operator[](const int &i) const;

    Vector2D operator+(const Vector2D &vec) const;

    Vector2D operator-(const Vector2D &vec) const;

    const Vector2D &operator+=(const Vector2D &vec);

    const Vector2D &operator-=(const Vector2D &vec);

    const Vector2D &operator=(const Vector2D &vec);

    bool operator==(const Vector2D &vec) const;

private:
    double p[2];
};

class Quaternion;

class Vector3D
{
public:
    Vector3D(const double &x, const double &y, const double &z);

    void set(const double &x, const double &y, const double &z);

    double &x();
    double x() const;
    double &y();
    double y() const;
    double &z();
    double z() const;

    double &operator[](const int &i);
    double operator[](const int &i) const;

    Vector3D operator+(const Vector3D &vec) const;

    Vector3D operator-(const Vector3D &vec) const;

    const Vector3D &operator+=(const Vector3D &vec);

    const Vector3D &operator-=(const Vector3D &vec);

    const Vector3D &operator=(const Vector3D &vec);

    bool operator==(const Vector3D &vec) const;

    Quaternion operator*(const Quaternion &quat) const;

private:
    double p[3];
};

class Quaternion
{
public:
    Quaternion(const double &wset, const double &xset, const double &yset, const double &zset);

    Quaternion(const double &angle, const Vector3D &vec);

    double &w();
    double w() const;
    double &x();
    double x() const;
    double &y();
    double y() const;
    double &z();
    double z() const;

    Quaternion T() const;

    Vector3D getVector() const;

    Quaternion operator*(const Quaternion &quat) const;
    Quaternion operator*(const Vector3D &vec) const;

    const Quaternion &operator=(const Quaternion &quat);

private:
    double _w, _x, _y, _z;
};

std::ostream &operator<<(std::ostream &, const Quaternion &);

class Transform
{
public:
    Transform();

    Transform(const Vector3D &vec, const Quaternion &quat);

    Vector3D &v();

    Vector3D v() const;

    Quaternion &q();

    Quaternion q() const;

    Transform operator*(const Transform &trans) const;

    const Transform &operator=(const Transform &trans);

protected:
    Vector3D _v;
    Quaternion _q;
};

class RoadPoint
{
public:
    RoadPoint(const double &p1 = 0, const double &p2 = 0, const double &p3 = 0, const double &n1 = 0, const double &n2 = 0, const double &n3 = 0)
    {
        p[0] = p1;
        p[1] = p2;
        p[2] = p3;

        n[0] = n1;
        n[1] = n2;
        n[2] = n3;
    }

    double &x()
    {
        return p[0];
    }
    double x() const
    {
        return p[0];
    }
    double &y()
    {
        return p[1];
    }
    double y() const
    {
        return p[1];
    }
    double &z()
    {
        return p[2];
    }
    double z() const
    {
        return p[2];
    }

    double &nx()
    {
        return n[0];
    }
    double nx() const
    {
        return n[0];
    }
    double &ny()
    {
        return n[1];
    }
    double ny() const
    {
        return n[1];
    }
    double &nz()
    {
        return n[2];
    }
    double nz() const
    {
        return n[2];
    }

    double &operator[](int i)
    {
        return p[i];
    }
    double operator[](int i) const
    {
        return p[i];
    }

    double &operator()(int i)
    {
        return n[i];
    }
    double operator()(int i) const
    {
        return n[i];
    }

protected:
    double p[3];
    double n[3];
};

typedef std::vector<RoadPoint> RoadPointVector;

typedef std::vector<double> DistanceVector;

class Curve
{
public:
    Curve()
    {
    }
    virtual ~Curve()
    {
    }

    Curve(double s)
    {
        start = s;
    }

    bool operator<(Curve *curve)
    {
        return start < (curve->getStart());
    }

    double getStart()
    {
        return start;
    }

protected:
    double start;
};

class Polynom : public Curve
{
public:
    Polynom(double, double = 0, double = 0, double = 0, double = 0);

    Vector2D getPoint(double);
    double getValue(double);
    double getSlope(double);

private:
    double a, b, c, d;
};

/// Interface PlaneCurve: Mapping distance s to two values
//typedef std::pair<double,double> Vector3D;
class PlaneCurve : public Curve
{
public:
    virtual ~PlaneCurve(){};

    double getLength();

    virtual Vector3D getPoint(double) = 0;
    virtual void movePoint(Vector3D &, double, double) = 0;
    virtual double getOrientation(double) = 0;

protected:
    double length;
};

// Derivatives of PlaneCurve
class PlaneStraightLine : public PlaneCurve
{
public:
    PlaneStraightLine(double, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    void movePoint(Vector3D &, double, double);
    double getOrientation(double);

private:
    double ax, bx;
    double ay, by;
    double hdgs;
};

class PlaneArc : public PlaneCurve
{
public:
    PlaneArc(double, double = 0, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    void movePoint(Vector3D &, double, double);
    double getOrientation(double);

private:
    double r;
    double sinhdg, coshdg;
    double xs, ys;
    double hdgs;
};

class PlaneClothoid : public PlaneCurve
{
public:
    PlaneClothoid(double, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    void movePoint(Vector3D &, double, double);
    double getOrientation(double);

private:
    void integrateClothoid(double, double &, double &);
    void approximateClothoid(double, double &, double &);

    double ax, ay;
    double xs, ys;
    double sinbeta, cosbeta;
    double ts;
    double hdgs;

    double tsn[9];
    double fca8, fca7, fca6, fca5, fca4, fca3, fca2, fca1, fca0;

    double sqrtpi;
};

class PlanePolynom : public PlaneCurve
{
public:
    PlanePolynom(double, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    void movePoint(Vector3D &, double, double);
    double getOrientation(double);

private:
    double getT(double);

    double a, b, c, d;
    double sinhdg, coshdg;
    double xs, ys;
    double hdgs;
};

class LateralProfile : public Curve
{
public:
    virtual double getAngle(double, double) = 0;

protected:
};

class SuperelevationPolynom : public LateralProfile
{
public:
    SuperelevationPolynom(double, double = 0, double = 0, double = 0, double = 0);

    double getAngle(double, double);

private:
    double a, b, c, d;
};

class CrossfallPolynom : public LateralProfile
{
public:
    CrossfallPolynom(double, double = 0, double = 0, double = 0, double = 0, double = -1.0, double = 1.0);

    double getAngle(double, double);

private:
    double a, b, c, d;
    double leftFactor, rightFactor;
};

inline Vector2D::Vector2D(const double &x, const double &y)
{
    p[0] = x;
    p[1] = y;
}

inline void Vector2D::set(const double &x, const double &y)
{
    p[0] = x;
    p[1] = y;
}

inline double &Vector2D::x()
{
    return p[0];
}
inline double Vector2D::x() const
{
    return p[0];
}
inline double &Vector2D::y()
{
    return p[1];
}
inline double Vector2D::y() const
{
    return p[1];
}

inline double &Vector2D::operator[](const int &i)
{
    return p[i];
}
inline double Vector2D::operator[](const int &i) const
{
    return p[i];
}

inline Vector2D Vector2D::operator+(const Vector2D &vec) const
{
    return Vector2D(p[0] + vec[0], p[1] + vec[1]);
}

inline Vector2D Vector2D::operator-(const Vector2D &vec) const
{
    return Vector2D(p[0] - vec[0], p[1] - vec[1]);
}

inline const Vector2D &Vector2D::operator+=(const Vector2D &vec)
{
    p[0] += vec[0];
    p[1] += vec[1];
    return *this;
}

inline const Vector2D &Vector2D::operator-=(const Vector2D &vec)
{
    p[0] -= vec[0];
    p[1] -= vec[1];
    return *this;
}

inline const Vector2D &Vector2D::operator=(const Vector2D &vec)
{
    p[0] = vec[0];
    p[1] = vec[1];
    return *this;
}

inline bool Vector2D::operator==(const Vector2D &vec) const
{
    return ((p[0] == vec[0]) && (p[1] == vec[1]));
}

inline Vector3D::Vector3D(const double &x, const double &y, const double &z)
{
    p[0] = x;
    p[1] = y;
    p[2] = z;
}

inline void Vector3D::set(const double &x, const double &y, const double &z)
{
    p[0] = x;
    p[1] = y;
    p[2] = z;
}

inline double &Vector3D::x()
{
    return p[0];
}
inline double Vector3D::x() const
{
    return p[0];
}
inline double &Vector3D::y()
{
    return p[1];
}
inline double Vector3D::y() const
{
    return p[1];
}
inline double &Vector3D::z()
{
    return p[2];
}
inline double Vector3D::z() const
{
    return p[2];
}

inline double &Vector3D::operator[](const int &i)
{
    return p[i];
}
inline double Vector3D::operator[](const int &i) const
{
    return p[i];
}

inline Vector3D Vector3D::operator+(const Vector3D &vec) const
{
    return Vector3D(p[0] + vec[0], p[1] + vec[1], p[2] + vec[2]);
}

inline Vector3D Vector3D::operator-(const Vector3D &vec) const
{
    return Vector3D(p[0] - vec[0], p[1] - vec[1], p[2] - vec[2]);
}

inline const Vector3D &Vector3D::operator+=(const Vector3D &vec)
{
    p[0] += vec[0];
    p[1] += vec[1];
    p[2] += vec[2];
    return *this;
}

inline const Vector3D &Vector3D::operator-=(const Vector3D &vec)
{
    p[0] -= vec[0];
    p[1] -= vec[1];
    p[2] -= vec[2];
    return *this;
}

inline const Vector3D &Vector3D::operator=(const Vector3D &vec)
{
    p[0] = vec[0];
    p[1] = vec[1];
    p[2] = vec[2];
    return *this;
}

inline bool Vector3D::operator==(const Vector3D &vec) const
{
    return ((p[0] == vec[0]) && (p[1] == vec[1]) && (p[2] == vec[2]));
}

inline Quaternion Vector3D::operator*(const Quaternion &quat) const
{
    return Quaternion(-p[0] * quat.x() - p[1] * quat.y() - p[2] * quat.z(),
                      p[0] * quat.w() + p[1] * quat.z() - p[2] * quat.y(),
                      p[1] * quat.w() + p[2] * quat.x() - p[0] * quat.z(),
                      p[2] * quat.w() + p[0] * quat.y() - p[1] * quat.x());
}

inline Quaternion::Quaternion(const double &wset, const double &xset, const double &yset, const double &zset)
    : _w(wset)
    , _x(xset)
    , _y(yset)
    , _z(zset)
{
}

inline Quaternion::Quaternion(const double &angle, const Vector3D &vec)
{
    _w = cos(angle * 0.5);

    double sinhalfangle = sin(angle * 0.5);
    _x = vec[0] * sinhalfangle;
    _y = vec[1] * sinhalfangle;
    _z = vec[2] * sinhalfangle;
}

inline double &Quaternion::w()
{
    return _w;
}
inline double Quaternion::w() const
{
    return _w;
}
inline double &Quaternion::x()
{
    return _x;
}
inline double Quaternion::x() const
{
    return _x;
}
inline double &Quaternion::y()
{
    return _y;
}
inline double Quaternion::y() const
{
    return _y;
}
inline double &Quaternion::z()
{
    return _z;
}
inline double Quaternion::z() const
{
    return _z;
}

inline Quaternion Quaternion::T() const
{
    return Quaternion(_w, -_x, -_y, -_z);
}

inline Vector3D Quaternion::getVector() const
{
    return Vector3D(_x, _y, _z);
}

inline Quaternion Quaternion::operator*(const Quaternion &quat) const
{
    return Quaternion(_w * quat.w() - _x * quat.x() - _y * quat.y() - _z * quat.z(),
                      _w * quat.x() + _x * quat.w() + _y * quat.z() - _z * quat.y(),
                      _w * quat.y() + _y * quat.w() + _z * quat.x() - _x * quat.z(),
                      _w * quat.z() + _z * quat.w() + _x * quat.y() - _y * quat.x());
}
inline Quaternion Quaternion::operator*(const Vector3D &vec) const
{
    return Quaternion(-_x * vec.x() - _y * vec.y() - _z * vec.z(),
                      _w * vec.x() + _y * vec.z() - _z * vec.y(),
                      _w * vec.y() + _z * vec.x() - _x * vec.z(),
                      _w * vec.z() + _x * vec.y() - _y * vec.x());
}

inline const Quaternion &Quaternion::operator=(const Quaternion &quat)
{
    _w = quat.w();
    _x = quat.x();
    _y = quat.y();
    _z = quat.z();
    return *this;
}

inline Transform::Transform(const Vector3D &vec, const Quaternion &quat)
    : _v(vec)
    , _q(quat)
{
}

inline Vector3D &Transform::v()
{
    return _v;
}
inline Vector3D Transform::v() const
{
    return _v;
}
inline Quaternion &Transform::q()
{
    return _q;
}
inline Quaternion Transform::q() const
{
    return _q;
}

inline Transform::Transform()
    : _v(Vector3D(0, 0, 0))
    , _q(Quaternion(1, 0, 0, 0))
{
}

inline Transform Transform::operator*(const Transform &trans) const
{
    return Transform(_v + trans.v(), _q * trans.q());
}

inline const Transform &Transform::operator=(const Transform &trans)
{
    _v = trans.v();
    _q = trans.q();
    return *this;
}

#endif
