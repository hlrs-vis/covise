/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Types_h
#define Types_h

#include <map>
#include <vector>
#include <osg/Matrix>
#include <limits>

#include "RoadSystemVisitor.h"

class VEHICLEUTILEXPORT Vector2D
{
public:
    Vector2D(const double &x, const double &y);

    void set(const double &x, const double &y);

    double &x();
    double x() const;
    double &y();
    double y() const;
    double &u();
    double u() const;
    double &v();
    double v() const;

    double &operator[](const int &i);
    double operator[](const int &i) const;

    Vector2D operator+(const Vector2D &vec) const;

    Vector2D operator-(const Vector2D &vec) const;

    const Vector2D &operator+=(const Vector2D &vec);

    const Vector2D &operator-=(const Vector2D &vec);

    const Vector2D &operator=(const Vector2D &vec);

    bool operator==(const Vector2D &vec) const;

    bool isNaV() const; //not a vector

    double length() const;

    static Vector2D NaV()
    {
        return Vector2D(std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN());
    }

private:
    double p[2];
};

class Quaternion;

class VEHICLEUTILEXPORT Vector3D
{
public:
    Vector3D(const double &val);

    Vector3D(const double &x, const double &y, const double &z);

    Vector3D(const Vector2D &vec, const double &z);

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

    Vector3D operator*(const double &scal) const;

    const Vector3D &operator+=(const Vector3D &vec);

    const Vector3D &operator-=(const Vector3D &vec);

    const Vector3D &operator=(const Vector3D &vec);

    bool operator==(const Vector3D &vec) const;

    Quaternion operator*(const Quaternion &quat) const;

    double dot(const Vector3D &vec) const;
    Vector3D cross(const Vector3D &vec) const;

    double length() const;

    void normalize();

    Vector3D normalized() const;

    bool isNaV() const; //not a vector

    static Vector3D NaV()
    {
        return Vector3D(std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN());
    }

private:
    double p[3];
};

std::ostream &operator<<(std::ostream &, const Vector3D &);

class VEHICLEUTILEXPORT Matrix2D2D
{
public:
    Matrix2D2D(const double &a);
    Matrix2D2D(const double &a11, const double &a12,
               const double &a21, const double &a22);

    double &operator()(const int &i, const int &j);
    double operator()(const int &i, const int &j) const;

    Vector2D operator*(const Vector2D &vec) const;
    Matrix2D2D operator*(const Matrix2D2D &mat) const;

private:
    double m[2][2];
};

std::ostream &operator<<(std::ostream &, const Matrix2D2D &);

class VEHICLEUTILEXPORT Matrix3D3D
{
public:
    Matrix3D3D(const double &a);
    Matrix3D3D(const double &a11, const double &a12, const double &a13,
               const double &a21, const double &a22, const double &a23,
               const double &a31, const double &a32, const double &a33);

    double &operator()(const int &i, const int &j);
    double operator()(const int &i, const int &j) const;

    Vector3D operator*(const Vector3D &vec) const;
    Matrix3D3D operator*(const Matrix3D3D &mat) const;

    Matrix3D3D multiplyElementByElement(const Matrix3D3D &mat) const;
    double getSumOverElements() const;

private:
    double m[3][3];
};

std::ostream &operator<<(std::ostream &, const Matrix3D3D &);

class VEHICLEUTILEXPORT Quaternion
{
public:
    Quaternion(const double &val);

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

    bool isNaQ() const; //not a quaternion

private:
    double _w, _x, _y, _z;
};

std::ostream &operator<<(std::ostream &, const Quaternion &);

class VEHICLEUTILEXPORT Transform
{
public:
    Transform();

    Transform(const Vector3D &vec, const Quaternion &quat);

    Transform(const Quaternion &quat);

    Vector3D &v();

    Vector3D v() const;

    Quaternion &q();

    Quaternion q() const;

    Transform operator*(const Transform &trans) const;

    const Transform &operator=(const Transform &trans);

    Vector3D operator*(const Vector3D &vec) const;

    bool isNaT() const;

    void print() const;

protected:
    Vector3D _v;
    Quaternion _q;
};

std::ostream &operator<<(std::ostream &, const Transform &);

class VEHICLEUTILEXPORT RoadPoint
{
public:
    RoadPoint(const double &p1 = 0, const double &p2 = 0, const double &p3 = 0, const double &n1 = 0, const double &n2 = 0, const double &n3 = 0)
        : p(p1, p2, p3)
        , n(n1, n2, n3)
    {
    }

    RoadPoint(const Vector3D &p_, const Vector3D &n_)
        : p(p_)
        , n(n_)
    {
    }
    Vector3D &pos()
    {
        return p;
    }
    const Vector3D &pos() const
    {
        return p;
    }
    Vector3D &normal()
    {
        return n;
    }
    const Vector3D &normal() const
    {
        return n;
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
    Vector3D p;
    Vector3D n;
};

typedef std::vector<RoadPoint> RoadPointVector;

typedef std::vector<double> DistanceVector;

class VEHICLEUTILEXPORT Curve
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

    double getStart()
    {
        return start;
    }

protected:
    double start;
};

class VEHICLEUTILEXPORT Polynom : public Curve
{
public:
    Polynom(double, double = 0, double = 0, double = 0, double = 0);

    Vector2D getPoint(double);
    double getValue(double);
    double getSlope(double);
    double getSlopeAngle(double);
    double getCurvature(double);
    void getCoefficients(double &, double &, double &, double &);

    void accept(XodrWriteRoadSystemVisitor *);

private:
    double a, b, c, d;
};

/// Interface PlaneCurve: Mapping distance s to two values
//typedef std::pair<double,double> Vector3D;
class VEHICLEUTILEXPORT PlaneCurve : public Curve
{
public:
    virtual ~PlaneCurve(){};

    double getLength();

    virtual Vector3D getPoint(double) = 0;
    virtual double getOrientation(double) = 0;
    virtual double getCurvature(double) = 0;
    virtual Vector2D getTangentVector(double) = 0;
    virtual Vector2D getNormalVector(double) = 0;

    virtual void accept(XodrWriteRoadSystemVisitor *) = 0;

protected:
    double length;
};

// Derivatives of PlaneCurve
class VEHICLEUTILEXPORT PlaneStraightLine : public PlaneCurve
{
public:
    PlaneStraightLine(double, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    double getOrientation(double);
    double getCurvature(double);
    Vector2D getTangentVector(double);
    Vector2D getNormalVector(double);

    void accept(XodrWriteRoadSystemVisitor *);

private:
    //double ax, bx;
    //double ay, by;
    Matrix2D2D A;
    Vector2D cs;
    double hdgs;
};

class VEHICLEUTILEXPORT PlaneArc : public PlaneCurve
{
public:
    PlaneArc(double, double = 0, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    double getOrientation(double);
    double getCurvature(double);
    Vector2D getTangentVector(double);
    Vector2D getNormalVector(double);

    void accept(XodrWriteRoadSystemVisitor *);

private:
    double r;
    //double sinhdg, coshdg;
    Matrix2D2D A;
    //double xs, ys;
    Vector2D cs;
    double hdgs;
};

class VEHICLEUTILEXPORT PlaneClothoid : public PlaneCurve
{
public:
    PlaneClothoid(double, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    double getOrientation(double);
    double getCurvature(double);
    Vector2D getTangentVector(double);
    Vector2D getNormalVector(double);

    void accept(XodrWriteRoadSystemVisitor *);

private:
    void integrateClothoid(double, double &, double &);
    void approximateClothoid(double, double &, double &);

    double ax, ay;
    //double xs, ys;
    Matrix2D2D A;
    Vector2D cs;
    //double sinbeta, cosbeta;
    double ts;
    double hdgs;

    double tsn[9];
    double fca8, fca7, fca6, fca5, fca4, fca3, fca2, fca1, fca0;

    double sqrtpi;
};

class VEHICLEUTILEXPORT PlanePolynom : public PlaneCurve
{
public:
    PlanePolynom(double, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0);

    Vector3D getPoint(double);
    double getOrientation(double);
    double getCurvature(double);
    Vector2D getTangentVector(double);
    Vector2D getNormalVector(double);

    double getCurveLength(double from, double to);

    void getCoefficients(double &, double &, double &, double &);

    void accept(XodrWriteRoadSystemVisitor *);

private:
    double getT(double);
    double g(double x, double factor, double delta);

    double a, b, c, d;
    //double sinhdg, coshdg;
    Matrix2D2D A;
    //double xs, ys;
    Vector2D cs;
    double hdgs;
};

class VEHICLEUTILEXPORT PlaneParamPolynom : public PlaneCurve
{
public:
	PlaneParamPolynom(double, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, double = 0, bool = true);

	Vector3D getPoint(double);
	double getOrientation(double);
	double getCurvature(double);
	Vector2D getTangentVector(double);
	Vector2D getNormalVector(double);

	double getCurveLength(double from, double to);

	void getCoefficients(double &, double &, double &, double &, double &, double &, double &, double &);

	void accept(XodrWriteRoadSystemVisitor *);
	bool isNormalized() { return normalized; };

private:
	double getT(double);
	double g(double x, double factor, double delta);

	double aU, bU, cU, dU;
	double aV, bV, cV, dV;
	bool normalized;
	//double sinhdg, coshdg;
	Matrix2D2D A;
	//double xs, ys;
	Vector2D cs;
	double hdgs;
};
class VEHICLEUTILEXPORT LateralProfile : public Curve
{
public:
    virtual double getAngle(double, double) = 0;

    virtual void accept(XodrWriteRoadSystemVisitor *) = 0;

protected:
};

class VEHICLEUTILEXPORT SuperelevationPolynom : public LateralProfile
{
public:
    SuperelevationPolynom(double, double = 0, double = 0, double = 0, double = 0);

    double getAngle(double, double);
    void getCoefficients(double &, double &, double &, double &);

    void accept(XodrWriteRoadSystemVisitor *);

private:
    double a, b, c, d;
};

class VEHICLEUTILEXPORT CrossfallPolynom : public LateralProfile
{
public:
    CrossfallPolynom(double, double = 0, double = 0, double = 0, double = 0, double = -1.0, double = 1.0);

    double getAngle(double, double);
    double getLeftFallFactor();
    double getRightFallFactor();
    void getCoefficients(double &, double &, double &, double &);

    void accept(XodrWriteRoadSystemVisitor *);

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

inline double &Vector2D::u()
{
    return p[0];
}
inline double Vector2D::u() const
{
    return p[0];
}
inline double &Vector2D::v()
{
    return p[1];
}
inline double Vector2D::v() const
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

inline double Vector2D::length() const
{
    return sqrt(p[0] * p[0] + p[1] * p[1]);
}

inline bool Vector2D::isNaV() const
{
    if (p[0] != p[0] || p[1] != p[1])
    {
        return true;
    }
    return false;
}

inline Vector3D::Vector3D(const double &val)
{
    p[0] = val;
    p[1] = val;
    p[2] = val;
}

inline Vector3D::Vector3D(const double &x, const double &y, const double &z)
{
    p[0] = x;
    p[1] = y;
    p[2] = z;
}

inline Vector3D::Vector3D(const Vector2D &vec, const double &z)
{
    p[0] = vec[0];
    p[1] = vec[1];
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

inline Vector3D Vector3D::operator*(const double &scal) const
{
    return Vector3D(p[0] * scal, p[1] * scal, p[2] * scal);
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

inline double Vector3D::dot(const Vector3D &vec) const
{
    return p[0] * vec[0] + p[1] * vec[1] + p[2] * vec[2];
}

inline Vector3D Vector3D::cross(const Vector3D &vec) const
{
    return Vector3D(p[1] * vec[2] - p[2] * vec[1], p[2] * vec[0] - p[0] * vec[2], p[0] * vec[1] - p[1] * vec[0]);
}

inline double Vector3D::length() const
{
    return sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
}

inline void Vector3D::normalize()
{
    double invLength = 1 / length();
    p[0] *= invLength;
    p[1] *= invLength;
    p[2] *= invLength;
}

inline Vector3D Vector3D::normalized() const
{
    double invLength = 1 / length();
    return Vector3D(p[0] * invLength, p[1] * invLength, p[2] * invLength);
}

inline bool Vector3D::isNaV() const
{
    if (p[0] != p[0] || p[1] != p[1] || p[2] != p[2])
    {
        return true;
    }
    return false;
}

inline Matrix2D2D::Matrix2D2D(const double &a)
{
    m[0][0] = a;
    m[0][1] = a;
    m[1][0] = a;
    m[1][1] = a;
}
inline Matrix2D2D::Matrix2D2D(const double &a11, const double &a12,
                              const double &a21, const double &a22)
{
    m[0][0] = a11;
    m[0][1] = a12;
    m[1][0] = a21;
    m[1][1] = a22;
}
inline double &Matrix2D2D::operator()(const int &i, const int &j)
{
    return m[i][j];
}
inline double Matrix2D2D::operator()(const int &i, const int &j) const
{
    return m[i][j];
}
inline Vector2D Matrix2D2D::operator*(const Vector2D &vec) const
{
    Vector2D back(0.0, 0.0);
    back[0] = m[0][0] * vec[0] + m[0][1] * vec[1];
    back[1] = m[1][0] * vec[0] + m[1][1] * vec[1];
    return back;
}
inline Matrix2D2D Matrix2D2D::operator*(const Matrix2D2D &mat) const
{
    Matrix2D2D mret(0.0);
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            mret(i, j) += m[i][0] * mat(0, j) + m[i][1] * mat(1, j);
        }
    }
    return mret;
}

inline Matrix3D3D::Matrix3D3D(const double &a)
{
    m[0][0] = a;
    m[0][1] = a;
    m[0][2] = a;
    m[1][0] = a;
    m[1][1] = a;
    m[1][2] = a;
    m[2][0] = a;
    m[2][1] = a;
    m[2][2] = a;
}
inline Matrix3D3D::Matrix3D3D(const double &a11, const double &a12, const double &a13,
                              const double &a21, const double &a22, const double &a23,
                              const double &a31, const double &a32, const double &a33)
{
    m[0][0] = a11;
    m[0][1] = a12;
    m[0][2] = a13;
    m[1][0] = a21;
    m[1][1] = a22;
    m[1][2] = a23;
    m[2][0] = a31;
    m[2][1] = a32;
    m[2][2] = a33;
}
inline double &Matrix3D3D::operator()(const int &i, const int &j)
{
    return m[i][j];
}
inline double Matrix3D3D::operator()(const int &i, const int &j) const
{
    return m[i][j];
}
inline Vector3D Matrix3D3D::operator*(const Vector3D &vec) const
{
    Vector3D back(0.0, 0.0, 0.0);
    back[0] = m[0][0] * vec[0] + m[0][1] * vec[1] + m[0][2] * vec[2];
    back[1] = m[1][0] * vec[0] + m[1][1] * vec[1] + m[1][2] * vec[2];
    back[2] = m[2][0] * vec[0] + m[2][1] * vec[1] + m[2][2] * vec[2];
    return back;
}
inline Matrix3D3D Matrix3D3D::operator*(const Matrix3D3D &mat) const
{
    Matrix3D3D mret(0.0);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            mret(i, j) += m[i][0] * mat(0, j) + m[i][1] * mat(1, j) + m[i][2] * mat(2, j);
        }
    }
    return mret;
}

inline Matrix3D3D Matrix3D3D::multiplyElementByElement(const Matrix3D3D &mat) const
{
    return Matrix3D3D(m[0][0] * mat(0, 0), m[0][1] * mat(0, 1), m[0][2] * mat(0, 2),
                      m[1][0] * mat(1, 0), m[1][1] * mat(1, 1), m[1][2] * mat(1, 2),
                      m[2][0] * mat(2, 0), m[2][1] * mat(2, 1), m[2][2] * mat(2, 2));
}

inline double Matrix3D3D::getSumOverElements() const
{
    return m[0][0] + m[0][1] + m[0][2]
           + m[1][0] + m[1][1] + m[1][2]
           + m[2][0] + m[2][1] + m[2][2];
}

inline Quaternion::Quaternion(const double &val)
    : _w(val)
    , _x(val)
    , _y(val)
    , _z(val)
{
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

inline bool Quaternion::isNaQ() const
{
    if (_w != _w || _x != _x || _y != _y || _z != _z)
    {
        return true;
    }
    return false;
}

inline Transform::Transform(const Vector3D &vec, const Quaternion &quat)
    : _v(vec)
    , _q(quat)
{
}

inline Transform::Transform(const Quaternion &quat)
    : _v(Vector3D(0.0, 0.0, 0.0))
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
    return Transform(_v + (_q * trans.v() * _q.T()).getVector(), _q * trans.q());
}

inline const Transform &Transform::operator=(const Transform &trans)
{
    _v = trans.v();
    _q = trans.q();
    return *this;
}

inline Vector3D Transform::operator*(const Vector3D &vec) const
{
    return _v + (_q * vec * _q.T()).getVector();
}

inline bool Transform::isNaT() const
{
    return (_v.isNaV() || _q.isNaQ());
}

#endif
