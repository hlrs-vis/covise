/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.02.2010
**
**************************************************************************/

#include "polynomial.hpp"

// Utils //
//
#include "math.h"
#include "src/util/odd.hpp"

/** Creates a cubic curve with the formula
* a + b*x + c*x^2 + d*x^3
*/
Polynomial::Polynomial(double a, double b, double c, double d)
    : a_(a)
    , b_(b)
    , c_(c)
    , d_(d)
    , degree_(0)
{
    recalculateDegree();
}

Polynomial::Polynomial(double f0, double df0, double f1, double df1, double length)
    : a_(f0)
    , b_(df0)
    , degree_(0)
{
    d_ = (df1 + df0 - 2.0 * f1 / length + 2.0 * f0 / length) / (length * length);
    c_ = (f1 - d_ * length * length * length - df0 * length - f0) / (length * length);

    recalculateDegree();
}

Polynomial::Polynomial(const Polynomial &polynomial)
    : a_(polynomial.a_)
    , b_(polynomial.b_)
    , c_(polynomial.c_)
    , d_(polynomial.d_)
    , degree_(0)
{
    recalculateDegree();
}

void
Polynomial::protectedSetParameters(double a, double b, double c, double d)
{
    a_ = a;
    b_ = b;
    c_ = c;
    d_ = d;

    recalculateDegree();
}

/** Evaluates the polynomial for a given argument x.
*/
double
Polynomial::f(double x) const
{
    return (a_ + b_ * x + c_ * x * x + d_ * x * x * x);
}

/** Calculates the first derivative of the polynomial
* for a given argument x.
*/
double
Polynomial::df(double x) const
{
    return (b_ + 2 * c_ * x + 3 * d_ * x * x);
}

/** Calculates the second derivative of the polynomial
* for a given argument x.
*/
double
Polynomial::ddf(double x) const
{
    return (2 * c_ + 6 * d_ * x);
}

/** Calculates the curvature of the polynomial
* for a given argument x.
*/
double
Polynomial::k(double x) const
{
    return ddf(x) / (pow(1 + df(x) * df(x), 1.5));
}

/** Calculates the heading [degrees] of the polynomial
* for a given argument x.
*/
double
Polynomial::hdg(double x) const
{
    return atan2(df(x), 1.0) * 360.0 / (2 * M_PI);
}

/** Calculates the heading [rad] of the polynomial
* for a given argument x.
*/
double
Polynomial::hdgRad(double x) const
{
    return atan2(df(x), 1.0);
}

/** Calculates the length [m] of a curve segment.
*
* Uses a Gauss-Legendre integration (n=5). Should be good enough.
*/
double
Polynomial::getCurveLength(double from, double to)
{
    double factor = (to - from) / 2.0; // new length = 2.0
    double deltaX = (to + from) / 2.0; // center
    double l = (0.236926885056189 * g(-0.906179845938664, factor, deltaX)
                + 0.478628670499366 * g(-0.538469310105683, factor, deltaX)
                + 0.568888888888889 * g(0.0, factor, deltaX)
                + 0.478628670499366 * g(0.538469310105683, factor, deltaX)
                + 0.236926885056189 * g(0.906179845938664, factor, deltaX)) * factor;
    return l;
}

/** Calculates the degree of the polynomial.
*
* \note If the polynomial is 0.0, degree will be -1.
*
*/
void
Polynomial::recalculateDegree()
{
    if (fabs(d_) > NUMERICAL_ZERO8)
    {
        degree_ = 3;
    }
    else if (fabs(c_) > NUMERICAL_ZERO8)
    {
        degree_ = 2;
    }
    else if (fabs(b_) > NUMERICAL_ZERO8)
    {
        degree_ = 1;
    }
    else if (fabs(a_) > NUMERICAL_ZERO8)
    {
        degree_ = 0;
    }
    else
    {
        degree_ = -1; // polynomial is 0.0
    }
}

double
Polynomial::g(double x, double factor, double delta)
{
    x = x * factor + delta;
    return sqrt(1.0 + (b_ + 2.0 * c_ * x + 3.0 * d_ * x * x) * (b_ + 2.0 * c_ * x + 3.0 * d_ * x * x));
}
