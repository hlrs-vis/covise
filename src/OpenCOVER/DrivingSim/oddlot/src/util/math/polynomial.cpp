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

#include <QPointF>
#include <QDebug>

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
Polynomial::operator+=(const Polynomial &g)
{
	a_ += g.a_;
	b_ += g.b_;
	c_ += g.c_;
	d_ += g.d_;
}

void
Polynomial::operator-=(const Polynomial &g)
{
	a_ -= g.a_;
	b_ -= g.b_;
	c_ -= g.c_;
	d_ -= g.d_;
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

double
Polynomial::getT(double y, double xApprox) const
{

	// Newton's Method //
	//
	double x;
	double crit = 0.000001; // stop if improvement is less than crit
	int iIteration = 0;
	do
	{
		x = xApprox;
		++iIteration;

		// Approximation //
		//
		double p = ax_ - y + bx_ * x + cx_ * x * x + dx_ * x * x * x;
		double q = bx_ + 2 * cx_ * x + 3 * dx_ * x * x;
		xApprox = x - (p / q);

		if (iIteration >= 50)
		{
			qDebug() << "y: " << y << " x not found";
		}
	} while ((iIteration < 50) && (fabs(x - xApprox) > crit));

	return xApprox;
}

/** Evaluates the polynomial for a given argument x.
*/
double
Polynomial::f(double x) const
{
	//double t = getT(x, 0.5);
	//return (a_ + b_ * t + c_ * t * t + d_ * t * t * t);
	// polynomials in OpenDrive are defined as function of x , not a different parameter t, x can be a local coordinate
	// this coordinate is sometimes caled u or t in the standard document but it is always the local "x" coordinate of the track or the local s coordinate (for the width or height polynom) 
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

double 
Polynomial::getX(double y, double n, double xApprox)
{
	double at = a_;
	double bt = b_ * n;
	double ct = c_ * n * n;
	double dt = d_ * n * n * n;

	// Newton's Method //
	//
	double x;
	double crit = 0.000001; // stop if improvement is less than crit
	int iIteration = 0;
	do
	{
		x = xApprox;
		++iIteration;

		// Approximation //
		//
		double p = at - y + bt * x + ct * x * x + dt * x * x * x;
		double q = bt + 2 * ct * x + 3 * dt * x * x;
		xApprox = x - (p / q);

		if (iIteration >= 50)
		{
			qDebug() << "y: " << y << " x not found";
		}
	} while ((iIteration < 50) && (fabs(x - xApprox) > crit));

	return xApprox * n;
}

bool
Polynomial::getPolynomialControlPoints(double l, QPointF &p0, QPointF &p1, QPointF &p2, QPointF &p3)
{

	// Control Points are initialized at 1/3 of the segment length
	//
	if (abs(l) > NUMERICAL_ZERO3)
	{
		p0 = QPointF(0, a_);
	//	p1 = QPointF(l / 3, (3 * b_) / l + a_);
//		p1 = QPointF(l / 3, b_ * l / 3 + a_);
		double cl = 2 * l / 3;
		p3 = QPointF(l, f(l));
//		p2 = QPointF(cl, l* (p3.y() - df(l)) / 3);

		p1.setY((b_ * l) / 3 + p0.y());
		p1.setY(getX(p1.y(), l, 1.0 / 3.0));
//		p1.setX((p1.y() - p0.y()) / b_);

		p2.setY((c_ * l * l) / 3 - p0.y() + 2 * p1.y());
		p2.setX((p2.y() - p3.y()) / df(p3.x()) + p3.x());


		return true;
	}

	return false;
}

void 
Polynomial::setParameters(double a, double b, double c, double d)
{
	a_ = a;
	b_ = b;
	c_ = c;
	d_ = d;

	recalculateDegree();
}

void 
Polynomial::setParametersFromControlPoints(QPointF p0, QPointF p1, QPointF p2, QPointF p3)
{
	double n = 1 / p3.x();

	 ax_ = p0.x();
	 bx_= 3 * (p1.x() - p0.x());
	 cx_ = 3 * (p0.x() - 2 * p1.x() + p2.x());
	 dx_ = -p0.x() + 3 * (p1.x() - p2.x()) + p3.x();


	 a_ = p0.y();
	 b_ = 3 * (p1.y() - p0.y());
	 c_ = 3 * (p0.y() - 2 * p1.y() + p2.y());
	 d_ = -p0.y() + 3 * (p1.y() - p2.y()) + p3.y();

/*	a_ = at;
	b_ = bt * n;
	c_ = ct * n * n;
	d_ = dt * n * n * n; 

	qDebug() << "P1: " << p1.x() << " aus Steigung: " << (p1.y() - p0.y()) / bt; */

	recalculateDegree();

/*	a_ = p0.y();
	double l = p1.x() - p0.x();
	if (abs(l) > NUMERICAL_ZERO3)
	{
		b_ = (p1.y() - a_) / l;
	}
	else
	{
		b_ = 0.0;
	}

	l = p3.x() - p2.x();
	if (abs(l) > NUMERICAL_ZERO3)
	{
		double h = (p3.y() - p2.y()) / l;
		d_ = (2 * (a_ - p3.y()) + p3.x() *(b_ + h)) / (p3.x() * p3.x() * p3.x());
		c_ = (h - b_ - 3 * d_ * p3.x() * p3.x()) / (2 * p3.x());
	}
	else
	{
		c_ = d_ = 0;
	} */
/*	qDebug() << "Points: " << p0.x() << "," << p0.y() << " " << p1.x() << "," << p1.y() << " " << p2.x() << "," << p2.y() << " " << p3.x() << "," << p3.y();
	qDebug() << "Parameters: " << a_ << "," << b_ << "," << c_ << "," << d_;
	qDebug() << "f(0)=" << f(0);
	qDebug() << "f(p3.x)=" << f(p3.x()); */
}

void 
Polynomial::setParameters(QPointF endPoint)
{
	double l = endPoint.x();
	double h0 = a_;
	double dh0 = b_;
	double h1 = endPoint.y();
	double dh1 = endPoint.y() / endPoint.x();

	double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
	double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);
	d_ = d;
	c_ = c;
}

double
Polynomial::getTLength(double s)
{

	double t = s * 0.5; // first approximation

	for (int i = 0; i < 30; ++i)  // 20 is not enough!
	{

		// Flo's code //
		//
		// Cut taylor series approximation (1-degree) of arc length integral, solved with Newton-Raphson for t with respect to s
		//		double b = getB();
		//		double c = getC();
		//		double d = getD();
		//		double f = t*sqrt(pow(((3*d*pow(t,2))/4+c*t+b),2)+1)-s;
		//		double df = sqrt(pow(((3*d*pow(t,2))/4+c*t+b),2)+1)+(t*((3*d*t)/2+c)*((3*d*pow(t,2))/4+c*t+b))/sqrt(pow(((3*d*pow(t,2))/4+c*t+b),2)+1);

		// New code with integration //
		//
		double f = getCurveLength(0.0, t) - s;
		double df = sqrt(1.0 + (b_ + 2.0 * c_ * t + 3.0 * d_ * t * t) * (b_ + 2.0 * c_ * t + 3.0 * d_ * t * t));
		t -= f / df;
	}
	return t;
}

