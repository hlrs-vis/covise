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

#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

class Polynomial
{
public:
    Polynomial(double a = 0.0, double b = 0.0, double c = 0.0, double d = 0.0);
    Polynomial(double f0, double df0, double f1, double df1, double length);
    Polynomial(const Polynomial &polynomial);

    virtual ~Polynomial()
    { /* does nothing */
    }

    double getA() const
    {
        return a_;
    }
    double getB() const
    {
        return b_;
    }
    double getC() const
    {
        return c_;
    }
    double getD() const
    {
        return d_;
    }

    // TODO: setFunctions with recalculateDegree

    int getDegree() const
    {
        return degree_;
    }

    double f(double x) const; // value
    double df(double x) const; // slope
    double ddf(double x) const; // curvature (2nd derivative)

    double k(double x) const; // curvature (the real one)

    double hdg(double x) const; // slope angle (heading) in [degrees]
    double hdgRad(double x) const; // slope angle (heading) in [rad]

    double getCurveLength(double fromX, double toX);

protected:
    // four doubles = 0 is the same Polynomial(); /* not allowed */
    Polynomial &operator=(const Polynomial &); /* not allowed */

    void protectedSetParameters(double a, double b, double c, double d);

private:
    void recalculateDegree();

    double g(double x, double factor, double delta);

protected:
    double a_;
    double b_;
    double c_;
    double d_;

    int degree_;
};

#endif // POLYNOMIAL_HPP
