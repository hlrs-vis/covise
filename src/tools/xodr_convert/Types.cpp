/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Types.h"

#include <cmath>
#include <iostream>

Polynom::Polynom(double s, double a, double b, double c, double d)
{
    start = s;

    this->a = a, this->b = b, this->c = c, this->d = d;
}

Vector2D Polynom::getPoint(double s)
{
    s -= start;
    return Vector2D(a + b * s + c * s * s + d * s * s * s, -atan(b + 2 * c * s + 3 * d * s * s));
}

double Polynom::getValue(double s)
{
    s -= start;
    return a + b * s + c * s * s + d * s * s * s;
}
double Polynom::getSlope(double s)
{
    s -= start;
    return atan(b + 2 * c * s + 3 * d * s * s);
}

double PlaneCurve::getLength()
{
    return length;
}

PlaneStraightLine::PlaneStraightLine(double s, double l, double xs, double ys, double hdg)
{
    start = s;
    length = l;

    ax = cos(hdg);
    ay = sin(hdg);
    bx = xs;
    by = ys;

    hdgs = hdg;
}

Vector3D PlaneStraightLine::getPoint(double s)
{
    s -= start;
    return Vector3D(ax * s + bx, ay * s + by, hdgs);
}

void PlaneStraightLine::movePoint(Vector3D &point, double ds, double)
{
    point += Vector3D(ax * ds, ay * ds, 0);
}

double PlaneStraightLine::getOrientation(double)
{
    return hdgs;
}

PlaneArc::PlaneArc(double s, double l, double xstart, double ystart, double hdg, double r)
{
    start = s;
    length = l;

    this->r = r;
    sinhdg = sin(hdg);
    coshdg = cos(hdg);
    xs = xstart - r * sinhdg;
    ys = ystart + r * coshdg;
    hdgs = hdg;
}

Vector3D PlaneArc::getPoint(double s)
{
    s -= start;
    double t = s / r;
    double sint = sin(t);
    double cost = cos(t);
    return Vector3D(xs + r * (coshdg * sint + sinhdg * cost), ys + r * (sinhdg * sint - coshdg * cost), s / r + hdgs);
}

void PlaneArc::movePoint(Vector3D &point, double ds, double s)
{
    s = s - start + ds;
    double t = s / r;
    double sint = sin(t);
    double cost = cos(t);
    point.set(xs + r * (coshdg * sint + sinhdg * cost), ys + r * (sinhdg * sint - coshdg * cost), s / r + hdgs);
}

double PlaneArc::getOrientation(double s)
{
    s -= start;
    return s / r + hdgs;
}

PlaneClothoid::PlaneClothoid(double s, double l, double xstart, double ystart, double hdg, double cs, double ce)
{
    sqrtpi = sqrt(M_PI);

    start = s;
    length = l;

    double cdelta = ce - cs;
    if (cdelta > 0)
    {
        ax = ay = sqrt(l / cdelta);
        ts = cs * ax / sqrtpi;
    }
    else if (cdelta < 0)
    {
        ax = sqrt(-l / cdelta);
        ay = -ax;
        ts = cs * ay / sqrtpi;
    }
    else
    {
        ax = ay = 0;
        ts = 0;
    }

    //double xsc, ysc;
    //integrateClothoid(0, ts, xsc, ysc);
    //xs = xstart - xsc;
    //ys = ystart - ysc;
    xs = xstart;
    ys = ystart;

    double chdg = ay / ax * M_PI * ts * ts * 0.5;
    double beta = hdg - chdg;
    hdgs = beta;
    sinbeta = sin(beta);
    cosbeta = cos(beta);

    double ts2 = pow(ts, 2);
    tsn[0] = ts;
    for (int i = 1; i < 9; ++i)
    {
        tsn[i] = tsn[i - 1] * ts2;
    }
    fca8 = 39 * pow(M_PI, 8) / ((double)6843432960);
    fca7 = 11 * pow(M_PI, 7) / ((double)106444800);
    fca6 = 11424 * pow(M_PI, 6) / ((double)6843432960);
    fca5 = 2520 * pow(M_PI, 5) / ((double)106444800);
    fca4 = 1980160 * pow(M_PI, 4) / ((double)6843432960);
    fca3 = 316800 * pow(M_PI, 3) / ((double)106444800);
    fca2 = 171085824 * pow(M_PI, 2) / ((double)6843432960);
    fca1 = 17740800 * M_PI / ((double)106444800);
    fca0 = 6843432960 / ((double)6843432960);
}

Vector3D PlaneClothoid::getPoint(double s)
{
    s -= start;
    double t = s / (ax * sqrtpi) + ts;
    double x, y;
    //integrateClothoid(t, x, y);
    approximateClothoid(t, x, y);

    return Vector3D(xs + x * cosbeta - y * sinbeta, ys + x * sinbeta + y * cosbeta, hdgs + ay / ax * 0.5 * M_PI * t * t);
}

void PlaneClothoid::movePoint(Vector3D &point, double ds, double s)
{
    s -= start;
    double dt = ds / (ax * sqrtpi);
    double thdt = s / (ax * sqrtpi) + ts + 0.5 * dt;

    point += Vector3D(ax * cos(0.5 * M_PI * pow(thdt, 2)), ay * sin(0.5 * M_PI * pow(thdt, 2)), ay / ax * M_PI * dt);
}

double PlaneClothoid::getOrientation(double s)
{
    s -= start;
    double t = s / (ax * sqrtpi) + ts;

    //std::cerr << "Clothoid hdgs: " << hdgs << ", hdg: " << -ay/ax*0.5*M_PI*t*t << ", t: " << t << std::endl;
    return hdgs + ay / ax * 0.5 * M_PI * t * t;
}

void PlaneClothoid::integrateClothoid(double t, double &x, double &y)
{
    x = 0;
    y = 0;

    double h = 0.0001;

    //std::cout << "t-ts: " << t-ts << std::endl;
    double n = ceil((t - ts) / h);

    h = (t - ts) / n;

    if (h == h)
    {
        for (int i = 1; i < (n - 1); ++i)
        {
            x += cos(M_PI * 0.5 * pow(ts + i * h, 2));
            y += sin(M_PI * 0.5 * pow(ts + i * h, 2));
        }
        x += (cos(M_PI * ts) + cos(M_PI * t)) * 0.5;
        y += (sin(M_PI * ts) + sin(M_PI * t)) * 0.5;
        x *= ax * sqrtpi * h;
        y *= ay * sqrtpi * h;
    }
}

void PlaneClothoid::approximateClothoid(double t, double &x, double &y)
{
    double t2 = pow(t, 2);
    double tn[9];
    tn[0] = t;
    for (int i = 1; i < 9; ++i)
    {
        tn[i] = tn[i - 1] * t2;
    }

    x = ax * sqrtpi * (fca8 * (tn[8] - tsn[8]) - fca6 * (tn[6] - tsn[6]) + fca4 * (tn[4] - tsn[4]) - fca2 * (tn[2] - tsn[2]) + fca0 * (tn[0] - tsn[0]));
    y = ay * sqrtpi * (fca7 * (tn[7] - tsn[7]) + fca5 * (tn[5] - tsn[5]) - fca3 * (tn[3] - tsn[3]) + fca1 * (tn[1] - tsn[1]));
}

PlanePolynom::PlanePolynom(double s, double l, double xstart, double ystart, double hdg, double a, double b, double c, double d)
{
    start = s;
    length = l;

    hdgs = hdg;
    sinhdg = sin(hdg);
    coshdg = cos(hdg);
    xs = xstart;
    ys = ystart;

    this->a = a, this->b = b, this->c = c, this->d = d;
}

Vector3D PlanePolynom::getPoint(double s)
{
    s -= start;
    double t = getT(s);

    double u = t;
    double v = (a + b * t + c * t * t + d * t * t * t);
    return Vector3D(u * coshdg - v * sinhdg + xs, u * sinhdg + v * coshdg + ys, atan(3 * d * t * t + 2 * c * t + b) + hdgs);
}

void PlanePolynom::movePoint(Vector3D &point, double ds, double)
{
    double tdt = point.x();
    double dt;
    for (int i = 0; i < 5; ++i)
    {
        dt = 1 / sqrt(pow((3 * d * tdt * tdt + 2 * c * tdt + b), 2) + 1) * ds;
        tdt = point.x() + dt;
    }

    double u = tdt;
    double v = (a + b * tdt + c * tdt * tdt + d * tdt * tdt * tdt);
    point.set(u * coshdg - v * sinhdg + xs, u * sinhdg + v * coshdg + ys, atan(3 * d * tdt * tdt + 2 * c * tdt + b) + hdgs);
}

double PlanePolynom::getOrientation(double s)
{
    s -= start;
    double t = getT(s);
    return atan(3 * d * t * t + 2 * c * t + b) + hdgs;
}

double PlanePolynom::getT(double s)
{
    int n = 100;
    double h = s / n;
    double t = 0;
    double dt = 0;
    for (int i = 0; i < n; ++i)
    {
        dt = 1 / sqrt(pow((3 * d * t * t + 2 * c * t + b), 2) + 1) * h;
        t += dt;
    }
    return t;
}

SuperelevationPolynom::SuperelevationPolynom(double s, double a, double b, double c, double d)
{
    start = s;

    this->a = a, this->b = b, this->c = c, this->d = d;
}

double SuperelevationPolynom::getAngle(double s, double)
{
    s -= start;
    return a + b * s + c * s * s + d * s * s * s;
}

CrossfallPolynom::CrossfallPolynom(double s, double a, double b, double c, double d, double lf, double rf)
{
    start = s;

    this->a = a, this->b = b, this->c = c, this->d = d;
    leftFactor = lf, rightFactor = rf;
}

double CrossfallPolynom::getAngle(double s, double t)
{
    s -= start;
    double absval = a + b * s + c * s * s + d * s * s * s;
    return (t < 0) ? rightFactor * absval : leftFactor * absval;
}

std::ostream &operator<<(std::ostream &os, const Quaternion &quat)
{
    os << "Hallllooooooooooooooooo ";
    os << "(" << quat.w() << ", [" << quat.x() << ", " << quat.y() << ", " << quat.z() << "])";

    return os;
}
