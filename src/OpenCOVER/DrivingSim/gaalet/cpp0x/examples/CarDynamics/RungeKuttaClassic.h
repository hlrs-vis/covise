/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Runge_Kutta_Classic_h
#define __Runge_Kutta_Classic_h

#include "TupleExpressions.h"

template <typename F, typename S>
class EulerIntegrator
{
public:
    EulerIntegrator(const F &f_, const double &t_ = 0.0)
        : f(f_)
        , t(t_)
    {
    }

    void operator()(const double &h, S &y)
    {
        y = eval<S>(wrap(y) + wrap(f(t, y)) * h);
        t += h;
    };

private:
    const F &f;
    double t;
};

template <typename F, typename S>
class RungeKuttaClassicIntegrator
{
public:
    RungeKuttaClassicIntegrator(const F &f_, const double &t_ = 0.0)
        : f(f_)
        , t(t_)
    {
    }

    void operator()(const double &h, S &y)
    {
        S k1 = f(t, y);
        S k2 = f(t + 0.5 * h, eval<S>(wrap(y) + 0.5 * h * wrap(k1)));
        S k3 = f(t + 0.5 * h, eval<S>(wrap(y) + 0.5 * h * wrap(k2)));
        S k4 = f(t + h, eval<S>(wrap(y) + h * wrap(k3)));
        y = eval<S>(wrap(y) + 1.0 / 6.0 * h * (wrap(k1) + 2.0 * wrap(k2) + 2.0 * wrap(k3) + wrap(k4)));
        t += h;
    };

private:
    const F &f;
    double t;
};

#endif
