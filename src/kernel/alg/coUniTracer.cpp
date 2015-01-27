/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cmath>
#include "coUniTracer.h"
#include <do/covise_gridmethods.h>
#include <algorithm>

#define SAFETY 0.9f
#define PGROW -0.2f
#define PSHRNK -0.25f
#define ERRCON 1.89e-4f

using std::vector;
using namespace covise;

coUniTracer::coUniTracer(float xmin, float xmax, float ymin, float ymax,
                         float zmin, float zmax, int nx, int ny, int nz,
                         const float *u, const float *v, const float *w)
    : xmin_(xmin)
    , ymin_(ymin)
    , zmin_(zmin)
    , xmax_(xmax)
    , ymax_(ymax)
    , zmax_(zmax)
    , nx_(nx)
    , ny_(ny)
    , nz_(nz)
    , u_(u)
    , v_(v)
    , w_(w)
{
    dx_ = (xmax - xmin) / (nx - 1);
    dy_ = (ymax - ymin) / (ny - 1);
    dz_ = (zmax - zmin) / (nz - 1);
    // model characteristic length
    model_length_ = xmax_ - xmin_;
    if (ymax_ - ymin_ > model_length_)
        model_length_ = ymax_ - ymin_;
    if (zmax_ - zmin_ > model_length_)
        model_length_ = zmax_ - zmin_;

    // model characteristic velocity
    float velminmax[6];
    velminmax[0] = fabs(*std::min_element(u_, u_ + nx_ * ny_ * nz_));
    velminmax[1] = fabs(*std::min_element(v_, v_ + nx_ * ny_ * nz_));
    velminmax[2] = fabs(*std::min_element(w_, w_ + nx_ * ny_ * nz_));
    velminmax[3] = fabs(*std::max_element(u_, u_ + nx_ * ny_ * nz_));
    velminmax[4] = fabs(*std::max_element(v_, v_ + nx_ * ny_ * nz_));
    velminmax[5] = fabs(*std::max_element(w_, w_ + nx_ * ny_ * nz_));
    max_vel_ = *(std::max_element(velminmax, velminmax + 6));
}

coUniTracer::~coUniTracer()
{
}

void
coUniTracer::solve(const float *yini,
                   vector<coUniState> &solu,
                   float eps_rel,
                   float max_length,
                   float vel_factor,
                   int ts) const
{
    float length(0.0);
    float y[3], x(0.0);
    const float TINY = 1.0e-30f;
    const float EPS_ABS_FACTOR = 1.0e-5f;
    std::copy(yini, yini + 3, y);
    float dydx[3];
    bool diagnose = interpolate(y, dydx);
    if (!diagnose) // initial point out of domain
    {
        solu.clear();
        return;
    }

    // add initial state
    coUniState state;
    std::copy(y, y + 3, state.point_);
    std::copy(dydx, dydx + 3, state.velocity_);
    state.time_ = 0.0;
    vector<coUniState> l_solu;
    l_solu.push_back(state);

    float h = ts * getInitialH(y, dydx);

    // dr
    // integriert so lange bis tracelen erreicht ist oder aber die geschwindigkecit kleiner als Abbruchkriterium ist
    while (length < max_length
           && !stop(vel_factor, l_solu)) // integration loop
    {
        float yscal[3];
        float hdid, hnext;
        int i;
        for (i = 0; i < 3; i++)
            yscal[i] = fabs(y[i]) + fabs(dydx[i] * h) + TINY;
        // rkqs is a runge-kutta stepper
        bool diagnose = rkqs(y, dydx, &x, h,
                             eps_rel, EPS_ABS_FACTOR * model_length_,
                             yscal, &hdid, &hnext);
        // if diagnose OK, then test last point
        // and if in domain, accumulate the new state,
        // increase length, reset h, and continue
        if (diagnose && interpolate(y, dydx))
        {
            coUniState state;
            std::copy(y, y + 3, state.point_);
            std::copy(dydx, dydx + 3, state.velocity_);
            state.time_ = x;
            const coUniState &prev_state = *(l_solu.rbegin());
            length += sqrt((prev_state.point_[0] - y[0]) * (prev_state.point_[0] - y[0]) + (prev_state.point_[1] - y[1]) * (prev_state.point_[1] - y[1]) + (prev_state.point_[2] - y[2]) * (prev_state.point_[2] - y[2]));
            l_solu.push_back(state);
            h = hnext;
            continue;
        }
        // otherwise...
        else
        {
            // prepare for throwing away last step restarting from last state in
            // l_solu
            const coUniState &last_state = *(l_solu.rbegin());
            std::copy(last_state.point_, last_state.point_ + 3, y);
            std::copy(last_state.velocity_, last_state.velocity_ + 3, dydx);
            x = last_state.time_;
            float h_crit_restart = ts * getInitialH(y, dydx);
            // and if hdid is not too small,
            // then restart from last state
            // with an appropriate h
            if (h > h_crit_restart)
            {
                float previous_h = h;
                h = -1 * h_crit_restart;
                if (h != previous_h)
                    continue;
                break;
            }
            // if h was already too small, interrupt integration
            else
                break; // out of domain
        }
    }
    l_solu.swap(solu);
}

bool
coUniTracer::interpolate(const float *y, float *dydx) const
{
    if (y[0] < xmin_ || y[0] > xmax_)
        return false;
    if (y[1] < ymin_ || y[1] > ymax_)
        return false;
    if (y[2] < zmin_ || y[2] > zmax_)
        return false;
    if (!dydx)
        return true; // no velocity is retrieved

    int cell[3]; // find cell in grid
    cell[0] = (dx_ > 0.0) ? int(floor((y[0] - xmin_) / dx_)) : 0;
    cell[1] = (dy_ > 0.0) ? int(floor((y[1] - ymin_) / dy_)) : 0;
    cell[2] = (dz_ > 0.0) ? int(floor((y[2] - zmin_) / dz_)) : 0;

    if (cell[0] < 0)
        cell[0] = 0;
    if (cell[1] < 0)
        cell[1] = 0;
    if (cell[2] < 0)
        cell[2] = 0;

    if (cell[0] && cell[0] >= nx_ - 1)
        cell[0] = nx_ - 2;
    if (cell[1] && cell[1] >= ny_ - 1)
        cell[1] = ny_ - 2;
    if (cell[2] && cell[2] >= nz_ - 1)
        cell[2] = nz_ - 2;

    float fem_c[3]; // find fem parameters (defined from -1 to 1)
    fem_c[0] = (dx_ > 0) ? (y[0] - float(xmin_) - dx_ * cell[0]) / dx_ : 0.5f;
    fem_c[1] = (dy_ > 0) ? (y[1] - float(ymin_) - dy_ * cell[1]) / dy_ : 0.5f;
    fem_c[2] = (dz_ > 0) ? (y[2] - float(zmin_) - dz_ * cell[2]) / dz_ : 0.5f;
    fem_c[0] -= 0.5f;
    fem_c[1] -= 0.5f;
    fem_c[2] -= 0.5f;
    fem_c[0] += fem_c[0];
    fem_c[1] += fem_c[1];
    fem_c[2] += fem_c[2];

    // prepare an array with velocities at the 8 nodes of the cell
    float velos[24];
    int i, j, k, base, count = 0;
    for (i = 0; i < 2; ++i)
    {
        for (j = 0; j < 2; ++j)
        {
            for (k = 0; k < 2; ++k)
            {
                base = (cell[0] + i) * ny_ * nz_ + (cell[1] + j) * nz_ + cell[2] + k;
                velos[count] = u_[base];
                ++count;
                velos[count] = v_[base];
                ++count;
                velos[count] = w_[base];
                ++count;
            }
        }
    }
    // use velos in grid_methods::interpElem
    grid_methods::interpElem(fem_c, dydx, 3, velos);
    return true;
}

float
coUniTracer::getInitialH(const float *, const float *dydx) const
{
    // y is not required
    float CELL_FACTOR = 0.25;
    float velo = sqrt(dydx[0] * dydx[0] + dydx[1] * dydx[1] + dydx[2] * dydx[2]);
    if (velo == 0.0)
        return 0.0;
    float mind = std::min(dx_, dy_);
    mind = std::min(mind, dz_);
    int minn = std::min(nx_, ny_);
    minn = std::min(minn, nz_);
    if (minn < 16)
        CELL_FACTOR /= (16 - minn);
    return CELL_FACTOR * mind / velo;
}

bool
coUniTracer::rkqs(float *y, const float *dydx, float *x,
                  float htry, float eps, float eps_abs,
                  const float *yscal,
                  float *hdid, float *hnext) const
{
    int i;
    bool ret;
    float errmax, h, htemp, xnew, yerr[3], ytemp[3];
    float abserr = 0.0;

    h = htry;
    for (;;)
    {
        ret = rkck(y, dydx, *x, h, ytemp, yerr);
        if (!ret)
        {
            return false;
        }
        errmax = 0.0;
        for (i = 0; i < 3; i++)
            errmax = std::max(errmax, float(fabs(yerr[i] / yscal[i])));
        for (i = 0; i < 3; i++)
            abserr = std::max(abserr, float(fabs(yerr[i])));

        errmax /= eps;

        if (errmax > 1.0 && abserr > eps_abs)
        {
            htemp = SAFETY * h * pow(errmax, (float)PSHRNK);
            h = (h >= 0.0 ? std::max(htemp, 0.1f * h) : std::min(htemp, 0.1f * h));
            xnew = (*x) + h;
            if (xnew == *x)
            {
                //Covise::sendWarning("stepsize underflow in stepper");
                return false;
            }
            continue;
        }
        else // FIXME: different in the original version
        {
            if (errmax > ERRCON)
                *hnext = SAFETY * h * pow(errmax, (float)PGROW);
            else
                *hnext = 5.0f * h;
        }
        *x += (*hdid = h);
        for (i = 0; i < 3; i++)
            y[i] = ytemp[i];
        break;
    }
    return true;
}

bool
coUniTracer::rkck(const float *y,
                  const float *dydx,
                  float /*x*/,
                  float h,
                  float yout[],
                  float yerr[]) const
{
    int i;
    bool ctrl1;
    bool ctrl2;
    bool ctrl3;
    bool ctrl4;
    bool ctrl5;
    // auskommentierte variablen w�ren relevant f�r instation�re berechnungen
    const float /*a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,*/ b21 = 0.2f,
                                                          b31 = 3.0f / 40.0f, b32 = 9.0f / 40.0f, b41 = 0.3f, b42 = -0.9f, b43 = 1.2f,
                                                          b51 = -11.0f / 54.0f, b52 = 2.5f, b53 = -70.0f / 27.0f, b54 = 35.0f / 27.0f,
                                                          b61 = 1631.0f / 55296.0f, b62 = 175.0f / 512.0f, b63 = 575.0f / 13824.0f,
                                                          b64 = 44275.0f / 110592.0f, b65 = 253.0f / 4096.0f, c1 = 37.0f / 378.0f,
                                                          c3 = 250.0f / 621.0f, c4 = 125.0f / 594.0f, c6 = 512.0f / 1771.0f,
                                                          dc5 = -277.00f / 14336.0f;
    const float dc1 = c1 - 2825.0f / 27648.0f, dc3 = c3 - 18575.0f / 48384.0f,
                dc4 = c4 - 13525.0f / 55296.0f, dc6 = c6 - 0.25f;
    float ak2[3], ak3[3], ak4[3], ak5[3], ak6[3], ytemp[3];

    for (i = 0; i < 3; i++)
        ytemp[i] = y[i] + b21 * h * dydx[i];
    ctrl1 = interpolate(ytemp, ak2);
    if (!ctrl1)
    {
        return false;
    }

    for (i = 0; i < 3; i++)
        ytemp[i] = y[i] + h * (b31 * dydx[i] + b32 * ak2[i]);
    ctrl2 = interpolate(ytemp, ak3);
    if (!ctrl2)
    {
        return false;
    }

    for (i = 0; i < 3; i++)
        ytemp[i] = y[i] + h * (b41 * dydx[i] + b42 * ak2[i] + b43 * ak3[i]);
    ctrl3 = interpolate(ytemp, ak4);
    if (!ctrl3)
    {
        return false;
    }

    for (i = 0; i < 3; i++)
        ytemp[i] = y[i] + h * (b51 * dydx[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i]);
    ctrl4 = interpolate(ytemp, ak5);
    if (!ctrl4)
    {
        return false;
    }

    for (i = 0; i < 3; i++)
        ytemp[i] = y[i] + h * (b61 * dydx[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i]);
    ctrl5 = interpolate(ytemp, ak6);
    if (!ctrl5)
    {
        return false;
    }

    for (i = 0; i < 3; i++)
        yout[i] = y[i] + h * (c1 * dydx[i] + c3 * ak3[i] + c4 * ak4[i] + c6 * ak6[i]);
    for (i = 0; i < 3; i++)
        yerr[i] = h * (dc1 * dydx[i] + dc3 * ak3[i] + dc4 * ak4[i] + dc5 * ak5[i] + dc6 * ak6[i]);

    return true;
}

bool
coUniTracer::stop(float factor, const vector<coUniState> &solu) const
{
    const coUniState &last = *(solu.rbegin());
    const float *velocity = last.velocity_;

    // dr, 05. Juli 2006
    // Rauchsonde soll interaktiv (schnell) sein und rechnet daher die Stromlinien nicht bis Gecshwindigkeit 0
    // aktuelles abbruchkriterium: gecshwindigkeitesbtrag kleiner als factor * max Betrag in eine Komponentenrichtung
    // bisheriger factor war 0.005, das reicht fuer aussenumstroemung bei Hochgeschwindigkeit nicht aus
    // neuer faktor ist 0.00005, das reicht fuer die aktuellen probleme
    // falls das irgendwann auch probleme macht, schlage ich vor, einen parameter mit absoluter abbruchgeschwindigkeit einzufuehren

    float velomag2 = velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2];
    //float velomag = sqrt(velomag2);
    float vgl2 = factor * factor * max_vel_ * max_vel_;
    //float vgl = factor*max_vel_;
    //fprintf(stderr,"velomag2=%f vgl2=%f velomag=%f vgl=%f max_vel=%f\n", velomag2, vgl2, velomag, vgl, max_vel_);
    return (velomag2 < vgl2);
}
