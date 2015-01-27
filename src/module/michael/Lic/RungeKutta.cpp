/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RungeKutta.h"

//see W.H. Press et al, Numerical Recipes in C, 2nd Edition
//Chap. 16.2: Adaptive Stepsize Control for Runge-Kutta, pg. 719-720

void rkqs(fvec s, fvec ds, int n, float *t, float htry, float eps,
          fvec sscal, float *hdid, float *hnext)
{
    float errmax = 0.0;
    float h = 0.0;
    float htemp = 0.0;
    float tnew = 0.0;

    fvec serr = fvec(n);
    fvec stemp = fvec(n);

    h = htry;

    int counter = 0;
    for (;;)
    {
        rkck(s, ds, n, *t, h, stemp, serr);

        {
            for (int i = 0; i < n; i++)
            {
                errmax = FMAX(errmax, fabs(serr[i] / sscal[i]));
            }
            errmax /= eps;

            if (errmax <= 1.0)
            {
                break;
            }
            else
            {
            }

            htemp = SAFETY * (h * pow(errmax, PSHRNK));
            h = (h >= 0.0 ? FMAX(htemp, 0.1 * h) : FMIN(htemp, 0.1 * h));

            tnew = (*t) + h;
            if (tnew == (*t))
            {
                cout << "\nstepsize underflow in Runge-Kutta integrator\n" << flush;
                exit(1);
            }
            else
            {
            }
        }

        //for development purposes only
        ++counter;
        if (counter > 999)
        {
            break;
        }
        else
        {
        }
    }

    if (errmax > ERRCON)
    {
        *hnext = SAFETY * (h * pow(errmax, PGROW));
    }
    else
    {
        *hnext = 5.0 * h;
    }

    *hdid = h;
    *t += (*hdid);
    {
        for (int i = 0; i < n; i++)
        {
            s[i] = stemp[i];
        }
    }

    return;
}

//Cash-Sharp embedded Runge-Kutta method
//see W.H.Press et al.: Numerical Recipes in C, 2nd ed., pg. 717 for tableau
// for tableau, pgs. 719/720 for implementation.
void rkck(fvec s, fvec ds, int n, float t, float h, fvec sout, fvec serr)
{
    static float a2 = 0.2;
    static float a3 = 0.3;
    static float a4 = 0.6;
    static float a5 = 1.0;
    static float a6 = 0.875;

    static float b21 = 0.2;

    static float b31 = 0.075;
    static float b32 = 0.225;

    static float b41 = 0.3;
    static float b42 = -0.9;
    static float b43 = 1.2;

    static float b51 = -11.0 / 54.0;
    static float b52 = 2.5;
    static float b53 = -70.0 / 27.0;
    static float b54 = 35.0 / 27.0;

    static float b61 = 1631.0 / 55296.0;
    static float b62 = 175.0 / 512.0;
    static float b63 = 575.0 / 13824.0;
    static float b64 = 44275.0 / 110592.0;
    static float b65 = 253.0 / 4096.0;

    static float c1 = 37.0 / 378.0;
    static float c3 = 250.0 / 621.0;
    static float c4 = 125.0 / 594.0;
    static float c6 = 512.0 / 1771.0;

    static float dc5 = -277.0 / 14336.0;

    float dc1 = c1 - (2825.0 / 27648.0);
    float dc3 = c3 - (18575.0 / 48384.0);
    float dc4 = c4 - (13525.0 / 55296.0);
    float dc6 = c6 - 0.25;

    fvec ak2 = fvec(n);
    fvec ak3 = fvec(n);
    fvec ak4 = fvec(n);
    fvec ak5 = fvec(n);
    fvec ak6 = fvec(n);

    fvec stemp = fvec(n);

    {
        //first step
        {
            for (int i = 0; i < n; i++)
            {
                stemp[i] = s[i] + (b21 * h * ds[i]);
            }
        }

        //second step
        derivs(t + a2 * h, stemp, ak2);
        {
            for (int i = 0; i < n; i++)
            {
                stemp[i] = s[i] + h * (b31 * ds[i] + b32 * ak2[i]);
            }
        }

        //third step
        derivs(t + a3 * h, stemp, ak3);
        {
            for (int i = 0; i < n; i++)
            {
                stemp[i] = s[i] + h * (b41 * ds[i] + b42 * ak2[i] + b43 * ak3[i]);
            }
        }

        //fourth step
        derivs(t + a4 * h, stemp, ak4);
        {
            for (int i = 0; i < n; i++)
            {
                stemp[i] = s[i] + h * (b51 * ds[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i]);
            }
        }

        //fifth step
        derivs(t + a5 * h, stemp, ak5);
        {
            for (int i = 0; i < n; i++)
            {
                stemp[i] = s[i] + h * (b61 * ds[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i]);
            }
        }

        //sixth step
        derivs(t + a6 * h, stemp, ak6);

        //accumulate increments with proper weights
        {
            for (int i = 0; i < n; i++)
            {
                sout[i] = s[i] + h * (c1 * ds[i] + c3 * ak3[i] + c4 * ak4[i] + c6 * ak6[i]);
            }
        }

        //estimate error
        {
            for (int i = 0; i < n; i++)
            {
                serr[i] = h * (dc1 * ds[i] + dc3 * ak3[i] + dc4 * ak4[i] + dc5 * ak5[i] + dc6 * ak6[i]);
            }
        }
    }

    return;
}

//used to calculate the derivative of s at step t
//interpolates v at s(t) here and returns it
fvec derivative(float t, const fvec &s, const fvec &v)
{
    //s are the three nodes of the triangle
    //v are the velocity values at these nodes
    //ds is the calculated derivative ds/dt at the current position

    fvec lambda = fvec(3, 0);
    fvec ds = fvec(3, 0);

    return ds;
}

//used to calculate the derivative of s at step t
//interpolates v at s(t) here and returs it
void derivs(float t, const fvec &s, const fvec &ds)
{
    return;
}
