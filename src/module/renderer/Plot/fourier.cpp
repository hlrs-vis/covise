/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: fourier.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * DFT by definition and FFT
 */

#include <stdio.h>
#include <math.h>
#include "extern.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int bit_swap(int i, int nu);
static void sswap(double *x1, double *x2);

/*
   DFT by definition
*/
void dft(double *jr, double *ji, int n, int iflag)
{
    int i, j, sgn;
    double sumr, sumi, tpi, *w, *xr, *xi, co, si, o, on = 1.0 / n;

    sgn = iflag ? -1 : 1;
    tpi = 2.0 * M_PI;
    w = (double *)calloc(n, sizeof(double));
    xr = (double *)calloc(n, sizeof(double));
    xi = (double *)calloc(n, sizeof(double));
    if (w == NULL || xr == NULL || xi == NULL)
    {
        errwin("Can't allocate temporary in DFT");
        free(w);
        free(xr);
        free(xi);
        return;
    }
    for (i = 0; i < n; i++)
    {
        w[i] = tpi * i * on;
        xr[i] = jr[i];
        xi[i] = ji[i];
    }
    for (j = 0; j < n; j++)
    {
        sumr = 0.0;
        sumi = 0.0;
        for (i = 0; i < n; i++)
        {
            o = w[j] * i;
            co = cos(o);
            si = sin(o);
            sumr = sumr + xr[i] * co + sgn * xi[i] * si;
            sumi = sumi + xi[i] * co - sgn * xr[i] * si;
        }
        jr[j] = sumr;
        ji[j] = sumi;
    }
    if (sgn == 1)
    {
        on = 2.0 * on;
    }
    else
    {
        on = 0.5;
    }
    for (i = 0; i < n; i++)
    {
        jr[i] = jr[i] * on;
        ji[i] = ji[i] * on;
    }
    free(w);
    free(xr);
    free(xi);
}

/*
   this came off the net - if you wrote it let me know and I'll
   give you credit - PJT
*/
/*
   real_data ... ptr. to real part of data to be transformed
   imag_data ... ptr. to imag  "   "   "   "  "      "
   inv ..... Switch to flag normal or inverse transform
   n_pts ... Number of real data points
   nu ...... logarithm in base 2 of n_pts e.g. nu = 5 if n_pts = 32.
*/

void fft(double *real_data, double *imag_data, int n_pts, int nu, int inv)
{
    int n2, j, l, i, ib, k, k1, k2;
    int sgn;
    double tr, ti, arg, nu1; /* intermediate values in calcs. */
    double c, s; /* cosine & sine components of Fourier trans. */
    double fac;

    n2 = n_pts / 2;
    nu1 = nu - 1.0;
    k = 0;
    /*
    * sign change for inverse transform
    */
    sgn = inv ? -1 : 1;
    /*
    * Calculate the componets of the Fourier series of the function
    */
    for (l = 0; l != nu; l++)
    {
        do
        {
            for (i = 0; i != n2; i++)
            {
                j = k / (int)(pow(2.0, nu1));
                ib = bit_swap(j, nu);
                arg = 2.0 * M_PI * ib / n_pts;
                c = cos(arg);
                s = sgn * sin(arg);
                k1 = k;
                k2 = k1 + n2;
                tr = *(real_data + k2) * c + *(imag_data + k2) * s;
                ti = *(imag_data + k2) * c - *(real_data + k2) * s;
                *(real_data + k2) = *(real_data + k1) - tr;
                *(imag_data + k2) = *(imag_data + k1) - ti;
                *(real_data + k1) = *(real_data + k1) + tr;
                *(imag_data + k1) = *(imag_data + k1) + ti;
                k++;
            }
            k += n2;
        } while (k < n_pts - 1);
        k = 0;
        nu1 -= 1.0;
        n2 /= 2;
    }
    for (k = 0; k != n_pts; k++)
    {
        ib = bit_swap(k, nu);
        if (ib > k)
        {
            sswap((real_data + k), (real_data + ib));
            sswap((imag_data + k), (imag_data + ib));
        }
    }
    /*
    * If calculating the inverse transform, must divide the data by the number of
    * data points.
    */
    if (inv)
        fac = 2.0 / n_pts;
    else
        fac = 0.5;
    for (k = 0; k != n_pts; k++)
    {
        *(real_data + k) *= fac;
        *(imag_data + k) *= fac;
    }
}

/*
 * Bit swaping routine in which the bit pattern of the integer i is reordered.
 * See Brigham's book for details
 */
static int bit_swap(int i, int nu)
{
    int ib, i1, i2;

    ib = 0;

    for (i1 = 0; i1 != nu; i1++)
    {
        i2 = i / 2;
        ib = ib * 2 + i - 2 * i2;
        i = i2;
    }
    return (ib);
}

/*
 * Simple exchange routine where *x1 & *x2 are swapped
 */
static void sswap(double *x1, double *x2)
{
    double temp_x;

    temp_x = *x1;
    *x1 = *x2;
    *x2 = temp_x;
}
