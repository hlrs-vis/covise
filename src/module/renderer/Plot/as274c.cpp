/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "as274c.h"
#include "extern.h"

int includ(int np, int, double w,
           double *xrow, double y, double *d, double *rbar,
           double *thetab, double *sserr)
{
    int ier = 0, i, k, nextr;
    double cbar, sbar, di, xi, xk, dpi, wxi;

    /* ALGORITHM AS274.1  APPL. STATIST. (1992) VOL 41, NO. 2
      Calling this routine updates d, rbar, thetab and sserr by the
      inclusion of xrow, yelem with the specified weight.  This version has
      been modified to make it slightly faster when the early elements of
      XROW are not zeroes.  *** WARNING *** The elements of XROW are
      over-written. */

    --thetab;
    --rbar;
    --d;
    --xrow;

#ifdef STRINGENT
    /*     Some checks. */
    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (ier)
        return ier;
#endif /* STRINGENT */
    /* the function includ() is called for each obs, and is where all
      the time is spent.  To make it faster, we skip the checks about
      whether nrbar is big enough for np.  If you'd like to have the
      checks in anyway, then compile with -DSTRINGENT */

    nextr = 1;
    /* Skip unnecessary transformations.  Test on exact zeroes must be
   used or stability can be destroyed. */
    for (i = 1; i <= np; ++i)
    {
        if (w == 0)
            return ier;
        xi = xrow[i];
        if (xi == 0)
            nextr += np - i;
        else
        {
            di = d[i];
            wxi = w * xi;
            dpi = di + wxi * xi;
            cbar = di / dpi;
            sbar = wxi / dpi;
            w = cbar * w;
            d[i] = dpi;
            if (i != np)
                for (k = i + 1; k <= np; ++k)
                {
                    xk = xrow[k];
                    xrow[k] = xk - xi * rbar[nextr];
                    rbar[nextr] = cbar * rbar[nextr] + sbar * xk;
                    ++nextr;
                }
            xk = y;
            y = xk - xi * thetab[i];
            thetab[i] = cbar * thetab[i] + sbar * xk;
        }
    }

    /* Y * SQRT(W) is now equal to the Brown, Durbin & Evans recursive residual. */
    *sserr += w * y * y;
    return ier;
} /* includ */

int clear(int np, int nrbar, double *d, double *rbar,
          double *thetab, double *sserr)
/* ALGORITHM AS274.2  APPL. STATIST. (1992) VOL.41, NO. 2
   Sets arrays to zero prior to calling AS75.1 */
{
    int ier = 0;

    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (ier)
        return ier;

    memset(d, 0, np * sizeof(double));
    memset(thetab, 0, np * sizeof(double));
    memset(rbar, 0, nrbar * sizeof(double));
    *sserr = 0;
    return 0;
}

int regcf(int np, int nrbar, double *d, double *rbar,
          double *thetab, double *tol, double *beta,
          int nreq)
{
    int ier = 0, i, j, nextr;

    /* ALGORITHM AS274.3  APPL. STATIST. (1992) VOL 41, NO.2
      Modified version of AS75.4 to calculate regression coefficients
      for the first NREQ variables, given an orthogonal reduction from
      AS75.1. */

    --beta;
    --tol;
    --thetab;
    --rbar;
    --d;

    /*     Some checks. */
    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (nreq < 1 || nreq > np)
        ier += 4;
    if (ier)
        return ier;

    for (i = nreq; i >= 1; --i)
        if (sqrt(d[i]) < tol[i])
        {
            beta[i] = d[i] = 0;
        }
        else
        {
            beta[i] = thetab[i];
            nextr = (i - 1) * (np + np - i) / 2 + 1;
            for (j = i + 1; j <= nreq; ++j)
                beta[i] -= rbar[nextr++] * beta[j];
        }

    return ier;
} /* regcf */

int tolset(int np, int nrbar, double *d, double *rbar, double *tol)
/* ALGORITHM AS274.4  APPL. STATIST. (1992) VOL 41, NO.2
   Sets up array TOL for testing for zeroes in an orthogonal
   reduction formed using AS75.1.
   EPS is a machine-dependent constant.   For compilers which use
   the IEEE format for floating-point numbers, recommended values
   are 1.E-06 for single precision and 1.E-15 for double precision. */
{
    const double eps = 1e-15;
    double d__1, sum, *work;
    int ier = 0, col, pos, row;

    --tol;
    --rbar;
    --d;

    /* Some checks. */
    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (ier)
        return ier;

    /* Set TOL(I) = sum of absolute values in column I of RBAR after
      scaling each element by the square root of its row multiplier. */

    work = -1 + (double *)malloc(np * sizeof(double));
    for (col = 1; col <= np; ++col)
        work[col] = sqrt(d[col]);
    for (col = 1; col <= np; ++col)
    {
        pos = col - 1;
        sum = work[col];
        for (row = 1; row <= col - 1; ++row)
        {
            sum += (d__1 = rbar[pos], fabs(d__1)) * work[row];
            pos += np - row - 1;
        }
        tol[col] = eps * sum;
    }
    free(work + 1);
    return 0;
} /* tolset */

int sing(int np, int nrbar, double *d,
         double *rbar, double *thetab, double *sserr,
         double *tol, int *lindep)
/* ALGORITHM AS274.5  APPL. STATIST. (1992) VOL 41, NO. 2
   Checks for singularities, reports, and adjusts orthogonal
   reductions produced by AS75.1. */
{
    double d__1, temp, *work;
    int ier = 0, np2, col, pos, row, pos2;

    --lindep;
    --tol;
    --thetab;
    --rbar;
    --d;

    if (np <= 0)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (ier)
        return ier;
    work = -1 + (double *)malloc(np * sizeof(double));

    /* Set elements within RBAR to zero if they are less than TOL(COL) in
      absolute value after being scaled by the square root of their row
      multiplier. */
    for (col = 1; col <= np; ++col)
        work[col] = sqrt(d[col]);
    for (col = 1; col <= np; ++col)
    {
        temp = tol[col];
        pos = col - 1;
        for (row = 1; row < col; ++row)
        {
            if ((d__1 = rbar[pos], fabs(d__1)) * work[row] < temp)
                rbar[pos] = 0;
            pos += np - row - 1;
        }

        /* If diagonal element is near zero, set it to zero, set appropriate
         element of LINDEP, and use INCLUD to augment the projections in
         the lower rows of the orthogonalization. */

        lindep[col] = 0;
        if (work[col] < temp)
        {
            lindep[col] = 1;
            --ier;
            if (col < np)
            {
                np2 = np - col;
                pos2 = pos + np2 + 1;
                ier = includ(np2, np2 * (np2 - 1) / 2, d[col], rbar + pos + 1,
                             thetab[col], d + col + 1, rbar + pos2,
                             thetab + col + 1, sserr);
            }
            else
            {
                d__1 = thetab[col]; /* for squaring */
                *sserr += d[col] * (d__1 * d__1);
            }
            d[col] = work[col] = thetab[col] = 0;
        }
    }
    free(work + 1);
    return ier;
} /* sing */

int ss(int np, double *d, double *thetab, double *sserr, double *rss)
/* ALGORITHM AS274.6  APPL. STATIST. (1992) VOL 41, NO. 2
   Calculates partial residual sums of squares from an orthogonal
   reduction from AS75.1. */
{
    double sum;
    int i;

    --rss;
    --thetab;
    --d;

    if (np < 1)
        return 1;
    sum = rss[np] = *sserr;
    for (i = np; i >= 2; --i)
    {
        sum += d[i] * thetab[i] * thetab[i];
        rss[i - 1] = sum;
    }
    return 0;
} /* ss */

int cov(int np, int nrbar, double *d,
        double *rbar, int nreq, double *rinv, double *var,
        double *covmat, int dimcov, double *sterr)
/* ALGORITHM AS274.7  APPL. STATIST. (1992) VOL 41, NO.2
   Calculate covariance matrix for regression coefficients for the
   first NREQ variables, from an orthogonal reduction produced from
   AS75.1. */
{
    int ier = 0, k, start, col, pos, row, pos1, pos2;
    double sum;

    --sterr;
    --covmat;
    --rinv;
    --rbar;
    --d;

    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (dimcov < nreq * (nreq + 1) / 2)
        ier += 4;

    for (row = 1; row <= nreq; ++row)
        if (d[row] == 0)
            ier = -row;
    if (ier)
        return ier;

    inv(np, nrbar, rbar + 1, nreq, rinv + 1);
    start = pos = 1;
    for (row = 1; row <= nreq; ++row)
    {
        pos2 = start;
        for (col = row; col <= nreq; ++col)
        {
            pos1 = start + col - row;
            sum = (row == col ? 1.0 : rinv[pos1 - 1]) / d[col];
            for (k = col + 1; k <= nreq; ++k)
                sum += rinv[pos1++] * rinv[pos2++] / d[k];
            covmat[pos] = sum * *var;
            if (row == col)
                sterr[row] = sqrt(covmat[pos]);
            ++pos;
        }
        start += nreq - row;
    }
    return 0;
} /* cov */

void inv(int np, int, double *rbar, int nreq, double *rinv)
/* ALGORITHM AS274.8  APPL. STATIST. (1992) VOL 41, NO. 2
   Invert first NREQ rows and columns of Cholesky factorization
   produced by AS75.1. */
{
    int pos, row;

    --rinv;
    --rbar;

    /* Invert RBAR ignoring row multipliers, from the bottom up. */
    pos = nreq * (nreq - 1) / 2;
    for (row = nreq - 1; row >= 1; --row)
    {
        int col, start = (row - 1) * (np + np - row) / 2 + 1;
        for (col = nreq; col >= row + 1; --col)
        {
            double sum = 0;
            int pos1 = start, pos2 = pos, k;
            for (k = row + 1; k <= col - 1; ++k)
            {
                pos2 += nreq - k;
                sum -= rbar[pos1++] * rinv[pos2];
            }
            rinv[pos] = sum - rbar[pos1];
            --pos;
        }
    }
} /* inv */

int pcorr(int np, int nrbar, double *d,
          double *rbar, double *thetab, double *sserr, int in,
          double *cormat, int dimc, double *ycorr)
/* ALGORITHM AS274.9  APPL. STATIST. (1992) VOL 41, NO. 2
   Calculate partial correlations after the first IN variables
   have been forced into the regression. */
{
    int ier = 0, i, start, in1;
    double *work;

    --cormat;
    --thetab;
    --rbar;
    --d;

    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (in < 0 || in > np - 1)
        ier += 4;
    if (dimc < (np - in) * (np - in - 1) / 2)
        ier += 8;
    if (ier)
        return ier;

    start = in * (np + np - in - 1) / 2 + 1;
    in1 = in + 1;
    work = (double *)malloc(np * sizeof(double));
    cor(np - in, d + in1, rbar + start, thetab + in1, sserr, work, cormat + 1, ycorr);

    /* Check for zeroes. */
    for (i = 0; i < (np - in); ++i)
        if (work[i] <= 0)
            ier = -1 - i;
    free(work);
    return ier;
} /* pcorr */

void cor(int np, double *d, double *rbar,
         double *thetab, double *sserr, double *work,
         double *cormat, double *ycorr)
/* ALGORITHM AS274.10  APPL. STATIST. (1992) VOL 41, NO.2
   Calculate correlations from an orthogonal reduction.   This
   routine will usually be called from PCORR, which will have
   removed the appropriate number of rows at the start. */
{
    int i__1, i__2, diff, pos, row, col1, col2, pos1, pos2;
    double sumy, sum;

    --ycorr;
    --cormat;
    --work;
    --thetab;
    --rbar;
    --d;

    /* process by columns, including the projections of the dependent
      variable (THETAB). */
    sumy = *sserr;
    i__1 = np;
    for (row = 1; row <= i__1; ++row)
        sumy += d[row] * thetab[row] * thetab[row];
    sumy = sqrt(sumy);

    pos = np * (np - 1) / 2;
    for (col1 = np; col1 >= 1; --col1) /* find length of column COL1. */
    {
        sum = d[col1];
        i__1 = pos1 = col1 - 1;
        for (row = 1; row <= i__1; ++row)
        {
            sum += d[row] * rbar[pos1] * rbar[pos1];
            pos1 += np - row - 1;
        }
        work[col1] = sqrt(sum);

        /* If SUM = 0, set all correlations with this variable to zero. */
        if (sum == 0)
        {
            ycorr[col1] = 0;
            i__1 = col1 + 1;
            for (col2 = np; col2 >= i__1; --col2)
            {
                cormat[pos] = 0;
                --pos;
            }
            goto L70;
        }

        /* Form cross-products, then divide by product of column lengths. */
        sum = d[col1] * thetab[col1];
        i__1 = pos1 = col1 - 1;
        for (row = 1; row <= i__1; ++row)
        {
            sum += d[row] * rbar[pos1] * thetab[row];
            pos1 += np - row - 1;
        }
        ycorr[col1] = sum / (sumy * work[col1]);

        i__1 = col1 + 1;
        for (col2 = np; col2 >= i__1; --col2)
        {
            if (work[col2] > 0)
            {
                pos1 = col1 - 1;
                pos2 = col2 - 1;
                diff = col2 - col1;
                i__2 = col1 - 1;
                for (sum = 0, row = 1; row <= i__2; ++row)
                {
                    sum += d[row] * rbar[pos1] * rbar[pos2];
                    pos1 += np - row - 1;
                    pos2 = pos1 + diff;
                }
                sum += d[col1] * rbar[pos2];
                cormat[pos] = sum / (work[col1] * work[col2]);
            }
            else
                cormat[pos] = 0;
            --pos;
        }
    L70:
        ;
    }
} /* cor */

int vmove(int np, int nrbar, int *vorder,
          double *d, double *rbar, double *thetab, double *rss,
          int from, int to, double *tol)
/* ALGORITHM AS274.11 APPL. STATIST. (1992) VOL 41, NO. 2
   Move variable from position FROM to position TO in an
   orthogonal reduction produced by AS75.1. */
{
    int ier = 0, last, m;
    double d__1, cbar, sbar, d1new, d2new, x, y, d1, d2;
    int first, m1, m2, mp1, inc, col, pos, row;

    --tol;
    --rss;
    --thetab;
    --rbar;
    --d;
    --vorder;

    if (np <= 0)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (from < 1 || from > np)
        ier += 4;
    if (to < 1 || to > np)
        ier += 8;
    if (ier)
        return ier;
    if (from == to)
        return ier;

    if (from < to)
    {
        first = from;
        last = to - 1;
        inc = 1;
    }
    else
    {
        first = from - 1;
        last = to;
        inc = -1;
    }
    /*     Find addresses of first elements of RBAR in rows M and (M+1). */
    for (m = first; inc < 0 ? m >= last : m <= last; m += inc)
    {
        m1 = (m - 1) * (np + np - m) / 2 + 1;
        m2 = m1 + np - m;
        mp1 = m + 1;
        d1 = d[m];
        d2 = d[mp1];

        /* Special cases. */
        if (d1 == 0 && d2 == 0)
            goto L40;
        x = rbar[m1];
        if (fabs(x) * sqrt(d1) < tol[mp1])
            x = 0;
        if (d1 == 0 || x == 0)
        {
            d[m] = d2;
            d[mp1] = d1;
            rbar[m1] = 0;
            for (col = m + 2; col <= np; ++col)
            {
                x = rbar[++m1];
                rbar[m1] = rbar[m2];
                rbar[m2++] = x;
            }
            x = thetab[m];
            thetab[m] = thetab[mp1];
            thetab[mp1] = x;
            goto L40;
        }
        else if (d2 == 0)
        {
            d__1 = x;
            d[m] = d1 * d__1 * d__1;
            rbar[m1] = 1.0 / x;
            for (col = m + 2; col <= np; ++col)
                rbar[++m1] /= x;
            thetab[m] /= x;
            goto L40;
        }

        /*     Planar rotation in regular case. */
        d__1 = x;
        d1new = d2 + d1 * (d__1 * d__1);
        cbar = d2 / d1new;
        sbar = x * d1 / d1new;
        d2new = d1 * cbar;
        d[m] = d1new;
        d[mp1] = d2new;
        rbar[m1] = sbar;
        for (col = m + 2; col <= np; ++col)
        {
            y = rbar[++m1];
            rbar[m1] = cbar * rbar[m2] + sbar * y;
            rbar[m2] = y - x * rbar[m2];
            ++m2;
        }
        y = thetab[m];
        thetab[m] = cbar * thetab[mp1] + sbar * y;
        thetab[mp1] = y - x * thetab[mp1];

    /*     Swap columns M and (M+1) down to row (M-1). */
    L40:
        if (m == 1)
            goto L60;
        pos = m;
        for (row = 1; row < m; ++row)
        {
            x = rbar[pos];
            rbar[pos] = rbar[pos - 1];
            rbar[pos - 1] = x;
            pos += np - row - 1;
        }

    /* Adjust variable order (VORDER), the tolerances (TOL) and the vector
         of residual sums of squares (RSS). */
    L60:
        m1 = vorder[m];
        vorder[m] = vorder[mp1];
        vorder[mp1] = m1;
        x = tol[m];
        tol[m] = tol[mp1];
        tol[mp1] = x;
        d__1 = thetab[mp1];
        rss[m] = rss[mp1] + d[mp1] * (d__1 * d__1);
    }

    return 0;
} /* vmove */

int reordr(int np, int nrbar, int *vorder,
           double *d, double *rbar, double *thetab, double *rss,
           double *tol, int *list, int n, int pos1)
/* ALGORITHM AS274.12 APPL. STATIST. (1992) VOL 41, NO.2 Re-order the
   variables in an orthogonal reduction produced by AS75.1 so that the N
   variables in LIST start at position POS1, though will not necessarily
   be in the same order as in LIST.  Any variables in VORDER before
   position POS1 are not moved.  Auxiliary routine called: VMOVE */
{
    int ier = 0, next, i, j;

    --list;
    --tol;
    --rss;
    --thetab;
    --rbar;
    --d;
    --vorder;

    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (n < 1 || n >= np + 1 - pos1)
        ier += 4;
    if (ier)
        return ier;

    /* Work through VORDER finding variables which are in LIST. */
    i = next = pos1;
    do
    {
        int l = vorder[i];
        for (j = 1; j <= n; j++)
            if (l == list[j])
            {
                if (i > next)
                    vmove(np, nrbar, vorder + 1, d + 1, rbar + 1, thetab + 1, rss + 1,
                          i, next, tol + 1);
                if (++next >= n + pos1)
                    return 0;
                break;
            }
    } while (++i <= np);
    return 8;
} /* reordr */

int hdiag(double *xrow, int np, int nrbar,
          double *d, double *rbar, double *tol, int nreq,
          double *hii)
/* ALGORITHM AS274.13  APPL. STATIST. (1992) VOL.41, NO.2 */
{
    int i__1, i__2, ier = 0, col, pos, row;
    double d__1, sum, *wk;

    --tol;
    --rbar;
    --d;
    --xrow;

    if (np < 1)
        ier = 1;
    if (nrbar < np * (np - 1) / 2)
        ier += 2;
    if (nreq > np)
        ier += 4;
    if (ier)
        return ier;

    /* The elements of XROW.inv(RBAR).sqrt(D) are calculated and stored in WK. */
    wk = -1 + (double *)malloc(np * sizeof(double));
    *hii = 0;
    i__1 = nreq;
    for (col = 1; col <= i__1; ++col)
        if (sqrt(d[col]) <= tol[col])
            wk[col] = 0;
        else
        {
            i__2 = pos = col - 1;
            sum = xrow[col];
            for (row = 1; row <= i__2; ++row)
            {
                sum -= wk[row] * rbar[pos];
                pos += np - row - 1;
            }
            d__1 = wk[col] = sum;
            *hii += d__1 * d__1 / d[col];
        }
    free(wk + 1);
    return 0;
} /* hdiag */

void pr_utdm_v(double *x, int N, int width, int precision)
/* "print a upper triangular double matrix stored as a vector"
   The matrix is N x N, the vector has N*(N+1)/2 elements.
   Each element is formatted using width and precision.
   There are no sanity checks at all. */
{
    int pos = 0, i, j, leavespace;
    char s[100], fmt[100]; /* will be used in making printf() formats */
    char buf[256];

    sprintf(fmt, "%%%d.%dg", width, precision);
    for (i = 0; i < N; i++)
    {
        leavespace = i * width;
        sprintf(s, "%%%ds", leavespace);
        sprintf(buf, s, "");
        stufftext(buf, 0);
        for (j = i; j < N; j++)
        {
            sprintf(buf, fmt, x[pos++]);
            stufftext(buf, 0);
        }
        stufftext((char *)"\n", 0);
    }
}

double *dvector(int l, int h)
{
    double *block;
    int size;

    size = h - l + 1;
    block = (double *)malloc(sizeof(double) * size);
    if (block == NULL)
    {
        fprintf(stderr, "malloc failure in dvector()\n");
        return NULL;
    }
    return block - l;
}

int *ivector(int l, int h)
{
    int *block, size;

    size = h - l + 1;
    block = (int *)malloc(sizeof(int) * size);
    if (block == NULL)
    {
        fprintf(stderr, "malloc failure in ivector()\n");
        return NULL;
    }
    return block - l;
}

double **dmatrix(int rl, int rh, int cl, int ch)
{
    double *block;
    double **m;
    int size, i, rowsize, numrows;

    rowsize = ch - cl + 1; /* #locations consumed by 1 row */
    numrows = rh - rl + 1;
    size = numrows * rowsize;
    block = (double *)malloc((unsigned)sizeof(double) * size);
    if (block == NULL)
    {
        fprintf(stderr, "malloc failure in matrix allocation\n");
        return NULL;
    }
    /* so we have the matrix. */

    /* Now for the row pointers */
    m = (double **)malloc((unsigned)sizeof(double *) * numrows);
    if (m == NULL)
    {
        fprintf(stderr, "malloc failure in matrix allocation\n");
        return NULL;
    }
    m -= rl; /* fixup m pointer so m[rl] == old m[0] */

    /* Finally, setup pointers to rows */
    block -= cl;
    for (i = rl; i <= rh; i++)
    {
        m[i] = block;
        block += rowsize;
    }
    return m;
}

void putdvec(const char *s, double *x, int l, int h)
{
    int i;
    char buf[512];
    sprintf(buf, "Vector %-10s: \n", s);
    stufftext(buf, 0);
    for (i = l; i <= h; i++)
    {
        sprintf(buf, " %d: %.4g \n", i, x[i]);
        stufftext(buf, 0);
    }
}

static double r_mod(float x, float y) /* used by f2c's version of rand() */
{
    double quotient = (double)x / y;
    quotient = quotient >= 0 ? floor(quotient) : -floor(-quotient);
    return x - y * quotient;
}

double ranwm()
{
    struct
    {
        int ix, iy, iz;
    } randc;

    float r;
    randc.ix = randc.ix * 171 % 30269;
    randc.iy = randc.iy * 172 % 30307;
    randc.iz = randc.iz * 170 % 30323;
    r = (float)randc.ix / (float)30269. + (float)randc.iy / (float)30307. + (float)randc.iz / (float)30323.;
    return r_mod(r, 1.0);
}

int dofitcurve(int cnt, double *xd, double *yd, int nd, double *c)
{
    double **x, *y, *d, *rbar, *thetab, *xrow, sserr, *beta;
    double *tol, *rss;
    int i, j, nvars = nd + 1, nrbar = 100, *vorder, *lindep, error;
    double xval;
    char buf[256];

    x = dmatrix(0, cnt - 1, 0, nvars - 1);
    y = dvector(0, cnt - 1);
    d = dvector(0, nvars - 1);
    rbar = dvector(0, nrbar);
    thetab = dvector(0, nvars - 1);
    xrow = dvector(0, nvars - 1);
    beta = dvector(0, nvars - 1);
    tol = dvector(0, nvars - 1);
    rss = dvector(0, nvars - 1);
    vorder = ivector(0, nvars - 1);
    lindep = ivector(0, nvars - 1);

    for (i = 0; i < nvars; i++)
        vorder[i] = i;

    for (i = 0; i < cnt; i++)
    {
        xval = xd[i];
        for (j = 1; j < nvars; j++)
        {
            x[i][j - 1] = pow(xval, (double)j);
        }
        y[i] = yd[i];
    }
    error = clear(nvars, nrbar, d, rbar, thetab, &sserr);
    if (error)
    {
        sprintf(buf, "as274c: clear() returned %d", error);
        errwin(buf);
        goto bustout;
    }

    for (i = 0; i < cnt; i++)
    {
        xrow[0] = 1; /* include constant */
        for (j = 0; j < nvars - 1; j++)
            xrow[j + 1] = x[i][j];
        error = includ(nvars, nrbar, 1.0, xrow, y[i], d, rbar, thetab, &sserr);
        if (error)
        {
            sprintf(buf, "as274c: includ() returned %d at row %d", error, i);
            errwin(buf);
            goto bustout;
        }
    }

    error = tolset(nvars, nrbar, d, rbar, tol);
    if (error)
    {
        sprintf(buf, "as274c: tolset() returned %d\n", error);
        errwin(buf);
        goto bustout;
    }
    error = sing(nvars, nrbar, d, rbar, thetab, &sserr, tol, lindep);
    if (error)
    {
        sprintf(buf, "as274c: sing() returned %d", error);
        errwin(buf);
        goto bustout;
    }
    sprintf(buf, "SSerr = %17g\n", sserr);
    stufftext(buf, 0);

    error = regcf(nvars, nrbar, d, rbar, thetab, tol, beta, nvars);
    if (error)
    {
        sprintf(buf, "as274c: regcf() returned %d", error);
        errwin(buf);
        goto bustout;
    }
    stufftext((char *)"\nVariable order:\n ", 0);
    for (j = 0; j < nvars; j++)
    {
        sprintf(buf, "   %d ", vorder[j]);
        stufftext(buf, 0);
    }
    stufftext((char *)"\n", 0);

    putdvec("Beta", beta, 0, nvars - 1);
    putdvec("d", d, 0, nvars - 1);
    sprintf(buf, "rbar matrix:\n");
    stufftext(buf, 0);
    pr_utdm_v(rbar, nvars - 1, 14, 6);
    putdvec("thetab", thetab, 0, nvars - 1);
    for (j = 0; j < nvars; j++)
    {
        c[j] = beta[j];
    }

bustout:
    ;
    free(x[0]);
    free(x);
    free(y);
    free(d);
    free(rbar);
    free(thetab);
    free(xrow);
    free(beta);
    free(tol);
    free(rss);
    free(vorder);
    free(lindep);

    return 0;
}
