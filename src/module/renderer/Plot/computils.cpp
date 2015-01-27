/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: computils.c,v 1.11 1994/11/04 06:02:10 pturner Exp pturner $
 *
 * procedures for performing transformations from the command
 * line interpreter and the GUI.
 *
 */

#include <stdio.h>
#include <math.h>
#include "extern.h"

#include "symdefs.h"
#include "globals.h"
#include "noxprotos.h"

static void forwarddiff(double *x, double *y, double *resx, double *resy, int n);
static void backwarddiff(double *x, double *y, double *resx, double *resy, int n);
static void centereddiff(double *x, double *y, double *resx, double *resy, int n);
int get_points_inregion(int rno, int invr, int len, double *x, double *y, int *cnt, double **xt, double **yt);

void do_running_command(int type, int setno, int rlen)
{
    switch (type)
    {
    case RUNAVG:
        type = 0;
        break;
    case RUNMED:
        type = 1;
        break;
    case RUNMIN:
        type = 2;
        break;
    case RUNMAX:
        type = 3;
        break;
    case RUNSTD:
        type = 4;
        break;
    }
    do_runavg(setno, rlen, type, -1, 0);
}

void do_fourier_command(int ftype, int setno, int ltype)
{
    // int type;

    switch (ftype)
    {
    case DFT:
        do_fourier(0, setno, 0, ltype, 0, 0, 0);
        break;
    case INVDFT:
        do_fourier(0, setno, 0, ltype, 1, 0, 0);
        break;
    case FFT:
        do_fourier(1, setno, 0, ltype, 0, 0, 0);
        break;
    case INVFFT:
        do_fourier(1, setno, 0, ltype, 1, 0, 0);
        break;
    }
}

void do_histo_command(int fromset, int toset, int tograph,
                      double minb, double binw, int nbins)
{
    do_histo(fromset, toset, tograph, binw, minb, minb + nbins * binw, 0);
}

/*
 * evaluate a formula
 */
void do_compute(int setno, int loadto, int graphto, char *fstr)
{
    int i, idraw = 0, itmp = setno;

    if (graphto < 0)
    {
        graphto = cg;
    }
    if (strlen(fstr) == 0)
    {
        errwin("Define formula first");
        return;
    }
    if (setno == SET_SELECT_ALL)
    {
        for (i = 0; i < g[cg].maxplot; i++)
        {
            if (isactive(cg, i))
            {
                if (formula(cg, i, fstr))
                {
                    sprintf(buf, "\nERROR: computing %s on set %d\n", fstr, i);
                    stufftext(buf, STUFF_START);
                    return;
                }
                sprintf(buf, "\nComputed %s on set %d\n", fstr, i);
                stufftext(buf, STUFF_START);
                idraw = 1;
            }
        }
    }
    else if (isactive(cg, setno))
    {
        /* both loadto and setno do double duty here */
        if (loadto)
        {
            loadto = nextset(graphto);
            if (loadto != -1)
            {
                do_copyset(cg, setno, graphto, loadto);
                setno = loadto;
            }
            else
            {
                return;
            }
        }
        else if (graphto != cg)
        {
            loadto = setno;
            if (isactive(graphto, loadto))
            {
                killset(graphto, loadto);
            }
            do_copyset(cg, setno, graphto, loadto);
            setno = loadto;
        }
        if (formula(graphto, setno, fstr))
        {
            sprintf(buf, "\nERROR: computing %s on set %d\n", fstr, setno);
            stufftext(buf, STUFF_START);
            if (loadto != setno)
            {
                killset(graphto, loadto);
            }
            return;
        }
        sprintf(buf, "\nComputed %s on set %d, result to set %d\n", fstr, itmp, setno);
        stufftext(buf, STUFF_START);
        if (!isactive_graph(graphto))
        {
            set_graph_active(graphto);
        }
        idraw = 1;
    }
    if (idraw)
    {
        drawgraph();
    }
    else
    {
        errwin("Set(s) not active");
    }
}

/*
 * load a set
 */
void do_load(int setno, int toval, char *startstr, char *stepstr)
{
    int i, ier = 0, idraw = 0;
    double x, y, a, b, c, d;
    extern double result;
    double start, step;
    char *s1, *s2;

    if (strlen(startstr) == 0)
    {
        errwin("Start item undefined");
        return;
    }
    /* fixupstr adds a newline, so add 2 rather than 1 */
    s1 = (char *)malloc((strlen(startstr) + 2) * sizeof(char));
    strcpy(s1, startstr);
    fixupstr(s1);
    scanner(s1, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &ier);
    if (ier)
    {
        free(s1);
        return;
    }
    start = result;

    if (strlen(stepstr) == 0)
    {
        errwin("Step item undefined");
        free(s1);
        return;
    }
    /* fixupstr adds a newline, so add 2 rather than 1 */
    s2 = (char *)malloc((strlen(stepstr) + 2) * sizeof(char));
    strcpy(s2, stepstr);
    fixupstr(s2);
    scanner(s2, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &ier);
    if (ier)
    {
        free(s1);
        free(s2);
        return;
    }
    step = result;

    if (setno == SET_SELECT_ALL)
    {
        for (i = 0; i < g[cg].maxplot; i++)
        {
            if (isactive(cg, i))
            {
                loadset(cg, i, toval, start, step);
                idraw = 1;
            }
        }
    }
    else if (isactive(cg, setno))
    {
        loadset(cg, setno, toval, start, step);
        idraw = 1;
    }
    if (idraw)
    {
        drawgraph();
    }
    else
    {
        errwin("Set(s) not active");
    }
    free(s1);
    free(s2);
}

/*
 * evaluate a formula loading the next set
 */
void do_compute2(char *fstrx, char *fstry, char *startstr, char *stopstr, int npts, int toval)
{
    int setno, ier;
    double start, stop, step, x, y, a, b, c, d;
    extern double result;

    if (npts < 1)
    {
        errwin("Number of points < 1");
        return;
    }
    /*
    * if npts is > maxarr, then increase length of scratch arrays
    */
    if (npts > maxarr)
    {
        if (init_scratch_arrays(npts))
        {
            return;
        }
    }
    setno = nextset(cg);
    if (setno < 0)
    {
        return;
    }
    activateset(cg, setno);
    setlength(cg, setno, npts);
    if (strlen(fstrx) == 0)
    {
        errwin("Undefined expression for X");
        return;
    }
    if (strlen(fstry) == 0)
    {
        errwin("Undefined expression for Y");
        return;
    }
    if (strlen(startstr) == 0)
    {
        errwin("Start item undefined");
        return;
    }
    fixupstr(startstr);
    scanner(startstr, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &ier);
    if (ier)
        return;
    start = result;

    if (strlen(stopstr) == 0)
    {
        errwin("Stop item undefined");
        return;
    }
    fixupstr(stopstr);
    scanner(stopstr, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &ier);
    if (ier)
    {
        return;
    }
    stop = result;

    if (npts - 1 == 0)
    {
        errwin("Number of points = 0");
        return;
    }
    step = (stop - start) / (npts - 1);
    loadset(cg, setno, toval, start, step);
    strcpy(buf, "X=");
    strcat(buf, fstrx);
    formula(cg, setno, buf);
    strcpy(buf, "Y=");
    strcat(buf, fstry);
    formula(cg, setno, buf);
    drawgraph();
}

/*
 * forward, backward and centered differences
 */
static void forwarddiff(double *x, double *y, double *resx, double *resy, int n)
{
    int i, eflag = 0;
    double h;

    for (i = 1; i < n; i++)
    {
        resx[i - 1] = x[i - 1];
        h = x[i - 1] - x[i];
        if (h == 0.0)
        {
            resy[i - 1] = MBIG;
            eflag = 1;
        }
        else
        {
            resy[i - 1] = (y[i - 1] - y[i]) / h;
        }
    }
    if (eflag)
    {
        errwin("Warning: infinite slope, check set status before proceeding");
    }
}

static void backwarddiff(double *x, double *y, double *resx, double *resy, int n)
{
    int i, eflag = 0;
    double h;

    for (i = 0; i < n - 1; i++)
    {
        resx[i] = x[i];
        h = x[i + 1] - x[i];
        if (h == 0.0)
        {
            resy[i] = MBIG;
            eflag = 1;
        }
        else
        {
            resy[i] = (y[i + 1] - y[i]) / h;
        }
    }
    if (eflag)
    {
        errwin("Warning: infinite slope, check set status before proceeding");
    }
}

static void centereddiff(double *x, double *y, double *resx, double *resy, int n)
{
    int i, eflag = 0;
    double h1, h2;

    for (i = 1; i < n - 1; i++)
    {
        resx[i - 1] = x[i];
        h1 = x[i] - x[i - 1];
        h2 = x[i + 1] - x[i];
        if (h1 + h2 == 0.0)
        {
            resy[i - 1] = MBIG;
            eflag = 1;
        }
        else
        {
            resy[i - 1] = (y[i + 1] - y[i - 1]) / (h1 + h2);
        }
    }
    if (eflag)
    {
        errwin("Warning: infinite slope, check set status before proceeding");
    }
}

static void seasonaldiff(double *x, double *y,
                         double *resx, double *resy, int n, int period)
{
    int i;

    for (i = 0; i < n - period; i++)
    {
        resx[i] = x[i];
        resy[i] = y[i] - y[i + period];
    }
}

/*
 * trapezoidal rule
 */
double trapint(double *x, double *y, double *resx, double *resy, int n)
{
    int i;
    double sum = 0.0;
    double h;

    for (i = 1; i < n; i++)
    {
        h = (x[i] - x[i - 1]);
        if (resx != NULL)
        {
            resx[i - 1] = (x[i - 1] + x[i]) * 0.5;
        }
        sum = sum + h * (y[i - 1] + y[i]) * 0.5;
        if (resy != NULL)
        {
            resy[i - 1] = sum;
        }
    }
    return sum;
}

/*
 * apply a digital filter
 */
void do_digfilter(int set1, int set2)
{
    int digfiltset;

    if (!(isactive(cg, set1) && isactive(cg, set2)))
    {
        errwin("Set not active");
        return;
    }
    if ((getsetlength(cg, set1) < 3) || (getsetlength(cg, set2) < 3))
    {
        errwin("Set length < 3");
        return;
    }
    digfiltset = nextset(cg);
    if (digfiltset != (-1))
    {
        activateset(cg, digfiltset);
        setlength(cg, digfiltset, getsetlength(cg, set1) - getsetlength(cg, set2) + 1);
        sprintf(buf, "Digital filter from set %d applied to set %d", set2, set1);
        filterser(getsetlength(cg, set1),
                  getx(cg, set1),
                  gety(cg, set1),
                  getx(cg, digfiltset),
                  gety(cg, digfiltset),
                  gety(cg, set2),
                  getsetlength(cg, set2));
        setcomment(cg, digfiltset, buf);
        log_results(buf);
        updatesetminmax(cg, digfiltset);
        update_set_status(cg, digfiltset);
        drawgraph();
    }
}

/*
 * linear convolution
 */
void do_linearc(int set1, int set2)
{
    int linearcset, i, itmp;
    double *xtmp;

    if (!(isactive(cg, set1) && isactive(cg, set2)))
    {
        errwin("Set not active");
        return;
    }
    if ((getsetlength(cg, set1) < 3) || (getsetlength(cg, set2) < 3))
    {
        errwin("Set length < 3");
        return;
    }
    linearcset = nextset(cg);
    if (linearcset != (-1))
    {
        activateset(cg, linearcset);
        setlength(cg, linearcset, (itmp = getsetlength(cg, set1) + getsetlength(cg, set2) - 1));
        linearconv(gety(cg, set2), gety(cg, set1), gety(cg, linearcset), getsetlength(cg, set2), getsetlength(cg, set1));
        xtmp = getx(cg, linearcset);
        for (i = 0; i < itmp; i++)
        {
            xtmp[i] = i;
        }
        sprintf(buf, "Linear convolution of set %d with set %d", set1, set2);
        setcomment(cg, linearcset, buf);
        log_results(buf);
        updatesetminmax(cg, linearcset);
        update_set_status(cg, linearcset);
        drawgraph();
    }
}

/*
 * cross correlation
 */
void do_xcor(int set1, int set2, int itype, int lag)
{
    int xcorset, i, ierr;
    double *xtmp;

    if (!(isactive(cg, set1) && isactive(cg, set2)))
    {
        errwin("Set not active");
        return;
    }
    if (lag == 0 || (getsetlength(cg, set1) - 1 < lag))
    {
        errwin("Lag incorrectly specified");
        return;
    }
    if ((getsetlength(cg, set1) < 3) || (getsetlength(cg, set2) < 3))
    {
        errwin("Set length < 3");
        return;
    }
    xcorset = nextset(cg);
    if (xcorset != (-1))
    {
        activateset(cg, xcorset);
        setlength(cg, xcorset, lag);
        if (set1 != set2)
        {
            sprintf(buf, "X-correlation of set %d and %d at lag %d", set1, set2, lag);
        }
        else
        {
            sprintf(buf, "Autocorrelation of set %d at lag %d", set1, lag);
        }
        ierr = crosscorr(gety(cg, set1), gety(cg, set2), getsetlength(cg, set1), lag, itype, getx(cg, xcorset), gety(cg, xcorset));
        xtmp = getx(cg, xcorset);
        for (i = 0; i < lag; i++)
        {
            xtmp[i] = i;
        }
        setcomment(cg, xcorset, buf);
        log_results(buf);
        updatesetminmax(cg, xcorset);
        update_set_status(cg, xcorset);
        drawgraph();
    }
}

/*
 * splines
 */
void do_spline(int set, double start, double stop, int n)
{
    int i, splineset, len;
    double delx, *x, *y, *b, *c, *d, *xtmp, *ytmp;
    double seval(int n, double u, double *x, double *y, double *b, double *c, double *d);

    if (!isactive(cg, set))
    {
        errwin("Set not active");
        return;
    }
    if ((len = getsetlength(cg, set)) < 3)
    {
        errwin("Improper set length");
        return;
    }
    if (n <= 1)
    {
        errwin("Number of steps must be > 1");
        return;
    }
    delx = (stop - start) / (n - 1);
    splineset = nextset(cg);
    if (splineset != -1)
    {
        activateset(cg, splineset);
        setlength(cg, splineset, n);
        sprintf(buf, "Spline fit from set %d", set);
        x = getx(cg, set);
        y = gety(cg, set);
        b = (double *)calloc(len, sizeof(double));
        c = (double *)calloc(len, sizeof(double));
        d = (double *)calloc(len, sizeof(double));
        if (b == NULL || c == NULL || d == NULL)
        {
            errwin("Not enough memory for splines");
            cxfree(b);
            cxfree(c);
            cxfree(d);
            killset(cg, splineset);
            return;
        }
        spline(len, x, y, b, c, d);
        xtmp = getx(cg, splineset);
        ytmp = gety(cg, splineset);

        for (i = 0; i < n; i++)
        {
            xtmp[i] = start + i * delx;
            ytmp[i] = seval(len, xtmp[i], x, y, b, c, d);
        }
        setcomment(cg, splineset, buf);
        log_results(buf);
        updatesetminmax(cg, splineset);
        update_set_status(cg, splineset);
        cxfree(b);
        cxfree(c);
        cxfree(d);
        drawgraph();
    }
}

void do_spline_command(int set, double start, double stop, int n)
{
    do_spline(set, start, stop, n);
}

/*
 * numerical integration
 */
double do_int(int setno, int itype)
{
    int intset;
    double sum = 0.0;

    if (!isactive(cg, setno))
    {
        errwin("Set not active");
        return 0.0;
    }
    if (getsetlength(cg, setno) < 3)
    {
        errwin("Set length < 3");
        return 0.0;
    }
    if (itype == 0)
    {
        intset = nextset(cg);
        if (intset != (-1))
        {
            activateset(cg, intset);
            setlength(cg, intset, getsetlength(cg, setno) - 1);
            sprintf(buf, "Cumulative sum of set %d", setno);
            sum = trapint(getx(cg, setno), gety(cg, setno), getx(cg, intset), gety(cg, intset), getsetlength(cg, setno));
            setcomment(cg, intset, buf);
            log_results(buf);
            updatesetminmax(cg, intset);
            update_set_status(cg, intset);
            drawgraph();
        }
        else
        {
            fprintf(stderr, "computils.cpp:do_int(): sum is used uninitialized\n");
        }
    }
    else
    {
        sum = trapint(getx(cg, setno), gety(cg, setno), NULL, NULL, getsetlength(cg, setno));
    }
    return sum;
}

/*
 * difference a set
 * itype means
 *  0 - forward
 *  1 - backward
 *  2 - centered difference
 */
void do_differ(int setno, int itype)
{
    int diffset;

    if (!isactive(cg, setno))
    {
        errwin("Set not active");
        return;
    }
    if (getsetlength(cg, setno) < 3)
    {
        errwin("Set length < 3");
        return;
    }
    diffset = nextset(cg);
    if (diffset != (-1))
    {
        activateset(cg, diffset);
        switch (itype)
        {
        case 0:
            sprintf(buf, "Forward difference of set %d", setno);
            setlength(cg, diffset, getsetlength(cg, setno) - 1);
            forwarddiff(getx(cg, setno), gety(cg, setno), getx(cg, diffset), gety(cg, diffset), getsetlength(cg, setno));
            break;
        case 1:
            sprintf(buf, "Backward difference of set %d", setno);
            setlength(cg, diffset, getsetlength(cg, setno) - 1);
            backwarddiff(getx(cg, setno), gety(cg, setno), getx(cg, diffset), gety(cg, diffset), getsetlength(cg, setno));
            break;
        case 2:
            sprintf(buf, "Centered difference of set %d", setno);
            setlength(cg, diffset, getsetlength(cg, setno) - 2);
            centereddiff(getx(cg, setno), gety(cg, setno), getx(cg, diffset), gety(cg, diffset), getsetlength(cg, setno));
            break;
        }
        setcomment(cg, diffset, buf);
        log_results(buf);
        updatesetminmax(cg, diffset);
        update_set_status(cg, diffset);
        drawgraph();
    }
}

/*
 * seasonally difference a set
 */
void do_seasonal_diff(int setno, int period)
{
    int diffset;

    if (!isactive(cg, setno))
    {
        errwin("Set not active");
        return;
    }
    if (getsetlength(cg, setno) < 2)
    {
        errwin("Set length < 2");
        return;
    }
    diffset = nextset(cg);
    if (diffset != (-1))
    {
        activateset(cg, diffset);
        setlength(cg, diffset, getsetlength(cg, setno) - period);
        seasonaldiff(getx(cg, setno), gety(cg, setno),
                     getx(cg, diffset), gety(cg, diffset),
                     getsetlength(cg, setno), period);
        sprintf(buf, "Seasonal difference of set %d, period %d", setno, period);
        setcomment(cg, diffset, buf);
        log_results(buf);
        updatesetminmax(cg, diffset);
        update_set_status(cg, diffset);
        drawgraph();
    }
}

/*
 * regression with restrictions to region rno if rno >= 0
 */
void do_regress(int setno, int ideg, int iresid, int rno, int invr)
{
    int len, fitset, i, sdeg = ideg, ifail;
    int cnt = 0;
    double *x, *y, *xt = NULL, *yt = NULL, *xr, *yr;
    // char rtype[256];
    char buf[256];

    if (!isactive(cg, setno))
    {
        errwin("Set not active");
        return;
    }
    len = getsetlength(cg, setno);
    x = getx(cg, setno);
    y = gety(cg, setno);
    if (rno == -1)
    {
        xt = x;
        yt = y;
    }
    else if (isactive_region(rno))
    {
        if (!get_points_inregion(rno, invr, len, x, y, &cnt, &xt, &yt))
        {
            if (cnt == 0)
            {
                errwin("No points found in region, operation cancelled");
            }
            else
            {
                errwin("Memory allocation failed for points in region");
            }
            return;
        }
        len = cnt;
    }
    else
    {
        errwin("Selected region is not active");
        return;
    }
    /*
    * first part for polynomials, second part for linear fits to transformed
    * data
    */
    if ((len < ideg && ideg <= 10) || (len < 2 && ideg > 10))
    {
        errwin("Too few points in set, operation cancelled");
        return;
    }
    fitset = nextset(cg);
    if (fitset != -1)
    {
        activateset(cg, fitset);
        setlength(cg, fitset, len);
        xr = getx(cg, fitset);
        yr = gety(cg, fitset);
        for (i = 0; i < len; i++)
        {
            xr[i] = xt[i];
        }
        if (ideg == 12) /* ln(y) = ln(A) + b * ln(x) */
        {
            ideg = 1;
            for (i = 0; i < len; i++)
            {
                if (xt[i] <= 0.0)
                {
                    errwin("One of X[i] <= 0.0");
                    return;
                }
                if (yt[i] <= 0.0)
                {
                    errwin("One of Y[i] <= 0.0");
                    return;
                }
            }
            for (i = 0; i < len; i++)
            {
                xt[i] = log(xt[i]);
                yt[i] = log(yt[i]);
            }
        }
        else if (ideg == 13)
        {
            ideg = 1;
            for (i = 0; i < len; i++)
            {
                if (yt[i] <= 0.0)
                {
                    errwin("One of Y[i] <= 0.0");
                    return;
                }
            }
            for (i = 0; i < len; i++)
            {
                yt[i] = log(yt[i]);
            }
        }
        else if (ideg == 14)
        {
            ideg = 1;
            for (i = 0; i < len; i++)
            {
                if (xt[i] <= 0.0)
                {
                    errwin("One of X[i] <= 0.0");
                    return;
                }
            }
            for (i = 0; i < len; i++)
            {
                xt[i] = log(xt[i]);
            }
        }
        else if (ideg == 15)
        {
            ideg = 1;
            for (i = 0; i < len; i++)
            {
                if (yt[i] == 0.0)
                {
                    errwin("One of Y[i] = 0.0");
                    return;
                }
            }
            for (i = 0; i < len; i++)
            {
                yt[i] = 1.0 / yt[i];
            }
        }

        ifail = fitcurve(xt, yt, len, ideg, yr);

        if (ifail)
        {
            killset(cg, fitset);
            goto bustout;
        }

        sprintf(buf, "\nRegression of set %d results to set %d\n", setno, fitset);
        stufftext(buf, STUFF_STOP);

        if (sdeg == 12) /* ln(y) = ln(A) + b * ln(x) */
        {
            for (i = 0; i < len; i++)
            {
                xt[i] = xr[i] = exp(xt[i]);
                yt[i] = exp(yt[i]);
                yr[i] = exp(yr[i]);
            }
        }
        else if (sdeg == 13)
        {
            for (i = 0; i < len; i++)
            {
                yt[i] = exp(yt[i]);
                yr[i] = exp(yr[i]);
            }
        }
        else if (sdeg == 14)
        {
            for (i = 0; i < len; i++)
            {
                xt[i] = xr[i] = exp(xt[i]);
            }
        }
        else if (sdeg == 15)
        {
            for (i = 0; i < len; i++)
            {
                yt[i] = 1.0 / yt[i];
                yr[i] = 1.0 / yr[i];
            }
        }
        switch (iresid)
        {
        case 1:
            for (i = 0; i < len; i++)
            {
                yr[i] = yt[i] - yr[i];
            }
            break;
        case 2:
            break;
        }
        sprintf(buf, "%d deg fit of set %d", ideg, setno);
        setcomment(cg, fitset, buf);
        log_results(buf);
        updatesetminmax(cg, fitset);
        update_set_status(cg, fitset);
    }
bustout:
    ;
    if (rno >= 0 && cnt != 0) /* had a region and allocated memory there */
    {
        cfree(xt);
        cfree(yt);
    }
}

/*
 * running averages, medians, min, max, std. deviation
 */
void do_runavg(int setno, int runlen, int runtype, int rno, int invr)
{
    int runset;
    int len, cnt = 0;
    double *x, *y, *xt = NULL, *yt = NULL, *xr, *yr;

    if (!isactive(cg, setno))
    {
        errwin("Set not active");
        return;
    }
    if (runlen < 2)
    {
        errwin("Length of running average < 2");
        return;
    }
    len = getsetlength(cg, setno);
    x = getx(cg, setno);
    y = gety(cg, setno);
    if (rno == -1)
    {
        xt = x;
        yt = y;
    }
    else if (isactive_region(rno))
    {
        if (!get_points_inregion(rno, invr, len, x, y, &cnt, &xt, &yt))
        {
            if (cnt == 0)
            {
                errwin("No points found in region, operation cancelled");
            }
            else
            {
                errwin("Memory allocation failed for points in region");
            }
            return;
        }
        len = cnt;
    }
    else
    {
        errwin("Selected region is not active");
        return;
    }
    if (runlen >= len)
    {
        errwin("Length of running average > set length");
        goto bustout;
    }
    runset = nextset(cg);
    if (runset != (-1))
    {
        activateset(cg, runset);
        setlength(cg, runset, len - runlen + 1);
        xr = getx(cg, runset);
        yr = gety(cg, runset);
        switch (runtype)
        {
        case 0:
            runavg(xt, yt, xr, yr, len, runlen);
            sprintf(buf, "%d-pt. avg. on set %d ", runlen, setno);
            break;
        case 1:
            runmedian(xt, yt, xr, yr, len, runlen);
            sprintf(buf, "%d-pt. median on set %d ", runlen, setno);
            break;
        case 2:
            runminmax(xt, yt, xr, yr, len, runlen, 0);
            sprintf(buf, "%d-pt. min on set %d ", runlen, setno);
            break;
        case 3:
            runminmax(xt, yt, xr, yr, len, runlen, 1);
            sprintf(buf, "%d-pt. max on set %d ", runlen, setno);
            break;
        case 4:
            runstddev(xt, yt, xr, yr, len, runlen);
            sprintf(buf, "%d-pt. std dev., set %d ", runlen, setno);
            break;
        }
        setcomment(cg, runset, buf);
        log_results(buf);
        updatesetminmax(cg, runset);
        update_set_status(cg, runset);
    }
bustout:
    ;
    if (rno >= 0 && cnt != 0) /* had a region and allocated memory there */
    {
        cfree(xt);
        cfree(yt);
    }
}

/*
 * DFT by FFT or definition
 */
void do_fourier(int fftflag, int setno, int load, int loadx, int invflag, int type, int wind)
{
    int i, ilen;
    double *x, *y, *xx, *yy, delt, T;
    int specset;

    if (!isactive(cg, setno))
    {
        errwin("Set not active");
        return;
    }
    ilen = getsetlength(cg, setno);
    if (ilen < 2)
    {
        errwin("Set length < 2");
        return;
    }
    int i2 = 0;
    if (fftflag)
    {
        i2 = ilog2(ilen);
        if (i2 <= 0)
        {
            errwin("Set length not a power of 2");
            return;
        }
    }
    specset = nextset(cg);
    if (specset != -1)
    {
        activateset(cg, specset);
        setlength(cg, specset, ilen);
        xx = getx(cg, specset);
        yy = gety(cg, specset);
        x = getx(cg, setno);
        y = gety(cg, setno);
        copyx(cg, setno, specset);
        copyy(cg, setno, specset);
        if (wind != 0) /* apply data window if needed */
        {
            apply_window(xx, yy, ilen, type, wind);
        }
        if (type == 0) /* real data */
        {
            for (i = 0; i < ilen; i++)
            {
                xx[i] = yy[i];
                yy[i] = 0.0;
            }
        }
        if (fftflag)
        {
            fft(xx, yy, ilen, i2, !invflag);
        }
        else
        {
            dft(xx, yy, ilen, invflag);
        }
        switch (load)
        {
        case 0:
            delt = x[1] - x[0];
            T = (ilen - 1) * delt;
            setlength(cg, specset, ilen / 2);
            xx = getx(cg, specset);
            yy = gety(cg, specset);
            for (i = 0; i < ilen / 2; i++)
            {
                yy[i] = my_hypot(xx[i], yy[i]);
                switch (loadx)
                {
                case 0:
                    xx[i] = i;
                    break;
                case 1:
                    /* xx[i] = 2.0 * M_PI * i / ilen; */
                    xx[i] = i / T;
                    break;
                case 2:
                    if (i == 0)
                    {
                        xx[i] = T + delt; /* the mean */
                    }
                    else
                    {
                        /* xx[i] = (double) ilen / (double) i; */
                        xx[i] = T / i;
                    }
                    break;
                }
            }
            break;
        case 1:
            delt = x[1] - x[0];
            T = (x[ilen - 1] - x[0]);
            setlength(cg, specset, ilen / 2);
            xx = getx(cg, specset);
            yy = gety(cg, specset);
            for (i = 0; i < ilen / 2; i++)
            {
                yy[i] = -atan2(yy[i], xx[i]);
                switch (loadx)
                {
                case 0:
                    xx[i] = i;
                    break;
                case 1:
                    /* xx[i] = 2.0 * M_PI * i / ilen; */
                    xx[i] = i / T;
                    break;
                case 2:
                    if (i == 0)
                    {
                        xx[i] = T + delt;
                    }
                    else
                    {
                        /* xx[i] = (double) ilen / (double) i; */
                        xx[i] = T / i;
                    }
                    break;
                }
            }
            break;
        }
        if (fftflag)
        {
            sprintf(buf, "FFT of set %d", setno);
        }
        else
        {
            sprintf(buf, "DFT of set %d", setno);
        }
        setcomment(cg, specset, buf);
        log_results(buf);
        updatesetminmax(cg, specset);
        update_set_status(cg, specset);
    }
}

/*
 * Apply a window to a set, result goes to a new set.
 */
void do_window(int setno, int type, int wind)
{
    int ilen;
    double *xx, *yy;
    int specset;

    if (!isactive(cg, setno))
    {
        errwin("Set not active");
        return;
    }
    ilen = getsetlength(cg, setno);
    if (ilen < 2)
    {
        errwin("Set length < 2");
        return;
    }
    specset = nextset(cg);
    if (specset != -1)
    {
        const char *wtype[6];
        wtype[0] = "Triangular";
        wtype[1] = "Hanning";
        wtype[2] = "Welch";
        wtype[3] = "Hamming";
        wtype[4] = "Blackman";
        wtype[5] = "Parzen";

        activateset(cg, specset);
        setlength(cg, specset, ilen);
        xx = getx(cg, specset);
        yy = gety(cg, specset);
        copyx(cg, setno, specset);
        copyy(cg, setno, specset);
        if (wind != 0)
        {
            apply_window(xx, yy, ilen, type, wind);
            sprintf(buf, "%s windowed set %d", wtype[wind - 1], setno);
        } /* shouldn't happen */
        else
        {
        }
        setcomment(cg, specset, buf);
        log_results(buf);
        updatesetminmax(cg, specset);
        update_set_status(cg, specset);
    }
}

void apply_window(double *xx, double *yy, int ilen, int type, int wind)
{
    int i;

    for (i = 0; i < ilen; i++)
    {
        switch (wind)
        {
        case 1: /* triangular */
            if (type != 0)
            {
                xx[i] *= 1.0 - fabs((i - 0.5 * (ilen - 1.0)) / (0.5 * (ilen - 1.0)));
            }
            yy[i] *= 1.0 - fabs((i - 0.5 * (ilen - 1.0)) / (0.5 * (ilen - 1.0)));

            break;
        case 2: /* Hanning */
            if (type != 0)
            {
                xx[i] = xx[i] * (0.5 - 0.5 * cos(2.0 * M_PI * i / (ilen - 1.0)));
            }
            yy[i] = yy[i] * (0.5 - 0.5 * cos(2.0 * M_PI * i / (ilen - 1.0)));
            break;
        case 3: /* Welch (from Numerical Recipes) */
            if (type != 0)
            {
                xx[i] *= 1.0 - pow((i - 0.5 * (ilen - 1.0)) / (0.5 * (ilen + 1.0)), 2.0);
            }
            yy[i] *= 1.0 - pow((i - 0.5 * (ilen - 1.0)) / (0.5 * (ilen + 1.0)), 2.0);
            break;
        case 4: /* Hamming */
            if (type != 0)
            {
                xx[i] = xx[i] * (0.54 - 0.46 * cos(2.0 * M_PI * i / (ilen - 1.0)));
            }
            yy[i] = yy[i] * (0.54 - 0.46 * cos(2.0 * M_PI * i / (ilen - 1.0)));
            break;
        case 5: /* Blackman */
            if (type != 0)
            {
                xx[i] = xx[i] * (0.42 - 0.5 * cos(2.0 * M_PI * i / (ilen - 1.0)) + 0.08 * cos(4.0 * M_PI * i / (ilen - 1.0)));
            }
            yy[i] = yy[i] * (0.42 - 0.5 * cos(2.0 * M_PI * i / (ilen - 1.0)) + 0.08 * cos(4.0 * M_PI * i / (ilen - 1.0)));
            break;
        case 6: /* Parzen (from Numerical Recipes) */
            if (type != 0)
            {
                xx[i] *= 1.0 - fabs((i - 0.5 * (ilen - 1)) / (0.5 * (ilen + 1)));
            }
            yy[i] *= 1.0 - fabs((i - 0.5 * (ilen - 1)) / (0.5 * (ilen + 1)));
            break;
        }
    }
}

/*
 * histograms
 */
void do_histo(int fromset, int toset, int tograph,
              double binw, double xmin, double xmax, int hist_type)
{
    if (!isactive(cg, fromset))
    {
        errwin("Set not active");
        return;
    }
    if (getsetlength(cg, fromset) <= 0)
    {
        errwin("Set length = 0");
        return;
    }
    if (binw <= 0.0)
    {
        errwin("Bin width <= 0");
        return;
    }
    if (tograph == -1)
    {
        tograph = cg;
    }
    if (g[tograph].active == OFF)
    {
        set_graph_active(tograph);
    }
    if (toset == SET_SELECT_NEXT)
    {
        toset = nextset(tograph);
        if (toset == -1)
        {
            return;
        }
    }
    else if (isactive_set(tograph, toset))
    {
        errwin("Target set not empty");
        return;
    }
    histogram(fromset, toset, tograph, binw, xmin, xmax, hist_type);
    drawgraph();
}

void histogram(int fromset, int toset, int tograph,
               double bins, double xmin, double xmax, int hist_type)
{
    int n, i, j, nbins;
    double sum = 0.0, spread, xi, *x, *y;
    int *ind;

    n = getsetlength(cg, fromset);
    spread = xmax - xmin;
    nbins = (int)(spread / bins);
    if (nbins <= 0)
    {
        errwin("No bins, no work to do");
        killset(tograph, toset);
        return;
    }
    ind = (int *)calloc(nbins, sizeof(int));
    if (ind == NULL)
    {
        errwin("Not enough memory for histogram");
        killset(tograph, toset);
        return;
    }
    activateset(tograph, toset);
    setlength(tograph, toset, nbins);
    j = 0;
    y = gety(cg, fromset);
    for (i = 0; i < n; i++)
    {
        xi = y[i];
        if (xi >= xmin && xi <= xmax)
        {
            j = (int)((xi - xmin) / bins);
            if (j < 0)
            {
                j = 0;
            }
            else
            {
                if (j >= nbins)
                {
                    j = nbins - 1;
                }
            }
            ind[j] = ind[j] + 1;
        }
    }
    x = getx(tograph, toset);
    y = gety(tograph, toset);
    for (i = 0; i < nbins; i++)
    {
        x[i] = i * bins + xmin;
        sum = sum * hist_type + ind[i]; /* hist_type = 0 => regular histo */
        y[i] = sum;
    }
    set_prop(tograph, SET, SETNUM, toset, SYMBOL, TYPE, SYM_HISTOX, 0);
    set_prop(tograph, SET, SETNUM, toset, LINESTYLE, 0, 0);
    updatesymbols(tograph, toset);
    updatesetminmax(tograph, toset);
    sprintf(buf, "Histogram from set # %d", fromset);
    setcomment(tograph, toset, buf);
    log_results(buf);
    update_set_status(tograph, toset);
    cxfree(ind);
    drawgraph();
}

/*
 * sample a set, by start/step or logical expression
 */
void do_sample(int setno, int typeno, char *exprstr, int startno, int stepno)
{
    int len, npts = 0, i, resset, ier;
    double *x, *y;
    double a, b, c, d;
    extern double result;

    if (ismaster)
        if (!isactive(cg, setno))
        {
            errwin("Set not active");
            return;
        }
    len = getsetlength(cg, setno);
    resset = nextset(cg);
    if (resset < 0)
    {
        return;
    }
    if (typeno == 0)
    {
        if (len <= 2)
        {
            errwin("Set has <= 2 points");
            return;
        }
        if (startno < 1)
        {
            errwin("Start point < 1 (locations in sets are numbered starting from 1)");
            return;
        }
        if (stepno < 1)
        {
            errwin("Step < 1");
            return;
        }
        x = getx(cg, setno);
        y = gety(cg, setno);
        for (i = startno - 1; i < len; i += stepno)
        {
            add_point(cg, resset, x[i], y[i], 0.0, 0.0, XY);
            npts++;
        }
        sprintf(buf, "Sample, %d, %d set #%d", startno, stepno, setno);
    }
    else
    {
        if (!strlen(exprstr))
        {
            errwin("Enter logical expression first");
            return;
        }
        x = getx(cg, setno);
        y = gety(cg, setno);
        npts = 0;
        fixupstr(exprstr);
        for (i = 0; i < len; i++)
        {
            scanner(exprstr, &x[i], &y[i], 1, &a, &b, &c, &d, 1, i, setno, &ier);
            if (ier)
            {
                killset(cg, resset);
                return;
            }
            if ((int)result)
            {
                add_point(cg, resset, x[i], y[i], 0.0, 0.0, XY);
                npts++;
            }
        }
        if (npts > 0)
        {
            sprintf(buf, "Sample from %d, using '%s'", setno, exprstr);
        }
    }
    if (npts > 0)
    {
        updatesetminmax(cg, resset);
        setcomment(cg, resset, buf);
        log_results(buf);
        update_set_status(cg, resset);
        drawgraph();
    }
}

int get_points_inregion(int rno, int invr, int len, double *x, double *y, int *cnt, double **xt, double **yt)
{
    int i, clen = 0;
    double *xtmp, *ytmp;
    *cnt = 0;
    if (isactive_region(rno))
    {
        for (i = 0; i < len; i++)
        {
            if (invr)
            {
                if (!inregion(rno, x[i], y[i]))
                {
                    clen++;
                }
            }
            else
            {
                if (inregion(rno, x[i], y[i]))
                {
                    clen++;
                }
            }
        }
        if (clen == 0)
        {
            return 0;
        }
        xtmp = (double *)calloc(clen, sizeof(double));
        if (xtmp == NULL)
        {
            return 0;
        }
        ytmp = (double *)calloc(clen, sizeof(double));
        if (ytmp == NULL)
        {
            cfree(xtmp);
            return 0;
        }
        clen = 0;
        for (i = 0; i < len; i++)
        {
            if (invr)
            {
                if (!inregion(rno, x[i], y[i]))
                {
                    xtmp[clen] = x[i];
                    ytmp[clen] = y[i];
                    clen++;
                }
            }
            else
            {
                if (inregion(rno, x[i], y[i]))
                {
                    xtmp[clen] = x[i];
                    ytmp[clen] = y[i];
                    clen++;
                }
            }
        }
    }
    else
    {
        return 0;
    }
    *cnt = clen;
    *xt = xtmp;
    *yt = ytmp;
    return 1;
}

void do_ntiles(int, int, int)
{
}
