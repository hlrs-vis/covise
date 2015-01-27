/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: setutils.c,v 1.10 1994/10/30 07:26:14 pturner Exp pturner $
 *
 * routines to allocate, manipulate, and return
 * information about sets.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "globals.h"
#include "noxprotos.h"

extern int yesno(const char *msg1, const char *s1, const char *s2, const char *helptext);
extern void errwin(const char *s);
extern "C" {
extern void cfree(void *);
}

extern void set_plotstr_string(plotstr *pstr, char *buf);
extern void updatesymbols(int gno, int value);
extern void updatelegendstr(int gno);
extern void update_set_status(int gno, int setno);
extern void drawgraph(void);
extern void log_results(const char *buf);
extern void set_prop(int gno, ...);
extern void set_wait_cursor();
extern void unset_wait_cursor();
extern void set_action(int act);

#define min(a, b) ((a) <= (b) ? (a) : (b))
#define max(a, b) ((a) >= (b) ? (a) : (b))

/* static int default_mono[MAXPLOT] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
}; */
static int default_color[MAXPLOT] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15
};
/* static int default_symbol[MAXPLOT] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static int default_linestyle[MAXPLOT] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};*/

int index_set_types[] = { XY, XYDX, XYDY, XYDXDX, XYDYDY, XYDXDY, XYZ, XYHILO, XYRT, XYUV, XYBOX, XYBOXPLOT, -1 };
int index_set_ncols[] = { 2, 3, 3, 4, 4, 4, 3, 5, 3, 4, 5, -1 };

/*
 * return the string version of the set type
 */
char *set_types(int it)
{
    const char *s = "XY";

    switch (it)
    {
    case XY:
        s = "xy";
        break;
    case XYDX:
        s = "xydx";
        break;
    case XYDY:
        s = "xydy";
        break;
    case XYDYDY:
        s = "xydydy";
        break;
    case XYDXDX:
        s = "xydxdx";
        break;
    case XYDXDY:
        s = "xydxdy";
        break;
    case XYZ:
        s = "xyz";
        break;
    case XYHILO:
        s = "xyhilo";
        break;
    case XYBOX:
        s = "xybox";
        break;
    case XYRT:
        s = "xyrt";
        break;
    case XYUV:
        s = "xyuv";
        break;
    case XYBOXPLOT:
        s = "xyboxplot";
        break;
    }
    return (char *)s;
}

/*
 * needed as initplots is called before
 * the number of planes is determined
 */
void setdefaultcolors(int gno)
{
    int i;

    for (i = 0; i < maxplot; i++)
    {
        g[gno].p[i].color = default_color[i % MAXPLOT];
    }
}

/*
 * allocate arrays for a set of length len.
 */
void allocxy(plotarr *p, int len)
{
    int i, ncols = 0;

    switch (p->type)
    {
    case XY:
        ncols = 2;
        break;
    case XYDX:
    case XYDY:
    case XYZ:
        ncols = 3;
        break;
    case XYDXDX:
    case XYDYDY:
    case XYDXDY:
    case XYZW:
    case XYRT:
    case XYUV:
    case XYX2Y2:
    case XYARC:
    case XYYY:
    case XYXX:
        ncols = 4;
        break;
    case XYHILO:
    case XYBOX:
        ncols = 5;
        break;
    case XYBOXPLOT:
        ncols = 6;
        break;
    }
    if (ncols == 0)
    {
        fprintf(stderr, "Set type not found in setutils.c:allocxy()!!\n");
        return;
    }
    for (i = 0; i < ncols; i++)
    {
        if (p->ex[i] == NULL)
        {
            if ((p->ex[i] = (double *)calloc(len, sizeof(double))) == NULL)
            {
                fprintf(stderr, "Insufficient memory to allocate for plots\n");
                exit(1);
            }
        }
        else
        {
            if ((p->ex[i] = (double *)realloc(p->ex[i], len * sizeof(double))) == NULL)
            {
                fprintf(stderr, "Insufficient memory to allocate for plots\n");
                exit(1);
            }
        }
    }
    p->len = len;
}

int init_array(double **a, int n)
{
    if (*a != NULL)
    {
        *a = (double *)realloc(*a, n * sizeof(double));
    }
    else
    {
        *a = (double *)calloc(n, sizeof(double));
    }
    return *a == NULL ? 1 : 0;
}

int init_scratch_arrays(int n)
{
    if (!init_array(&ax, n))
    {
        if (!init_array(&bx, n))
        {
            if (!init_array(&cx, n))
            {
                if (!init_array(&dx, n))
                {
                    maxarr = n;
                    return 0;
                }
                free(cx);
            }
            free(bx);
        }
        free(ax);
    }
    return 1;
}

/*
 * get the min/max fields of a set
 */
void getsetminmax(int gno, int setno, double *x1, double *x2, double *y1, double *y2)
{
    *x1 = g[gno].p[setno].xmin;
    *x2 = g[gno].p[setno].xmax;
    *y1 = g[gno].p[setno].ymin;
    *y2 = g[gno].p[setno].ymax;
}

/*
 * get a bounding box for the set
 * over all columns
 */
void getminmaxall(int gno, int setno)
{
    int i, n, ncols;
    double *x;

    ncols = getncols(gno, setno);
    n = getsetlength(gno, setno);
    for (i = 0; i < ncols; i++)
    {

        if (n == 0)
        {
            g[gno].p[setno].emin[i] = 0.0;
            g[gno].p[setno].emax[i] = 0.0;
            g[gno].p[setno].imin[i] = 0;
            g[gno].p[setno].imax[i] = 0;
        }
        else
        {
            x = getcol(gno, setno, i);
            minmax(x, n, &g[gno].p[setno].emin[i],
                   &g[gno].p[setno].emax[i],
                   &g[gno].p[setno].imin[i],
                   &g[gno].p[setno].imax[i]);
        }
    }
}

/*
 * compute the mins and maxes of a vector x
 */
void minmax(double *x, int n, double *xmin, double *xmax, int *imin, int *imax)
{
    int i;
    *xmin = x[0];
    *xmax = x[0];
    *imin = 1;
    *imax = 1;
    for (i = 1; i < n; i++)
    {
        if (x[i] < *xmin)
        {
            *xmin = x[i];
            *imin = i + 1;
        }
        if (x[i] > *xmax)
        {
            *xmax = x[i];
            *imax = i + 1;
        }
    }
}

/*
 * compute the mins and maxes of a vector x
 */
double vmin(double *x, int n)
{
    int i;
    double xmin;
    if (n <= 0)
    {
        return 0.0;
    }
    xmin = x[0];
    for (i = 1; i < n; i++)
    {
        if (x[i] < xmin)
        {
            xmin = x[i];
        }
    }
    return xmin;
}

double vmax(double *x, int n)
{
    int i;
    double xmax;
    if (n <= 0)
    {
        return 0.0;
    }
    xmax = x[0];
    for (i = 1; i < n; i++)
    {
        if (x[i] > xmax)
        {
            xmax = x[i];
        }
    }
    return xmax;
}

void getsetdxdyminmax(int gno, int setno, double *dx1, double *dx2, double *dy1, double *dy2)
{
    int itmp;

    if (getcol(gno, setno, 2) != NULL)
    {
        minmax(getcol(gno, setno, 2), getsetlength(gno, setno), dx1, dx2, &itmp, &itmp);
    }
    if (getcol(gno, setno, 3) != NULL)
    {
        minmax(getcol(gno, setno, 3), getsetlength(gno, setno), dy1, dy2, &itmp, &itmp);
    }
}

/*
 * compute the mins/maxes and update the appropriate fields of
 * set i.
 */
void updatesetminmax(int gno, int setno)
{
    double *tmp = (double *)NULL;
    double b1, b2;
    int i, n, itmp1, itmp2;

    if (isactive_set(gno, setno))
    {
        n = getsetlength(gno, setno);
        /* compute min/max for each column in the set */
        getminmaxall(gno, setno);
        /* compute global min max (applies over all columns) */
        g[gno].p[setno].xmin = g[gno].p[setno].emin[0];
        g[gno].p[setno].xmax = g[gno].p[setno].emax[0];
        g[gno].p[setno].ymin = g[gno].p[setno].emin[1];
        g[gno].p[setno].ymax = g[gno].p[setno].emax[1];
        tmp = (double *)calloc(getsetlength(gno, setno), sizeof(double));
        if (tmp == (double *)NULL)
        {
            errwin("Error: Unable to malloc temporary in updatesetminmax()");
            return;
        }
        switch (g[gno].p[setno].type)
        {
        case XY:
            break;
        case XYDX:
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[0][i] + g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].xmin = min(b1, g[gno].p[setno].xmin);
            g[gno].p[setno].xmax = max(b2, g[gno].p[setno].xmax);
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[0][i] - g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].xmin = min(b1, g[gno].p[setno].xmin);
            g[gno].p[setno].xmax = max(b2, g[gno].p[setno].xmax);
            break;
        case XYDY:
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[1][i] + g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[1][i] - g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            break;
        case XYDXDX:
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[0][i] + g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].xmin = min(b1, g[gno].p[setno].xmin);
            g[gno].p[setno].xmax = max(b2, g[gno].p[setno].xmax);
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[0][i] - g[gno].p[setno].ex[3][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].xmin = min(b1, g[gno].p[setno].xmin);
            g[gno].p[setno].xmax = max(b2, g[gno].p[setno].xmax);
            break;
        case XYDYDY:
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[1][i] + g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[1][i] - g[gno].p[setno].ex[3][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            break;
        case XYDXDY:
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[0][i] + g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].xmin = min(b1, g[gno].p[setno].xmin);
            g[gno].p[setno].xmax = max(b2, g[gno].p[setno].xmax);
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[0][i] - g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].xmin = min(b1, g[gno].p[setno].xmin);
            g[gno].p[setno].xmax = max(b2, g[gno].p[setno].xmax);
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[1][i] + g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            for (i = 0; i < n; i++)
            {
                tmp[i] = g[gno].p[setno].ex[1][i] - g[gno].p[setno].ex[2][i];
            }
            minmax(tmp, n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            break;
        case XYZW:
            break;
        case XYRT:
            break;
        case XYUV:
            break;
        case XYX2Y2:
            break;
        case XYBOX:
            minmax(g[gno].p[setno].ex[2], n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].xmin = min(b1, g[gno].p[setno].xmin);
            g[gno].p[setno].xmax = max(b2, g[gno].p[setno].xmax);
            minmax(g[gno].p[setno].ex[3], n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            break;
        case XYARC:
            break;
        case XYYY:
            break;
        case XYXX:
            break;
        case XYHILO:
            minmax(g[gno].p[setno].ex[2], n, &b1, &b2, &itmp1, &itmp2);
            g[gno].p[setno].ymin = min(b1, g[gno].p[setno].ymin);
            g[gno].p[setno].ymax = max(b2, g[gno].p[setno].ymax);
            break;
        case XYBOXPLOT:
            g[gno].p[setno].ymin = g[gno].p[setno].emin[4];
            g[gno].p[setno].ymax = g[gno].p[setno].emax[5];
            break;
        }
        if (tmp)
        {
            cfree(tmp);
        }
    }
    else
    {
        g[gno].p[setno].xmin = 0.0;
        g[gno].p[setno].xmax = 0.0;
        g[gno].p[setno].ymin = 0.0;
        g[gno].p[setno].ymax = 0.0;
    }
}

void set_point(int gno, int setn, int seti, double wx, double wy)
{
    g[gno].p[setn].ex[0][seti] = wx;
    g[gno].p[setn].ex[1][seti] = wy;
    updatesetminmax(gno, setn);
}

void get_point(int gno, int setn, int seti, double *wx, double *wy)
{
    *wx = g[gno].p[setn].ex[0][seti];
    *wy = g[gno].p[setn].ex[1][seti];
}

void setcol(int gno, double *x, int setno, int len, int col)
{
    g[gno].p[setno].ex[col] = x;
    g[gno].p[setno].len = len;
}

int getncols(int gno, int setno)
{
    int i = 0;

    while (g[gno].p[setno].ex[i])
    {
        i++;
    }
    return i;
}

void *geteditpoints(int gno, int setno)
{
    return g[gno].p[setno].ep;
}

void setxy(int gno, double **ex, int setno, int len, int ncols)
{
    int i;

    for (i = 0; i < ncols; i++)
    {
        g[gno].p[setno].ex[i] = ex[i];
    }
    g[gno].p[setno].len = len;
}

void setlength(int gno, int i, int length)
{
    allocxy(&g[gno].p[i], length);
}

void copycol(int gno, int setfrom, int setto, int col)
{
    int i, n;
    double *x1, *x2;

    n = g[gno].p[setfrom].len;
    x1 = getcol(gno, setfrom, col);
    x2 = getcol(gno, setto, col);
    for (i = 0; i < n; i++)
    {
        x2[i] = x1[i];
    }
}

void copycol2(int gfrom, int setfrom, int gto, int setto, int col)
{
    int i, n;
    double *x1, *x2;

    n = g[gfrom].p[setfrom].len;
    x1 = getcol(gfrom, setfrom, col);
    x2 = getcol(gto, setto, col);
    for (i = 0; i < n; i++)
    {
        x2[i] = x1[i];
    }
}

/*
 * moveset assumes both sets exist, have their length
 * properly set, and that they are both active
 */
void moveset(int gnofrom, int setfrom, int gnoto, int setto)
{
    int k;

    memcpy(&g[gnoto].p[setto], &g[gnofrom].p[setfrom], sizeof(plotarr));
    for (k = 0; k < MAX_SET_COLS; k++)
    {
        g[gnofrom].p[setfrom].ex[k] = NULL;
    }
}

/*
 * copyset assumes both sets exist, have their length
 * properly set, and that they are both active
 */
void copyset(int gnofrom, int setfrom, int gnoto, int setto)
{
    int k;
    double *savec[MAX_SET_COLS];
    int len = getsetlength(gnofrom, setfrom);

    for (k = 0; k < MAX_SET_COLS; k++)
    {
        savec[k] = g[gnoto].p[setto].ex[k];
    }
    memcpy(&g[gnoto].p[setto], &g[gnofrom].p[setfrom], sizeof(plotarr));
    for (k = 0; k < MAX_SET_COLS; k++)
    {
        g[gnoto].p[setto].ex[k] = savec[k];
        if (g[gnofrom].p[setfrom].ex[k] != NULL && g[gnoto].p[setto].ex[k] != NULL)
        {
            memcpy(g[gnoto].p[setto].ex[k], g[gnofrom].p[setfrom].ex[k], len * sizeof(double));
        }
    }
}

/*
 * copy everything but the data
 */
void copysetprops(int gnofrom, int setfrom, int gnoto, int setto)
{
    int k;
    double *savec[MAX_SET_COLS];

    for (k = 0; k < MAX_SET_COLS; k++)
    {
        savec[k] = g[gnoto].p[setto].ex[k];
    }
    memcpy(&g[gnoto].p[setto], &g[gnofrom].p[setfrom], sizeof(plotarr));
    for (k = 0; k < MAX_SET_COLS; k++)
    {
        g[gnoto].p[setto].ex[k] = savec[k];
    }
}

/*
 * copy data only
 */
void copysetdata(int gnofrom, int setfrom, int gnoto, int setto)
{
    int k;
    int len = getsetlength(gnofrom, setfrom);

    for (k = 0; k < MAX_SET_COLS; k++)
    {
        if (g[gnofrom].p[setfrom].ex[k] != NULL && g[gnoto].p[setto].ex[k] != NULL)
        {
            memcpy(g[gnoto].p[setto].ex[k], g[gnofrom].p[setfrom].ex[k], len * sizeof(double));
        }
    }
}

/*
 * pack all sets leaving no gaps in the set structure
 */
void packsets(int gno)
{
    int i, j, k;

    i = 0;
    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (isactive_set(gno, i))
        {
            j = 0;
            while (j < i)
            {
                if (!isactive_set(gno, j))
                {
                    memcpy(&g[gno].p[j], &g[gno].p[i], sizeof(plotarr));
                    if (j < MAXPLOT && i < MAXPLOT)
                    {
                        set_plotstr_string(&g[gno].l.str[j], g[gno].l.str[i].s);
                    }
                    for (k = 0; k < MAX_SET_COLS; k++)
                    {
                        g[gno].p[i].ex[k] = NULL;
                    }
                    killset(gno, i);
                    updatesymbols(gno, j);
                    updatesymbols(gno, i);
                    updatelegendstr(gno);
                    updatesetminmax(gno, j);
                    updatesetminmax(gno, i);
                    update_set_status(gno, j);
                    update_set_status(gno, i);
                }
                j++;
            }
        }
    }
}

/*
 * action proc for menu item
 */
void do_packsets(void)
{
    packsets(cg);
}

/*
 * return the next available set in graph gno
 * ignoring deactivated sets.
 */
int nextset(int gno)
{
    int i;

    i = 0;
    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (!isactive_set(gno, i) && !g[gno].p[i].deact)
        {
            return (i);
        }
    }
    errwin("Error - no sets available");
    return (-1);
}

/*
 * kill a set
 */
void killset(int gno, int setno)
{
#ifdef HAS_XBAE
    if (g[gno].p[setno].ep != NULL)
    {
        epdtor(g[gno].p[setno].ep);
        g[gno].p[setno].ep = NULL;
    }
#endif
    set_default_plotarr(&g[gno].p[setno]);
    g[gno].p[setno].active = OFF;
    g[gno].p[setno].deact = 0; /* just in case */
}

/*
 * kill a set, but preserve the parameter settings
 */
void softkillset(int gno, int setno)
{
    int i;

#ifdef HAS_XBAE
    if (g[gno].p[setno].ep != NULL)
    {
        epdtor(g[gno].p[setno].ep);
        g[gno].p[setno].ep = NULL;
    }
#endif
    for (i = 0; i < MAX_SET_COLS; i++)
    {
        if (g[gno].p[setno].ex[i] != NULL)
        {
            cfree(g[gno].p[setno].ex[i]);
        }
        g[gno].p[setno].ex[i] = NULL;
    }
    g[gno].p[setno].active = OFF;
    g[gno].p[setno].deact = 0;
}

/*
 * activate a set
 */
void activateset(int gno, int setno)
{
    g[gno].p[setno].active = ON;
    g[gno].p[setno].deact = 0;
    g[gno].p[setno].gno = gno;
    g[gno].p[setno].setno = setno;
    sprintf(g[gno].p[setno].name, "S%1d", setno);
}

/*
 * return the active status of a set
 */
int activeset(int gno)
{
    int i;

    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (g[gno].p[i].active == ON)
        {
            return (1);
        }
    }
    return (0);
}

/*
 * drop points from a set
 */
void droppoints(int gno, int setno, int, int endno, int dist)
{
    double *x;
    int i, j, len, ncols;

    len = getsetlength(gno, setno);
    ncols = getncols(gno, setno);
    for (j = 0; j < ncols; j++)
    {
        x = getcol(gno, setno, j);
        for (i = endno + 1; i < len; i++)
        {
            x[i - dist] = x[i];
        }
    }
    setlength(gno, setno, len - dist);
}

/*
 * join 2 sets together
 */
void joinsets(int g1, int j1, int g2, int j2)
{
    int i, j, len1, len2, ncols1, ncols2, ncols;
    double *x1, *x2;

    len1 = getsetlength(g1, j1);
    len2 = getsetlength(g2, j2);
    setlength(g2, j2, len1 + len2);
    ncols1 = getncols(g1, j1);
    ncols2 = getncols(g2, j2);
    ncols = (ncols2 < ncols1) ? ncols2 : ncols1;
    for (j = 0; j < ncols; j++)
    {
        x1 = getcol(g1, j1, j);
        x2 = getcol(g2, j2, j);
        for (i = len2; i < len2 + len1; i++)
        {
            x2[i] = x1[i - len2];
        }
    }
}

/*
 * sort a set
 */

static double *vptr;

/*
 * for ascending and descending sorts
 */
int compare_points1(int *i, int *j)
{
    if (vptr[*i] < vptr[*j])
    {
        return -1;
    }
    if (vptr[*i] > vptr[*j])
    {
        return 1;
    }
    return 0;
}

int compare_points2(int *i, int *j)
{
    if (vptr[*i] > vptr[*j])
    {
        return -1;
    }
    if (vptr[*i] < vptr[*j])
    {
        return 1;
    }
    return 0;
}

void sortset(int gno, int setno, int sorton, int stype)
{
    int i, j, nc, len, *ind;
    double *dtmp, *stmp;
    double *getvptr(int gno, int setno, int v);

    /*
    * get the vector to sort on
    */
    vptr = getvptr(gno, setno, sorton);
    if (vptr == NULL)
    {
        errwin("NULL vector in sort, operation cancelled, check set type");
    }
    len = getsetlength(gno, setno);
    if (len <= 1)
    {
        errwin("Setlength <= 1, nothing to do!");
    }
    /*
    * allocate memory for permuted indices
    */
    ind = (int *)calloc(len, sizeof(int));
    if (ind == NULL)
    {
        errwin("Unable to allocate memory for sort");
        return;
    }
    /*
    * allocate memory for temporary array
    */
    dtmp = (double *)calloc(len, sizeof(double));
    if (dtmp == NULL)
    {
        cfree(ind);
        errwin("Unable to allocate memory for sort");
        return;
    }
    /*
    * initialize indices
    */
    for (i = 0; i < len; i++)
    {
        ind[i] = i;
    }

    /*
    * sort
    */
    qsort(ind, len, sizeof(int), stype ? (int (*)(const void *, const void *))compare_points1 : (int (*)(const void *, const void *))compare_points2);

    /*
    * straighten things out - done one vector at a time for storage.
    */
    nc = getncols(gno, setno);
    /* loop over the number of columns */
    for (j = 0; j < nc; j++)
    {
        /* get this vector and put into the temporary vector in the right order */
        stmp = getcol(gno, setno, j);
        for (i = 0; i < len; i++)
        {
            dtmp[i] = stmp[ind[i]];
        }
        /* load it back to the set */
        for (i = 0; i < len; i++)
        {
            stmp[i] = dtmp[i];
        }
    }
}

/*
 * sort a set - only does type XY
 */
void sort_xy(double *tmp1, double *tmp2, int up, int sorton, int stype)
{

    int d, i, j;
    int lo = 0;
    double t1, t2;

    if (sorton == 1)
    {
        double *ttmp;

        ttmp = tmp1;
        tmp1 = tmp2;
        tmp2 = ttmp;
    }
    up--;

    for (d = up - lo + 1; d > 1;)
    {
        if (d < 5)
            d = 1;
        else
            d = (5 * d - 1) / 11;
        for (i = up - d; i >= lo; i--)
        {
            t1 = tmp1[i];
            t2 = tmp2[i];
            if (!stype)
            {
                for (j = i + d; j <= up && (t1 > tmp1[j]); j += d)
                {
                    tmp1[j - d] = tmp1[j];
                    tmp2[j - d] = tmp2[j];
                }
                tmp1[j - d] = t1;
                tmp2[j - d] = t2;
            }
            else
            {
                for (j = i + d; j <= up && (t1 < tmp1[j]); j += d)
                {
                    tmp1[j - d] = tmp1[j];
                    tmp2[j - d] = tmp2[j];
                }
                tmp1[j - d] = t1;
                tmp2[j - d] = t2;
            }
        }
    }
}

/*
 * locate a point and the set the point is in
 */
void findpoint(int gno, double x, double y, double *xs, double *ys, int *setno, int *loc)
{
    double dx = g[gno].w.xg2 - g[gno].w.xg1, dy = g[gno].w.yg2 - g[gno].w.yg1, *xtmp, *ytmp, tmp, tmin = 1.0e307;
    int i, j, len;

    *setno = -1;
    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (isactive(gno, i))
        {
            xtmp = getx(gno, i);
            ytmp = gety(gno, i);
            len = getsetlength(gno, i);
            for (j = 0; j < len; j++)
            {
                if ((tmp = my_hypot((x - xtmp[j]) / dx, (y - ytmp[j]) / dy)) < tmin)
                {
                    *setno = i;
                    *loc = j + 1;
                    *xs = xtmp[j];
                    *ys = ytmp[j];
                    tmin = tmp;
                }
            }
        }
    }
}

/*
 * locate a point in setno nearest (x, y)
 */
void findpoint_inset(int gno, int setno, double x, double y, int *loc)
{
    double dx = g[gno].w.xg2 - g[gno].w.xg1, dy = g[gno].w.yg2 - g[gno].w.yg1, *xtmp, *ytmp, tmp, tmin = 1.0e307;
    int j, len;

    if (isactive(gno, setno))
    {
        xtmp = getx(gno, setno);
        ytmp = gety(gno, setno);
        len = getsetlength(gno, setno);
        for (j = 0; j < len; j++)
        {
            if ((tmp = my_hypot((x - xtmp[j]) / dx, (y - ytmp[j]) / dy)) < tmin)
            {
                *loc = j + 1;
                tmin = tmp;
            }
        }
    }
    else
    {
        *loc = -1;
    }
}

/*
 * delete the point pt in setno
 */
void del_point(int gno, int setno, int pt)
{
    int i, j, len, ncols;
    double *tmp;

    ncols = getncols(gno, setno);
    len = getsetlength(gno, setno);
    if (pt > len)
    {
        return;
    }
    if (pt != len)
    {
        for (i = pt - 1; i < len - 1; i++)
        {
            for (j = 0; j < ncols; j++)
            {
                tmp = g[gno].p[setno].ex[j];
                tmp[i] = tmp[i + 1];
            }
        }
    }
    if (len > 1)
    {
        setlength(gno, setno, len - 1);
    }
    else
    {
        softkillset(gno, setno);
    }
    updatesetminmax(gno, setno);
    drawgraph();
}

/*
 * add a point to setno
 */
void add_point(int gno, int setno, double px, double py, double, double, int type)
{
    int len = 0;
    double *x, *y;

    if (isactive(gno, setno))
    {
        x = getx(gno, setno);
        y = gety(gno, setno);
        len = getsetlength(gno, setno);
        x = (double *)realloc(x, (len + 1) * sizeof(double));
        y = (double *)realloc(y, (len + 1) * sizeof(double));
        setcol(gno, x, setno, len + 1, 0);
        setcol(gno, y, setno, len + 1, 1);
        x[len] = px;
        y[len] = py;
    }
    else
    {
        g[gno].active = ON;
        activateset(gno, setno);
        g[gno].p[setno].type = type;
        allocxy(&g[gno].p[setno], 1);
        x = getx(gno, setno);
        y = gety(gno, setno);
        x[0] = px;
        y[0] = py;
    }
    updatesetminmax(gno, setno);
}

/*
 * add a point to setno after or before ind
 */
void add_point_at(int gno, int setno, int ind, int where, double px, double py, double, double, int type)
{
    int i, len = 0;
    double *x, *y;

    len = getsetlength(gno, setno);
    if (isactive(gno, setno) && len > 0)
    {
        x = getx(gno, setno);
        y = gety(gno, setno);
        x = (double *)realloc(x, (len + 1) * sizeof(double));
        y = (double *)realloc(y, (len + 1) * sizeof(double));
        setcol(gno, x, setno, len + 1, 0);
        setcol(gno, y, setno, len + 1, 1);
        if (where) /* add after ind */
        {
            for (i = len - 1; i > ind; i--)
            {
                x[i + 1] = x[i];
                y[i + 1] = y[i];
            }
            x[ind + 1] = px;
            y[ind + 1] = py;
        } /* add point before ind (at ind) */
        else
        {
            for (i = len - 1; i >= ind; i--)
            {
                x[i + 1] = x[i];
                y[i + 1] = y[i];
            }
            x[ind] = px;
            y[ind] = py;
        }
    }
    else
    {
        g[gno].active = ON;
        activateset(gno, setno);
        g[gno].p[setno].type = type;
        allocxy(&g[gno].p[setno], 1);
        x = getx(gno, setno);
        y = gety(gno, setno);
        x[0] = px;
        y[0] = py;
    }
    updatesetminmax(gno, setno);
}

/*
 * copy a set to another set, if the to set doesn't exist
 * get a new one, if it does, ask if it is okay to overwrite
 */
void do_copyset(int gfrom, int j1, int gto, int j2)
{
    if (!isactive_graph(gto))
    {
        set_graph_active(gto);
    }
    if (!isactive(gfrom, j1))
    {
        return;
    }
    if (j1 == j2 && gfrom == gto)
    {
        return;
    }
    if (isactive(gto, j2))
    {
        killset(gto, j2);
    }
    activateset(gto, j2);
    settype(gto, j2, dataset_type(gfrom, j1));
    setlength(gto, j2, getsetlength(gfrom, j1));
    copyset(gfrom, j1, gto, j2);
    sprintf(buf, "copy of set %d", j1);
    setcomment(gto, j2, buf);
    log_results(buf);
    updatesetminmax(gto, j2);
    update_set_status(gto, j2);
}

/*
 * move a set to another set, in possibly another graph
 */
void do_moveset(int gfrom, int j1, int gto, int j2)
{

    if (!isactive_graph(gto))
    {
        set_graph_active(gto);
    }
    if (!isactive(gfrom, j1))
    {
        return;
    }
    if (j1 == j2 && gto == gfrom)
    {
        return;
    }
    if (isactive(gto, j2))
    {
        killset(gto, j2);
    }
    moveset(gfrom, j1, gto, j2);
    updatesymbols(gto, j2);
    updatesymbols(gfrom, j1);
    updatelegendstr(gto);
    updatesetminmax(gto, j2);
    update_set_status(gto, j2);
    killset(gfrom, j1);
    update_set_status(gfrom, j1);
}

/*
 * swap a set with another set
 */
void do_swapset(int gfrom, int j1, int gto, int j2)
{
    plotarr p;

    if (j1 == j2 && gto == gfrom)
    {
        errwin("Set from and set to are the same");
        return;
    }
    memcpy(&p, &g[gto].p[j1], sizeof(plotarr));
    memcpy(&g[gto].p[j1], &g[gfrom].p[j2], sizeof(plotarr));
    memcpy(&g[gfrom].p[j2], &p, sizeof(plotarr));
    updatesetminmax(gfrom, j1);
    updatesymbols(gfrom, j1);
    updatelegendstr(gfrom);
    update_set_status(gfrom, j1);
    updatesetminmax(gto, j2);
    updatesymbols(gto, j2);
    updatelegendstr(gto);
    update_set_status(gto, j2);
    drawgraph();
}

/*
 * activate a set and set its length
 */
void do_activateset(int gno, int setno, int len)
{
    if (isactive(gno, setno))
    {
        sprintf(buf, "Set %d already active", setno);
        errwin(buf);
        return;
    }
    if (len <= 0)
    {
        sprintf(buf, "Improper set length = %d", len);
        errwin(buf);
        return;
    }
    activateset(gno, setno);
    setlength(gno, setno, len);
    updatesetminmax(gno, setno);
    update_set_status(gno, setno);
}

/*
 * split a set into lpart length sets
 */
void do_splitsets(int gno, int setno, int lpart)
{
    int i, j, k, nsets, ncols, len, nleft, tmpset, psets, stype;
    char s[256];
    double *x[MAX_SET_COLS], *xtmp[MAX_SET_COLS], *xt[MAX_SET_COLS];
    plotarr p;

    if (!activeset(gno))
    {
        errwin("No active sets");
        return;
    }
    if (!isactive(gno, setno))
    {
        sprintf(s, "Set %d not active", setno);
        errwin(s);
        return;
    }
    if ((len = getsetlength(gno, setno)) < 3)
    {
        errwin("Set length < 3");
        return;
    }
    if (lpart >= len)
    {
        errwin("Split length >= set length");
        return;
    }
    if (lpart == 0)
    {
        errwin("Split length = 0");
        return;
    }
    psets = len / lpart;
    nleft = len % lpart;
    if (nleft)
    {
        psets++;
    }
    nsets = 0;

    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (isactive(gno, i))
        {
            nsets++;
        }
    }
    if (psets > (g[gno].maxplot - nsets + 1))
    {
        errwin("Not enough sets for split");
        return;
    }
    /* get number of columns in this set */
    ncols = getncols(gno, setno);

    /* copy the contents to a temporary buffer */
    for (j = 0; j < ncols; j++)
    {
        x[j] = getcol(gno, setno, j);
        xtmp[j] = (double *)calloc(len, sizeof(double));
        if (xtmp[j] == NULL)
        {
            errwin("Not enough memory for split");
            for (k = 0; k < j; k++)
            {
                cxfree(xtmp[k]);
            }
            return;
        }
    }
    for (j = 0; j < ncols; j++)
    {
        for (i = 0; i < len; i++)
        {
            xtmp[j][i] = x[j][i];
        }
    }

    /* save the set type */
    stype = dataset_type(gno, setno);
    /*
    * load the props for this set into a temporary set, set the columns to
    * NULL
    */
    p = g[gno].p[setno];
    p.len = 0;
    for (k = 0; k < MAX_SET_COLS; k++)
    {
        p.ex[k] = NULL;
    }

    /* return the set to the heap */
    killset(gno, setno);
    /* now load each set */

    for (i = 0; i < psets - 1; i++)
    {
        tmpset = nextset(gno);
        /* set the plot parameters includes the set type */
        g[gno].p[tmpset] = p;
        activateset(gno, tmpset);
        settype(gno, tmpset, stype);
        setlength(gno, tmpset, lpart);
        /* load the data into each column */
        for (k = 0; k < ncols; k++)
        {
            xt[k] = getcol(gno, tmpset, k);
            for (j = 0; j < lpart; j++)
            {
                xt[k][j] = xtmp[k][i * lpart + j];
            }
        }
        sprintf(s, "partition %d of set %d", i + 1, setno);
        setcomment(gno, tmpset, s);
        log_results(buf);
        updatesetminmax(gno, tmpset);
        update_set_status(gno, tmpset);
    }
    if (nleft == 0)
    {
        nleft = lpart;
    }
    tmpset = nextset(gno);
    memcpy(&g[gno].p[tmpset], &p, sizeof(plotarr));
    activateset(gno, tmpset);
    settype(gno, tmpset, stype);
    setlength(gno, tmpset, nleft);

    /* load the data into each column */
    for (k = 0; k < ncols; k++)
    {
        xt[k] = getcol(gno, tmpset, k);
        for (j = 0; j < nleft; j++)
        {
            xt[k][j] = xtmp[k][i * lpart + j];
        }
    }

    sprintf(s, "partition %d of set %d", i + 1, setno);
    setcomment(gno, tmpset, s);
    log_results(buf);
    updatesetminmax(gno, tmpset);
    update_set_status(gno, tmpset);
    for (k = 0; k < ncols; k++)
    {
        cfree(xtmp[k]);
    }

    drawgraph();
}

/*
 * break a set at a point
 */
void do_breakset(int gno, int setno, int ind)
{
    int j, k, ncols, len, tmpset, stype;
    int n1, n2;
    char s[256];
    double *e1, *e2;

    if (!activeset(gno))
    {
        errwin("No active sets");
        return;
    }
    if (!isactive(gno, setno))
    {
        sprintf(s, "Set %d not active", setno);
        errwin(s);
        return;
    }
    if ((len = getsetlength(gno, setno)) < ind + 1)
    {
        errwin("Set length less than point index");
        return;
    }
    /* get number of columns in this set */
    ncols = getncols(gno, setno);
    stype = dataset_type(gno, setno);

    n2 = len - ind; /* upper part of new set */
    n1 = len - n2; /* lower part of old set */
    if (n1 <= 0 || n2 <= 0)
    {
        errwin("Break set length <= 0");
        return;
    }
    tmpset = nextset(gno);
    if (tmpset == -1)
    {
        return;
    }
    activateset(gno, tmpset);
    settype(gno, tmpset, stype);
    setlength(gno, tmpset, n2);

    /* load the data into each column */
    for (k = 0; k < ncols; k++)
    {
        e1 = getcol(gno, setno, k);
        e2 = getcol(gno, tmpset, k);
        for (j = ind; j < len; j++)
        {
            e2[j - ind] = e1[j];
        }
    }

    setlength(gno, setno, n1);
    updatesetminmax(gno, setno);
    update_set_status(gno, setno);

    sprintf(s, "split set %d at point %d", setno, ind);
    setcomment(gno, tmpset, s);
    log_results(buf);
    updatesetminmax(gno, tmpset);
    update_set_status(gno, tmpset);
    drawgraph();
}

/*
 * write out a set
 */
void do_writesets(int gno, int setno, int imbed, char *fn, char *format)
{
    int i, j, k, n, which_graph = gno, save_cg = cg, start, stop, set_start, set_stop;
    FILE *cp;
    double *x, *y, *dx, *dy, *dz, *dw;

    if (!fn[0])
    {
        errwin("Define file name first");
        return;
    }
    if (fexists(fn))
    {
        return;
    }
    if ((cp = fopen(fn, "w")) == NULL)
    {
        char s[192];

        sprintf(s, "Unable to open file %s", fn);
        errwin(s);
        return;
    }
    if (which_graph == maxgraph)
    {
        start = 0;
        stop = maxgraph - 1;
    }
    else if (which_graph == -1)
    {
        start = cg;
        stop = cg;
    }
    else
    {
        start = which_graph;
        stop = which_graph;
    }
    if (imbed)
    {
        if (start != stop)
        {
            putparms(-1, cp, imbed);
        }
        else
        {
            putparms(start, cp, imbed);
        }
    }
    for (k = start; k <= stop; k++)
    {
        if (isactive_graph(k))
        {
            if (start != stop)
            {
                fprintf(cp, "@WITH G%1d\n", k);
                fprintf(cp, "@G%1d ON\n", k);
            }
            if (setno == -1)
            {
                set_start = 0;
                set_stop = g[cg].maxplot - 1;
            }
            else
            {
                set_start = setno;
                set_stop = setno;
            }
            for (j = set_start; j <= set_stop; j++)
            {
                if (isactive(k, j))
                {
                    fprintf(cp, "@TYPE %s\n", set_types(dataset_type(k, j)));
                    x = getx(k, j);
                    y = gety(k, j);
                    n = getsetlength(k, j);
                    switch (dataset_type(k, j))
                    {
                    case XY:
                        for (i = 0; i < n; i++)
                        {
                            fprintf(cp, format, x[i], y[i]);
                            fputc('\n', cp);
                        }
                        break;
                    case XYDX:
                    case XYDY:
                    case XYZ:
                    case XYRT:
                        dx = getcol(k, j, 2);
                        for (i = 0; i < n; i++)
                        {
                            fprintf(cp, "%lg %lg %lg", x[i], y[i], dx[i]);
                            fputc('\n', cp);
                        }
                        break;
                    case XYDXDX:
                    case XYDYDY:
                    case XYDXDY:
                    case XYUV:
                        dx = getcol(k, j, 2);
                        dy = getcol(k, j, 3);
                        for (i = 0; i < n; i++)
                        {
                            fprintf(cp, "%lg %lg %lg %lg", x[i], y[i], dx[i], dy[i]);
                            fputc('\n', cp);
                        }
                        break;
                    case XYHILO:
                        dx = getcol(k, j, 2);
                        dy = getcol(k, j, 3);
                        dz = getcol(k, j, 4);
                        for (i = 0; i < n; i++)
                        {
                            fprintf(cp, "%lg %lg %lg %lg %lg", x[i], y[i], dx[i], dy[i], dz[i]);
                            fputc('\n', cp);
                        }
                        break;
                    case XYBOX:
                        dx = getcol(k, j, 2);
                        dy = getcol(k, j, 3);
                        dz = getcol(k, j, 4);
                        for (i = 0; i < n; i++)
                        {
                            fprintf(cp, "%lg %lg %lg %lg %d", x[i], y[i], dx[i], dy[i], (int)dz[i]);
                            fputc('\n', cp);
                        }
                        break;
                    case XYBOXPLOT:
                        dx = getcol(k, j, 2);
                        dy = getcol(k, j, 3);
                        dz = getcol(k, j, 4);
                        dw = getcol(k, j, 5);
                        for (i = 0; i < n; i++)
                        {
                            fprintf(cp, "%lg %lg %lg %lg %lg %lg", x[i], y[i], dx[i], dy[i], dz[i], dw[i]);
                            fputc('\n', cp);
                        }
                        break;
                    }
                    fprintf(cp, "&\n");
                }
            }
        }
    }
    fclose(cp);
    cg = save_cg;
}

/*
 * activate a set and set its length
 */
void do_activate(int setno, int type, int len)
{
    type = index_set_types[type];
    if (isactive(cg, setno))
    {
        sprintf(buf, "Set %d already active", setno);
        errwin(buf);
        return;
    }
    if (len <= 0)
    {
        sprintf(buf, "Improper set length = %d", len);
        errwin(buf);
        return;
    }
    activateset(cg, setno);
    settype(cg, setno, type);
    setlength(cg, setno, len);
    updatesetminmax(cg, setno);
    update_set_status(cg, setno);
}

/*
 * de-activate a set
 */
void do_deactivate(int gno, int setno)
{
    set_prop(gno, SET, SETNUM, setno, ACTIVE, OFF, 0);
    g[gno].p[setno].deact = 1;
    update_set_status(gno, setno);
    drawgraph();
}

/*
 * re-activate a set
 */
void do_reactivate(int gno, int setno)
{
    set_prop(gno, SET, SETNUM, setno, ACTIVE, ON, 0);
    g[gno].p[setno].deact = 0;
    update_set_status(gno, setno);
    drawgraph();
}

/*
 * change the type of a set
 */
void do_changetype(int setno, int type)
{
    type = index_set_types[type];
    settype(cg, setno, type);
    setlength(cg, setno, getsetlength(cg, setno));
    updatesetminmax(cg, setno);
    update_set_status(cg, setno);
}

/*
 * set the length of an active set - contents are destroyed
 */
void do_setlength(int setno, int len)
{
    if (!isactive(cg, setno))
    {
        sprintf(buf, "Set %d not active", setno);
        errwin(buf);
        return;
    }
    if (len <= 0)
    {
        sprintf(buf, "Improper set length = %d", len);
        errwin(buf);
        return;
    }
    setlength(cg, setno, len);
    updatesetminmax(cg, setno);
    update_set_status(cg, setno);
}

/*
 * copy a set to another set, if the to set doesn't exist
 * get a new one, if it does, ask if it is okay to overwrite
 */
void do_copy(int j1, int gfrom, int j2, int gto)
{
    if (!isactive(gfrom, j1))
    {
        sprintf(buf, "Set %d not active", j1);
        errwin(buf);
        return;
    }
    gto--;
    if (gto == -1)
    {
        gto = cg;
    }
    if (!isactive_graph(gto))
    {
        set_graph_active(gto);
    }
    if (j1 == j2 && gfrom == gto)
    {
        errwin("Set from and set to are the same");
        return;
    }
    /* select next set */
    if (j2 == SET_SELECT_NEXT)
    {
        if ((j2 = nextset(gto)) != -1)
        {
            activateset(gto, j2);
            settype(gto, j2, dataset_type(gfrom, j1));
            setlength(gto, j2, getsetlength(gfrom, j1));
        }
        else
        {
            return;
        }
    }
    /* use user selected set */
    else
    {
        if (isactive(gto, j2))
        {
            sprintf(buf, "Set %d active, overwrite?", j2);
            if (!yesno(buf, NULL, NULL, NULL))
            {
                return;
            }
            killset(gto, j2);
        }
        activateset(gto, j2);
        settype(gto, j2, dataset_type(gfrom, j1));
        setlength(gto, j2, getsetlength(gfrom, j1));
    }
    copyset(gfrom, j1, gto, j2);
    sprintf(buf, "copy of set %d", j1);
    setcomment(gto, j2, buf);
    log_results(buf);
    updatesetminmax(gto, j2);
    update_set_status(gto, j2);
    drawgraph();
}

/*
 * move a set to another set, if the to set doesn't exist
 * get a new one, if it does, ask if it is okay to overwrite
 */
void do_move(int j1, int gfrom, int j2, int gto)
{
    if (!isactive(gfrom, j1))
    {
        sprintf(buf, "Set %d not active", j1);
        errwin(buf);
        return;
    }
    gto--;
    if (gto == -1)
    {
        gto = cg;
    }
    if (!isactive_graph(gto))
    {
        set_graph_active(gto);
    }
    if (j2 == SET_SELECT_NEXT)
    {
        if ((j2 = nextset(gto)) == -1)
        {
            return;
        }
    }
    if (j1 == j2 && gto == gfrom)
    {
        errwin("Set from and set to are the same");
        return;
    }
    if (isactive(gto, j2))
    {
        sprintf(buf, "Set %d active, overwrite?", j2);
        if (!yesno(buf, NULL, NULL, NULL))
        {
            return;
        }
        killset(gto, j2);
    }
    moveset(gfrom, j1, gto, j2);
    updatesymbols(gto, j2);
    updatesymbols(gfrom, j1);
    updatelegendstr(gto);
    updatesetminmax(gto, j2);
    update_set_status(gto, j2);
    killset(gfrom, j1);
    update_set_status(gfrom, j1);
    drawgraph();
}

/*
 * swap a set with another set
 */
void do_swap(int j1, int gfrom, int j2, int gto)
{
    gfrom--;
    if (gfrom == -1)
    {
        gfrom = cg;
    }
    gto--;
    if (gto == -1)
    {
        gto = cg;
    }
    if (j1 == j2 && gfrom == gto)
    {
        errwin("Set from and set to are the same");
        return;
    }
    do_swapset(gfrom, j1, gto, j2);
}

/*
 * drop points from an active set
 */
void do_drop_points(int setno, int startno, int endno)
{
    int dist;

    if (!isactive(cg, setno))
    {
        sprintf(buf, "Set %d not active", setno);
        errwin(buf);
        return;
    }
    dist = endno - startno + 1;
    if (startno < 0)
    {
        errwin("Start # < 1");
        return;
    }
    if (endno >= getsetlength(cg, setno))
    {
        errwin("Ending # > set length");
        return;
    }
    if (startno > endno)
    {
        errwin("Starting # > ending #");
        return;
    }
    if (dist == getsetlength(cg, setno))
    {
        errwin("# of points to drop = set length, use kill");
        return;
    }
    droppoints(cg, setno, startno, endno, dist);
    updatesetminmax(cg, setno);
    update_set_status(cg, setno);
    drawgraph();
}

/*
 * append one set to another
 */
void do_join_sets(int gfrom, int j1, int gto, int j2)
{
    int i;

    if (j1 == -1)
    {
        if (!isactive(gfrom, j2))
        {
            activateset(gfrom, j2);
            setlength(gfrom, j2, 0);
        }
        for (i = 0; i < g[gfrom].maxplot; i++)
        {
            if (isactive(gfrom, i) && i != j2)
            {
                joinsets(gfrom, i, gfrom, j2);
                killset(gfrom, i);
                update_set_status(gfrom, i);
            }
        }
    }
    else
    {
        if (!isactive(gfrom, j1))
        {
            sprintf(buf, "Set %d not active", j1);
            errwin(buf);
            return;
        }
        if (!isactive(gto, j2))
        {
            sprintf(buf, "Set %d not active", j2);
            errwin(buf);
            return;
        }
        joinsets(gfrom, j1, gto, j2);
        killset(gfrom, j1);
        update_set_status(gfrom, j1);
    }
    updatesetminmax(gto, j2);
    update_set_status(gto, j2);
    drawgraph();
}

/*
 * reverse the order of a set
 */
void do_reverse_sets(int setno)
{
    int n, i, j, k, ncols;
    double *x;

    if (!isactive(cg, setno))
    {
        sprintf(buf, "Set %d not active", setno);
        errwin(buf);
        return;
    }
    n = getsetlength(cg, setno);
    ncols = getncols(cg, setno);
    for (k = 0; k < ncols; k++)
    {
        x = getcol(cg, setno, k);
        for (i = 0; i < n / 2; i++)
        {
            j = (n - 1) - i;
            fswap(&x[i], &x[j]);
        }
    }
    update_set_status(cg, setno);
    drawgraph();
}

/*
 * coalesce sets
 */
void do_coalesce_sets(int setno)
{
    int i, first = 1;

    if (!activeset(cg))
    {
        errwin("No active sets");
        return;
    }
    if (isactive(cg, setno))
    {
        sprintf(buf, "Set %d active, need an inactive set", setno);
        errwin(buf);
        return;
    }
    else
    {
        if ((setno = nextset(cg)) != -1)
        {
            activateset(cg, setno);
        }
        else
        {
            return;
        }
        for (i = 0; i < g[cg].maxplot; i++)
        {
            if (isactive(cg, i) && i != setno)
            {
                if (first)
                {
                    first = 0;
                    setlength(cg, setno, getsetlength(cg, i));
                    copyset(cg, i, cg, setno);
                    killset(cg, i);
                    update_set_status(cg, i);
                }
                else
                {
                    joinsets(cg, i, cg, setno);
                    killset(cg, i);
                    update_set_status(cg, i);
                }
            }
        }
    }
    updatesetminmax(cg, setno);
    update_set_status(cg, setno);
    drawgraph();
}

/*
 * kill a set
 */
void do_kill(int gno, int setno, int soft)
{
    int redraw = 0, i;

    if (setno == g[gno].maxplot || setno == -1)
    {
        for (i = 0; i < g[gno].maxplot; i++)
        {
            if (isactive(gno, i))
            {
                if (soft)
                {
                    softkillset(gno, i);
                }
                else
                {
                    killset(gno, i);
                }
                redraw = 1;
                update_set_status(gno, i);
            }
        }
        if (redraw)
        {
            drawgraph();
        }
        else
        {
            errwin("No sets to kill");
        }
    }
    else
    {
        if (!isactive(gno, setno))
        {
            sprintf(buf, "Set %d already dead", setno);
            errwin(buf);
            return;
        }
        else
        {
            if (soft)
            {
                softkillset(gno, setno);
            }
            else
            {
                killset(gno, setno);
            }
            update_set_status(gno, setno);
            drawgraph();
        }
    }
}

/*
 * kill all active sets
 */
void do_flush(void)
{
    int i;

    if (yesno("Flush all active sets, are you sure? ", NULL, NULL, NULL))
    {
        set_wait_cursor();
        for (i = 0; i < g[cg].maxplot; i++)
        {
            if (isactive(cg, i))
            {
                killset(cg, i);
                update_set_status(cg, i);
            }
        }
        unset_wait_cursor();
        drawgraph();
    }
}

/*
 * sort sets, only works on sets of type XY
 */
void do_sort(int setno, int sorton, int stype)
{
    int i;

    if (setno == -1)
    {
        for (i = 0; i < g[cg].maxplot; i++)
        {
            if (isactive(cg, i))
            {
                sort_set(i, sorton, stype);
            }
        }
    }
    else
    {
        if (!isactive(cg, setno))
        {
            sprintf(buf, "Set %d not active", setno);
            errwin(buf);
            return;
        }
        else
        {
            sort_set(setno, sorton, stype);
        }
    }
    drawgraph();
}

void sort_set(int setno, int sorton, int stype)
{
    int up;

    up = getsetlength(cg, setno);
    if (up < 2)
    {
        return;
    }
    sortset(cg, setno, sorton, stype);
}

void autoon_proc(void)
{
    set_action(0);
    set_action(AUTO_NEAREST);
}

/*
 * write out all sets in binary
 */
void do_writesets_binary(int gno, int, char *fn)
{
    int i, j, n, scnt;
    FILE *cp;
    double *x, *y;
    float *xf, *yf;

    if (fn == NULL || !fn[0])
    {
        errwin("Define file name first");
        return;
    }
    if (!isactive_graph(gno))
    {
    }
    if (fexists(fn))
    {
        return;
    }
    if ((cp = fopen(fn, "w")) == NULL)
    {
        char s[192];

        sprintf(s, "Unable to open file %s", fn);
        errwin(s);
        return;
    }
    scnt = 0;
    for (i = 0; i < g[gno].maxplot; i++)
    {
        if (isactive(gno, i) && getsetlength(gno, i))
        {
            scnt++;
        }
    }
    fwrite(&scnt, sizeof(int), 1, cp);
    for (j = 0; j < g[gno].maxplot; j++)
    {
        if (isactive(gno, j))
        {
            x = getx(gno, j);
            y = gety(gno, j);
            n = getsetlength(gno, j);
            xf = (float *)calloc(n, sizeof(float));
            yf = (float *)calloc(n, sizeof(float));
            for (i = 0; i < n; i++)
            {
                xf[i] = x[i];
                yf[i] = y[i];
            }
            fwrite(&n, sizeof(int), 1, cp);
            fwrite(xf, sizeof(float), n, cp);
            fwrite(yf, sizeof(float), n, cp);
            cfree(xf);
            cfree(yf);
        }
    }
    fclose(cp);
}

void outputset(int gno, int setno, char *fname, char *dformat)
{
    int i, n;
    FILE *cp;
    double *x, *y, *dx, *dy, *dz, *dw;
    char format[256];
    if (fname == NULL)
    {
        cp = stdout;
    }
    else if ((cp = fopen(fname, "w")) == NULL)
    {
        char s[256];
        sprintf(s, "Unable to open file %s", fname);
        errwin(s);
        return;
    }
    if (dformat == NULL)
    {
        strcpy(format, "%lf %lf");
    }
    else
    {
        strcpy(format, dformat);
    }
    if (isactive(cg, setno))
    {
        x = getx(cg, setno);
        y = gety(cg, setno);
        n = getsetlength(cg, setno);
        switch (dataset_type(cg, setno))
        {
        case XY:
            for (i = 0; i < n; i++)
            {
                fprintf(cp, format, x[i], y[i]);
                fputc('\n', cp);
            }
            break;
        case XYDX:
        case XYDY:
        case XYZ:
        case XYRT:
            dx = getcol(cg, setno, 2);
            for (i = 0; i < n; i++)
            {
                fprintf(cp, "%lg %lg %lg", x[i], y[i], dx[i]);
                fputc('\n', cp);
            }
            break;
        case XYDXDX:
        case XYDYDY:
        case XYDXDY:
        case XYUV:
            dx = getcol(gno, setno, 2);
            dy = getcol(gno, setno, 3);
            for (i = 0; i < n; i++)
            {
                fprintf(cp, "%lg %lg %lg %lg", x[i], y[i], dx[i], dy[i]);
                fputc('\n', cp);
            }
            break;
        case XYHILO:
            dx = getcol(gno, setno, 2);
            dy = getcol(gno, setno, 3);
            dz = getcol(gno, setno, 4);
            for (i = 0; i < n; i++)
            {
                fprintf(cp, "%lg %lg %lg %lg %lg", x[i], y[i], dx[i], dy[i], dz[i]);
                fputc('\n', cp);
            }
            break;
        case XYBOXPLOT:
            dx = getcol(gno, setno, 2);
            dy = getcol(gno, setno, 3);
            dz = getcol(gno, setno, 4);
            dw = getcol(gno, setno, 5);
            for (i = 0; i < n; i++)
            {
                fprintf(cp, "%lg %lg %lg %lg %lg %lg", x[i], y[i], dx[i], dy[i], dz[i], dw[i]);
                fputc('\n', cp);
            }
            break;
        case XYBOX:
            dx = getcol(gno, setno, 2);
            dy = getcol(gno, setno, 3);
            dz = getcol(gno, setno, 4);
            for (i = 0; i < n; i++)
            {
                fprintf(cp, "%lg %lg %lg %lg %d", x[i], y[i], dx[i], dy[i], (int)dz[i]);
                fputc('\n', cp);
            }
            break;
        }
    }
    if (fname != NULL)
    {
        fclose(cp);
    }
}

void set_hotlink(int gno, int setno, int onoroff, char *fname, int src)
{
    g[gno].p[setno].hotlink = onoroff;
    if (onoroff && fname != NULL)
    {
        strcpy(g[gno].p[setno].hotfile, fname);
        g[gno].p[setno].hotsrc = src;
    }
}

int is_hotlinked(int gno, int setno)
{
    return (g[gno].p[setno].hotlink && strlen(g[gno].p[setno].hotfile) > 0);
}

char *get_hotlink_file(int gno, int setno)
{
    return g[gno].p[setno].hotfile;
}

int get_hotlink_src(int gno, int setno)
{
    return g[gno].p[setno].hotsrc;
}

void do_update_hotlink(int gno, int setno)
{
    read_set_fromfile(gno, setno, g[gno].p[setno].hotfile, g[gno].p[setno].hotsrc);
}
