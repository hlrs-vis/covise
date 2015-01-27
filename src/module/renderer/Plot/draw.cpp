/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: draw.c,v 1.3 1994/07/17 05:24:03 pturner Exp pturner $
 *
 * interface to device drivers
 *
 */

/*
   as of 2.05, the default line clipping algorithm is liang-barsky
   This was done as there were rare cases where the S-C algorithm
   would go into an infinite loop. In case there are problems with
   L-B, comment out this define to go back to the S-C clip

   PJT - 12-7-91
*/

#define LIANG_BARSKY

#include <stdio.h>
#include <math.h>
#include "symdefs.h"
#include "draw.h"
#include "extern.h"
#include <Xm/Xm.h>
#include "xprotos.h"
#include "defines.h"
#include "noxprotos.h"

extern int debuglevel;
extern double my_hypot(double x, double y);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define DRAWNORMAL 0
#define DRAWLOG 1
#define DRAWPOLAR 2

/* maximum size of polygon for fills and patterns */
#define MAX_POLY 15000

double charsize = 1.0;
static double xg1, xg2, yg1, yg2; /* world coordinates */
static double rxg1, rxg2, ryg1, ryg2;
/* real world coordinates (different
 * if log plots) */
static double dxg, dyg; /* delta x, y of world co-ordinates */
static double xv1, xv2, yv1, yv2; /* viewpoint coordinates */
static double dxv, dyv; /* distance covered by viewport */
static double rx, ry; /* dxv/dxg & dyv/dyg */
static double crx, cry; /* constants for world -> normal */
static double curx, cury; /* current location of pen */
static int color; /* current color */
static int lines; /* current linestyle */
static int linew; /* current line width */
// static int ylabpos;		/* info for positioning y-axis label */
static int devtype; /* current device */
static int clipflag = 1; /* clip in my_draw2 */
static int scaletypex = DRAWNORMAL; /* NORMAL, LOG, DRAWPOLAR */
static int scaletypey = DRAWNORMAL; /* NORMAL, LOG, DRAWPOLAR */
//static int swappedx = FALSE;	/* if axis is in reverse */
//static int swappedy = FALSE;	/* if axis is in reverse */

extern int hardcopyflag;

/*
 * globals. for device drivers
 */
double devcharsize; /* device charsize */
int (*devsetcolor)(int); /* device set color */
int (*devsetfont)(int); /* device set font */
int (*devsetline)(int); /* device set line style */
int (*devsetlinew)(int); /* device set line width */
int (*devsetpat)(int); /* device set fill pattern */
void (*devdrawarc)(int, int, int); /* device arc routine */
void (*devfillarc)(int, int, int); /* device fill arc routine */
void (*devdrawellipse)(int, int, int, int); /* device ellipse routine */
void (*devfillellipse)(int, int, int, int); /* device ellipse arc routine */
void (*devfill)(int, int *, int *); /* device fill polygon with a pattern */
void (*devfillcolor)(int, int *, int *); /* device fill polygon with color */
int (*devconvx)(double); /* viewport to device mapping for x */
int (*devconvy)(double); /* viewport to device mapping for y */
int (*devvector)(int, int, int); /* device draw line */
/* device write string */
int (*devwritestr)(int, int, int, char *, int, int);
int (*devdrawtic)(int, int, int, int); /* device draw tic */
int (*devleavegraphics)(); /* device exit */
int devsymsize; /* device default symbol size */
int devxticl; /* device tic length for x-axis */
int devyticl; /* device tic length for y-axis */
int devarrowlength; /* device arrowlength */
int devfontsrc; /* device or hershey font selector */
int devwidthmm; /* device width in mm */
int devheightmm; /* device height in mm */
int devwidth; /* device number of points in width */
int devheight; /* device number of points in height */
int devoffsx; /* device offset in x */
int devoffsy; /* device offset in y */
int devorient;
/* = 1 if y starts at top and increases
 * downward (terminal devices) */

static int savexticl; /* save device tic length for x-axis */
static int saveyticl; /* save device tic length for y-axis */

static int curfontd = 0; /* current font */
static int current_pattern;
static double hdelta;

int stringextentx(double scale, const char *s); /* in chersh.c */
int stringextenty(double scale, const char *s);
#ifndef LIANG_BARSKY
static int regions(double xend, double yend);
#endif
static int goodpointxy(double x, double y);
static int clip2d(double *x0, double *y0, double *x1, double *y1);
static int inside(double x, double y, int which);
static void intersect(double *px, double *py, double x1, double y1, double x2, double y2, int which);
static void clip_edge(double *px1, double *py1, double *px2, double *py2, int n1, int *n2, int which);
static void polygon_clip(double *px1, double *py1, int *n);
static void xhistbox(double x, double y);
static void yhistbox(double x, double y);

/*
 * device2world - given (x,y) in screen coordinates, return the world coordinates
 *             in (wx,wy) used for the display only
 */
void device2world(int x, int y, double *wx, double *wy)
{
    double vx, vy;

    device2view(x, y, &vx, &vy);
    view2world(vx, vy, wx, wy);
}

/*
 * device2view - given (x,y) in screen coordinates, return the viewport coordinates
 */
void device2view(int x, int y, double *vx, double *vy)
{
    if (devwidth == 0)
    {
        *vx = (double)0.0;
    }
    else
    {
        *vx = (double)x / devwidth;
    } /* if devwidth */

    if (devheight == 0)
    {
        *vy = (double)0.0;
    }
    else
    {
        if (devorient)
        {
            *vy = (double)(y - devheight) / (-devheight);
        }
        else
        {
            *vy = (double)y / devheight;
        } /* if devorient */
    } /* if devheight */

    /*
    *vx = (double) x / devwidth;
       if (devorient) {
      *vy = (double) (y - devheight) / (-devheight);
       } else {
      *vy = (double) y / devheight;
       }
   */
}

/*
 * view2world - given (vx,vy) in viewport coordinates, return world coordinates
 *            in (x,y)
 */
void view2world(double vx, double vy, double *x, double *y)
{
    *x = (vx - crx) / rx;
    if (scaletypex == DRAWLOG)
    {
        *x = pow(10.0, *x);
    }
    *y = (vy - cry) / ry;
    if (scaletypey == DRAWLOG)
    {
        *y = pow(10.0, *y);
    }
}

/*
 * world2deviceabs - given world coordinates (wx,wy) return the device coordinates,
 *              for the display only
 */
int world2deviceabs(double wx, double wy, int *x, int *y)
{
    if (scaletypex == DRAWLOG)
    {
        if (wx > 0.0)
        {
            wx = log10(wx);
        }
        else
        {
            return 0;
        }
    }
    *x = (int)((rx * wx + crx) * devwidth);
    if (scaletypey == DRAWLOG)
    {
        if (wy > 0.0)
        {
            wy = log10(wy);
        }
        else
        {
            return 0;
        }
    }
    if (devorient)
    {
        *y = (int)(devheight - (ry * wy + cry) * devheight);
    }
    else
    {
        *y = (int)((ry * wy + cry) * devheight);
    }
    return 1;
}

/*
 * world2device - given world coordinates (wx,wy) return the device coordinates,
 *              for the display only
 */
int world2device(double wx, double wy, int *x, int *y)
{
    if (scaletypex == DRAWLOG)
    {
        if (wx > 0.0)
        {
            wx = log10(wx);
        }
        else
        {
            return 0;
        }
    }
    *x = (int)((rx * wx + crx) * (devwidth + devoffsx));
    if (scaletypey == DRAWLOG)
    {
        if (wy > 0.0)
        {
            wy = log10(wy);
        }
        else
        {
            return 0;
        }
    }
    *y = (int)((ry * wy + cry) * (devheight + devoffsy));
    return 1;
}

int world2view(double x, double y, double *vx, double *vy)
{
    if (scaletypex == DRAWLOG)
    {
        if (x > 0.0)
        {
            x = log10(x);
        }
        else
        {
            return 0;
        }
    }
    *vx = rx * x + crx;
    if (scaletypey == DRAWLOG)
    {
        if (y > 0.0)
        {
            y = log10(y);
        }
        else
        {
            return 0;
        }
    }
    *vy = ry * y + cry;
    return 1;
}

/*
 * map world co-ordinates to viewport
 */
double xconv(double x)
{
    if (scaletypex == DRAWLOG)
    {
        if (x > 0.0)
        {
            x = log10(x);
        }
        else
        {
            return 0;
        }
    }
    return (rx * x + crx);
}

double yconv(double y)
{
    if (scaletypey == DRAWLOG)
    {
        if (y > 0.0)
        {
            y = log10(y);
        }
        else
        {
            return 0;
        }
    }
    return (ry * y + cry);
}

void set_coordmap(int mapx, int mapy)
{
    scaletypex = mapx;
    scaletypey = mapy;
}

void setfixedscale(double xv1, double yv1, double xv2, double yv2,
                   double *xg1, double *yg1, double *xg2, double *yg2)
{
    double dx, dy, dxv, dyv, dxi, dyi, scalex, scaley;

    dx = *xg2 - *xg1;
    dy = *yg2 - *yg1;
    dxv = xv2 - xv1;
    dyv = yv2 - yv1;
    dxi = dxv * devwidthmm;
    dyi = dyv * devheightmm;
    scalex = dx / dxi;
    scaley = dy / dyi;
    if (scalex > scaley)
    {
        *yg2 = *yg1 + scalex * dyi;
    }
    else
    {
        *xg2 = *xg1 + scaley * dxi;
    }
}

/*
 * defineworld - really should be called definewindow, defines the scaling
 *               of the plotting rectangle to be used for clipping
 */
void defineworld(double x1, double y1, double x2, double y2, int mapx, int mapy)
{
    rxg1 = x1;
    ryg1 = y1;
    rxg2 = x2;
    ryg2 = y2;
    set_coordmap(mapx, mapy);
    switch (scaletypex)
    {
    case DRAWNORMAL:
        xg1 = x1;
        xg2 = x2;
        dxg = xg2 - xg1;
        break;
    case DRAWLOG:
        xg1 = log10(x1);
        xg2 = log10(x2);
        dxg = xg2 - xg1;
        break;
    }
    switch (scaletypey)
    {
    case DRAWNORMAL:
        yg1 = y1;
        yg2 = y2;
        dyg = yg2 - yg1;
        break;
    case DRAWLOG:
        yg1 = log10(y1);
        yg2 = log10(y2);
        dyg = yg2 - yg1;
        break;
    }
    if (debuglevel == 3)
    {
        printf("%lf %lf %lf %lf\n", x1, x2, y1, y2);
        printf("%lf %lf %lf %lf\n", xg1, xg2, yg1, yg2);
    }
}

/*
 * viewport - define the location of the clipping rectangle defined by
 *            defineworld on the display device
 */
void viewport(double x1, double y1, double x2, double y2)
{
    xv1 = x1;
    xv2 = x2;
    yv1 = y1;
    yv2 = y2;
    dxv = x2 - x1;
    dyv = y2 - y1;
    rx = dxv / dxg;
    ry = dyv / dyg;
    crx = -rx * xg1 + xv1;
    cry = -ry * yg1 + yv1;
}

/*
static int reversedx()
{
    return (xv2 < xv1);
}
*/
static int reversedy()
{
    return (yv2 < yv1);
}

/*
 * clip if clipflag = TRUE
 */
void setclipping(int fl)
{
    clipflag = fl;
}

#ifndef LIANG_BARSKY
/*
 * clip lines within rectangle defined in world coordinates
 */
static int regions(double xend, double yend)
{
    int endpoint;

    endpoint = 0;
    if (xend < xg1)
        endpoint = 1; /* left */
    else
    {
        if (xend > xg2)
            endpoint = 2; /* right */
    }
    if (yend < yg1)
        endpoint |= 4; /* bottom */
    else
    {
        if (yend > yg2)
            endpoint |= 8; /* top */
    }
    return (endpoint);
}
#endif

/*
 * liang-barsky line clipping, the default as of v2.05
 */
int clipt(double d, double n, double *te, double *tl)
{
    double t;
    int ac = 1;

    if (d > 0.0)
    {
        t = n / d;
        if (t > *tl)
        {
            ac = 0;
        }
        else if (t > *te)
        {
            *te = t;
        }
    }
    else if (d < 0.0)
    {
        t = n / d;
        if (t < *te)
        {
            ac = 0;
        }
        else if (t < *tl)
        {
            *tl = t;
        }
    }
    else
    {
        if (n > 0.0)
        {
            ac = 0;
        }
    }
    return ac;
}

static int goodpointxy(double x, double y)
{
    return ((x >= xg1) && (x <= xg2) && (y >= yg1) && (y <= yg2));
}

static int clip2d(double *x0, double *y0, double *x1, double *y1)
{
    double te = 0.0, tl = 1.0, dx = (*x1 - *x0), dy = (*y1 - *y0);

    if (dx == 0.0 && dy == 0.0 && goodpointxy(*x1, *y1))
    {
        return 1;
    }
    if (clipt(dx, xg1 - *x0, &te, &tl))
    {
        if (clipt(-dx, *x0 - xg2, &te, &tl))
        {
            if (clipt(dy, yg1 - *y0, &te, &tl))
            {
                if (clipt(-dy, *y0 - yg2, &te, &tl))
                {
                    if (tl < 1.0)
                    {
                        *x1 = *x0 + tl * dx;
                        *y1 = *y0 + tl * dy;
                    }
                    if (te > 0.0)
                    {
                        *x0 = *x0 + te * dx;
                        *y0 = *y0 + te * dy;
                    }
                    return 1;
                }
            }
        }
    }
    return 0;
}

/*
 * my_draw2 - draw a line from the current point (curx,cury) to (x2,y2) clipping the
 *         line to the viewport providing clipflag is TRUE
 */
void my_draw2(double x2, double y2)
{
    double x1, y1, xtmp, ytmp;

    x1 = curx;
    y1 = cury;
    switch (scaletypex)
    {
    case DRAWNORMAL:
        break;
    case DRAWLOG:
        x2 = log10(x2);
        break;
    }
    switch (scaletypey)
    {
    case DRAWNORMAL:
        break;
    case DRAWLOG:
        y2 = log10(y2);
        break;
    }
    xtmp = x2;
    ytmp = y2;
    if (!clipflag)
    {
        switch (scaletypex)
        {
        case DRAWNORMAL:
            break;
        case DRAWLOG:
            x1 = pow(10.0, x1);
            x2 = pow(10.0, x2);
            break;
        }
        switch (scaletypey)
        {
        case DRAWNORMAL:
            break;
        case DRAWLOG:
            y1 = pow(10.0, y1);
            y2 = pow(10.0, y2);
            break;
        }
        my_move2(x1, y1);
        (*devvector)((*devconvx)(x2), (*devconvy)(y2), 1);
        curx = xtmp;
        cury = ytmp;
        return;
    }
#ifdef LIANG_BARSKY
    if (!clip2d(&x1, &y1, &x2, &y2))
    {
        curx = xtmp;
        cury = ytmp;
        return;
    }
#else
    dir1 = regions(x1, y1);
    dir2 = regions(x2, y2);
    if (x1 != x2)
    {
        m = (y2 - y1) / (x2 - x1);
    }
    if (y2 != y1)
    {
        minverse = (x2 - x1) / (y2 - y1);
    }
    while ((dir1 != 0) || (dir2 != 0))
    {
        if (dir1 & dir2)
        {
            curx = xtmp;
            cury = ytmp;
            return;
        }
        if (dir1 == 0)
        {
            dir = dir2;
            x = x2;
            y = y2;
        }
        else
        {
            dir = dir1;
            x = x1;
            y = y1;
        }
        if (1 & dir)
        {
            y = m * (xg1 - x1) + y1;
            x = xg1;
        }
        else
        {
            if (2 & dir)
            {
                y = m * (xg2 - x1) + y1;
                x = xg2;
            }
            else
            {
                if (4 & dir)
                {
                    x = minverse * (yg1 - y1) + x1;
                    y = yg1;
                }
                else
                {
                    if (8 & dir)
                    {
                        x = minverse * (yg2 - y1) + x1;
                        y = yg2;
                    }
                }
            }
        }
        if (dir == dir1)
        {
            x1 = x;
            y1 = y;
            dir1 = regions(x1, y1);
        }
        else
        {
            x2 = x;
            y2 = y;
            dir2 = regions(x2, y2);
        }
    }
#endif
    switch (scaletypex)
    {
    case DRAWNORMAL:
        break;
    case DRAWLOG:
        x1 = pow(10.0, x1);
        x2 = pow(10.0, x2);
        break;
    }
    switch (scaletypey)
    {
    case DRAWNORMAL:
        break;
    case DRAWLOG:
        y1 = pow(10.0, y1);
        y2 = pow(10.0, y2);
        break;
    }
    my_move2(x1, y1);
    (*devvector)((*devconvx)(x2), (*devconvy)(y2), 1);
    curx = xtmp;
    cury = ytmp;
}

/*
 * my_move2 - make (x,y) the current point (curx,cury)
 */
void my_move2(double x, double y)
{
    (*devvector)((*devconvx)(x), (*devconvy)(y), 0);
    switch (scaletypex)
    {
    case DRAWNORMAL:
        break;
    case DRAWLOG:
        x = log10(x);
        break;
    }
    switch (scaletypey)
    {
    case DRAWNORMAL:
        break;
    case DRAWLOG:
        y = log10(y);
        break;
    }
    curx = x;
    cury = y;
}

#define EDGE_LEFT 0
#define EDGE_RIGHT 1
#define EDGE_ABOVE 2
#define EDGE_BELOW 3

static int inside(double x, double y, int which)
{
    switch (which)
    {
    case EDGE_ABOVE:
        return (y <= ryg2);
    case EDGE_BELOW:
        return (y >= ryg1);
    case EDGE_LEFT:
        return (x >= rxg1);
    case EDGE_RIGHT:
        return (x <= rxg2);
    }
    return 0;
}

static void intersect(double *px, double *py,
                      double x1, double y1, double x2, double y2, int which)
{
    double m = (x2 - x1), b = 0.0;

    if (m != 0.0)
    {
        m = (y2 - y1) / m;
        b = y2 - m * x2;
    }
    switch (which)
    {
    case EDGE_ABOVE:
        *py = ryg2;
        if (m != 0.0)
        {
            *px = (ryg2 - b) / m;
        }
        else
        {
            *px = x1;
        }
        break;
    case EDGE_BELOW:
        *py = ryg1;
        if (m != 0.0)
        {
            *px = (ryg1 - b) / m;
        }
        else
        {
            *px = x1;
        }
        break;
    case EDGE_LEFT:
        *px = rxg1;
        *py = m * rxg1 + b;
        break;
    case EDGE_RIGHT:
        *px = rxg2;
        *py = m * rxg2 + b;
        break;
    }
}

static void clip_edge(double *px1, double *py1, double *px2, double *py2,
                      int n1, int *n2, int which)
{
    int i, k, n = n1;
    double px, py, pxprev, pyprev;

    i = 0;
    *n2 = 0;
    k = 0;
    pxprev = px1[n - 1];
    pyprev = py1[n - 1];
    for (i = 0; i < n; i++)
    {
        if (inside(px1[i], py1[i], which))
        {
            if (inside(pxprev, pyprev, which))
            {
                px2[k] = px1[i];
                py2[k] = py1[i];
                k++;
            }
            else
            {
                intersect(&px, &py, pxprev, pyprev, px1[i], py1[i], which);
                px2[k] = px;
                py2[k] = py;
                k++;
                px2[k] = px1[i];
                py2[k] = py1[i];
                k++;
            }
        }
        else if (inside(pxprev, pyprev, which))
        {
            intersect(&px, &py, pxprev, pyprev, px1[i], py1[i], which);
            px2[k] = px;
            py2[k] = py;
            k++;
        }
        else
        {
            /* do nothing */
        }
        pxprev = px1[i];
        pyprev = py1[i];
    }
    *n2 = k;
}

static void polygon_clip(double *px1, double *py1, int *n)
{
    int n1 = *n, n2;
    double px2[MAX_POLY], py2[MAX_POLY];
    if (*n >= MAX_POLY)
    {
        return;
    }
    clip_edge(px1, py1, px2, py2, n1, &n2, EDGE_ABOVE);
    clip_edge(px2, py2, px1, py1, n2, &n1, EDGE_BELOW);
    clip_edge(px1, py1, px2, py2, n1, &n2, EDGE_LEFT);
    clip_edge(px2, py2, px1, py1, n2, &n1, EDGE_RIGHT);
    *n = n1;
}

int setpattern(int k)
{
    int savepat = current_pattern;

    (*devsetpat)(current_pattern = k);
    return (savepat);
}

void fillpattern(int n, double *px, double *py)
{
    int i;
    int ipx[MAX_POLY], ipy[MAX_POLY];
    double pxtmp[MAX_POLY], pytmp[MAX_POLY];

    if (n >= MAX_POLY)
    {
        return;
    }
    if (clipflag)
    {
        for (i = 0; i < n; i++)
        {
            pxtmp[i] = px[i];
            pytmp[i] = py[i];
        }
        polygon_clip(pxtmp, pytmp, &n);
        if (n == 0)
        {
            return;
        }
    }
    for (i = 0; i < n; i++)
    {
        ipx[i] = (*devconvx)(pxtmp[i]);
        ipy[i] = (*devconvy)(pytmp[i]);
    }
    ipx[n] = ipx[0];
    ipy[n] = ipy[0];
    n++;
    (*devfill)(n, ipx, ipy);
}

void fillcolor(int n, double *px, double *py)
{
    int i;
    int ipx[MAX_POLY], ipy[MAX_POLY];
    double pxtmp[MAX_POLY], pytmp[MAX_POLY];
    if (n >= MAX_POLY)
    {
        return;
    }
    if (devtype == 7)
    {
        for (i = 0; i < n; i++)
        {
            ipx[i] = (*devconvx)(px[i]);
            ipy[i] = (*devconvy)(py[i]);
        }
        (*devfillcolor)(n, ipx, ipy);
    }
    else
    {
        if (clipflag)
        {
            for (i = 0; i < n; i++)
            {
                pxtmp[i] = px[i];
                pytmp[i] = py[i];
            }
            polygon_clip(pxtmp, pytmp, &n);
            if (n == 0)
            {
                return;
            }
        }
        for (i = 0; i < n; i++)
        {
            ipx[i] = (*devconvx)(pxtmp[i]);
            ipy[i] = (*devconvy)(pytmp[i]);
        }
        ipx[n] = ipx[0];
        ipy[n] = ipy[0];
        n++;
        (*devfillcolor)(n, ipx, ipy);
    }
}

void fillrectcolor(double x1, double y1, double x2, double y2)
{
    int i, n = 4;
    int ipx[5], ipy[5];
    double pxtmp[5], pytmp[5];

    pxtmp[0] = x1;
    pytmp[0] = y1;
    pxtmp[1] = x1;
    pytmp[1] = y2;
    pxtmp[2] = x2;
    pytmp[2] = y2;
    pxtmp[3] = x2;
    pytmp[3] = y1;

    if (clipflag)
    {
        polygon_clip(pxtmp, pytmp, &n);
        if (n == 0)
        {
            return;
        }
    }
    for (i = 0; i < n; i++)
    {
        ipx[i] = (*devconvx)(pxtmp[i]);
        ipy[i] = (*devconvy)(pytmp[i]);
    }
    ipx[n] = ipx[0];
    ipy[n] = ipy[0];
    n++;
    (*devfillcolor)(n, ipx, ipy);
}

void fillrectpat(double x1, double y1, double x2, double y2)
{
    int i, n = 4;
    int ipx[5], ipy[5];
    double pxtmp[5], pytmp[5];

    pxtmp[0] = x1;
    pytmp[0] = y1;
    pxtmp[1] = x1;
    pytmp[1] = y2;
    pxtmp[2] = x2;
    pytmp[2] = y2;
    pxtmp[3] = x2;
    pytmp[3] = y1;

    if (clipflag)
    {
        polygon_clip(pxtmp, pytmp, &n);
        if (n == 0)
        {
            return;
        }
    }
    for (i = 0; i < n; i++)
    {
        ipx[i] = (*devconvx)(pxtmp[i]);
        ipy[i] = (*devconvy)(pytmp[i]);
    }
    ipx[n] = ipx[0];
    ipy[n] = ipy[0];
    n++;
    (*devfill)(n, ipx, ipy);
}

/*
 * setfont - make f the current font to use for writing strings
 */
int setfont(int f)
{
    int itmp = curfontd;

    curfontd = f;
    (*devsetfont)(f);
    return (itmp);
}

/*
 * rect - draw a rectangle using the current color and linestyle
 */
void rect(double x1, double y1, double x2, double y2)
{
    my_move2(x1, y1);
    my_draw2(x1, y2);
    my_draw2(x2, y2);
    my_draw2(x2, y1);
    my_draw2(x1, y1);
}

/*
 * symok - return TRUE if (x,y) lies within the currently defined window in
 *         world coordinates. used to clip symbols that bypass the clipping
 *         done in my_draw2.
 */
int symok(double x, double y)
{
    if (clipflag)
    {
        return ((x >= rxg1) && (x <= rxg2) && (y >= ryg1) && (y <= ryg2));
    }
    else
    {
        return 1;
    }
}

/*
 * histbox - draw a box of width 2*hdelta centered at x with height y starting
 *           from y = 0.0
 */
static void xhistbox(double x, double y)
{
    double tmpx[4];
    double tmpy[4];
    int i;
    // int hist_pattern = 5;

    tmpx[0] = x - hdelta;
    tmpy[0] = 0.0;
    tmpx[1] = x - hdelta;
    tmpy[1] = y;
    tmpx[2] = x + hdelta;
    tmpy[2] = y;
    tmpx[3] = x + hdelta;
    tmpy[3] = 0.0;
    for (i = 0; i < 4; i++)
    {
        if (tmpx[i] < rxg1)
            tmpx[i] = rxg1;
        else if (tmpx[i] > rxg2)
            tmpx[i] = rxg2;
        if (tmpy[i] < ryg1)
            tmpy[i] = ryg1;
        else if (tmpy[i] > ryg2)
            tmpy[i] = ryg2;
    }
    /*
    * setpattern(hist_pattern); fillpattern(4, tmpx, tmpy); setpattern(0);
    */
    my_move2(tmpx[0], tmpy[0]);
    for (i = 0; i < 4; i++)
    {
        my_draw2(tmpx[(i + 1) % 4], tmpy[(i + 1) % 4]);
    }
}

/*
 * histbox3 - draw a box of width 2*hdelta centered at y with length x starting
 *           from x = 0.0
 */
static void yhistbox(double x, double y)
{
    int i;

    double tmpx[4];
    double tmpy[4];

    tmpy[0] = y - hdelta;
    tmpx[0] = 0.0;
    tmpy[1] = y - hdelta;
    tmpx[1] = x;
    tmpy[2] = y + hdelta;
    tmpx[2] = x;
    tmpy[3] = y + hdelta;
    tmpx[3] = 0.0;
    for (i = 0; i < 4; i++)
    {
        if (tmpx[i] < xg1)
            tmpx[i] = xg1;
        else if (tmpx[i] > xg2)
            tmpx[i] = xg2;
        if (tmpy[i] < yg1)
            tmpy[i] = yg1;
        else if (tmpy[i] > yg2)
            tmpy[i] = yg2;
    }
    /*
    * setpattern(current_pattern); fillpattern(4, tmpx, tmpy);
    */
    my_move2(tmpx[0], tmpy[0]);
    for (i = 0; i < 4; i++)
    {
        my_draw2(tmpx[(i + 1) % 4], tmpy[(i + 1) % 4]);
    }
}

/*
 * lengthpoly - return the length of a polyline in device coords
 */
int lengthpoly(double *x, double *y, int n)
{
    int i;
    double dist = 0.0, xtmp, ytmp;

    for (i = 1; i < n; i++)
    {
        xtmp = (*devconvx)(x[i]) - (*devconvx)(x[i - 1]);
        ytmp = (*devconvy)(y[i]) - (*devconvy)(y[i - 1]);
        dist = dist + my_hypot(xtmp, ytmp);
    }
    return ((int)dist);
}

/*
 * drawpoly - draw a connected line in the current color and linestyle
 *            with nodes given by (x[],y[])
 */
void drawpoly(double *x, double *y, int n)
{
    int i;

    my_move2(x[0], y[0]);
    for (i = 1; i < n; i++)
    {
        my_draw2(x[i], y[i]);
    }
}

/*
 * drawpolarpoly - draw a connected line in the current color and linestyle
 *            with nodes given by (x[] = r,y[] = theta) in polar coordinates
 */
void drawpolarpoly(double *x, double *y, int n)
{
    int i;
    double xt, yt;

    xt = x[0] * cos(y[0]);
    yt = x[0] * sin(y[0]);
    my_move2(xt, yt);
    for (i = 1; i < n; i++)
    {
        xt = x[i] * cos(y[i]);
        yt = x[i] * sin(y[i]);
        my_draw2(xt, yt);
    }
}

/*
 * drawpolyseg - draw segments, treating each successive pairs of points
 *               as a line segment
 */
void drawpolyseg(double *x, double *y, int n)
{
    int i, itmp;

    for (i = 0; i < (n / 2); i++)
    {
        itmp = i * 2;
        my_move2(x[itmp], y[itmp]);
        my_draw2(x[itmp + 1], y[itmp + 1]);
    }
}

void openclose(double x, double y1, double y2, double ebarlen, int xy)
{
    int ilen;

    if (xy)
    {
        ilen = stringextentx(ebarlen * devcharsize, "M");
        if (symok(x, y1))
        {
            (*devvector)((*devconvx)(x), (*devconvy)(y1), 0);
            (*devvector)((*devconvx)(x)-ilen, (*devconvy)(y1), 1);
        }
        if (symok(x, y2))
        {
            (*devvector)((*devconvx)(x), (*devconvy)(y2), 0);
            (*devvector)((*devconvx)(x) + ilen, (*devconvy)(y2), 1);
        }
    }
    else
    {
        ilen = stringextenty(ebarlen * devcharsize, "M");
        if (symok(y1, x))
        {
            (*devvector)((*devconvx)(y1), (*devconvy)(x), 0);
            (*devvector)((*devconvx)(y1), (*devconvy)(x)-ilen, 1);
        }
        if (symok(y2, x))
        {
            (*devvector)((*devconvx)(y2), (*devconvy)(x), 0);
            (*devvector)((*devconvx)(y2), (*devconvy)(x) + ilen, 1);
        }
    }
}

void errorbar(double x, double y, double ebarlen, int xy)
{
    int ilen;

    if (symok(x, y))
    {
        if (xy)
        {
            ilen = stringextentx(ebarlen * devcharsize, "M");
            (*devvector)((*devconvx)(x)-ilen, (*devconvy)(y), 0);
            (*devvector)((*devconvx)(x) + ilen, (*devconvy)(y), 1);
        }
        else
        {
            ilen = stringextenty(ebarlen * devcharsize, "M");
            (*devvector)((*devconvx)(x), (*devconvy)(y)-ilen, 0);
            (*devvector)((*devconvx)(x), (*devconvy)(y) + ilen, 1);
        }
    }
}

void boxplotsym(double x, double med, double il, double iu, double ol, double ou)
{
    int ilen;

    ilen = stringextentx(devcharsize, "M");
    /* median line */
    (*devvector)((*devconvx)(x)-ilen, (*devconvy)(med), 0);
    (*devvector)((*devconvx)(x) + ilen, (*devconvy)(med), 1);
    /* inner box */
    (*devvector)((*devconvx)(x)-ilen, (*devconvy)(il), 0);
    (*devvector)((*devconvx)(x) + ilen, (*devconvy)(il), 1);
    (*devvector)((*devconvx)(x) + ilen, (*devconvy)(iu), 1);
    (*devvector)((*devconvx)(x)-ilen, (*devconvy)(iu), 1);
    (*devvector)((*devconvx)(x)-ilen, (*devconvy)(il), 1);
    /* whiskers */
    (*devvector)((*devconvx)(x), (*devconvy)(il), 0);
    (*devvector)((*devconvx)(x), (*devconvy)(ol), 1);
    (*devvector)((*devconvx)(x), (*devconvy)(iu), 0);
    (*devvector)((*devconvx)(x), (*devconvy)(ou), 1);
}

/*
 * make the current color col
 */
int setcolor(int col)
{
    int scol = color;

    color = (*devsetcolor)(col);
    return scol;
}

/*
 * make the current linestyle style
 */
int setlinestyle(int style)
{
    int slin = lines;

    lines = (*devsetline)(style);
    return slin;
}

/*
 * make the current line width wid
 */
int setlinewidth(int wid)
{
    int slinw = linew;

    if (devsetlinew != NULL)
    {
        linew = (*devsetlinew)(wid);
    }
    return slinw;
}

void getlineprops(int *col, int *style, int *width)
{
    *col = color;
    *style = lines;
    *width = linew;
}

/*
 * drawtic - interface to device driver routines, done this way
 *           as there were problems with low resolution devices
 *           (this is probably not necessary now)
 *      dir = 0 = draw in up (for x-axis) or right (for y-axis)
 *      axis = 0 = draw x-axis tic,   1 = y-axis
 */
void drawtic(double x, double y, int dir, int axis)
{
    (*devdrawtic)((*devconvx)(x), (*devconvy)(y), dir, axis);
}

/*
 * set the current character size to size
 */
double setcharsize(double size)
{
    double s = charsize;

    if (devtype == 1 || devtype == 2)
    {
        pssetfontsize(size);
    }
    else if (devtype == 3 || devtype == 4)
    {
        mifsetfontsize(size);
    }
    else if (devtype == 7 || devtype == 8)
    {
        leafsetfontsize(size);
    }
    charsize = size;
    return s;
}

/*
 * set the current length of the tick marks
 */

void setticksize(double sizex, double sizey)
{
    devxticl = (int)(savexticl * sizex);
    devyticl = (int)(saveyticl * sizey);
}

/*
 * writestr - user interface to the current device text drawing routine
 */
void writestr(double x, double y, int dir, int just, char *s)
{
    if (s == NULL)
    {
        return;
    }
    if (strlen(s) == 0)
    {
        return;
    }
    (*devwritestr)((*devconvx)(x), (*devconvy)(y), dir, s, just, devtype);
}

/*
 * draw the title, subtitle
 */
void drawtitle(char *title, int which)
{
    int fudge, ix, iy;
    int tmp;
    double y = reversedy() ? ryg1 : ryg2;

    fudge = 4;
    tmp = ((*devconvx)(rxg2) + (*devconvx)(rxg1)) / 2;
    /* center x-axis label
    * and title */
    if (title[0] && !which)
    {
        iy = (int)(3.0 * stringextenty(charsize * devcharsize, "X"));
    }
    else
    {
        iy = stringextenty(charsize * devcharsize, "Xy");
    }
    ix = stringextentx(charsize * devcharsize, title);
    (*devwritestr)(tmp, (*devconvy)(y) + iy, 0, title, 2, 0);
}

/*
        draw grid lines rather than ticmarks
*/
void drawgrid(int dir, double start, double end, double y1, double y2, double step,
              int cy, int ly, int wy)
{
    int slin, swid, scol;

    slin = lines;
    swid = linew;
    scol = color;
    setcolor(cy);
    setlinestyle(ly);
    setlinewidth(wy);
    while (start <= end)
    {
        if (dir)
        {
            my_move2(y1, start);
            my_draw2(y2, start);
        }
        else
        {
            my_move2(start, y1);
            my_draw2(start, y2);
        }
        start += step;
    }
    setcolor(scol);
    setlinestyle(slin);
    setlinewidth(swid);
}

/*
 * draw a circle
 */
void my_circle(double xc, double yc, double s)
{
    int x = (*devconvx)(xc), y = (*devconvy)(yc), xm, ym;
    xm = (int)fabs((double)(x - (*devconvx)(xc + s)));
    ym = (int)fabs((double)(y - (*devconvy)(yc + s)));
    (*devdrawellipse)(x, y, xm, ym);
}

void my_filledcircle(double xc, double yc, double s)
{
    int x = (*devconvx)(xc), y = (*devconvy)(yc), xm, ym;
    xm = (int)fabs((double)(x - (*devconvx)(xc + s)));
    ym = (int)fabs((double)(y - (*devconvy)(yc + s)));
    (*devfillellipse)(x, y, xm, ym);
}

/*
 * draw a circle symbol
 */
void drawcircle(double xc, double yc, double s, int f)
{
    int x = (*devconvx)(xc), y = (*devconvy)(yc), xm, ym, cs;

    xm = (int)fabs((double)(x - (*devconvx)(xc + s)));
    ym = (int)fabs((double)(y - (*devconvy)(yc + s)));
    switch (f)
    {
    case 0:
        (*devdrawellipse)(x, y, xm, ym);
        break;
    case 1:
        (*devfillellipse)(x, y, xm, ym);
        if (devtype == 0)
        {
            (*devdrawellipse)(x, y, xm, ym);
        }
        break;
    case 2:
        cs = setcolor(0);
        (*devfillellipse)(x, y, xm, ym);
        setcolor(cs);
        (*devdrawellipse)(x, y, xm, ym);
        break;
    }
}

/*
        place symbols at the vertices of a polygon specified by x & y.
*/

double barwid = 0.01;

void symcircle(int x, int y, double s, int f)
{
    int cs;
    int side = (int)(devsymsize * s);

    switch (f)
    {
    case 0:
        (*devdrawarc)(x, y, side);
        break;
    case 1:
        (*devfillarc)(x, y, side);
        if (devtype == 0)
        {
            (*devdrawarc)(x, y, side);
        }
        break;
    case 2:
        cs = setcolor(0);
        (*devfillarc)(x, y, side);
        setcolor(cs);
        (*devdrawarc)(x, y, side);
        break;
    }
}

void symsquare(int x, int y, double s, int f)
{
    int side = (int)(devsymsize * s * 0.85);
    int sx[4], sy[4], sc;

    switch (f)
    {
    case 0:
        (*devvector)(x - side, y - side, 0);
        (*devvector)(x - side, y + side, 1);
        (*devvector)(x + side, y + side, 1);
        (*devvector)(x + side, y - side, 1);
        (*devvector)(x - side, y - side, 1);
        break;
    case 1:
        sx[0] = x - side;
        sx[1] = sx[0];
        sx[2] = x + side;
        sx[3] = sx[2];
        sy[0] = y - side;
        sy[1] = y + side;
        sy[2] = sy[1];
        sy[3] = sy[0];
        (*devfillcolor)(4, sx, sy);
        if (devtype == 0)
        {
            (*devvector)(x - side, y - side, 0);
            (*devvector)(x - side, y + side, 1);
            (*devvector)(x + side, y + side, 1);
            (*devvector)(x + side, y - side, 1);
            (*devvector)(x - side, y - side, 1);
        }
        break;
    case 2:
        sc = setcolor(0);
        sx[0] = x - side;
        sx[1] = sx[0];
        sx[2] = x + side;
        sx[3] = sx[2];
        sy[0] = y - side;
        sy[1] = y + side;
        sy[2] = sy[1];
        sy[3] = sy[0];
        (*devfillcolor)(4, sx, sy);
        setcolor(sc);
        (*devvector)(sx[0], sy[0], 0);
        (*devvector)(sx[0], sy[1], 1);
        (*devvector)(sx[2], sy[1], 1);
        (*devvector)(sx[2], sy[0], 1);
        (*devvector)(sx[0], sy[0], 1);
    }
}

void symtriangle1(int x, int y, double s, int f)
{
    int side = (int)(devsymsize * s);
    int sx[3], sy[3], sc;

    switch (f)
    {
    case 0:
        (*devvector)(x, y + side, 0);
        (*devvector)(x - side, y - side, 1);
        (*devvector)(x + side, y - side, 1);
        (*devvector)(x, y + side, 1);
        break;
    case 1:
        sx[0] = x;
        sx[1] = x - side;
        sx[2] = x + side;
        sy[0] = y + side;
        sy[1] = y - side;
        sy[2] = sy[1];
        (*devfillcolor)(3, sx, sy);
        if (devtype == 0)
        {
            (*devvector)(x, y + side, 0);
            (*devvector)(x - side, y - side, 1);
            (*devvector)(x + side, y - side, 1);
            (*devvector)(x, y + side, 1);
        }
        break;
    case 2:
        sc = setcolor(0);
        sx[0] = x;
        sx[1] = x - side;
        sx[2] = x + side;
        sy[0] = y + side;
        sy[1] = y - side;
        sy[2] = sy[1];
        (*devfillcolor)(3, sx, sy);
        setcolor(sc);
        (*devvector)(sx[0], sy[0], 0);
        (*devvector)(sx[1], sy[1], 1);
        (*devvector)(sx[2], sy[2], 1);
        (*devvector)(sx[0], sy[0], 1);
    }
}

void symtriangle2(int x, int y, double s, int f)
{
    int side = (int)(devsymsize * s);
    int sx[3], sy[3], sc;

    switch (f)
    {
    case 0:
        (*devvector)(x - side, y, 0);
        (*devvector)(x + side, y - side, 1);
        (*devvector)(x + side, y + side, 1);
        (*devvector)(x - side, y, 1);
        break;
    case 1:
        sx[0] = x - side;
        sx[1] = x + side;
        sx[2] = x + side;
        sy[0] = y;
        sy[1] = y - side;
        sy[2] = y + side;
        (*devfillcolor)(3, sx, sy);
        if (devtype == 0)
        {
            (*devvector)(x - side, y, 0);
            (*devvector)(x + side, y - side, 1);
            (*devvector)(x + side, y + side, 1);
            (*devvector)(x - side, y, 1);
        }
        break;
    case 2:
        sc = setcolor(0);
        sx[0] = x - side;
        sx[1] = x + side;
        sx[2] = x + side;
        sy[0] = y;
        sy[1] = y - side;
        sy[2] = y + side;
        (*devfillcolor)(3, sx, sy);
        setcolor(sc);
        (*devvector)(sx[0], sy[0], 0);
        (*devvector)(sx[1], sy[1], 1);
        (*devvector)(sx[2], sy[2], 1);
        (*devvector)(sx[0], sy[0], 1);
    }
}

void symtriangle3(int x, int y, double s, int f)
{
    int side = (int)(devsymsize * s);
    int sx[3], sy[3], sc;

    switch (f)
    {
    case 0:
        (*devvector)(x - side, y + side, 0);
        (*devvector)(x, y - side, 1);
        (*devvector)(x + side, y + side, 1);
        (*devvector)(x - side, y + side, 1);
        break;
    case 1:
        sx[0] = x - side;
        sx[1] = x;
        sx[2] = x + side;
        sy[0] = y + side;
        sy[1] = y - side;
        sy[2] = y + side;
        (*devfillcolor)(3, sx, sy);
        if (devtype == 0)
        {
            (*devvector)(x - side, y + side, 0);
            (*devvector)(x, y - side, 1);
            (*devvector)(x + side, y + side, 1);
            (*devvector)(x - side, y + side, 1);
        }
        break;
    case 2:
        sc = setcolor(0);
        sx[0] = x - side;
        sx[1] = x;
        sx[2] = x + side;
        sy[0] = y + side;
        sy[1] = y - side;
        sy[2] = y + side;
        (*devfillcolor)(3, sx, sy);
        setcolor(sc);
        (*devvector)(sx[0], sy[0], 0);
        (*devvector)(sx[1], sy[1], 1);
        (*devvector)(sx[2], sy[2], 1);
        (*devvector)(sx[0], sy[0], 1);
    }
}

void symtriangle4(int x, int y, double s, int f)
{
    int side = (int)(devsymsize * s);
    int sx[3], sy[3], sc;

    switch (f)
    {
    case 0:
        (*devvector)(x - side, y + side, 0);
        (*devvector)(x - side, y - side, 1);
        (*devvector)(x + side, y, 1);
        (*devvector)(x - side, y + side, 1);
        break;
    case 1:
        sx[0] = x - side;
        sx[1] = x - side;
        sx[2] = x + side;
        sy[0] = y + side;
        sy[1] = y - side;
        sy[2] = y;
        (*devfillcolor)(3, sx, sy);
        if (devtype == 0)
        {
            (*devvector)(x - side, y + side, 0);
            (*devvector)(x - side, y - side, 1);
            (*devvector)(x + side, y, 1);
            (*devvector)(x - side, y + side, 1);
        }
        break;
    case 2:
        sc = setcolor(0);
        sx[0] = x - side;
        sx[1] = x - side;
        sx[2] = x + side;
        sy[0] = y + side;
        sy[1] = y - side;
        sy[2] = y;
        (*devfillcolor)(3, sx, sy);
        setcolor(sc);
        (*devvector)(sx[0], sy[0], 0);
        (*devvector)(sx[1], sy[1], 1);
        (*devvector)(sx[2], sy[2], 1);
        (*devvector)(sx[0], sy[0], 1);
    }
}

void symdiamond(int x, int y, double s, int f)
{
    int side = (int)(devsymsize * s);
    int sx[4], sy[4], sc;

    switch (f)
    {
    case 0:
        (*devvector)(x, y + side, 0);
        (*devvector)(x - side, y, 1);
        (*devvector)(x, y - side, 1);
        (*devvector)(x + side, y, 1);
        (*devvector)(x, y + side, 1);
        break;
    case 1:
        sx[0] = x;
        sx[1] = x - side;
        sx[2] = sx[0];
        sx[3] = x + side;
        sy[0] = y + side;
        sy[1] = y;
        sy[2] = y - side;
        sy[3] = sy[1];
        (*devfillcolor)(4, sx, sy);
        if (devtype == 0)
        {
            (*devvector)(x, y + side, 0);
            (*devvector)(x - side, y, 1);
            (*devvector)(x, y - side, 1);
            (*devvector)(x + side, y, 1);
        }
        break;
    case 2:
        sc = setcolor(0);
        sx[0] = x;
        sx[1] = x - side;
        sx[2] = sx[0];
        sx[3] = x + side;
        sy[0] = y + side;
        sy[1] = y;
        sy[2] = y - side;
        sy[3] = sy[1];
        (*devfillcolor)(4, sx, sy);
        setcolor(sc);
        (*devvector)(sx[0], sy[0], 0);
        (*devvector)(sx[1], sy[1], 1);
        (*devvector)(sx[2], sy[2], 1);
        (*devvector)(sx[3], sy[3], 1);
        (*devvector)(sx[0], sy[0], 1);
    }
}

void symplus(int x, int y, double s, int)
{
    int side = (int)(devsymsize * s);

    (*devvector)(x - side, y, 0);
    (*devvector)(x + side, y, 1);
    (*devvector)(x, y + side, 0);
    (*devvector)(x, y - side, 1);
}

void symx(int x, int y, double s, int)
{
    int side = (int)(devsymsize * s * 0.707);

    (*devvector)(x - side, y - side, 0);
    (*devvector)(x + side, y + side, 1);
    (*devvector)(x - side, y + side, 0);
    (*devvector)(x + side, y - side, 1);
}

void symstar(int /*x*/, int /*y*/, double /*s*/, int /*f*/)
{
}

void symsplat(int x, int y, double s, int f)
{
    symplus(x, y, s, f);
    symx(x, y, s, f);
}

/*
case SYM_HILOX:
case SYM_HILOY:
case SYM_OPENCLOSEX:
case SYM_OPENCLOSEY:
case SYM_CLOSEX:
case SYM_CLOSEY:
case SYM_HILO_OPCLX:
case SYM_HILO_OPCLX:
case SYM_ERRORX1:
case SYM_ERRORY1:
case SYM_ERRORX2:
case SYM_ERRORY2:
case SYM_ERRORXY:
*/
void drawpolysym(double *x, double *y, int len, int sym, int skip, int fill, double size)
{
    int i;
    char s[10];
    double xtmp, ytmp;

    skip += 1;
    switch (sym)
    {
    case SYM_DOT:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                my_move2(x[i], y[i]);
                my_draw2(x[i], y[i]);
            }
        }
        break;
    case SYM_CIRCLE:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symcircle((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_SQUARE:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symsquare((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_DIAMOND:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symdiamond((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_TRIANG1:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symtriangle1((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_TRIANG2:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symtriangle2((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_TRIANG3:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symtriangle3((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_TRIANG4:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symtriangle4((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_PLUS:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symplus((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_X:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symx((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_SPLAT:
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                symsplat((*devconvx)(x[i]), (*devconvy)(y[i]), size, fill);
            }
        }
        break;
    case SYM_IMPULSEY:
        for (i = 0; i < len; i += skip)
        {
            if (rxg1 < 0.0 && rxg2 > 0.0)
            {
                xtmp = 0.0;
            }
            else
            {
                xtmp = rxg1;
            }
            my_move2(xtmp, y[i]);
            my_draw2(x[i], y[i]);
        }
        break;
    case SYM_IMPULSEX:
        for (i = 0; i < len; i += skip)
        {
            if (ryg1 < 0.0 && ryg2 > 0.0)
            {
                ytmp = 0.0;
            }
            else
            {
                ytmp = ryg1;
            }
            my_move2(x[i], ytmp);
            my_draw2(x[i], y[i]);
        }
        break;
    case SYM_VERTX:
        for (i = 0; i < len; i += skip)
        {
            my_move2(x[i], ryg1);
            my_draw2(x[i], ryg2);
        }
        break;
    case SYM_VERTY:
        for (i = 0; i < len; i += skip)
        {
            my_move2(rxg1, y[i]);
            my_draw2(rxg2, y[i]);
        }
        break;
    case SYM_HISTOX: /* histogram x */
        for (i = 0; i < len - 1; i += skip)
        {
            rect(x[i], y[i], x[i + 1], 0.0);
        }
        break;
    case SYM_BARX: /* bar x */
        hdelta = barwid * (xg2 - xg1);
        for (i = 0; i < len; i++)
        {
            xhistbox(x[i], y[i]);
        }
        break;
    case SYM_HISTOY: /* histogram y */
        for (i = 0; i < len - 1; i += skip)
        {
            rect(x[i], y[i], 0.0, y[i + 1]);
        }
        break;
    case SYM_BARY: /* bar y */
        hdelta = barwid * (yg2 - yg1);
        for (i = 0; i < len; i++)
        {
            yhistbox(x[i], y[i]);
        }
        break;
    case SYM_STAIRX: /* stairstep */
        my_move2(x[0], y[0]);
        for (i = 0; i < len - 1; i += skip)
        {
            my_draw2(x[i + 1], y[i]);
            my_draw2(x[i + 1], y[i + 1]);
        }
        break;
    case SYM_STAIRY:
        my_move2(x[0], y[0]);
        for (i = 0; i < len - 1; i += skip)
        {
            my_draw2(x[i], y[i + 1]);
            my_draw2(x[i + 1], y[i + 1]);
        }
        break;
    case SYM_LOC: /* index of point */
        for (i = 0; i < len; i += skip)
        {
            if (symok(x[i], y[i]))
            {
                sprintf(s, "%d", i + 1);
                writestr(x[i], y[i], 0, 0, s);
            }
        }
        break;
    }
}

/*
 * draw the head of an arrow
 */
void draw_head(int ix1, int iy1, int ix2, int iy2, int sa, int type)
{
    double dist, ang, dx, dy;
    int xr, yr, xl, yl, s1, s2;
    int varrx[4], varry[4];

    dx = ix2 - ix1;
    dy = iy2 - iy1;
    if (dx == 0.0 && dy == 0.0)
    {
        errwin("Can't draw arrow, dx = dy = 0.0");
        return;
    }
    dist = my_hypot(dx, dy);
    ang = atan2(dy, dx);
    s1 = (int)((dist - sa) * cos(ang));
    s2 = (int)((dist - sa) * sin(ang));
    xr = (int)(sa * cos(ang + M_PI / 2.0) / 2.0);
    yr = (int)(sa * sin(ang + M_PI / 2.0) / 2.0);
    varrx[0] = s1 + ix1 - xr;
    varry[0] = s2 + iy1 - yr;
    varrx[1] = ix2;
    varry[1] = iy2;
    xl = (int)(sa * cos(ang - M_PI / 2.0) / 2.0);
    yl = (int)(sa * sin(ang - M_PI / 2.0) / 2.0);
    varrx[2] = s1 + ix1 - xl;
    varry[2] = s2 + iy1 - yl;
    switch (type)
    {
    case 0:
        (*devvector)(varrx[0], varry[0], 0);
        (*devvector)(varrx[1], varry[1], 1);
        (*devvector)(varrx[2], varry[2], 1);
        break;
    case 1:
        (*devfillcolor)(3, varrx, varry);
        break;
    case 2:
        (*devvector)(varrx[0], varry[0], 0);
        (*devvector)(varrx[1], varry[1], 1);
        (*devvector)(varrx[2], varry[2], 1);
        (*devvector)(varrx[0], varry[0], 1);
        break;
    }
}

/*
 * draw an arrow
 * end = 0 = none 1 = arrow at x1, y1  2 = arrow at
 * x2, y2 3 arrow at both ends
 */
void draw_arrow(double x1, double y1, double x2, double y2, int end, double asize, int type)
{
    int ix1, iy1, ix2, iy2;

    int sa = (int)(asize * devarrowlength);

    ix1 = (*devconvx)(x1);
    ix2 = (*devconvx)(x2);
    iy1 = (*devconvy)(y1);
    iy2 = (*devconvy)(y2);
    (*devvector)(ix1, iy1, 0);
    (*devvector)(ix2, iy2, 1);
    switch (end)
    {
    case 0:
        break;
    case 1:
        draw_head(ix2, iy2, ix1, iy1, sa, type);
        break;
    case 2:
        draw_head(ix1, iy1, ix2, iy2, sa, type);
        break;
    case 3:
        draw_head(ix2, iy2, ix1, iy1, sa, type);
        draw_head(ix1, iy1, ix2, iy2, sa, type);
        break;
    }
}

/*
 * flow fields are reserved for local use
 */

#define ADJ (8.0 * M_PI / 180.0)

void velplt(double xx, double yy, double u, double v, double vscale, int type)
{
    double x, y, s, xl, yl, xr, yr;
    double theta, i1, i2;
    int ix1, iy1;
    int ix2, iy2;
    double asize = 1.0;
    int sa = (int)(asize * devarrowlength);

    if (!symok(xx, yy))
    {
        return;
    }
    if ((fabs(u) > 1e-10) || (fabs(v) > 1e-10))
    {
        ix1 = (*devconvx)(xx);
        iy1 = (*devconvy)(yy);
        theta = atan2(v, u);
        x = u * vscale * devwidth / (double)devwidthmm;
        y = v * vscale * devheight / (double)devheightmm;
        ix2 = (int)(ix1 + x);
        iy2 = (int)(iy1 + y);
        s = my_hypot(x, y);
        (*devvector)(ix1, iy1, 0);
        (*devvector)(ix2, iy2, 1);
        switch (type)
        {
        case 0:
            i1 = theta + ADJ;
            i2 = theta - ADJ;
            xl = ix1 + s * 0.6 * cos(i1);
            yl = iy1 + s * 0.6 * sin(i1);
            xr = ix1 + s * 0.6 * cos(i2);
            yr = iy1 + s * 0.6 * sin(i2);
            (*devvector)((int)xl, (int)yl, 0);
            (*devvector)((int)(ix1 + x), (int)(iy1 + y), 1);
            (*devvector)((int)xr, (int)yr, 1);
            break;
        case 1:
            draw_head(ix1, iy1, ix2, iy2, sa, 0);
            break;
        case 2:
            draw_head(ix1, iy1, ix2, iy2, sa, 1);
            break;
        case 3:
            draw_head(ix1, iy1, ix2, iy2, sa, 2);
            break;
        }
    }
}

void drawsym(int x, int y, int sym, double size, int fill)
{
    switch (sym)
    {
    case SYM_DOT:
        (*devvector)(x, y, 0);
        (*devvector)(x, y, 1);
        break;
    case SYM_CIRCLE:
        symcircle(x, y, size, fill);
        break;
    case SYM_SQUARE:
        symsquare(x, y, size, fill);
        break;
    case SYM_DIAMOND:
        symdiamond(x, y, size, fill);
        break;
    case SYM_TRIANG1:
        symtriangle1(x, y, size, fill);
        break;
    case SYM_TRIANG2:
        symtriangle2(x, y, size, fill);
        break;
    case SYM_TRIANG3:
        symtriangle3(x, y, size, fill);
        break;
    case SYM_TRIANG4:
        symtriangle4(x, y, size, fill);
        break;
    case SYM_PLUS:
        symplus(x, y, size, fill);
        break;
    case SYM_X:
        symx(x, y, size, fill);
        break;
    case SYM_SPLAT:
        symsplat(x, y, size, fill);
        break;
    }
}

static int xm1, xm2, ym1, ym2; /* bounding box for rectangle around legend */

/*
 * draw the legend at world coordinates (x,y)
 */
void putlegend(int i, /* which set */
               int d,
               /* flag, 1 = no draw, just compute min/max
 * for bounding box */
               int xlen, /* length of legend */
               int ylen, /* distance between entries */
               double size, /* symbol size */
               double x, /* location x */
               double y, /* location y */
               int sy, /* symbol */
               int ly, /* line style */
               int cy, /* line color */
               int wy, /* line width */
               char *s, /* legend string */
               int fill, /* symbol fill */
               int sc, /* symbol color */
               int sw, /* symbol linewidth */
               int sl)
{ /* symbol linestyle */
    int itmp, scol, slins, slinw, xtmp, ytmp;
    static int ifudgex, ifudgey;

    xtmp = (*devconvx)(x);
    ytmp = (*devconvy)(y)-i *ylen *ifudgey;
    if (i == 0)
    {
        ifudgey = stringextenty(charsize * devcharsize, "N");
        ifudgex = stringextentx(charsize * devcharsize, "N");
        xm1 = xtmp;
        xm2 = xtmp + (xlen + 1) * ifudgex + stringextentx(charsize * devcharsize, s);
        ym2 = ytmp;
        ym1 = ytmp;
    }
    else
    {
        ym1 = ytmp;
        itmp = xtmp + (xlen + 1) * ifudgex + stringextentx(charsize * devcharsize, s);
        if (xm2 < itmp)
        {
            xm2 = itmp;
        }
    }
    if (d)
    {
        return;
    }
    scol = setcolor(cy);
    slins = setlinestyle(ly);
    slinw = setlinewidth(wy);
    (*devvector)(xtmp, ytmp, 0);
    if (ly)
    {
        (*devvector)(xtmp + xlen * ifudgex, ytmp, 1);
    }
    if ((sy > 0) && (sy <= SYM_SPLAT))
    {
        setcolor(sc);
        setlinestyle(sl);
        setlinewidth(sw);
        if (ly)
        {
            drawsym(xtmp, ytmp, sy, size, fill);
            drawsym(xtmp + xlen * ifudgex, ytmp, sy, size, fill);
        }
        else
        {
            drawsym(xtmp + xlen * ifudgex, ytmp, sy, size, fill);
        }
    }
    setcolor(scol);
    setlinestyle(slins);
    setlinewidth(slinw);
    (*devwritestr)(xtmp + (xlen + 1) * ifudgex, ytmp, 0, s, 0, 1);
}

/*
 * draw the legend for barcharts at world coordinates (x,y)
 */
void putbarlegend(int i, /* which set */
                  int d,
                  /* flag, 1 = no draw, just compute min/max
 * for bounding box */
                  int xlen, /* length of legend */
                  int ylen, /* distance between entries */
                  double, /* symbol size */
                  double x, /* location x */
                  double y, /* location y */
                  int, /* symbol */
                  int ly, /* line style */
                  int cy, /* line color */
                  int wy, /* line width */
                  char *s, /* legend string */
                  int, /* symbol fill */
                  int fu, /* fill using pattern or color */
                  int fc, /* fill color */
                  int fp)
{ /* fill pattern */
    int ipx[4], ipy[4], itmp, scol, slins, slinw, xtmp, ytmp;
    static int ifudgex, ifudgey;

    xtmp = (*devconvx)(x);
    ytmp = (*devconvy)(y)-i *ylen *ifudgey;
    if (i == 0)
    {
        ifudgey = stringextenty(charsize * devcharsize, "N");
        ifudgex = stringextentx(charsize * devcharsize, "N");
        xm1 = xtmp;
        xm2 = xtmp + (xlen + 1) * ifudgex + stringextentx(charsize * devcharsize, s);
        ym2 = ytmp;
        ym1 = ytmp;
    }
    else
    {
        ym1 = ytmp;
        itmp = xtmp + (xlen + 1) * ifudgex + stringextentx(charsize * devcharsize, s);
        if (xm2 < itmp)
        {
            xm2 = itmp;
        }
    }
    if (d)
    {
        return;
    }
    scol = setcolor(cy);
    slins = setlinestyle(ly);
    slinw = setlinewidth(wy);
    ipx[0] = xtmp;
    ipy[0] = ytmp - ifudgey;
    ipx[1] = xtmp + xlen * ifudgex;
    ipy[1] = ytmp - ifudgey;
    ipx[2] = xtmp + xlen * ifudgex;
    ipy[2] = ytmp + ifudgey;
    ipx[3] = xtmp;
    ipy[3] = ytmp + ifudgey;
    if (fu)
    {
        setpattern(fp);
        (*devfill)(4, ipx, ipy);
    }
    else
    {
        setcolor(fc);
        (*devfillcolor)(4, ipx, ipy);
    }
    if (ly && wy)
    {
        setcolor(cy);
        setlinestyle(ly);
        setlinewidth(wy);
        (*devvector)(ipx[0], ipy[0], 0);
        (*devvector)(ipx[1], ipy[1], 1);
        (*devvector)(ipx[2], ipy[2], 1);
        (*devvector)(ipx[3], ipy[3], 1);
        (*devvector)(ipx[0], ipy[0], 1);
    }
    setcolor(scol);
    setlinestyle(slins);
    setlinewidth(slinw);
    (*devwritestr)(xtmp + (xlen + 1) * ifudgex, ytmp, 0, s, 0, 1);
}

void putlegendrect(int fill, int fillusing, int fillcolor, int fillpat, int cy, int wy, int ly)
{
    int ifudgex, ifudgey;
    int ipx[4], ipy[4];
    int scol, slins, slinw;

    scol = setcolor(cy);
    slins = setlinestyle(ly);
    slinw = setlinewidth(wy);
    ifudgey = stringextenty(charsize * devcharsize, "N");
    ifudgex = stringextentx(charsize * devcharsize, "N");
    if (fill)
    {
        ipx[0] = xm1 - ifudgex;
        ipy[0] = ym1 - ifudgey;
        ipx[1] = xm1 - ifudgex;
        ipy[1] = ym2 + ifudgey;
        ipx[2] = xm2 + ifudgex;
        ipy[2] = ym2 + ifudgey;
        ipx[3] = xm2 + ifudgex;
        ipy[3] = ym1 - ifudgey;
        if (fillusing)
        {
            setcolor(fillcolor);
            (*devfillcolor)(4, ipx, ipy);
        }
        else
        {
            setpattern(fillpat);
            (*devfill)(4, ipx, ipy);
        }
    }
    setcolor(cy);
    (*devvector)(xm1 - ifudgex, ym1 - ifudgey, 0);
    (*devvector)(xm1 - ifudgex, ym2 + ifudgey, 1);
    (*devvector)(xm2 + ifudgex, ym2 + ifudgey, 1);
    (*devvector)(xm2 + ifudgex, ym1 - ifudgey, 1);
    (*devvector)(xm1 - ifudgex, ym1 - ifudgey, 1);
    setcolor(scol);
    setlinestyle(slins);
    setlinewidth(slinw);
}

void my_doublebuffer(int mode)
{
    xlibdoublebuffer(mode);
}

void my_frontbuffer(int mode)
{
    xlibfrontbuffer(mode);
}

void my_backbuffer(int mode)
{
    xlibbackbuffer(mode);
}

void my_swapbuffer(void)
{
    xlibswapbuffer();
}

/*
 * initialize the graphics device device
 * return -1 if unable to open device
 */
int initgraphics(int device)
{
    int retval;

    switch (device)
    {
    case 0:
        retval = xlibinitgraphics(1);
        break;
    case 1:
        retval = psinitgraphics(1);
        break;
    case 2:
        retval = psinitgraphics(3);
        break;
    case 3:
        retval = mifinitgraphics(1);
        break;
    case 4:
        retval = mifinitgraphics(3);
        break;
    case 5:
        retval = hpinitgraphics(5);
        break;
    case 6:
        retval = hpinitgraphics(3);
        break;
    case 7:
        retval = leafinitgraphics(1);
        break;
    case 8:
        retval = leafinitgraphics(3);
        break;
    case 11:
        retval = xlibinitgraphics(2);
        break;
    default:
        retval = -1;
    }
    if (retval != -1)
    {
        devtype = device;
        savexticl = devxticl;
        saveyticl = devyticl;
    }
    return retval;
}

void leavegraphics(void)
{
    (*devleavegraphics)();
}
