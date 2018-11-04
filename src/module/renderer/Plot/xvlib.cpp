/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: xvlib.c,v 1.2 1994/07/30 04:35:02 pturner Exp pturner $
 *
 * driver for xlib for gr
 *
 */

#include <stdio.h>
#include <ctype.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <Xm/Xm.h>
#include "extern.h"

#include "externs.h"
#include "patterns.h"
#include "xprotos.h"

double xconv(double x);
double yconv(double y);

/* external variables */
extern Display *disp;
extern GC gc;
extern GC gcxor;
extern GC gcclr;
extern Window xwin;
extern XGCValues gc_val;

extern int use_colors;
extern int use_defaultcmap;
extern int revflag; /* defined in main.c */
extern int inwin;
extern int use_xvertext;

static int save_images;
char saveimage_prstr[256] = "ACEgr.xwd";

extern double devcharsize;
extern double charsize;

#define NUM_COLORS 256

unsigned char red[NUM_COLORS], green[NUM_COLORS], blue[NUM_COLORS];
unsigned long colors[NUM_COLORS];
XColor xc[NUM_COLORS];
int ncolors;
Colormap cmap, mycmap;

extern Widget canvas;
extern Widget app_shell;
XColor cmscolors[256];

#define MINCOLOR 0
#define MAXLINEW 15

extern int maxcolors;

static int xlibcolor = 1;
static int xliblinewidth = 0;
static int xlibdmode;
static int xlibfont = 0;
static int xliblinestyle = 1;
static int doublebuff = 0; /* no double buffering by default */
static Pixmap backbuff, displaybuff;
int win_h, win_w;

extern int backingstore;
Pixmap backpix;

static void xlibinit(void);
//static void xlib_scrunch(XPoint * p, int *n);
void flush_pending(void);

/*
 * the following is a tunable parameter and may
 * need to be adjusted
 */
#ifdef HIRES
double xlibcharsize = 0.80;

#else
double xlibcharsize = 0.60;
#endif

/*
 * fix for dotted/dashed linestyles
 */
XPoint polypoints[MAXLINELEN];
int xpoints[MAXLINELEN], ypoints[MAXLINELEN];
/* global, used in other
 * drivers */
int pathlength = 0;

static char solid[1] = { 1 };
static char dotted[2] = { 3, 1 };
static char shortdashed[2] = { 3, 3 };
static char longdashed[2] = { 7, 7 };
static char dot_dashed[4] = { 1, 3, 7, 3 };

static char *dash_list[] = {
    solid,
    dotted,
    shortdashed,
    longdashed,
    dot_dashed
};

static int dash_list_length[] = { 1, 2, 2, 2, 4 };

void get_xlib_dims(int *w, int *h)
{
    Arg args;
    Dimension ww, wh;

    XtSetArg(args, XmNwidth, &ww);
    XtGetValues(canvas, &args, 1);
    XtSetArg(args, XmNheight, &wh);
    XtGetValues(canvas, &args, 1);
    *w = ww;
    *h = wh;
}

static void xlibinit(void)
{
    double wx1, wx2, wyy1, wy2;
    static int inc = 1;
    extern int overlay, doclear, bgcolor;

    Arg args;
    Dimension ww, wh;

    disp = XtDisplay(canvas);
    xwin = XtWindow(canvas);
    XtSetArg(args, XmNwidth, &ww);
    XtGetValues(canvas, &args, 1);
    XtSetArg(args, XmNheight, &wh);
    XtGetValues(canvas, &args, 1);
    win_w = ww;
    win_h = wh;

    devwidth = win_w;
    devheight = win_h;
    wx1 = DisplayWidth(disp, DefaultScreen(disp));
    wx2 = DisplayWidthMM(disp, DefaultScreen(disp));
    wyy1 = DisplayHeight(disp, DefaultScreen(disp));
    wy2 = DisplayHeightMM(disp, DefaultScreen(disp));
    devwidthmm = (int)(wx2 / wx1 * win_w);
    devheightmm = (int)(wy2 / wyy1 * win_h);
    if (inc)
    {
        /*
         gc = DefaultGC(disp, DefaultScreen(disp));
         gc_val.foreground = WhitePixel(disp, DefaultScreen(disp));
         if (invert) {
             gc_val.function = GXinvert;
         } else {
             gc_val.function = GXxor;
         }
         gcxor = XCreateGC(disp, xwin, GCFunction | GCForeground, &gc_val);
         gc_val.foreground = colors[0];
         gc_val.function = GXcopy;
      gcclr = XCreateGC(disp, xwin, GCFunction | GCForeground, &gc_val);
      */
        xlibinit_tiles();
        if (backingstore)
        {
            backpix = XCreatePixmap(disp, DefaultRootWindow(disp), win_w, win_h, DisplayPlanes(disp, DefaultScreen(disp)));
        }
        inc = 0;
    }
    if (doublebuff)
    {
        xlibdoublebuffer(doublebuff);
        displaybuff = backbuff;
    }
    else
    {
        displaybuff = xwin;
    }
    if (doclear && !overlay)
    {
        xlibsetcolor(bgcolor);
        XFillRectangle(disp, displaybuff, gc, 0, 0, win_w, win_h);
        if (backingstore)
        {
            XFillRectangle(disp, backpix, gc, 0, 0, win_w, win_h);
        }
    }
}

void xlibdoublebuffer(int mode)
{
    extern int inwin;

    doublebuff = mode;
    if (!inwin)
    {
        return;
    }
    if (mode)
    {
        if (!backbuff)
        {
            backbuff = XCreatePixmap(disp, DefaultRootWindow(disp), win_w, win_h, DisplayPlanes(disp, DefaultScreen(disp)));
        }
        displaybuff = backbuff;
    }
    else
    {
        if (backbuff)
        {
            XFreePixmap(disp, backbuff);
            backbuff = (Pixmap)NULL;
            displaybuff = xwin;
        }
    }
}

void xlibfrontbuffer(int mode)
{
    extern int inwin;

    if (!inwin)
    {
        return;
    }
    if (mode)
    {
        displaybuff = xwin;
    }
    else
    {
        if (doublebuff && backbuff)
        {
            displaybuff = backbuff;
        }
    }
}

void xlibbackbuffer(int mode)
{
    extern int inwin;

    if (!inwin)
    {
        return;
    }
    if (mode && doublebuff && backbuff)
    {
        displaybuff = backbuff;
    }
    else
    {
        displaybuff = xwin;
    }
}

void xlibswapbuffer(void)
{
    extern int inwin;

    if (!inwin)
    {
        return;
    }
    if (doublebuff && backbuff)
    {
        XCopyArea(disp, displaybuff, xwin, gc, 0, 0, win_w, win_h, 0, 0);
    }
}

void refresh_from_backpix(void)
{
    if (backpix)
    {
        XCopyArea(disp, backpix, xwin, gc, 0, 0, win_w, win_h, 0, 0);
    }
}

void resize_backpix(void)
{
    XFreePixmap(disp, backpix);
    backpix = XCreatePixmap(disp, DefaultRootWindow(disp), win_w, win_h, DisplayPlanes(disp, DefaultScreen(disp)));
}

static int xlib_write_mode = 1;

void set_write_mode(int m)
{
    flush_pending();
    xlib_write_mode = m;
}

void xlibsetmode(int mode)
{
    switch (mode)
    {
    case 1:
        xlibinit();
        break;
    case 2:
        flush_pending();
        if (doublebuff && backbuff)
        {
            xlibswapbuffer();
        }
        if (save_images)
        {
            save_image_on_disk(disp, xwin, displaybuff, 0, 0, win_w, win_h, saveimage_prstr, (Colormap)NULL);
        }
        break;
    }
}

/* scrunch a pair of integer arrays */
void scrunch_points(int *x, int *y, int *n)
{
    int i, cnt = 0;
    cnt = 0;
    for (i = 0; i < *n - 1; i++)
    {
        if (x[cnt] == x[i + 1] && y[cnt] == y[i + 1])
        {
        }
        else
        {
            cnt++;
            x[cnt] = x[i + 1];
            y[cnt] = y[i + 1];
        }
    }
    cnt++;
    if (cnt < 2)
    {
        cnt = 2;
        x[1] = x[*n + 1];
        y[1] = y[*n + 1];
    }
    *n = cnt;
}

/* scrunch the array of XPoints */
void scrunch_xpoints(XPoint *p, int *n)
{
    int i, cnt = 0;
    cnt = 0;
    for (i = 0; i < *n - 1; i++)
    {
        if (p[cnt].x == p[i + 1].x && p[cnt].y == p[i + 1].y)
        {
        }
        else
        {
            cnt++;
            p[cnt] = p[i + 1];
        }
    }
    cnt++;
    if (cnt <= 2)
    {
        cnt = 2;
        p[1] = p[*n - 1];
    }
    *n = cnt;
}

void flush_pending(void)
{
    if (pathlength > 1)
    {
        if (pathlength > 3)
        {
            scrunch_xpoints(polypoints, &pathlength);
        }
        if (xlib_write_mode)
        {
            XDrawLines(disp, displaybuff, gc, polypoints, pathlength, CoordModeOrigin);
            if (backingstore)
            {
                XDrawLines(disp, backpix, gc, polypoints, pathlength, CoordModeOrigin);
            }
        }
        else
        {
            XDrawLines(disp, displaybuff, gcclr, polypoints, pathlength, CoordModeOrigin);
            if (backingstore)
            {
                XDrawLines(disp, backpix, gcclr, polypoints, pathlength, CoordModeOrigin);
            }
        }
    }
    pathlength = 0;
}

static int x1, yy1;

void drawxlib(int x, int y, int mode)
{
    if (mode)
    {
        polypoints[pathlength].x = x;
        polypoints[pathlength].y = win_h - y;
        pathlength++;
        if (pathlength == MAXLINELEN)
        {
            flush_pending();
            polypoints[pathlength].x = x;
            polypoints[pathlength].y = win_h - y;
            pathlength = 1;
        }
    }
    else
    {
        if ((x == x1 && y == yy1))
        {
            return;
        }
        else
        {
            flush_pending();
            polypoints[pathlength].x = x;
            polypoints[pathlength].y = win_h - y;
            pathlength = 1;
        }
    }
    x1 = x;
    yy1 = y;
}

int xconvxlib(double x)
{
    return ((int)(win_w * xconv(x)));
}

int yconvxlib(double y)
{
    return ((int)(win_h * yconv(y)));
}

/*
 * initialize_cms_data()
 *    Initialize the colormap segment data and setup the RGB values.
 */
void initialize_cms_data(void)
{
    int i, del;

    /* white  */
    red[0] = 255;
    green[0] = 255;
    blue[0] = 255;
    /* black    */
    red[1] = 0;
    green[1] = 0;
    blue[1] = 0;
    /* red    */
    red[2] = 255;
    green[2] = 0;
    blue[2] = 0;
    /* green  */
    red[3] = 0;
    green[3] = 255;
    blue[3] = 0;
    /* blue   */
    red[4] = 0;
    green[4] = 0;
    blue[4] = 255;
    /* yellow */
    red[5] = 255;
    green[5] = 255;
    blue[5] = 0;
    /* brown  */
    red[6] = 188;
    green[6] = 143;
    blue[6] = 143;
    /* gray   */
    red[7] = 220;
    green[7] = 220;
    blue[7] = 220;
    /* violet  */
    red[8] = 148;
    green[8] = 0;
    blue[8] = 211;
    /* cyan  */
    red[9] = 0;
    green[9] = 255;
    blue[9] = 255;
    /* magenta  */
    red[10] = 255;
    green[10] = 0;
    blue[10] = 211;
    /* orange  */
    red[11] = 255;
    green[11] = 138;
    blue[11] = 0;
    /* blue violet  */
    red[12] = 114;
    green[12] = 33;
    blue[12] = 188;
    /* maroon  */
    red[13] = 103;
    green[13] = 7;
    blue[13] = 72;
    /* turquoise  */
    red[14] = 72;
    green[14] = 209;
    blue[14] = 204;
    /* forest green  */
    red[15] = 85;
    green[15] = 192;
    blue[15] = 52;
    del = (maxcolors - 16) / 3;
    for (i = 16; i < maxcolors; i++)
    {
        red[i] = (i - 16) * 4 * ((i - 16) < del);
        green[i] = (i - 16) * 3 * ((i - 16) < 2 * del);
        blue[i] = (i - 16) * 2 * ((i - 16) <= maxcolors);
    }
    for (i = 0; i < maxcolors; i++)
    {
        cmscolors[i].red = red[i];
        cmscolors[i].green = green[i];
        cmscolors[i].blue = blue[i];
    }
}

void write_image(char *fname)
{
    set_wait_cursor();
    save_image_on_disk(disp, xwin, displaybuff, 0, 0, win_w, win_h, fname, mycmap);
    unset_wait_cursor();
}

/* NOTE: not called by xvgr */
void xlibinitcmap(void)
{
    int i;

    ncolors = DisplayCells(disp, DefaultScreen(disp));
    if (ncolors > 256)
    {
        ncolors = 256;
    }
    if (ncolors > 16)
    {
        cmap = DefaultColormap(disp, DefaultScreen(disp));
        for (i = 0; i < ncolors; i++)
        {
            xc[i].pixel = i;
            xc[i].flags = DoRed | DoGreen | DoBlue;
        }
        if (!use_defaultcmap)
        {
            XQueryColors(disp, cmap, xc, ncolors);
            mycmap = XCreateColormap(disp, xwin, DefaultVisual(disp, DefaultScreen(disp)), AllocAll);
        }
        else
        {
            mycmap = cmap;
        }
        for (i = 2; i < maxcolors; i++)
        {
            xc[i].red = red[i] << 8;
            xc[i].green = green[i] << 8;
            xc[i].blue = blue[i] << 8;
            if (use_defaultcmap)
            {
                if (!XAllocColor(disp, cmap, &xc[i]))
                {
                    fprintf(stderr, " Can't allocate color\n");
                }
            }
            colors[i] = xc[i].pixel;
        }
        if (!use_defaultcmap)
        {
            XStoreColors(disp, mycmap, xc, ncolors);
        }
    }
    if (revflag)
    {
        colors[1] = WhitePixel(disp, DefaultScreen(disp));
        colors[0] = BlackPixel(disp, DefaultScreen(disp));
    }
    else
    {
        colors[0] = WhitePixel(disp, DefaultScreen(disp));
        colors[1] = BlackPixel(disp, DefaultScreen(disp));
    }
}

void xlibsetcmap(int i, int r, int g, int b)
{
    XColor xct;

    red[i] = r;
    green[i] = g;
    blue[i] = b;
    cmscolors[i].red = red[i];
    cmscolors[i].green = green[i];
    cmscolors[i].blue = blue[i];
    if (inwin && use_colors > 4 && i >= 2)
    {
        xct.green = g << 8;
        xct.blue = b << 8;
        xct.red = r << 8;
        xct.flags = DoRed | DoGreen | DoBlue;
        xct.pixel = colors[i];
        xct.pad = 0;

        XStoreColor(disp, mycmap, &xct);
    }
}

int xlibsetlinewidth(int c)
{
    flush_pending();
    x1 = yy1 = 99999;
    if (c)
    {
        c = c % MAXLINEW;
        if (c == 0)
        {
            c = 1;
        }
        if (xliblinestyle <= 1)
        {
            XSetLineAttributes(disp, gc, c - 1 == 0 ? 0 : c, LineSolid, CapButt, JoinRound);
        }
        else
        {
            XSetLineAttributes(disp, gc, c - 1 == 0 ? 0 : c, LineOnOffDash, CapButt, JoinRound);
            XSetDashes(disp, gc, 0,
                       dash_list[xliblinestyle - 1],
                       dash_list_length[xliblinestyle - 1]);
        }
    }
    return (xliblinewidth = c);
}

int xlibsetlinestyle(int style)
{
    flush_pending();
    x1 = yy1 = 99999;
    if (style > 1 && xliblinewidth)
    {
        XSetLineAttributes(disp, gc, xliblinewidth - 1 == 0 ? 0 : xliblinewidth, LineOnOffDash, CapButt, JoinRound);
        XSetDashes(disp, gc, 0, dash_list[style - 1], dash_list_length[style - 1]);
    }
    else if (style == 1 && xliblinewidth)
    {
        XSetLineAttributes(disp, gc, xliblinewidth - 1 == 0 ? 0 : xliblinewidth, LineSolid, CapButt, JoinRound);
    }
    return (xliblinestyle = style);
}

int xlibsetcolor(int c)
{

    flush_pending();

    x1 = yy1 = 99999;
    c = c % maxcolors;

    if (use_colors > 4)
    {
        XSetForeground(disp, gc, colors[c]);
    }
    else
    {
        XSetForeground(disp, gc, colors[c == 0 ? 0 : 1]);
    }
    xlibcolor = c;
    return c;
}

void xlibdrawtic(int x, int y, int dir, int updown)
{
    switch (dir)
    {
    case 0:
        switch (updown)
        {
        case 0:
            drawxlib(x, y, 0);
            drawxlib(x, y + devxticl, 1);
            break;
        case 1:
            drawxlib(x, y, 0);
            drawxlib(x, y - devxticl, 1);
            break;
        }
        break;
    case 1:
        switch (updown)
        {
        case 0:
            drawxlib(x, y, 0);
            drawxlib(x + devyticl, y, 1);
            break;
        case 1:
            drawxlib(x, y, 0);
            drawxlib(x - devyticl, y, 1);
            break;
        }
        break;
    }
}

void xlibsetfont(int n)
{
    flush_pending();
    x1 = yy1 = 99999;
    hselectfont(xlibfont = n);
}

void dispstrxlib(int x, int y, int rot, char *s, int just, int fudge)
{
    (void)fudge;
    flush_pending();
    x1 = yy1 = 99999;
    puthersh(x, y, xlibcharsize * charsize, rot, just, xlibcolor, drawxlib, s);
    flush_pending();
    x1 = yy1 = 99999;
}

#define MAXPATTERNS 16

static int patno = 0;

static Pixmap tiles[30];
static Pixmap curtile;

void xlibinit_tiles(void)
{
    int i;
    Pixmap ptmp;

    for (i = 0; i < MAXPATTERNS; i++)
    {
        tiles[i] = XCreatePixmap(disp, xwin, 16, 16, DisplayPlanes(disp, DefaultScreen(disp)));
    }
    for (i = 0; i < MAXPATTERNS; i++)
    {
        if (tiles[i] == (Pixmap)NULL)
        {
            printf("bad tile %d\n", i);
        }
        else
        {
            XFillRectangle(disp, tiles[i], gcclr, 0, 0, 16, 16);
        }
    }
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat0_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[0], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat1_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[1], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat2_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[2], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat3_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[3], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat4_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[4], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat5_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[5], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat6_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[6], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat7_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[7], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat8_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[8], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat9_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[9], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat10_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[10], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat11_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[11], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat12_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[12], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat13_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[13], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat14_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[14], gc, 0, 0, 16, 16, 0, 0, 1);
    ptmp = XCreateBitmapFromData(disp, xwin, (char *)pat15_bits, 16, 16);
    XCopyPlane(disp, ptmp, tiles[15], gc, 0, 0, 16, 16, 0, 0, 1);
    curtile = tiles[0];
}

int xlibsetpat(int k)
{
    patno = k;
    if (k > MAXPATTERNS)
    {
        k = 1;
    }
    if (patno != 0)
    {
        curtile = tiles[k - 1];
    }
    return 0;
}

void xlibfill(int n, int *px, int *py)
{
    int i;
    XPoint *p;

    p = (XPoint *)calloc(n, sizeof(XPoint));
    if (p == NULL)
    {
        return;
    }
    if (patno == 0)
    {
        return;
    }
    XSetFillStyle(disp, gc, FillTiled);
    XSetTile(disp, gc, curtile);
    for (i = 0; i < n; i++)
    {
        p[i].x = px[i];
        p[i].y = win_h - py[i];
    }
    XFillPolygon(disp, displaybuff, gc, p, n, Nonconvex, CoordModeOrigin);
    if (backingstore)
    {
        XFillPolygon(disp, backpix, gc, p, n, Nonconvex, CoordModeOrigin);
    }
    XSetFillStyle(disp, gc, FillSolid);
    free(p);
}

void xlibfillcolor(int n, int *px, int *py)
{
    int i;
    XPoint *p;

    p = (XPoint *)calloc(n, sizeof(XPoint));
    if (p == NULL)
    {
        return;
    }
    for (i = 0; i < n; i++)
    {
        p[i].x = px[i];
        p[i].y = win_h - py[i];
    }
    XFillPolygon(disp, displaybuff, gc, p, n, Nonconvex, CoordModeOrigin);
    if (backingstore)
    {
        XFillPolygon(disp, backpix, gc, p, n, Nonconvex, CoordModeOrigin);
    }
    free(p);
}

void xlibdrawarc(int x, int y, int r)
{
    XDrawArc(disp, displaybuff, gc, x - r, win_h - (y + r), 2 * r, 2 * r, 0, 360 * 64);
    if (backingstore)
    {
        XDrawArc(disp, backpix, gc, x - r, win_h - (y + r), 2 * r, 2 * r, 0, 360 * 64);
    }
}

void xlibfillarc(int x, int y, int r)
{
    XFillArc(disp, displaybuff, gc, x - r, win_h - (y + r), 2 * r, 2 * r, 0, 360 * 64);
    if (backingstore)
    {
        XFillArc(disp, backpix, gc, x - r, win_h - (y + r), 2 * r, 2 * r, 0, 360 * 64);
    }
}

void xlibdrawellipse(int x, int y, int xm, int ym)
{
    if (xm == 0)
    {
        xm = 1;
    }
    if (ym == 0)
    {
        ym = 1;
    }
    XDrawArc(disp, displaybuff, gc, x - xm, win_h - (y + ym), 2 * xm, 2 * ym, 0, 360 * 64);
    if (backingstore)
    {
        XDrawArc(disp, backpix, gc, x - xm, win_h - (y + ym), 2 * xm, 2 * ym, 0, 360 * 64);
    }
}

void xlibfillellipse(int x, int y, int xm, int ym)
{
    if (xm == 0)
    {
        xm = 1;
    }
    if (ym == 0)
    {
        ym = 1;
    }
    XFillArc(disp, displaybuff, gc, x - xm, win_h - (y + ym), 2 * xm, 2 * ym, 0, 360 * 64);
    if (backingstore)
    {
        XFillArc(disp, backpix, gc, x - xm, win_h - (y + ym), 2 * xm, 2 * ym, 0, 360 * 64);
    }
}

void xlibleavegraphics(void)
{
    flush_pending();
    x1 = yy1 = 99999;
    xlibsetmode(2);
    save_images = 0;
    XFlush(disp);
}

int xlibinitgraphics(int dmode)
{
    pathlength = 0;
    x1 = 99999;
    yy1 = 99999;
    if (dmode > 1)
    {
        save_images = 1;
        dmode = 0;
    }
    xlibdmode = dmode;
    xlibsetmode(1);
    devorient = 1;
    devconvx = xconvxlib;
    devconvy = yconvxlib;
    devvector = drawxlib;
    devwritestr = dispstrxlib;
    devsetcolor = xlibsetcolor;
    devsetfont = xlibsetfont;
    devsetline = xlibsetlinestyle;
    devsetlinew = xlibsetlinewidth;
    devdrawtic = xlibdrawtic;
    devsetpat = xlibsetpat;
    devdrawarc = xlibdrawarc;
    devfillarc = xlibfillarc;
    devdrawellipse = xlibdrawellipse;
    devfillellipse = xlibfillellipse;
    devfill = xlibfill;
    devfillcolor = xlibfillcolor;
    devleavegraphics = xlibleavegraphics;
    devxticl = 12;
    devyticl = 12;
    devarrowlength = 12;
    devsymsize = 6;
    devcharsize = xlibcharsize;
    (*devsetcolor)(1);
    xlibsetlinestyle(1);
    return 0;
}

/*
 * cursors
 */

#include <X11/cursorfont.h>

static Cursor wait_cursor;
static Cursor line_cursor;
static Cursor find_cursor;
static Cursor move_cursor;
static Cursor text_cursor;
static Cursor kill_cursor;
static int cur_cursor = -1;
static int waitcursoron = FALSE;

void DefineDialogCursor(Cursor c);
void UndefineDialogCursor();

void set_wait_cursor()
{
    XDefineCursor(disp, xwin, wait_cursor);
    DefineDialogCursor(wait_cursor);
    waitcursoron = TRUE;
}

void unset_wait_cursor()
{
    UndefineDialogCursor();
    if (cur_cursor == -1)
    {
        XUndefineCursor(disp, xwin);
    }
    else
    {
        set_cursor(cur_cursor);
    }
    waitcursoron = FALSE;
}

void set_cursor(int c)
{
    XUndefineCursor(disp, xwin);
    cur_cursor = -1;
    switch (c)
    {
    case 0:
        XDefineCursor(disp, xwin, line_cursor);
        cur_cursor = 0;
        break;
    case 1:
        XDefineCursor(disp, xwin, find_cursor);
        cur_cursor = 1;
        break;
    case 2:
        XDefineCursor(disp, xwin, text_cursor);
        cur_cursor = 2;
        break;
    case 3:
        XDefineCursor(disp, xwin, kill_cursor);
        cur_cursor = 3;
        break;
    case 4:
        XDefineCursor(disp, xwin, move_cursor);
        cur_cursor = 4;
        break;
    }
    XFlush(disp);
}

void set_window_cursor(Window xwin, int c)
{
    XUndefineCursor(disp, xwin);
    switch (c)
    {
    case 0:
        XDefineCursor(disp, xwin, line_cursor);
        break;
    case 1:
        XDefineCursor(disp, xwin, find_cursor);
        break;
    case 2:
        XDefineCursor(disp, xwin, text_cursor);
        break;
    case 3:
        XDefineCursor(disp, xwin, kill_cursor);
        break;
    case 4:
        XDefineCursor(disp, xwin, move_cursor);
        break;
    case 5:
        XDefineCursor(disp, xwin, wait_cursor);
        break;
    }
    XFlush(disp);
}

void init_cursors(void)
{
    wait_cursor = XCreateFontCursor(disp, XC_watch);
    line_cursor = XCreateFontCursor(disp, XC_crosshair);
    find_cursor = XCreateFontCursor(disp, XC_hand2);
    text_cursor = XCreateFontCursor(disp, XC_xterm);
    kill_cursor = XCreateFontCursor(disp, XC_pirate);
    move_cursor = XCreateFontCursor(disp, XC_fleur);
    cur_cursor = -1;
}
