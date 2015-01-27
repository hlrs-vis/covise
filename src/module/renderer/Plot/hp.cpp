/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: hp.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * driver for an hp 7550, and HPGL in general
 * modified by A. Feldt, winter 1992, to be more efficient and
 * also to assume that the expanded HPGL instruction set is used
 * (as on our HP plotter)
 * also added the drawarc, fillarc, fillcolor routines (using extended
 * HPGL inst. set)
 *
 */

#include <stdio.h>
#include "externs.h"
#include "extern.h"
extern double charsize;
extern double devcharsize;
extern int ptofile;
extern char printstr[];
extern char *curprint; /* curprint = hp_prstr */

/*
 * spool using these
 */
#ifndef HP_PRSTR1
char hp_prstr1[128] = "/usr/ucb/lpr -Php -h";

#else
char hp_prstr1[128] = HP_PRSTR1;
#endif

#ifndef HP_PRSTR2
char hp_prstr2[128] = "/usr/ucb/lpr -h";

#else
char hp_prstr2[128] = HP_PRSTR2;
#endif

// static char tmpbuf[64];

#define HPXMIN 0
#define HPXMAX 10170
#define HPYMIN 0
#define HPYMAX 7840
#define DXHP 10170
#define DYHP 7840

#define HP2XMIN 0
#define HP2XMAX 16450
#define HP2YMIN 0
#define HP2YMAX 10170
#define DXHP2 16450
#define DYHP2 10170

#define MINCOLOR 0
#define MAXCOLOR 8
#define MAXLINEWIDTH 1
#define MAXLINESTYLE 14

// static int hpxmin = HPXMIN;
// static int hpxmax = HPXMAX;
// static int hpymin = HPYMIN;
static int hpymax = HPYMAX;
static int hpdx = DXHP;
static int hpdy = DYHP;

static int hpcolor;
static int hpdmode;
static int hpfont = 0;
static double hpcharsize = 6.1;
static int hplinestyle;
static int hpfillpat = 0;

double xconv(double x), yconv(double y);

static FILE *hpout;

void putstrhp(char *s)
{
    fprintf(hpout, "%s", s);
}

static char *fname;
static char tbuf[128];

static int orientflag = 0;

int hpsetmode(int mode)
{
    char sysbuf[128];
    //char *mktemp(char *);

    if (mode % 2)
    {
        if (!ptofile)
        {
            strcpy(tbuf, "/tmp/ACEgrXXXXXX");
            mkstemp(tbuf);
            fname = tbuf;
        }
        else
        {
            fname = printstr;
        }
        hpout = fopen(fname, "w");
        if (hpout != NULL)
        {
            if ((mode == 5) || (mode == 7))
                putstrhp((char *)"IN;DF;SP1\n");
            else
                putstrhp((char *)"PG;IN;DF;SP1\n");
        }
        else
        {
            return 0;
        }
    }
    switch (mode)
    {
    case 3: /* HP portrait */
    case 7: /* HPGL to LJII portrait */
        orientflag = 1;
    case 1: /* HP landscape */
    case 5: /* HPGL to LJII landscape */
        // hpxmin = HPXMIN;
        // hpxmax = HPXMAX;
        // hpymin = HPYMIN;
        hpymax = HPYMAX;
        hpdx = DXHP;
        hpdy = DYHP;
        break;
    case 11: /* portrait 11x17 */
        orientflag = 1;
    case 9: /* HP landscape 11x17 */
        // hpxmin = HP2XMIN;
        // hpxmax = HP2XMAX;
        // hpymin = HP2YMIN;
        hpymax = HP2YMAX;
        hpdx = DXHP2;
        hpdy = DYHP2;
        break;
    case 2: /* HP landscape */
    case 4: /* portrait */
    case 10: /* HP landscape 11x17 */
    case 12: /* portrait */
        putstrhp((char *)"PU0,0;SP 0;\n");
        fclose(hpout);
        if (!ptofile)
        {
            sprintf(sysbuf, "%s %s", curprint, fname);
            system(sysbuf);
            unlink(fname);
        }
        orientflag = 0;
        break;
    case 6: /* HPGL to Laserjet series II landscape */
    case 8: /* HPGL tp Laserjet Series II portrait */
        putstrhp((char *)"PU0,0;SP 0;\n");
        fclose(hpout);
        if (!ptofile)
        {
            sprintf(sysbuf, "%s %s", curprint, fname);
            system(sysbuf);
            unlink(fname);
        }
        orientflag = 0;
        break;
    }
    return 1;
}

static int x1 = 99999, yy1 = 99999;

void drawhp(int x2, int y2, int mode)
{
    if (orientflag)
    {
        int xtmp, ytmp;
        char stmp[30];

        xtmp = y2;
        ytmp = (-x2) + hpymax;

        if (mode)
        {
            sprintf(stmp, "PD%1d,%1d\n", xtmp, ytmp);
            putstrhp(stmp);
        }
        else
        {
            if (!(x1 == xtmp && yy1 == ytmp))
            {
                sprintf(stmp, "PU%1d,%1d\n", xtmp, ytmp);
                putstrhp(stmp);
            }
        }
        x1 = xtmp;
        yy1 = ytmp;
    }
    else
    {
        char stmp[30];

        if (mode)
        {
            sprintf(stmp, "PD%1d,%1d\n", x2, y2);
            putstrhp(stmp);
        }
        else
        {
            if (!(x1 == x2 && yy1 == y2))
            {
                sprintf(stmp, "PU%1d,%1d\n", x2, y2);
                putstrhp(stmp);
            }
        }
        x1 = x2;
        yy1 = y2;
    }
}

int xconvhp(double x)
{
    if (orientflag)
    {
        return ((int)(hpdy * xconv(x)));
    }
    else
    {
        return ((int)(hpdx * xconv(x)));
    }
}

int yconvhp(double y)
{
    if (orientflag)
    {
        return ((int)(hpdx * yconv(y)));
    }
    else
    {
        return ((int)(hpdy * yconv(y)));
    }
}

void hpsetfont(int n)
{
    hselectfont(hpfont = n);
}

int hpsetcolor(int c)
{
    if (c)
    {
        char stmp[10];

        c = (c - 1) % MAXCOLOR + 1;
        sprintf(stmp, "SP%d;PU\n", c);
        putstrhp(stmp);
    }
    hpcolor = c;
    return c;
}

/* only one line width */
int hpsetlinewidth(int c)
{
    return c;
}

void hpdrawtic(int x, int y, int dir, int updown)
{
    switch (dir)
    {
    case 0:
        switch (updown)
        {
        case 0:
            drawhp(x, y, 0);
            drawhp(x, y + devxticl, 1);
            break;
        case 1:
            drawhp(x, y, 0);
            drawhp(x, y - devxticl, 1);
            break;
        }
        break;
    case 1:
        switch (updown)
        {
        case 0:
            drawhp(x, y, 0);
            drawhp(x + devyticl, y, 1);
            break;
        case 1:
            drawhp(x, y, 0);
            drawhp(x - devyticl, y, 1);
            break;
        }
        break;
    }
}

int hpsetlinestyle(int style)
{
    char stmp[20];

    switch (style)
    {
    case 1:
        strcpy(stmp, "LT;");
        break;
    case 2:
        strcpy(stmp, "LT2,1;");
        break;
    case 3:
        strcpy(stmp, "LT2,2;");
        break;
    case 4:
        strcpy(stmp, "LT2,3;");
        break;
    case 5:
        strcpy(stmp, "LT2,4;");
        break;
    case 6:
        strcpy(stmp, "LT4,2;");
        break;
    case 7:
        strcpy(stmp, "LT5,2;");
        break;
    case 8:
        strcpy(stmp, "LT6,2;");
        break;
    default:
        strcpy(stmp, "LT;");
        break;
    }
    putstrhp(stmp);
    return (hplinestyle = style);
}

void dispstrhp(int x, int y, int rot, char *s, int just, int)
{
    puthersh(x, y, hpcharsize * charsize, rot, just, hpcolor, drawhp, s);
}

int hpsetpat(int pat)
{
    return (hpfillpat = pat % 7);
}

int setpathp(int pat)
{
    char stmp[20];

    switch (pat)
    {
    case 0:
        return (0);
    case 1:
        strcpy(stmp, "FT1,100,0;");
        break;
    case 2:
        strcpy(stmp, "FT3,100,45;");
        break;
    case 3:
        strcpy(stmp, "FT4,100,0;");
        break;
    case 4:
        strcpy(stmp, "FT3,100,90;");
        break;
    case 5:
        strcpy(stmp, "FT3,100,0;");
        break;
    case 6:
        strcpy(stmp, "FT4,100,45;");
        break;
    default:
        return (0);
    }
    putstrhp(stmp);
    return (pat);
}

void hpfill(int n, int *px, int *py)
{
    int j, xtmp, ytmp;
    char stmp[30];

    if (setpathp(hpfillpat) == 0)
        return;
    if (orientflag)
    {
        xtmp = py[0];
        ytmp = (-px[0]) + hpymax;
        sprintf(stmp, "PU%1d,%1d\n", xtmp, ytmp);
        putstrhp(stmp);
        strcpy(stmp, "PM0;PD");
        putstrhp(stmp);
        for (j = 1; j < n; j++)
        {
            xtmp = py[j];
            ytmp = (-px[j]) + hpymax;
            sprintf(stmp, "%1d,%1d,", xtmp, ytmp);
            putstrhp(stmp);
        }
    }
    else
    {
        sprintf(stmp, "PU%1d,%1d\n", px[0], py[0]);
        putstrhp(stmp);
        strcpy(stmp, "PM0;PD");
        putstrhp(stmp);
        for (j = 1; j < n; j++)
        {
            sprintf(stmp, "%1d,%1d,", px[j], py[j]);
            putstrhp(stmp);
        }
    }
    sprintf(stmp, ";PM2;FP\n");
    putstrhp(stmp);
}

void hpfillcolor(int n, int *px, int *py)
{
    int oldpat;

    oldpat = hpfillpat;
    hpfillpat = 1;
    hpfill(n, px, py);
    hpfillpat = oldpat;
}

void hpleavegraphics(void)
{
    hpsetmode(hpdmode + 1);
}

void hpdrawarc(int x, int y, int r)
{
    char stmp[30];

    if (orientflag)
    {
        int xtmp, ytmp;

        xtmp = y;
        ytmp = (-x) + hpymax;
        sprintf(stmp, "PU%1d,%1d;", xtmp, ytmp);
    }
    else
    {
        sprintf(stmp, "PU%1d,%1d;", x, y);
    }
    putstrhp(stmp);
    sprintf(stmp, "CI%1d\n", r);
    putstrhp(stmp);
}

void hpfillarc(int x, int y, int r)
{
    char stmp[30];

    setpathp(1);
    if (orientflag)
    {
        int xtmp, ytmp;

        xtmp = y;
        ytmp = (-x) + hpymax;
        sprintf(stmp, "PU%1d,%1d;", xtmp, ytmp);
    }
    else
    {
        sprintf(stmp, "PU%1d,%1d;", x, y);
    }
    putstrhp(stmp);
    sprintf(stmp, "PM0;CI%1d;PM2;FP\n", r);
    putstrhp(stmp);
}

void hpdrawellipse(int /*x*/, int /*y*/, int /*xm*/, int /*ym*/)
{
}

void hpfillellipse(int /*x*/, int /*y*/, int /*xm*/, int /*ym*/)
{
}

int hpinitgraphics(int dmode)
{
    hpdmode = dmode;
    if (!hpsetmode(hpdmode))
    {
        return -1;
    }
    devconvx = xconvhp;
    devconvy = yconvhp;
    devvector = drawhp;
    devwritestr = dispstrhp;
    devsetcolor = hpsetcolor;
    devsetfont = hpsetfont;
    devsetline = hpsetlinestyle;
    devsetlinew = hpsetlinewidth;
    devdrawtic = hpdrawtic;
    devsetpat = hpsetpat;
    devfill = hpfill;
    devdrawarc = hpdrawarc;
    devfillarc = hpfillarc;
    devfillcolor = hpfillcolor;
    devdrawellipse = hpdrawellipse;
    devfillellipse = hpfillellipse;
    devleavegraphics = hpleavegraphics;
    devcharsize = hpcharsize;
    devsymsize = 60;
    devxticl = 60;
    devyticl = 60;
    devarrowlength = 60;
    setfont(2);
    setcolor(1);
    setlinestyle(0);
    return 0;
}
