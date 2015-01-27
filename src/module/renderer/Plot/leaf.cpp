/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*  Improved driver for Interleaf ASCII document format
 *
 *  by
 *
 *  Kidambi Sreenivas (sree@erc.msstate.edu)
 *
 *  with help from
 *
 *  Sudarshan Kadambi (sudi@erc.msstate.edu)
 *
 *  Enhancements over the Interleaf driver of Mike Putnam (putnam@nosc.mil):
 *
 *  (1) Font changes are recognized within a text string.
 *  (2) Modifiers (\S\s\u\U\+\-\b\N) are all recognized.
 *  (3) The original text string is broken up into pieces and shows up in
 *      Interleaf with respective fonts and sizes (see (2) below).
 *  (4) Subscripts and Superscripts are smaller in size than regular text.
 *  (5) Successive levels of subscripting/superscripting reduces font size.
 *
 *  Known Bugs:
 *
 *  (1) Have to end modifiers (\s\S\+\-\b) with a "\N".
 *  (2) Modifiers might not show up at the exact right places, but they will
 *      all be there!
 *  (3) A string begining with a "\u" runs into problems with font type & size.
 *  (4) The other limitations of Mike's original driver still exist.
 *
 */

/* driver for Inteleaf ASCII document format
 * by Mike Putnam
 * putnam@nosc.mil
 *
 * based upon:
 *
 *      Framemaker driver by
 *      A. Snyder
 *      asnyder@artorg.hmc.psu.edu
 *
 *	postscript printer driver by
 *	Jim Hudgens
 *	hudgens@ray.met.fsu.edu
 *
 *
 * Notes:
 *
 *	Pattern fills on bar graphs are a bit flakey, sometimes they show
 *        sometimes not. Color fills are more dependable.
 *
 *      You can't change fonts within a string.
 *
 *      Center and right justification on lines of text rotated other
 *        than in 90 deg increments may not line up correctly.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "externs.h"
#include "extern.h"
// #include <Xm/Xm.h>
// #include "xprotos.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern char version[];
extern double charsize;
extern double devcharsize;
extern int ptofile;
extern char printstr[];
extern unsigned char red[], green[], blue[];

static int strsplit(char *s1, char *s2);
static int strchkslash(char *s);
static void newcoord(char *s, int *x, int *y, int rot, int flag, int fontsize);

/*
 * printer control string
 */
#ifndef LEAF_PRSTR
char leaf_prstr[128] = "cat >acegr.leaf <";

#else
char leaf_prstr[128] = LEAF_PRSTR;
#endif

/* Assume a landscape-shaped area 5.5 x 4.25 " (half size).  Address in 0.001" increments */
#define LEAFXMIN 0
#define LEAFXMAX 10500
#define LEAFYMIN 0
#define LEAFYMAX 8000
#define DXLEAF 10500
#define DYLEAF 8000
#define CHARS 6.0
#define TICL 80

/* Alternative is a portrait-shaped area 5.5 x 6 " */
#define LEAFXMINP 0
#define LEAFXMAXP 8000
#define LEAFYMINP 0
#define LEAFYMAXP 10500
#define DXLEAFP 8000
#define DYLEAFP 10500
#define CHARSP 6.0
#define TICLP 80

#define MINCOLOR 0
#define MAXCOLOR 15
#define MINLINEWIDTH 1
#define MAXLINEWIDTH 9
#define MINLINESTYLE 0
#define MAXLINESTYLE 6
#define MINPATTERN 0
#define MAXPATTERN 15

#define LINEWIDTHINC 0.5

// #define PORTRAIT 0
// #define LANDSCAPE 1

static int leafxmin = LEAFXMIN;
// static int leafxmax = LEAFXMAX;
static int leafymin = LEAFYMIN;
static int leafymax = LEAFYMAX;
static int leafdx = DXLEAF;
static int leafdy = DYLEAF;
static int leafcolor = 0;
static int leaflinewidth = -1;
static int leafdmode;
static int leafpattern = 5;
static int leaffont = 0;
static double leafcharsize = 1.5;
static int leafticl;
static int leaflinestyle;
static char *fname;
// static int orientflag = PORTRAIT;
static int styles[5] = { 0, 5, 4, 1, 2 };
static int pattern[16] = { 5, 10, 9, 8, 3, 2, 1, 4, 5, 11, 10, 9, 8, 3, 2, 1 };

struct ileaffonts
{
    char fontnames[14];
};
static struct ileaffonts fontlist[] = {
    { "wst:timsps\0" },
    { "wst:timsps\0" },
    { "wst:timsps\0" },
    { "wst:timsps\0" },
    { "wst:helvps\0" },
    { "wst:helvps\0" },
    { "wst:helvps\0" },
    { "wst:helvps\0" },
    { "grk:dutchbs\0" },
    { "sym:clas\0" },
    { "sym:clas\0" }
};
static struct ileaffonts fonttypelist[] = {
    { " \0" },
    { "b\0" },
    { "i\0" },
    { "bi\0" },
    { " \0" },
    { "b\0" },
    { "i\0" },
    { "bi\0" },
    { " \0" },
    { " \0" },
    { " \0" }
};

static int x_current = 99999, y_current = 99999;
double xconv(double x), yconv(double y);
static void my_dispstrleaf(int x, int y, int rot, char *string, int just, int fudge);
static FILE *leafout;

static void stroke(void)
{
    int i;

    if (pathlength > 1)
    {
        fprintf(leafout, "(p8,1,0,,%d,%d,127\n", leafpattern, leafcolor);
        fprintf(leafout, "  (g9,1,0\n");
        fprintf(leafout, "    (g9,1,0\n");
        for (i = 1; i < pathlength; i++)
        {
            fprintf(leafout, "      (v7,1,0,%6.3f,%6.3f,%6.3f,%6.3f,%d,0,%d,%d)\n",
                    (double)xpoints[i - 1] * 0.001, (double)(leafymax - ypoints[i - 1]) * 0.001,
                    (double)xpoints[i] * 0.001, (double)(leafymax - ypoints[i]) * 0.001,
                    leafcolor, leaflinewidth, styles[leaflinestyle - 1]);
        } /* for i */
        fprintf(leafout, ")))\n");
        xpoints[0] = xpoints[--pathlength];
        ypoints[0] = ypoints[pathlength];
        pathlength = 1;

    } /* if pathlength */
}

void otherdefs(void)
{
    int i;

    fprintf(leafout, "<!Color Definitions,\n");
    for (i = 0; i <= MAXCOLOR; i++)
    {
        fprintf(leafout, "C%d =  %5.1f, %5.1f, %5.1f, 0.0,\n",
                i, (float)((255.0 - (float)red[i]) * 0.3922),
                (float)((255.0 - (float)green[i]) * 0.3922),
                (float)((255.0 - (float)blue[i]) * 0.3922));
    } /* for ncolors */
    fprintf(leafout, ">\n");
    fprintf(leafout, "\n");
    fprintf(leafout, "\n");
    fprintf(leafout, "<\"para\">\n");
    fprintf(leafout, "<Frame,\n");
    fprintf(leafout, "	Placement =		Overlay,\n");
}

static char tbuf[128];
int leafsetmode(int mode)
{
    char sysbuf[128];
    // char *mktemp(char *);

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
        if ((leafout = fopen(fname, "w")) == NULL)
        {
            return 0;
        }
    }
    devoffsx = devoffsy = 0;
    switch (mode)
    {
    case 1: /* Interleaf landscape */
        // orientflag = 1 /* LANDSCAPE */;
        leafcharsize = CHARS;
        leafxmin = LEAFXMIN;
        // leafxmax = LEAFXMAX;
        leafymin = LEAFYMIN;
        leafymax = LEAFYMAX;
        leafdx = DXLEAF;
        leafdy = DYLEAF;
        devwidth = DXLEAF;
        devwidthmm = (int)(DXLEAF / 1000.0 * 25.4);
        devheight = DYLEAF;
        devheightmm = (int)(DYLEAF / 1000.0 * 25.4);
        leafticl = TICL;
        fprintf(leafout, "<!OPS, Version = 8.0>\n");
        fprintf(leafout, "<!Class, \"para\">\n");
        fprintf(leafout, "<!Page,\n");
        fprintf(leafout, "  Height = 8.5 Inches,\n");
        fprintf(leafout, "  Width = 11.0 Inches>\n");
        otherdefs();
        fprintf(leafout, "	Width =			11.0 Inches,\n");
        fprintf(leafout, "	Height =		8.5 Inches,\n");
        fprintf(leafout, "	Diagram =\n");
        fprintf(leafout, "V11,\n");
        fprintf(leafout, "(g9,1,0,\n");
        fprintf(leafout, "(g9,1,0,\n");
        break;
    case 3: /* Interleaf portrait */
        // orientflag = 0 /*PORTRAIT*/;
        leafcharsize = CHARSP;
        leafxmin = LEAFXMINP;
        // leafxmax = LEAFXMAXP;
        leafymin = LEAFYMINP;
        leafymax = LEAFYMAXP;
        leafdx = DXLEAFP;
        leafdy = DYLEAFP;
        devwidth = DXLEAFP;
        devwidthmm = (int)(DXLEAFP / 1000.0 * 25.4);
        devheight = DYLEAFP;
        devheightmm = (int)(DYLEAFP / 1000.0 * 25.4);
        leafticl = TICLP;
        fprintf(leafout, "<!OPS, Version = 8.0>\n");
        fprintf(leafout, "<!Class, \"para\">\n");
        fprintf(leafout, "<!Page,\n");
        fprintf(leafout, "  Height = 11.0 Inches,\n");
        fprintf(leafout, "  Width = 8.5 Inches>\n");
        otherdefs();
        fprintf(leafout, "	Width =			8.50 Inches,\n");
        fprintf(leafout, "	Height =		11.0 Inches,\n");
        fprintf(leafout, "	Diagram =\n");
        fprintf(leafout, "V11,\n");
        fprintf(leafout, "(g9,1,0,\n");
        fprintf(leafout, "(g9,1,0,\n");
        break;
    case 2:
    case 4:
        stroke();
        fprintf(leafout, "))>\n");
        fclose(leafout);
        if (!ptofile)
        {
            sprintf(sysbuf, "%s %s", leaf_prstr, fname);
            system(sysbuf);
            unlink(fname);
        }
        // orientflag = 0 /*PORTRAIT*/;
        break;
    }
    return mode;
}

void drawleaf(int x2, int y2, int mode)
{
    int xtmp, ytmp;

    if (x2 < 0 || y2 < 0) /* Eliminate garbage on output */
        return;

    xtmp = x2;
    ytmp = y2;

    if (mode) /* draw */
    {
        if (pathlength == MAXLINELEN)
        {
            stroke();
            xpoints[0] = xpoints[MAXLINELEN - 1];
            ypoints[0] = ypoints[MAXLINELEN - 1];
        }
    } /* moveto */
    else
    {
        /* Avoid excessive moveto's generated by grtool */
        if (xtmp == x_current && ytmp == y_current)
            return;
        stroke();
        pathlength = 0;
    }
    xpoints[pathlength] = xtmp;
    ypoints[pathlength++] = ytmp;
    x_current = xtmp;
    y_current = ytmp;
}

int xconvleaf(double x)
{
    return ((int)(leafxmin + leafdx * xconv(x)));
}

int yconvleaf(double y)
{
    return ((int)(leafymin + leafdy * yconv(y)));
}

int leafsetcolor(int c)
{
    if (c != leafcolor)
    {
        stroke();
        if ((leafcolor = c) > MAXCOLOR)
        {
            leafcolor = 1;
        }
        else if (leafcolor < MINCOLOR)
        {
            leafcolor = 1;
        }
    }
    return c;
}

int leafsetlinewidth(int c)
{
    if (c != leaflinewidth)
    {
        stroke();
        if ((c = c % MAXLINEWIDTH) < MINLINEWIDTH)
            c = MINLINEWIDTH;
    }
    leaflinewidth = c;
    return c;
}

void leafdrawtic(int x, int y, int dir, int updown)
{
    switch (dir)
    {
    case 0:
        switch (updown)
        {
        case 0:
            drawleaf(x, y, 0);
            drawleaf(x, y + devxticl, 1);
            break;
        case 1:
            drawleaf(x, y, 0);
            drawleaf(x, y - devxticl, 1);
            break;
        }
        break;
    case 1:
        switch (updown)
        {
        case 0:
            drawleaf(x, y, 0);
            drawleaf(x + devyticl, y, 1);
            break;
        case 1:
            drawleaf(x, y, 0);
            drawleaf(x - devyticl, y, 1);
            break;
        }
        break;
    }
}

int leafsetlinestyle(int style)
{
    if (style == leaflinestyle)
    {
        return (leaflinestyle);
    }
    stroke();
    if ((leaflinestyle = style) < MINLINESTYLE)
        leaflinestyle = MINLINESTYLE;
    else if (leaflinestyle > MAXLINESTYLE)
        leaflinestyle = MAXLINESTYLE;
    return (leaflinestyle = style);
}

char leafcurfont[128];
static int leaffontsize = 15;

void leafsetfont(int n)
{
    leaffont = n;
}

void leafsetfontsize(double size)
{
    leaffontsize = (int)(size * 15);
}

void dispstrleaf(int x, int y, int rot, char *s, int just, int fudge, char underline, int *leafsavefontsize, int *leafsavefonttype)
{
    int ilenx, ileny, oldx, oldy;
    double si, co, scale, rotrad;
    static int leafjust[] = { 0, 2, 1 };

    rotrad = M_PI / 180.0 * rot;
    si = sin(rotrad);
    co = cos(rotrad);

    scale = leaffontsize / 2.0;
    *leafsavefontsize = leaffontsize;
    *leafsavefonttype = leaffont;

    ilenx = stringextentx(scale, s);
    ileny = 0;
    oldx = x;
    oldy = y;
    switch (just)
    {
    case 1:
        x = (int)(x - co * ilenx);
        y = (int)(y - si * ilenx);
        break;
    case 2:
        x = (int)(x - (co * ilenx) / 2);
        y = (int)(y - (si * ilenx) / 2);
        break;
    }
    if (strlen(s) > 0)
    {
        if ((rot == 0) || (rot == 90) || (rot == 180) || (rot == 270))
        {
            if (fudge)
            {
                fprintf(leafout, "   (t14,1,0,%lf6.3,%lf6.3,%d,%d,0,%d,%c,%s%d%s,",
                        oldx * 0.001, (leafymax - (oldy - leaffontsize / 216.0 * 1000.0)) * 0.001,
                        leafjust[just], leafcolor, -rot, underline, (char *)&fontlist[leaffont], leaffontsize, (char *)&fonttypelist[leaffont]);
            }
            else
            {
                fprintf(leafout, "   (t14,1,0,%lf6.3,%lf6.3,%d,%d,0,%d,%c,%s%d%s,",
                        oldx * 0.001, (leafymax - oldy) * 0.001,
                        leafjust[just], leafcolor, -rot, underline, (char *)&fontlist[leaffont], leaffontsize, (char *)&fonttypelist[leaffont]);
            }
        }
        /* if rotrad */
        else
        {
            if (fudge)
            {
                fprintf(leafout, "   (o4,1,0,%lf6.3,%lf6.3,%lf6.3,%s%d%s,",
                        x * 0.001, (leafymax - (y - leaffontsize / 216.0 * 1000.0)) * 0.001, -rotrad,
                        (char *)&fontlist[leaffont], leaffontsize, (char *)&fonttypelist[leaffont]);
            }
            else
            {
                fprintf(leafout, "   (o4,1,0,%lf6.3,%lf6.3,%lf6.3,%s%d%s,",
                        x * 0.001, (leafymax - y) * 0.001, -rotrad,
                        (char *)&fontlist[leaffont], leaffontsize, (char *)&fonttypelist[leaffont]);
            }
        } /* else rotrad */
        putleaf(s);
    }
}

void putleaf(char *s)
{
    int i, slen = strlen(s);
    double saves = leaffontsize / 15.0 /*, scale = leaffontsize / 15.0*/;

    for (i = 0; i < slen; i++)
    {
        if (isoneof(s[i], (char *)"<>(), "))
        {
            fprintf(leafout, "\\%c", s[i]);
        }
        else if (s[i] == '\\')
        {
            fprintf(leafout, "\\\\");
        }
        else
        {
            fprintf(leafout, "%c", s[i]);
        };
    } /* for s[i] */
    fprintf(leafout, ")\n");
    leafsetfontsize(saves);
}

int leafsetpat(int k)
{
    if (k > 15)
    {
        k = 15;
    }
    else if (k < 0)
    {
        k = 0;
        stroke();
    }
    return (leafpattern = pattern[k]);
}

void leaffill(int n, int *px, int *py)
{
    int i;

    stroke();
    if (n)
    {
        fprintf(leafout, "(p8,1,0,,%d,%d,0\n", leafpattern, leafcolor);
        fprintf(leafout, "  (g9,1,0\n");
        fprintf(leafout, "    (g9,1,0\n");
        for (i = 1; i < n; i++)
        {
            fprintf(leafout, "      (v7,1,0,%6.3f,%6.3f,%6.3f,%6.3f,7,127,%d,%d)\n",
                    (double)px[i - 1] * 0.001, (double)(leafymax - py[i - 1]) * 0.001,
                    (double)px[i] * 0.001, (double)(leafymax - py[i]) * 0.001,
                    leaflinewidth, styles[leaflinestyle - 1]);
        } /* for i */
        fprintf(leafout, ")))\n");
        pathlength = 0;

    } /* if n */
}

void leaffillcolor(int n, int *px, int *py)
{
    int i;

    stroke();
    if (n)
    {
        fprintf(leafout, "(p8,1,0,,%d,%d,0\n", 5, leafcolor);
        fprintf(leafout, "(g9,1,0\n");
        fprintf(leafout, "  (g9,1,0\n");
        for (i = 1; i < n; i++)
        {
            fprintf(leafout, "    (v7,1,0,%6.3f,%6.3f,%6.3f,%6.3f,7,127,%d,%d)\n",
                    (double)px[i - 1] * 0.001, (double)(leafymax - py[i - 1]) * 0.001,
                    (double)px[i] * 0.001, (double)(leafymax - py[i]) * 0.001,
                    leaflinewidth, styles[leaflinestyle - 1]);
        } /* for i */
        fprintf(leafout, ")))\n");
        pathlength = 0;
    } /* if n */
}

void leafdrawarc(int x, int y, int r)
{

    stroke();
    fprintf(leafout, "(e8,1,0,\n");
    fprintf(leafout, " %6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,\n",
            (x - r) * 0.001, (leafymax - y - r) * 0.001,
            (x + r) * 0.001, (leafymax - y - r) * 0.001,
            (x - r) * 0.001, (leafymax - y + r) * 0.001);
    fprintf(leafout, "  %d,127,%d,%d,0,%d,%d)\n", leafcolor, leafpattern, leafcolor, leaflinewidth, styles[leaflinestyle - 1]);
    pathlength = 0;
}

void leaffillarc(int x, int y, int r)
{

    stroke();
    fprintf(leafout, "(e8,1,0,\n");
    fprintf(leafout, " %6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,\n",
            (x - r) * 0.001, (leafymax - y - r) * 0.001,
            (x + r) * 0.001, (leafymax - y - r) * 0.001,
            (x - r) * 0.001, (leafymax - y + r) * 0.001);
    fprintf(leafout, "  %d,0,%d,%d,0,%d,%d)\n", leafcolor, leafpattern, leafcolor, leaflinewidth, styles[leaflinestyle - 1]);

    pathlength = 0;
}

void leafdrawellipse(int x, int y, int xm, int ym)
{

    stroke();
    fprintf(leafout, "(e8,1,0,\n");
    fprintf(leafout, " %6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,\n",
            (x - xm) * 0.001, (leafymax - y - ym) * 0.001,
            (x + xm) * 0.001, (leafymax - y - ym) * 0.001,
            (x - xm) * 0.001, (leafymax - y + ym) * 0.001);
    fprintf(leafout, "  %d,127,%d,%d,0,%d,%d)\n", leafcolor, leafpattern, leafcolor, leaflinewidth, styles[leaflinestyle - 1]);
    pathlength = 0;
}

void leaffillellipse(int x, int y, int xm, int)
{
    stroke();
    fprintf(leafout, "(e8,1,0,\n");
    fprintf(leafout, " %6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,\n",
            (x - xm) * 0.001, (leafymax - y - xm) * 0.001,
            (x + xm) * 0.001, (leafymax - y - xm) * 0.001,
            (x - xm) * 0.001, (leafymax - y + xm) * 0.001);
    fprintf(leafout, "  %d,0,%d,%d,0,%d,%d)\n", leafcolor, leafpattern, leafcolor, leaflinewidth, styles[leaflinestyle - 1]);
    pathlength = 0;
}

void leafleavegraphics(void)
{
    leafsetmode(leafdmode + 1);
}

/*           leaf initialization routine  */
int leafinitgraphics(int dmode)
{
    leafdmode = dmode;
    pathlength = 0;
    if (!leafsetmode(leafdmode))
    {
        return -1;
    }
    devconvx = xconvleaf;
    devconvy = yconvleaf;
    devvector = drawleaf;
    /*
       devwritestr = dispstrleaf;
   */
    devwritestr = my_dispstrleaf;
    devsetcolor = leafsetcolor;
    devsetfont = leafsetfont;
    devsetline = leafsetlinestyle;
    devsetlinew = leafsetlinewidth;
    devdrawtic = leafdrawtic;
    devsetpat = leafsetpat;
    devdrawarc = leafdrawarc;
    devfillarc = leaffillarc;
    devdrawellipse = leafdrawellipse;
    devfillellipse = leaffillellipse;
    devfill = leaffill;
    devfillcolor = leaffillcolor;
    devleavegraphics = leafleavegraphics;
    devcharsize = leafcharsize;
    devxticl = leafticl;
    devyticl = leafticl;
    devsymsize = leafticl;
    devarrowlength = 80;
    leafsetcolor(1);
    leafsetlinewidth(2);
    setlinestyle(0);
    setfont(2);

    return (0);
}

/*
 * additional functions
 */
static int fonttype;
static int savefontsize, fontsize;
static double size;
static int saveleaffontsize;
static int xsave, ysave;
static int xtot, ytot;
static int kount = 0;
// static int savefonttype;
static int saveleaffonttype;
static char underline = ' ';

static void my_dispstrleaf(int x, int y, int rot, char *string, int just, int fudge)
{
    int sstype;
    int flag;
    int kounter = 0;
    char s[256], sd[256], st[256];

    //savefonttype = leaffont;
    fonttype = leaffont;
    fontsize = leaffontsize;

    fudge = 0;
    xtot = 0;
    ytot = 0;
    xsave = x;
    ysave = y;

    strcpy(s, string);
    st[0] = '\0';
    flag = strchkslash(s);

    while ((sstype = strsplit(s, sd)) == 1)
    {
        dispstrleaf(x, y, rot, s, just, fudge, underline, &saveleaffontsize, &saveleaffonttype);
        kounter++;
        if (kounter == 1)
        {
            fontsize = saveleaffontsize;
            fonttype = saveleaffonttype;
        }
        strcpy(st, s);
        strcpy(s, sd);
        sd[0] = '\0';
        flag = strchkslash(s);
        newcoord(st, &x, &y, rot, flag, fontsize);
    }

    dispstrleaf(x, y, rot, s, just, fudge, underline, &saveleaffontsize, &saveleaffonttype);
}

/*
 * Checks for font and attribute changes and sets fonttype and fontsize
 */
static int strchkslash(char *s)
{
    char st[2];
    int i, slen, flag = 0;

    slen = strlen(s);
    if (s[0] == '\\')
    {
        st[0] = s[1];
        st[1] = '\0';
        flag = 0;

        switch (st[0])
        {

        /* Check for fontchange */

        case '0':
            fonttype = 0;
            break;
        case '1':
            fonttype = 1;
            break;
        case '2':
            fonttype = 2;
            break;
        case '3':
            fonttype = 3;
            break;
        case '4':
            fonttype = 4;
            break;
        case '5':
            fonttype = 5;
            break;
        case '6':
            fonttype = 6;
            break;
        case '7':
            fonttype = 7;
            break;
        case '8':
            fonttype = 8;
            break;
        case '9':
            fonttype = 9;
            break;
        case 'x':
            fonttype = 10;
            break;

        /* Check for attribute changes */

        case 'b':
            if (kount == 0)
            {
                savefontsize = fontsize;
                kount++;
            }
            flag = 3;
            break;
        case 's':
            if (kount == 0)
            {
                savefontsize = fontsize;
                kount++;
            }
            fontsize = (int)(fontsize * 0.6 + 0.5);
            flag = -1;
            break;
        case 'S':
            if (kount == 0)
            {
                savefontsize = fontsize;
                kount++;
            }
            fontsize = (int)(fontsize * 0.6 + 0.5);
            flag = 1;
            break;
        case 'u':
            underline = 'u';
            break;
        case 'U':
            underline = ' ';
            break;
        case 'N':
            fontsize = savefontsize;
            kount = 0;
            flag = 2;
            break;
        case '+':
            if (kount == 0)
            {
                savefontsize = fontsize;
                kount++;
            }
            fontsize += 3;
            break;
        case '-':
            if (kount == 0)
            {
                savefontsize = fontsize;
                kount++;
            }
            fontsize -= 3;
            break;
        }
        leafsetfont(fonttype);
        if (fontsize == 0)
            fontsize = 2;
        size = fontsize / 15.0;
        leafsetfontsize(size);
        if (isoneof(st[0], (char *)"0123456789xsS+-NbuU"))
            for (i = 2; i <= slen; i++)
                s[i - 2] = s[i];
    }
    return (flag);
}

/*
 *  Computes new coordinates for the broken up text strings
 */
static void newcoord(char *s, int *x, int *y, int rot, int flag, int fontsize)
{

    int ilenx, ileny;
    int xinc, yinc;
    int facux = 0, facuy = 0, facdx = 0, facdy = 0;
    int moveup, movedn;
    int moveudist, moveddist;
    int bkspce, bkspc;
    int bkspc1, bkspc2;
    double scale;
    double si, co, rotrad;
    static char bksp[] = { ' ', '\0' };
    static int prevlenx;

    if (strlen(s) > 0)
    {

        rotrad = M_PI * rot / 180.0;
        si = sin(rotrad);
        co = cos(rotrad);

        scale = fontsize / 2.0;
        bkspc = stringextentx(scale, bksp);
        ilenx = stringextentx(scale, s);
        prevlenx += bkspc;
        ilenx = ilenx + bkspc;
        ileny = 0;

        moveudist = fontsize;
        moveddist = (int)(scale + 0.5);

        switch (flag)
        {

        case -1:
            moveup = 0;
            movedn = 1;
            bkspce = 0;
            break;
        case 0:
            moveup = 0;
            movedn = 0;
            bkspce = 0;
            break;
        case 1:
            moveup = 1;
            movedn = 0;
            bkspce = 0;
            break;
        case 2:
            moveup = 0;
            movedn = 0;
            bkspce = 0;
            break;
        case 3:
            moveup = 0;
            movedn = 0;
            bkspce = -2 * bkspc;
            break;
        default:
            moveup = 0;
            movedn = 0;
            bkspce = 0;
            break;
        }

        switch (rot)
        {

        case 0:
            facux = 0;
            facuy = 1;
            facdx = 0;
            facdy = -1;
            break;
        case 90:
            facux = -1;
            facuy = 0;
            facdx = 1;
            facdy = 0;
            break;
        case 180:
            facux = 0;
            facuy = -1;
            facdx = 0;
            facdy = 1;
            break;
        case 270:
            facux = 1;
            facuy = 0;
            facdx = -1;
            facdy = 0;
            break;
        }

        xinc = (int)((ilenx + bkspce) * co);
        yinc = (int)((ilenx + bkspce) * si);

        if (flag == 2)
        {

            bkspc1 = (int)(2.0 * prevlenx * co);
            bkspc2 = (int)(2.0 * prevlenx * si);

            *x = xsave + xtot - bkspc1 + xinc;
            *y = ysave + ytot - bkspc2 + yinc;

            xtot = xtot - bkspc1 + xinc;
            ytot = ytot - bkspc2 + yinc;

            prevlenx = 0;
        }
        else
        {

            *x = *x + xinc + moveudist * moveup * facux + moveddist * movedn * facdx;
            *y = *y + yinc + moveudist * moveup + facuy + moveddist * movedn * facdy;

            xtot = xtot + xinc;
            ytot = ytot + yinc;
        }
    }
}

/*
 *  Splits a given string into two at a "\" followed by the standard xv(m)gr modifiers
 */
static int strsplit(char *s1, char *s2)
{

    int ls1, i, j;

    ls1 = strlen(s1);

    if (ls1 > 1)
    {
        for (i = 0; i < ls1; i++)
        {

            if (s1[i] == '\\')
            {
                if (isoneof(s1[i + 1], (char *)"0123456789xsS+-NbuU"))
                {
                    for (j = i; j < ls1; j++)
                        s2[j - i] = s1[j];
                    s1[i] = '\0';
                    s2[ls1 - i] = '\0';
                    return (1);
                }
            }
        }
        s2[0] = '\0';
        return (0);
    }
    return (-1);
}
