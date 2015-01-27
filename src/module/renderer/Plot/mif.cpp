/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: mif.c,v 1.2 1994/06/08 02:09:02 pturner Exp pturner $
 * driver for FrameMaker .mif format
 * Alan J. Snyder
 * asnyder@artorg.hmc.psu.edu
 *
 * based upon:
 *	postscript printer driver by
 *	Jim Hudgens
 *	hudgens@ray.met.fsu.edu
 *
 * 	Further modifications by,
 * 	Ole Holm Nielsen
 * 	ohnielse@ltf.dth.dk
 *
 * Notes:
 *	Color is not implemented.
 *	Area fills 0-15 are done with FrameMaker patterns 15, 1-14 respectively.
 *	Pen patterns 1-5 map to Frame pen patterns 0, 1, 4, 10, 13.
 *
 * Area fills are untested.
 */

#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include "externs.h"

#include "extern.h"

extern char version[];
extern double charsize;
extern double devcharsize;
extern int ptofile;
extern char printstr[];

void putmif(char *s);

/*
 * printer control string
 */
#ifndef MIF_PRSTR
char mif_prstr[128] = "cat >acegr.mif <";

#else
char mif_prstr[128] = MIF_PRSTR;
#endif

/* Assume a landscape-shaped area 5.5 x 4.25 " (half size).  Address in 0.001" increments */
#define MIFXMIN 0
#define MIFXMAX 10500
#define MIFYMIN 0
#define MIFYMAX 8000
#define DXMIF 10500
#define DYMIF 8000
#define CHARS 6.0
#define TICL 80

/* Alternative is a portrait-shaped area 5.5 x 6 " */
#define MIFXMINP 0
#define MIFXMAXP 8000
#define MIFYMINP 0
#define MIFYMAXP 10500
#define DXMIFP 8000
#define DYMIFP 10500
#define CHARSP 6.0
#define TICLP 80

#define MINCOLOR 0
#define MAXCOLOR 7
#define MINLINEWIDTH 1
#define MAXLINEWIDTH 9
#define MINLINESTYLE 1
#define MAXLINESTYLE 6
#define MINPATTERN 0
#define MAXPATTERN 15

#define LINEWIDTHINC 0.5

// #define PORTRAIT 0
// #define LANDSCAPE 1

static int mifxmin = MIFXMIN;
// static int mifxmax = MIFXMAX;
static int mifymin = MIFYMIN;
static int mifymax = MIFYMAX;
static int mifdx = DXMIF;
static int mifdy = DYMIF;
static int mifcolor = 0;
static int miflinewidth = -1;
static int mifdmode;
static int mifpattern = 0;
static int miffont = 0;
static double mifcharsize = 1.5;
static int mifticl;
static int miflinestyle;
static char *fname;
// static int orientflag = 0 /*PORTRAIT*/;
static int styles[5] = { 0, 1, 4, 10, 13 };

static int x_current = 99999, y_current = 99999;
double xconv(double x), yconv(double y);
static FILE *mifout;

static void stroke(void)
{
    int i, cnt = 0;
    double x, y;

    if (pathlength > 1)
    {
        fprintf(mifout, "  <PolyLine\n");
        fprintf(mifout, "     <GroupID 1>\n");
        fprintf(mifout, "     <Pen %d>\n", styles[miflinestyle - 1]);
        fprintf(mifout, "     <Fill 15>\n");
        if (mifcolor == 0)
        {
            fprintf(mifout, "     <Separation 1>\n");
        }
        else if (mifcolor == 1)
        {
            fprintf(mifout, "     <Separation 0>\n");
        }
        else
        {
            fprintf(mifout, "     <Separation %d>\n", mifcolor);
        }
        fprintf(mifout, "     <PenWidth %4.1f pt>\n", miflinewidth * LINEWIDTHINC);
        fprintf(mifout, "     <HeadCap Round>\n");
        fprintf(mifout, "     <TailCap Round>\n");
        /* do a little compression */
        if (pathlength > 3)
        {
            scrunch_points(xpoints, ypoints, &pathlength);
        }
        cnt = pathlength;
        fprintf(mifout, "     <NumPoints %d>\n", cnt);
        for (i = 0; i < cnt; i++)
        {
            x = xpoints[i] * 0.001;
            y = (mifymax - ypoints[i]) * 0.001;
            fprintf(mifout, "     <Point %6.3f \" %6.3f \">\n", x, y);
        }
        fprintf(mifout, "  >\n");
        xpoints[0] = xpoints[pathlength - 1];
        ypoints[0] = ypoints[pathlength - 1];
        pathlength = 1;
    }
}

static char tbuf[128];
int mifsetmode(int mode)
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
        if ((mifout = fopen(fname, "w")) == NULL)
        {
            return 0;
        }
    }
    devoffsx = devoffsy = 0;
    switch (mode)
    {
    case 1: /* MIF landscape */
        // orientflag = 1 /*LANDSCAPE*/;
        mifcharsize = CHARS;
        mifxmin = MIFXMIN;
        // mifxmax = MIFXMAX;
        mifymin = MIFYMIN;
        mifymax = MIFYMAX;
        mifdx = DXMIF;
        mifdy = DYMIF;
        devwidth = DXMIF;
        devwidthmm = (int)(DXMIF / 1000.0 * 25.4);
        devheight = DYMIF;
        devheightmm = (int)(DYMIF / 1000.0 * 25.4);
        mifticl = TICL;
        fprintf(mifout, "<MIFFile 2.00> # %s\n", version);
        fprintf(mifout, "<Units Uin>\n");
        break;
    case 3: /* MIF portrait */
        // orientflag = 0 /*PORTRAIT*/;
        mifcharsize = CHARSP;
        mifxmin = MIFXMINP;
        // mifxmax = MIFXMAXP;
        mifymin = MIFYMINP;
        mifymax = MIFYMAXP;
        mifdx = DXMIFP;
        mifdy = DYMIFP;
        devwidth = DXMIFP;
        devwidthmm = (int)(DXMIFP / 1000.0 * 25.4);
        devheight = DYMIFP;
        devheightmm = (int)(DYMIFP / 1000.0 * 25.4);
        mifticl = TICLP;
        fprintf(mifout, "<MIFFile 2.00> # %s\n", version);
        fprintf(mifout, "<Units Uin>\n");
        break;
    case 2:
    case 4:
        stroke();
        fprintf(mifout, "<Group <ID 1>>\n");
        /*	fprintf(mifout,">\n");*/
        fclose(mifout);
        if (!ptofile)
        {
            sprintf(sysbuf, "%s %s", mif_prstr, fname);
            system(sysbuf);
            unlink(fname);
        }
        // orientflag = 0 /*PORTRAIT*/;
        break;
    }
    return mode;
}

void drawmif(int x2, int y2, int mode)
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

int xconvmif(double x)
{
    return ((int)(mifxmin + mifdx * xconv(x)));
}

int yconvmif(double y)
{
    return ((int)(mifymin + mifdy * yconv(y)));
}

int mifsetcolor(int c)
{
    if (c != mifcolor)
    {
        stroke();
        if ((mifcolor = c) > MAXCOLOR)
        {
            mifcolor = 1;
        }
        else if (mifcolor < MINCOLOR)
        {
            mifcolor = 1;
        }
    }
    return c;
}

int mifsetlinewidth(int c)
{
    if (c != miflinewidth)
    {
        stroke();
        if ((c = c % MAXLINEWIDTH) < MINLINEWIDTH)
            c = MINLINEWIDTH;
    }
    miflinewidth = c;
    return c;
}

void mifdrawtic(int x, int y, int dir, int updown)
{
    switch (dir)
    {
    case 0:
        switch (updown)
        {
        case 0:
            drawmif(x, y, 0);
            drawmif(x, y + devxticl, 1);
            break;
        case 1:
            drawmif(x, y, 0);
            drawmif(x, y - devxticl, 1);
            break;
        }
        break;
    case 1:
        switch (updown)
        {
        case 0:
            drawmif(x, y, 0);
            drawmif(x + devyticl, y, 1);
            break;
        case 1:
            drawmif(x, y, 0);
            drawmif(x - devyticl, y, 1);
            break;
        }
        break;
    }
}

int mifsetlinestyle(int style)
{
    if (style == miflinestyle)
    {
        return (miflinestyle);
    }
    stroke();
    if ((miflinestyle = style) < MINLINESTYLE)
        miflinestyle = MINLINESTYLE;
    else if (miflinestyle > MAXLINESTYLE)
        miflinestyle = MAXLINESTYLE;
    return (miflinestyle = style);
}

char mifcurfont[128] = "<Font\n  <FFamily `Times'>\n<FSize 14>\n<FBold No>\n<FAngle `Regular'>\n> # end of font";
static int miffontsize = 15;

void mifsetfont(int n)
{
    switch (n)
    {
    case 0:
        sprintf(mifcurfont, "<Font\n  <FFamily `Times'>\n<FSize %d>\n<FBold No>\n<FAngle `Regular'>\n> # end of font", miffontsize);
        break;
    case 1:
        sprintf(mifcurfont, "<Font\n  <FFamily `Times'>\n<FSize %d>\n <FBold Yes>\n<FAngle `Regular'>\n> # end of font", miffontsize);
        break;
    case 2:
        sprintf(mifcurfont, "<Font\n  <FFamily `Times'>\n<FAngle `Italic'>\n<FSize %d>\n<FBold Yes>\n> # end of font", miffontsize);
        break;
    case 3:
        sprintf(mifcurfont, "<Font\n  <FFamily `Times'>\n<FAngle `Italic'>\n<FSize %d>\n<FBold Yes>\n> # end of font", miffontsize);
        break;
    case 4:
        sprintf(mifcurfont, "<Font\n  <FFamily `Helvetica'>\n<FSize %d>\n <FBold No>\n<FAngle `Regular'>\n> # end of font", miffontsize);
        break;
    case 5:
        sprintf(mifcurfont, "<Font\n  <FFamily `Helvetica'>\n<FSize %d>\n <FBold Yes>\n<FAngle `Regular'>\n> # end of font", miffontsize);
        break;
    case 6:
        sprintf(mifcurfont, "<Font\n  <FFamily `Helvetica'>\n<FSize %d>\n <FBold No>\n<FAngle `Italic'>\n> # end of font", miffontsize);
        break;
    case 7:
        sprintf(mifcurfont, "<Font\n  <FFamily `Helvetica'>\n<FSize %d>\n <FBold Yes>\n<FAngle `Italic'>\n> # end of font", miffontsize);
        break;
    case 8:
        sprintf(mifcurfont, "<Font\n  <FFamily `Symbol'>\n<FSize %d>\n <FBold No>\n<FAngle `Regular'>\n> # end of font", miffontsize);
        break;
    case 9:
        sprintf(mifcurfont, "<Font\n  <FFamily `Symbol'>\n<FSize %d>\n <FBold No>\n<FAngle `Regular'>\n> # end of font", miffontsize);
        break;
    case 10:
        sprintf(mifcurfont, "<Font\n  <FFamily `Symbol'>\n<FSize %d>\n <FBold No>\n<FAngle `Regular'>\n> # end of font", miffontsize);
        break;
    }
    hselectfont(miffont = n);
}

void mifsetfontsize(double size)
{
    miffontsize = (int)(size * 15);
}

void dispstrmif(int x, int y, int rot, char *s, int just, int fudge)
{
    fprintf(mifout, "%s\n", mifcurfont);
    fprintf(mifout, "<TextLine\n");
    fprintf(mifout, "<GroupID 1>\n");
    if (fudge)
    {
        fprintf(mifout, "<TLOrigin %lf %lf>\n", x * 0.001, (mifymax - (y - miffontsize / 216.0 * 1000.0)) * 0.001);
    }
    else
    {
        fprintf(mifout, "<TLOrigin %lf %lf>\n", x * 0.001, (mifymax - y) * 0.001);
    }
    fprintf(mifout, "<Angle %d>\n", rot);
    switch (just)
    {
    case 0:
        fprintf(mifout, "<TLAlignment Left>\n");
        break;
    case 1:
        fprintf(mifout, "<TLAlignment Right>\n");
        break;
    case 2:
        fprintf(mifout, "<TLAlignment Center>\n");
        break;
    }
    putmif(s);
}

void putmif(char *s)
{
    int i, slen = strlen(s), curcnt = 0;
    int underline = 0, offset = 0;
    double saves = miffontsize / 15.0, scale = miffontsize / 15.0;
    char curstr[256];

    for (i = 0; i < slen; i++)
    {
        if (s[i] == '\\' && isdigit(s[i + 1]))
        {
            curstr[curcnt] = 0;
            if (curcnt >= 1)
            {
                fprintf(mifout, "<String `%s'> # end of substring\n", curstr);
            }
            curcnt = 0;
            mifsetfont(s[i + 1] - '0');
            i++;
            continue;
        }
        else if (s[i] == '\'')
        {
            curstr[curcnt++] = '\\';
            curstr[curcnt++] = 'q';
            i++;
            continue;
        }
        else if (s[i] == '`')
        {
            curstr[curcnt++] = '\\';
            curstr[curcnt++] = 'Q';
            i++;
            continue;
        }
        else if (s[i] == '>')
        {
            curstr[curcnt++] = '\\';
            curstr[curcnt++] = '>';
            i++;
            continue;
        }
        else if (s[i] == '\\' && s[i + 1] == '\\')
        {
            curstr[curcnt++] = '\\';
            curstr[curcnt++] = s[i];
            i++;
            continue;
        }
        else if (s[i] == '\\' && isoneof(s[i + 1], (char *)"cCbxsSNuU+-"))
        {
            switch (s[i + 1])
            {
            case 'x':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(mifout, "<String `%s'> # end of substring\n", curstr);
                }
                curcnt = 0;
                mifsetfont(10);
                i++;
                break;
            case 's':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(mifout, "<String `%s'>\n<Font <FSupScript No> <FSubScript Yes >>\n", curstr);
                }
                else
                {
                    fprintf(mifout, "<Font <FSupScript No> <FSubScript Yes >>\n");
                }
                curcnt = 0;
                mifsetfontsize(scale = 0.6 * saves);
                offset -= miffontsize;
                i++;
                break;
            case 'S':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(mifout, "<String `%s'>\n<Font <FSubScript No> <FSupScript Yes >>\n", curstr);
                }
                else
                {
                    fprintf(mifout, "<Font <FSubScript No> <FSupScript Yes >>");
                }
                curcnt = 0;
                mifsetfontsize(scale = 0.6 * saves);
                offset += miffontsize;
                i++;
                break;
            case 'N':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(mifout, "<String `%s'>\n<Font <FSubScript No> <FSupScript No >>\n", curstr);
                }
                else
                {
                    fprintf(mifout, "<Font <FSubScript No> <FSupScript No >>\n");
                }
                curcnt = 0;
                scale = saves;
                mifsetfontsize(scale);
                offset = 0;
                i++;
                break;
            case 'b':
                i++;
                break;
            case 'c':
                i++;
                break;
            case 'C':
                i++;
                break;
            case 'u':
                underline = 1;
                i++;
                break;
            case 'U':
                underline = 0;
                i++;
                break;
            case '-':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(mifout, "<String `%s'> # end of substring\n", curstr);
                }
                curcnt = 0;
                scale -= 0.2;
                if (scale <= 0.2)
                {
                    scale = 0.2;
                }
                mifsetfontsize(scale);
                i++;
                break;
            case '+':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(mifout, "<String `%s'> # end of substring\n", curstr);
                }
                curcnt = 0;
                scale += 0.2;
                mifsetfontsize(scale);
                i++;
                break;
            }
            continue;
        }
        curstr[curcnt++] = s[i];
    }
    if (curcnt > 0)
    {
        curstr[curcnt] = 0;
        fprintf(mifout, "<String `%s'>\n>  # end of textline\n", curstr);
    }
    else
    {
        fprintf(mifout, "\n>  # end of textline\n");
    }
    mifsetfontsize(saves);
}

int mifsetpat(int k)
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
    return (mifpattern = (k + 14) % 16);
}

void miffill(int n, int *px, int *py)
{
    int i;

    stroke();
    if (n)
    {
        fprintf(mifout, "  <Polygon\n");
        fprintf(mifout, "     <GroupID 1>\n");
        fprintf(mifout, "     <Pen %d>\n", mifcolor);
        fprintf(mifout, "     <PenWidth 0.0 pt>\n");
        fprintf(mifout, "     <Fill %d>\n", mifpattern);
        fprintf(mifout, "     <HeadCap Round>\n");
        fprintf(mifout, "     <TailCap Round>\n");
        fprintf(mifout, "     <NumPoints %d>\n", pathlength);
        for (i = 0; i < n; i++)
        {
            fprintf(mifout, "     <Point %6.3f \" %6.3f \">\n", (double)px[i] * 0.001, (double)(mifymax - py[i]) * 0.001);
        }
        fprintf(mifout, "  >\n");
        pathlength = 0;
    }
    fprintf(mifout, "     <Fill 15>\n");
}

void miffillcolor(int n, int *px, int *py)
{
    int i;

    stroke();
    if (n)
    {
        fprintf(mifout, "  <Polygon\n");
        fprintf(mifout, "     <GroupID 1>\n");
        fprintf(mifout, "     <Pen %d>\n", mifcolor);
        if (mifcolor == 0)
        {
            fprintf(mifout, "     <Separation 1>\n");
        }
        else if (mifcolor == 1)
        {
            fprintf(mifout, "     <Separation 0>\n");
        }
        else
        {
            fprintf(mifout, "     <Separation %d>\n", mifcolor);
        }
        fprintf(mifout, "     <PenWidth 0.0 pt>\n");
        /*
       * fprintf(mifout, "     <PenWidth 0.0 pt>\n", miflinewidth *
       * LINEWIDTHINC);
       */
        fprintf(mifout, "     <Fill %d>\n", (mifcolor % 2) ? 0 : 7);
        /* fprintf(mifout, "     <Fill %d>\n", mifpattern); */
        fprintf(mifout, "     <HeadCap Round>\n");
        fprintf(mifout, "     <TailCap Round>\n");
        fprintf(mifout, "     <NumPoints %d>\n", pathlength);
        for (i = 0; i < n; i++)
        {
            fprintf(mifout, "     <Point %6.3f \" %6.3f \">\n", (double)px[i] * 0.001, (double)(mifymax - py[i]) * 0.001);
        }
        fprintf(mifout, "  >\n");
        pathlength = 0;
    }
}

void mifdrawarc(int x, int y, int r)
{

    stroke();
    fprintf(mifout, "  <Ellipse\n");
    fprintf(mifout, "     <GroupID 1>\n");
    fprintf(mifout, "     <BRect %6.3lf %6.3lf %6.3lf %6.3lf>\n",
            (x - r) * 0.001,
            (mifymax - (y + r)) * 0.001,
            (2 * r) * 0.001,
            (2 * r) * 0.001);
    fprintf(mifout, "     <PenWidth %4.1f pt>\n", miflinewidth * LINEWIDTHINC);
    fprintf(mifout, "     <Fill 15>\n");
    fprintf(mifout, "     <HeadCap Round>\n");
    fprintf(mifout, "     <TailCap Round>\n");
    fprintf(mifout, "  >\n");
    pathlength = 0;
}

void miffillarc(int x, int y, int r)
{

    stroke();
    fprintf(mifout, "  <Ellipse\n");
    fprintf(mifout, "     <GroupID 1>\n");
    fprintf(mifout, "     <BRect %6.3lf %6.3lf %6.3lf %6.3lf>\n",
            (x - r) * 0.001,
            (mifymax - (y + r)) * 0.001,
            (2 * r) * 0.001,
            (2 * r) * 0.001);
    fprintf(mifout, "     <PenWidth 0.0 pt>\n");
    fprintf(mifout, "     <Fill %d>\n", (mifcolor % 2) ? 0 : 7);
    fprintf(mifout, "     <HeadCap Round>\n");
    fprintf(mifout, "     <TailCap Round>\n");
    fprintf(mifout, "  >\n");
    pathlength = 0;
}

void mifleavegraphics(void)
{
    mifsetmode(mifdmode + 1);
}

/*           mif initialization routine  */
int mifinitgraphics(int dmode)
{
    mifdmode = dmode;
    if (!mifsetmode(mifdmode))
    {
        return -1;
    }
    devconvx = xconvmif;
    devconvy = yconvmif;
    devvector = drawmif;
    devwritestr = dispstrmif;
    devsetcolor = mifsetcolor;
    devsetfont = mifsetfont;
    devsetline = mifsetlinestyle;
    devsetlinew = mifsetlinewidth;
    devdrawtic = mifdrawtic;
    devsetpat = mifsetpat;
    devdrawarc = mifdrawarc;
    devfillarc = miffillarc;
    devfill = miffill;
    devfillcolor = miffillcolor;
    devleavegraphics = mifleavegraphics;
    devcharsize = mifcharsize;
    devxticl = mifticl;
    devyticl = mifticl;
    devsymsize = mifticl;
    devarrowlength = 80;
    mifsetcolor(1);
    mifsetlinewidth(2);
    setlinestyle(0);
    setfont(2);

    return (0);
}
