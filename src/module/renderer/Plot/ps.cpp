/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: ps.c,v 1.4 1994/10/26 04:47:29 pturner Exp pturner $
 *
 * driver for postscript printer
 *
 * courtesy of:
 *
 * Jim Hudgens
 * hudgens@ray.met.fsu.edu
 *
 * Further modifications by,
 * Ole Holm Nielsen
 * ohnielse@ltf.dth.dk
 *
 * ISO Latin encoding by
 * Przemek Klosowski
 * przemek@rrdbartok.nist.gov
 *
 */

#include <stdio.h>
#include <ctype.h>
#include "extern.h"

#include "externs.h"
#include "defines.h"

#define MAX_BUF_LEN 128

extern char version[];
extern double charsize;
extern double devcharsize;
extern int ptofile;
extern char printstr[];
extern int monomode; /* allow 2 colors */

int epsflag = 0;

static void stroke(void);
static void escape_paren(char *s);
static void putps(char *s);

/*
 * printer control string
 */
#ifndef PS_PRSTR
char ps_prstr[MAX_BUF_LEN] = "lpr -h";

#else
char ps_prstr[MAX_BUF_LEN] = "lpr -h";
#endif

/* postscript page at scale = 0.25 */

/*
 * the following defines are tuned for our HP-LJ-III
 * and may need adjustment for other printers
 */
/* 8.5 x 11 letter */
#define PSXMIN 50
#define PSXMAX 2450
#define PSYMIN 50
#define PSYMAX 3200
#define DXPS 2400
#define DYPS 3150
#define PSWIDTHMM 204
#define PSHEIGHTMM 266

/* 8.5 x 14 legal */
/*
#define PSXMIN 50
#define PSXMAX 2450
#define PSYMIN 50
#define PSYMAX 4100
#define DXPS 2400
#define DYPS 4050
#define PSWIDTHMM 204
#define PSHEIGHTMM 343
*/

/* A4 */
/*
#define PSXMIN 50
#define PSXMAX 2430
#define PSYMIN 50
#define PSYMAX 3457
#define DXPS 2380
#define DYPS 3450
#define PSWIDTHMM 201
#define PSHEIGHTMM 292
*/

#define CHARS 1.8

#define MINCOLOR 0
#define MAXCOLOR 30
#define PSMAXPAT 30
#define MINLINEWIDTH 0
#define MAXLINEWIDTH 9
#define MAXLINESTYLE 14

/*
#define PORTRAIT 0
#define LANDSCAPE 1
*/

static int psxmin = PSXMIN;
// static int psxmax = PSXMAX;
static int psymin = PSYMIN;
// static int psymax = PSYMAX;
static int psdx = DXPS;
static int psdy = DYPS;

static int pscolor = -1;
static int pslinewidth = -1;
int pslwfactor = 3; /* factor for setting linewidth increment */
static int pslinestyle;

static int psdmode;
static int pspattern = 0;
static int psfont = 0;
static double pscharsize = 1.7;

static char *fname;
static FILE *psout;

// static int orientflag = PORTRAIT;

static char tbuf[MAX_BUF_LEN];

double xconv(double x), yconv(double y);

static int prevx = 99999, prevy = 99999, prevmode;

static void stroke(void)
{
    if (pathlength)
    {
        fprintf(psout, "stroke\n");
        prevx = 99999;
        prevy = 99999;
        pathlength = 0;
    }
}

int pssetmode(int mode)
{
    char sysbuf[MAX_BUF_LEN];
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
        if ((psout = fopen(fname, "w")) == NULL)
        {
            return 0;
        }
    }
    switch (mode)
    {

    case 3: /* EPS   portrait */
        // orientflag = PORTRAIT;
        pscharsize = CHARS;
        if (epsflag)
        {
            psxmin = PSXMIN;
            psymin = PSYMIN;
            if (devwidth == 0)
            {
                devwidth = PSXMAX - PSXMIN;
            }
            if (devheight == 0)
            {
                devheight = PSYMAX - PSYMIN;
            }
            if (((double)devheight) / devwidth > ((double)(PSYMAX - PSYMIN)) / (PSXMAX - PSXMIN))
            {
                // psymax = PSYMAX;
                // psxmax = (PSYMAX - PSYMIN) * devwidth / devheight + PSXMIN;
                psdy = PSYMAX - PSYMIN;
                psdx = (PSYMAX - PSYMIN) * devwidth / devheight;
            }
            else
            {
                // psxmax = PSXMAX;
                // psymax = (PSXMAX - PSXMIN) * devheight / devwidth + PSYMIN;
                psdx = PSXMAX - PSXMIN;
                psdy = (PSXMAX - PSXMIN) * devheight / devwidth;
            }

            devwidth = psdx;
            devheight = psdy;
            devwidthmm = (int)(25.4 * devwidth / 300.0);
            devheightmm = (int)(25.4 * devheight / 300.0);
        }
        else
        {
            psxmin = PSXMIN;
            // psxmax = PSXMAX;
            psymin = PSYMIN;
            // psymax = PSYMAX;
            psdx = DXPS;
            psdy = DYPS;
            devwidth = DXPS;
            devheight = DYPS;
            devwidthmm = PSWIDTHMM;
            devheightmm = PSHEIGHTMM;
        }
        devoffsx = 0;
        devoffsy = 0;
        break;

    case 1: /* EPS landscape */
        // orientflag = LANDSCAPE;
        pscharsize = CHARS;
        if (epsflag)
        {
            psxmin = PSYMIN;
            psymin = PSXMIN;
            if (devwidth == 0)
            {
                devwidth = PSYMAX - PSYMIN;
            }
            if (devheight == 0)
            {
                devheight = PSXMAX - PSXMIN;
            }
            if (((double)devheight) / devwidth > ((double)(PSXMAX - PSXMIN)) / (PSYMAX - PSYMIN))
            {
                // psymax = PSXMAX;
                // psxmax = (PSXMAX - PSXMIN) * devwidth / devheight + PSYMIN;
                psdy = PSXMAX - PSXMIN;
                psdx = (PSXMAX - PSXMIN) * devwidth / devheight;
            }
            else
            {
                // psxmax = PSYMAX;
                // psymax = (PSYMAX - PSYMIN) * devheight / devwidth + PSXMIN;
                psdx = PSYMAX - PSYMIN;
                psdy = (PSYMAX - PSYMIN) * devheight / devwidth;
            }
            devwidth = psdx;
            devheight = psdy;
            devwidthmm = (int)(25.4 * devwidth / 300.0);
            devheightmm = (int)(25.4 * devheight / 300.0);
        }
        else
        {
            psxmin = PSYMIN;
            // psxmax = PSYMAX;
            psymin = PSXMIN;
            // psymax = PSXMAX;
            psdx = DYPS;
            psdy = DXPS;
            devwidth = DYPS;
            devheight = DXPS;
            devwidthmm = PSHEIGHTMM;
            devheightmm = PSWIDTHMM;
        }
        devoffsx = 0;
        devoffsy = 0;
        break;

    case 2:
    case 4:
        stroke();
        fprintf(psout, "showpage\n");
        fprintf(psout, "%%%%Trailer\n");
        fclose(psout);
        if (!ptofile)
        {
            sprintf(sysbuf, "%s %s", ps_prstr, fname);
            system(sysbuf);
            unlink(fname);
        }
        // orientflag = PORTRAIT;
        break;
    }
    return mode;
}

void drawps(int x2, int y2, int mode)
{
    int xtmp, ytmp;

    if (x2 < 0 || y2 < 0) /* Eliminate garbage on output */
    {
        return;
    }
    xtmp = x2;
    ytmp = y2;

    if (mode)
    {
        if (prevmode && xtmp == prevx && ytmp == prevy)
        {
            return;
            /* previous mode was draw and points are the
          * same */
        }
        fprintf(psout, "%d %d l\n", xtmp, ytmp); /* lineto */
    }
    else
    {
        /* Avoid excessive moveto's generated by grtool */
        if (xtmp == prevx && ytmp == prevy)
        {
            return;
        }
        fprintf(psout, "%d %d m\n", xtmp, ytmp); /* moveto */
    }
    pathlength++;
    prevx = xtmp;
    prevy = ytmp;

    /*
    * Printers have some maximum number of points in a path. See PostScript
    * Language Reference Manual (Red book), p. 261. Hence the fix that
    * follows
    */

    prevmode = mode;
    if (pathlength > MAXLINELEN)
    {
        stroke();
        prevmode = 0;
        fprintf(psout, "%d %d m\n", xtmp, ytmp); /* moveto */
    }
}

int xconvps(double x)
{
    return ((int)(psxmin + psdx * xconv(x)));
}

int yconvps(double y)
{
    return ((int)(psymin + psdy * yconv(y)));
}

int pssetcolor(int c)
{
    extern unsigned char red[], green[], blue[];
    int itmp;

    stroke();
    if (monomode)
    {
        itmp = c > 0 ? 1 : 0;
    }
    else
    {
        itmp = c;
    }
    if (c != pscolor)
    {
        if (c >= 0)
        {
            fprintf(psout, "%lf %lf %lf setrgbcolor\n",
                    (double)red[itmp] / 255.0,
                    (double)green[itmp] / 255.0,
                    (double)blue[itmp] / 255.0);
        }
    }
    pscolor = c;
    return c;
}

int pssetlinewidth(int c)
{
    stroke();
    if (c != pslinewidth)
    {
        c = c % (MAXLINEWIDTH + 1);
        if (c == 1)
        {
            fprintf(psout, "1 setlinewidth\n");
        }
        else
        {
            fprintf(psout, "%d setlinewidth\n", (int)(pslwfactor * c + 0.51));
        }
    }
    pslinewidth = c;
    return c;
}

void psdrawtic(int x, int y, int dir, int updown)
{
    switch (dir)
    {
    case 0:
        switch (updown)
        {
        case 0:
            drawps(x, y, 0);
            drawps(x, y + devxticl, 1);
            break;
        case 1:
            drawps(x, y, 0);
            drawps(x, y - devxticl, 1);
            break;
        }
        break;
    case 1:
        switch (updown)
        {
        case 0:
            drawps(x, y, 0);
            drawps(x + devyticl, y, 1);
            break;
        case 1:
            drawps(x, y, 0);
            drawps(x - devyticl, y, 1);
            break;
        }
        break;
    }
}

int pssetlinestyle(int style)
{
    stroke();
    if (style == pslinestyle)
    {
        return (pslinestyle);
    }
    switch (style)
    {
    case 1: /* solid */
        fprintf(psout, "[] 0 setdash\n");
        break;
    case 2: /* dotted */
        fprintf(psout, "[4 8] 0 setdash\n");
        break;
    case 3: /* long dash */
        fprintf(psout, "[20 20] 0 setdash\n");
        break;
    case 4: /* short dash */
        fprintf(psout, "[40 20] 0 setdash\n");
        break;
    case 5: /* dot-dashed */
        fprintf(psout, "[40 20 12 20] 0 setdash\n");
        break;
    }
    return (pslinestyle = style);
}

char pscurfont[MAX_BUF_LEN] = "/Times-Roman-ISOLatin1 findfont 60 scalefont setfont";
int psfontsize = 60;

void pssetfont(int n)
{
    if (psfont == n)
    {
        return;
    }
    switch (n)
    {
    case 0:
        sprintf(pscurfont, "/Times-Roman-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 1:
        sprintf(pscurfont, "/Times-Bold-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 2:
        sprintf(pscurfont, "/Times-Italic-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 3:
        sprintf(pscurfont, "/Times-BoldItalic-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 4:
        sprintf(pscurfont, "/Helvetica-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 5:
        sprintf(pscurfont, "/Helvetica-Bold-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 6:
        sprintf(pscurfont, "/Helvetica-Oblique-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 7:
        sprintf(pscurfont, "/Helvetica-BoldOblique-ISOLatin1 findfont %d scalefont setfont", psfontsize);
        break;
    case 8:
        sprintf(pscurfont, "/Symbol findfont \n%d scalefont\n setfont", psfontsize);
        break;
    case 9:
        sprintf(pscurfont, "/Symbol findfont \n%d scalefont\n setfont", psfontsize);
        break;
    case 10:
        sprintf(pscurfont, "/Symbol findfont \n%d scalefont\n setfont", psfontsize);
        break;
    }
    fprintf(psout, "%s\n", pscurfont);
    hselectfont(psfont = n);
}

void pssetfontsize(double size)
{
    int sf = psfont;

    psfontsize = (int)(size * 60);
    psfont = -1;
    pssetfont(sf);
}

static void escape_paren(char *s)
{
    char t[256];
    int i, cnt = 0;
    for (i = 0; i < strlen(s); i++)
    {
        if (s[i] == '(' || s[i] == ')')
        {
            t[cnt++] = '\\';
        }
        t[cnt++] = s[i];
    }
    t[cnt] = 0;
    strcpy(s, t);
}

void dispstrps(int x, int y, int rot, char *s, int just, int fudge)
{
    char tmpstr[256];
    // int i, cnt = 0;

    stroke();
    if (psfontsize == 0 || s == NULL || strlen(s) == 0)
    {
        return;
    }
    fprintf(psout, "%d %d m\n", x, y);
    fprintf(psout, "gsave\n");
    fprintf(psout, "%d %d translate\n", x, y);
    fprintf(psout, "%d rotate\n", rot);
    if (fudge)
    {
        fprintf(psout, "0 %d  m\n", -psfontsize / 3);
    }
    else
    {
        fprintf(psout, "0 0  m\n");
    }
    switch (just)
    {
    case 0:
        break;
    case 1:
        stripspecial(s, tmpstr);
        escape_paren(tmpstr);
        fprintf(psout, "(%s) RJ\n", tmpstr);
        break;
    case 2:
        stripspecial(s, tmpstr);
        escape_paren(tmpstr);
        fprintf(psout, "(%s) CS\n", tmpstr);
        break;
    }
    putps(s);
    fprintf(psout, "grestore\n");
    fprintf(psout, "newpath\n");
}

static void putps(char *s)
{
    int i, slen = strlen(s), curcnt = 0;
    int underline = 0, offset = 0;
    double saves = psfontsize / 60.0, scale = psfontsize / 60.0;
    char curstr[256];
    int upperset = 0;
    int symfont = 0;

    if (psfont == 9)
    {
        symfont = 1;
        upperset = 0x80;
    }
    else
    {
        symfont = 0;
        upperset = 0;
    }
    for (i = 0; i < slen; i++)
    {
        if (s[i] == '-' && isdigit(s[i + 1]))
        {
            /* s[i] = 0261; */
        }
        else if (s[i] == '\\' && isdigit(s[i + 1]))
        {
            curstr[curcnt] = 0;
            if (curcnt >= 1)
            {
                fprintf(psout, "(%s) show\n", curstr);
            }
            curcnt = 0;
            if (symfont)
            {
                symfont = 0;
                upperset = 0;
            }
            pssetfont(s[i + 1] - '0');
            if (psfont == 9)
            {
                symfont = 1;
                upperset = 0x80;
            }
            i++;
            continue;
        }
        else if (s[i] == '(' || s[i] == ')')
        {
            curstr[curcnt++] = '\\';
        }
        else if (s[i] == '\\' && isoneof(s[i + 1], (char *)"cCbxsSNuU+-"))
        {
            switch (s[i + 1])
            {
            case 'x':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(psout, "(%s) show\n", curstr);
                }
                curcnt = 0;
                if (symfont == 0)
                {
                    symfont = 1;
                    upperset = 0x80;
                }
                pssetfont(10);
                i++;
                break;
            case 's':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(psout, "(%s) show\n", curstr);
                }
                curcnt = 0;
                pssetfontsize(scale = 0.6 * saves);
                offset -= psfontsize / 2;
                fprintf(psout, "0 %d rmoveto\n", -(psfontsize / 2));
                i++;
                break;
            case 'S':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(psout, "(%s) show\n", curstr);
                }
                curcnt = 0;
                pssetfontsize(scale = 0.6 * saves);
                offset += psfontsize;
                fprintf(psout, "0 %d rmoveto\n", psfontsize);
                i++;
                break;
            case 'N':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(psout, "(%s) show\n", curstr);
                }
                curcnt = 0;
                scale = saves;
                pssetfontsize(scale);
                fprintf(psout, "0 %d rmoveto\n", -offset);
                offset = 0;
                /*
                     fprintf(psout, "0 %d rmoveto\n", psfontsize);
               */
                i++;
                break;
            case 'b':
                i++;
                break;
            case 'c':
                upperset = 0x80;
                i++;
                break;
            case 'C':
                upperset = 0;
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
                    fprintf(psout, "(%s) show\n", curstr);
                }
                curcnt = 0;
                scale -= 0.2;
                if (scale < 0.2)
                {
                    scale = 0.2;
                }
                pssetfontsize(scale);
                i++;
                break;
            case '+':
                curstr[curcnt] = 0;
                if (curcnt >= 1)
                {
                    fprintf(psout, "(%s) show\n", curstr);
                }
                curcnt = 0;
                scale += 0.2;
                pssetfontsize(scale);
                i++;
                break;
            }
            continue;
        }
        else if (s[i] == '\\' && s[i + 1] == '\\')
        {
            curstr[curcnt++] = '\\';
            curstr[curcnt++] = s[i];
            i++;
            continue;
        }
        curstr[curcnt++] = s[i] + upperset;
    }
    curstr[curcnt] = 0;
    fprintf(psout, "(%s) show\n", curstr);
}

int pssetpat(int k)
{
    stroke();
    if (k > PSMAXPAT)
    {
        k = PSMAXPAT;
    }
    else if (k < 0)
    {
        k = 0;
        fprintf(psout, "0.0 setgray\n");
    }
    return (pspattern = k);
}

void psfill(int n, int *px, int *py)
{
    int i;

    stroke();
    drawps(px[0], py[0], 0);
    for (i = 1; i < n; i++)
    {
        drawps(px[i], py[i], 1);
    }
    fprintf(psout, "closepath\n");
    fprintf(psout, "%lf setgray\n", 1.0 - pspattern / (double)PSMAXPAT);
    fprintf(psout, "gsave eofill grestore\n");
    stroke();
    fprintf(psout, "0 setgray\n");
}

void psfillcolor(int n, int *px, int *py)
{
    int i;

    stroke();
    drawps(px[0], py[0], 0);
    for (i = 1; i < n; i++)
    {
        drawps(px[i], py[i], 1);
    }
    fprintf(psout, "closepath\n");
    fprintf(psout, "gsave eofill grestore\n");
    stroke();
}

void psdrawarc(int x, int y, int r)
{
    stroke();
    fprintf(psout, "%d %d %d %d %d arc\n", x, y, r, 0, 360);
    fprintf(psout, "stroke\n");
}

void psfillarc(int x, int y, int r)
{
    stroke();
    fprintf(psout, "%d %d %d %d %d arc\n", x, y, r, 0, 360);
    fprintf(psout, "gsave fill grestore\n");
    fprintf(psout, "stroke\n");
}

void psdrawellipse(int x, int y, int xm, int ym)
{
    double scalex, scaley = 1.0;
    /*
       if (xm == 0 || ym == 0) {
      return;
       }
   */
    if (xm == 0)
    {
        xm = 1;
    }
    if (ym == 0)
    {
        ym = 1;
    }
    scalex = (double)xm / (double)ym;

    stroke();
    fprintf(psout, "gsave\n");
    fprintf(psout, "%lf %lf scale\n", scalex, scaley);
    fprintf(psout, "%d %d %d %d %d arc\n", (int)(x * 1.0 / scalex), y, ym, 0, 360);
    fprintf(psout, "stroke\n");
    fprintf(psout, "grestore\n");
}

void psfillellipse(int x, int y, int xm, int ym)
{
    double scalex /*, scaley = 1.0*/;

    /*
       if (xm == 0 || ym == 0) {
      return;
       }
   */
    if (xm == 0)
    {
        xm = 1;
    }
    if (ym == 0)
    {
        ym = 1;
    }
    scalex = (double)xm / (double)ym;
    stroke();
    fprintf(psout, "gsave\n");
    fprintf(psout, "%lf %lf scale\n", (double)xm, scalex);
    fprintf(psout, "%d %d %d %d %d arc\n", x, y, ym, 0, 360);
    fprintf(psout, "gsave fill grestore\n");
    fprintf(psout, "stroke\n");
    fprintf(psout, "grestore\n");
}

void psleavegraphics(void)
{
    pssetmode(psdmode + 1);
}

/*           postscript initialization routine  */
int psinitgraphics(int dmode)
{
    // FILE *fp;
    // char datebuf[MAX_BUF_LEN];

    pathlength = 0;
    psdmode = dmode;
    if (!pssetmode(psdmode))
    {
        return -1;
    }
    devconvx = xconvps;
    devconvy = yconvps;
    devvector = drawps;
    devwritestr = dispstrps;
    devsetcolor = pssetcolor;
    devsetfont = pssetfont;
    devsetline = pssetlinestyle;
    devsetlinew = pssetlinewidth;
    devdrawtic = psdrawtic;
    devsetpat = pssetpat;
    devdrawarc = psdrawarc;
    devfillarc = psfillarc;
    devfill = psfill;
    devfillcolor = psfillcolor;
    devdrawellipse = psdrawellipse;
    devfillellipse = psfillellipse;
    devleavegraphics = psleavegraphics;
    devcharsize = pscharsize;
    devsymsize = 20;
    devxticl = 20;
    devyticl = 20;
    devarrowlength = 20;

    if (epsflag)
    {
        /* fprintf(psout, "%%!PS-Adobe-3.0 EPSF-3.0\n"); */
        fprintf(psout, "%%!PS-Adobe-2.0 EPSF-1.2\n");
        if (dmode == 1)
        {
            fprintf(psout, "%%%%BoundingBox: %d %d %d %d\n",
                    psxmin * 72 / 300,
                    psymin * 72 / 300,
                    psdy * 72 / 300,
                    psdx * 72 / 300);
        }
        else
        {
            fprintf(psout, "%%%%BoundingBox: %d %d %d %d\n",
                    psxmin * 72 / 300,
                    psymin * 72 / 300,
                    psdx * 72 / 300,
                    psdy * 72 / 300);
        }
    }
    else
    {
        fprintf(psout, "%%!PS-Adobe-2.0\n");
    }
    fprintf(psout, "%%%%Creator: %s\n", version);
    fprintf(psout, "%%%%Title: %s\n", fname);
    /*
       if ((fp = popen("date", "r")) != NULL) {
      while (fgets(datebuf, MAX_BUF_LEN, fp) != NULL) {
          fprintf(psout, "%%%%CreationDate: %s", datebuf);
      }
      pclose(fp);
       }
   */
    fprintf(psout, "%%%%EndComments\n");
    fprintf(psout, "/m {moveto} bind def\n");
    fprintf(psout, "/l {lineto} bind def\n");
    fprintf(psout, "/RJ {\n");
    fprintf(psout, " stringwidth neg exch neg exch\n");
    fprintf(psout, " rmoveto\n");
    fprintf(psout, "} bind def\n");

    fprintf(psout, "/CS {\n");
    fprintf(psout, " stringwidth\n");
    fprintf(psout, " 2 div neg exch 2 div neg exch\n");
    fprintf(psout, " rmoveto\n");
    fprintf(psout, "} bind def\n");

    fprintf(psout, "0.25 0.25 scale\n");
    fprintf(psout, "1 setlinecap\n");
    /*
    * CREATE ISOLatin1Encoding if not
    * there already; actually, this differs slightly from ISO, since ISO
    * leaves /hyphen at pos. 8#055 (decimal 45), plus it has /hyphen at
    * 8#255 Here, we put /minus at 8#055. Phew.
    */
    fprintf(psout, "\n\
mark             \n\
/ISOLatin1Encoding \n\
  8#000 1 8#054 {StandardEncoding exch get} for \n\
  /minus \n\
  8#056 1 8#217 {StandardEncoding exch get} for \n\
  /dotlessi \n\
  8#301 1 8#317 {StandardEncoding exch get} for \n\
  /space /exclamdown /cent /sterling /currency /yen /brokenbar /section \n\
  /dieresis /copyright /ordfeminine /guillemotleft /logicalnot /hyphen \n\
  /registered /macron /degree /plusminus /twosuperior /threesuperior /acute \n\
  /mu /paragraph /periodcentered /cedilla /onesuperior /ordmasculine \n\
  /guillemotright /onequarter /onehalf /threequarters /questiondown /Agrave \n\
  /Aacute /Acircumflex /Atilde /Adieresis /Aring /AE /Ccedilla /Egrave /Eacute \n\
  /Ecircumflex /Edieresis /Igrave /Iacute /Icircumflex /Idieresis /Eth /Ntilde \n\
  /Ograve /Oacute /Ocircumflex /Otilde /Odieresis /multiply /Oslash /Ugrave \n\
  /Uacute /Ucircumflex /Udieresis /Yacute /Thorn /germandbls /agrave /aacute \n\
  /acircumflex /atilde /adieresis /aring /ae /ccedilla /egrave /eacute \n\
  /ecircumflex /edieresis /igrave /iacute /icircumflex /idieresis /eth /ntilde \n\
  /ograve /oacute /ocircumflex /otilde /odieresis /divide /oslash /ugrave \n\
  /uacute /ucircumflex /udieresis /yacute /thorn /ydieresis \n\
  /ISOLatin1Encoding where not {256 array astore def} if \n\
 cleartomark \n");
    /* Define makeISOEncoded procedure expecting 'fontname' and returning the
    * font on stack; the procedure installs new font 'fontname-ISOLatin1' for
    * later use.  this can be used as either
        /Times-Roman makeISOEncoded  pop ...
        /Times-Roman-ISOL1 findfont 55 scalefont setfont
    * (when fonts are found many times) or
        /Times-Roman makeISOEncoded 55 scalefont setfont
    * (for one-shot deals)
    */
    fprintf(psout, "\n\
/makeISOEncoded \n\
{ findfont /curfont exch def \n\
  /newfont curfont maxlength dict def  \n\
  /ISOLatin1 (-ISOLatin1) def\n\
  /curfontname curfont /FontName get dup length string cvs def \n\
  /newfontname curfontname length ISOLatin1 length add string \n\
     dup 0                  curfontname putinterval \n\
     dup curfontname length ISOLatin1   putinterval \n\
  def \n\
  curfont   \n\
  { exch dup /FID ne  \n\
    { dup /Encoding eq  \n\
      { exch pop ISOLatin1Encoding exch }  \n\
      if  \n\
      dup /FontName eq  \n\
      { exch pop newfontname exch }  \n\
      if  \n\
      exch newfont 3 1 roll put  \n\
    }  \n\
    { pop pop }  \n\
    ifelse  \n\
  }  \n\
  forall \n\
  newfontname newfont definefont \n\
} def \n");

    fprintf(psout, "\n\
/Times-Roman makeISOEncoded pop \n\
/Times-Bold makeISOEncoded pop \n\
/Times-Italic makeISOEncoded pop \n\
/Times-BoldItalic makeISOEncoded pop \n\
/Helvetica makeISOEncoded pop \n\
/Helvetica-Bold makeISOEncoded pop \n\
/Helvetica-Oblique makeISOEncoded pop \n\
/Helvetica-BoldOblique makeISOEncoded pop \n");
    /*
    * rotate if in landscape mode
    */
    if (dmode == 1)
    {
        fprintf(psout, "%d 0 translate\n", 2 * psymin + psdy);
        fprintf(psout, "90 rotate\n");
    }
    pssetcolor(1);
    pssetlinewidth(1);
    setlinestyle(0);
    psfont = -1;
    setfont(2);
    return 0;
}
