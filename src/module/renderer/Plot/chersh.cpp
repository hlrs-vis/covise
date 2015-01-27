/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: chersh.c,v 1.2 1994/10/13 05:44:07 pturner Exp pturner $
 *
 * hershey fonts
 *
 */

#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include "hersh.h" /* character defs */
#include "special.h" /* character defs */
#include "extern.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * TODO - change the mapping of Greek and special chars
 */
static struct
{
    unsigned char *h_tables;
    int *h_indices;
}

hershey_fonts[12] = {

    { Romanc_table, Romanc_indices },
    { Romant_table, Romant_indices },
    { Italicc_table, Italicc_indices },
    { Italict_table, Italict_indices },
    { Romans_table, Romans_indices },
    { Romand_table, Romand_indices },
    { Italicc_table, Italicc_indices },
    { Italict_table, Italict_indices },
    { Greekc_table, Greekc_indices },
    { Special_table, Special_indices },
    { Special_table, Special_indices },
    { 0, 0 }
};

static unsigned char *chartable = Romanc_table;
static int *indices = Romanc_indices;

static int curfont;
static double hscale = 1.0;

/*
   select a font to use below
*/
void hselectfont(int f)
{
    chartable = hershey_fonts[f].h_tables;
    indices = hershey_fonts[f].h_indices;
    curfont = f;
    /* fudge factors to make for differences in hershey and PS fonts */
    switch (curfont)
    {
    case 0:
    case 1:
    case 2:
    case 3:
        hscale = 0.8;
        break;
    case 4:
    case 5:
        hscale = 0.9;
        break;
    case 6:
    case 7:
        hscale = 0.8;
        break;
    case 8:
    case 9:
    case 10:
    case 11:
        hscale = 0.9;
        break;
    default:
        hscale = 1.0;
    }
    /* if you are used to the old scaling then uncomment the following */
    /* hscale = 1.0; */
}

/*
   write s at xpos, ypos in device coordinates
   of size scale, direction dir, color color, using vector
   TODO - fix underlining
*/
void puthersh(int xpos, int ypos, double scale, int dir, int just, int color, void (*fvector)(int, int, int), char *s)
{
    int i, j, len = 0, ind, it1, it2, tind, sfont = curfont, slen = strlen(s);
    int sscript = 0, underline = 0;
    int ilenx, ileny;
    unsigned char charx, chary;
    double charw, x, y, xtmp, ytmp, saves = scale, slastx = 0.0, slasty = 0.0;
    double si = sin(M_PI / 180.0 * dir);
    double co = cos(M_PI / 180.0 * dir);
    double dxpos, dypos;

    ilenx = stringextentx(scale, s);
    /*
       ileny = stringextenty(scale, s);

       switch (just) {
       case 1:
           xpos = xpos - co * ilenx + si * ileny;
           ypos = ypos - si * ilenx - co * ileny;
           break;
       case 2:
           xpos = xpos - (co * ilenx - si * ileny) / 2;
           ypos = ypos - (si * ilenx + co * ileny) / 2;
   break;
   }
   */
    ileny = 0;

    switch (just)
    {
    case 1:
        xpos = (int)(xpos - co * ilenx);
        ypos = (int)(ypos - si * ilenx);
        break;
    case 2:
        xpos = (int)(xpos - (co * ilenx) / 2);
        ypos = (int)(ypos - (si * ilenx) / 2);
        break;
    }

    dxpos = xpos;
    dypos = ypos;

    setcolor(color);
    color = 1;
    for (i = 0; i < slen; i++)
    {
        if (s[i] < 32)
        {
            continue;
        }
        if (s[i] == '\\' && isdigit(s[i + 1]))
        {
            hselectfont(s[i + 1] - '0');
            i++;
            continue;
        }
        else if (s[i] == '\\' && s[i + 1] == '\\')
        {
            continue;
        }
        else if (s[i] == '\\' && isoneof(s[i + 1], (char *)"cCbxsSNuU+-"))
        {
            switch (s[i + 1])
            {
            case 'x':
                hselectfont(10);
                i++;
                break;
            case 's':
                scale = 0.6 * saves;
                sscript += 20;
                i++;
                break;
            case 'S':
                scale = 0.6 * saves;
                sscript += -20;
                i++;
                break;
            case 'N':
                scale = saves;
                sscript = 0;
                i++;
                break;
            case 'b':
                xpos = (int)(dxpos = dxpos - slastx);
                ypos = (int)(dypos = dypos - slasty);
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
                scale -= 0.2;
                i++;
                break;
            case '+':
                scale += 0.2;
                i++;
                break;
            }
            continue;
        }
        ind = s[i] - ' ';
        len = indices[ind + 1] - indices[ind];
        tind = 2 * indices[ind];
        it1 = chartable[tind];
        it2 = chartable[tind + 1];
        x = (it1 - 'R');
        y = (it2 - 'R');
        charw = y - x;
        for (j = 1; j < len; j++)
        {
            charx = chartable[tind + 2 * j];
            chary = chartable[tind + 2 * j + 1] + sscript;
            if (charx & 128)
            {
                charx &= 127;
                it1 = charx;
                it2 = chary;
                xtmp = hscale * scale * (it1 - 'R' - x);
                ytmp = (-scale * (it2 - 'R'));
                (*fvector)((int)(xpos + xtmp * co - ytmp * si), (int)(ypos + xtmp * si + ytmp * co), 0);
            }
            else
            {
                it1 = charx;
                it2 = chary;
                xtmp = hscale * scale * (it1 - 'R' - x);
                ytmp = (-scale * (it2 - 'R'));
                (*fvector)((int)(xpos + xtmp * co - ytmp * si), (int)(ypos + xtmp * si + ytmp * co), 1);
            }
        }
        if (underline)
        {
            (*fvector)((int)xpos, (int)(ypos - scale * 12.0), 0);
        }
        xpos = (int)(dxpos = dxpos + (slastx = hscale * scale * charw * co));
        ypos = (int)(dypos = dypos + (slasty = scale * charw * si));
        if (underline)
        {
            (*fvector)((int)xpos, (int)(ypos - scale * 12.0), 1);
        }
    }
    hselectfont(sfont);
}

/*
   get the x extent of the string in hershey coordinates given size
*/
int stringextentx(double scale, const char *s)
{
    int i, ind, xpos = 0, it1, it2, sfont = curfont, slen = strlen(s);
    double charw, x, y, dxpos, saves = scale, slastx = 0.0;

    dxpos = xpos;

    for (i = 0; i < slen; i++)
    {
        if (s[i] < 32)
        {
            continue;
        }
        if (s[i] == '\\' && isdigit(s[i + 1]))
        {
            hselectfont(s[i + 1] - '0');
            i++;
            continue;
        }
        else if (s[i] == '\\' && s[i + 1] == '\\')
        {
            continue;
        }
        else if (s[i] == '\\' && (isalpha(s[i + 1]) || s[i + 1] == '+' || s[i + 1] == '-'))
        {
            switch (s[i + 1])
            {
            case 'x':
                hselectfont(10);
                i++;
                break;
            case 's':
                scale = 0.6 * saves;
                i++;
                break;
            case 'S':
                scale = 0.6 * saves;
                i++;
                break;
            case 'N':
                scale = saves;
                i++;
                break;
            case 'b':
                xpos = (int)(dxpos = dxpos - slastx);
                i++;
                break;
            case 'c':
                i++;
                break;
            case 'C':
                i++;
                break;
            case '-':
                scale -= 0.2;
                i++;
                break;
            case '+':
                scale += 0.2;
                i++;
                break;
            }
            continue;
        }
        ind = s[i] - ' ';

        it1 = chartable[2 * indices[ind]];
        it2 = chartable[2 * indices[ind] + 1];
        x = it1 - 'R';
        y = it2 - 'R';
        charw = y - x;
        xpos = (int)(dxpos = dxpos + (slastx = hscale * scale * charw));
    }
    hselectfont(sfont);
    return xpos;
}

/*
   get the y extent of the string in hershey coordinates given size
*/
int stringextenty(double scale, const char *s)
{
    int i, j, len = 0, ind, it2;
    char charx, chary;
    double ytmp, ymin = 0, ymax = 0;

    for (i = 0; i < strlen(s); i++)
    {
        if (s[i] < 32)
        {
            continue;
        }
        ind = s[i] - ' ';
        len = indices[ind + 1] - indices[ind];

        for (j = 1; j < len; j++)
        {
            charx = chartable[2 * indices[ind] + 2 * j];
            chary = chartable[2 * indices[ind] + 2 * j + 1];
            if (charx & 128)
            {
                charx &= 127;
                it2 = chary;
                ytmp = (-scale * (it2 - 'R'));
            }
            else
            {
                it2 = chary;
                ytmp = (-scale * (it2 - 'R'));
            }
            if (ymax < ytmp)
                ymax = ytmp;
            if (ymin > ytmp)
                ymin = ytmp;
        }
    }
    return ((int)(ymax - ymin));
}
