/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvUIRegion.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include "InvUIRegion.h"
#include <GL/gl.h>

// ??? doing a GL_LINE_LOOP seems to be missing the top right
// ??? pixel due to subpixel == TRUE in openGL.
#define RECT(x1, y1, x2, y2) \
    glBegin(GL_LINE_STRIP);  \
    glVertex2s(x2, y2);      \
    glVertex2s(x1, y2);      \
    glVertex2s(x1, y1);      \
    glVertex2s(x2, y1);      \
    glVertex2s(x2, y2 + 1);  \
    glEnd();

void drawDownUIBorders(short x1, short y1, short x2, short y2, SbBool blackLast)
{
    DARK1_UI_COLOR;
    glBegin(GL_LINE_STRIP);
    glVertex2s(x1, y1 + 1);
    glVertex2s(x1, y2);
    glVertex2s(x2 + 1, y2);
    glEnd();
    LIGHT1_UI_COLOR;
    glBegin(GL_LINE_STRIP);
    glVertex2s(x1, y1);
    glVertex2s(x2, y1);
    glVertex2s(x2, y2);
    glEnd();

    x1++;
    y1++;
    x2--;
    y2--;
    DARK2_UI_COLOR;
    glBegin(GL_LINE_STRIP);
    glVertex2s(x1, y1 + 1);
    glVertex2s(x1, y2);
    glVertex2s(x2 + 1, y2);
    glEnd();
    WHITE_UI_COLOR;
    glBegin(GL_LINE_STRIP);
    glVertex2s(x1, y1);
    glVertex2s(x2, y1);
    glVertex2s(x2, y2);
    glEnd();

    x1++;
    y1++;
    x2--;
    y2--;
    if (blackLast)
    {
        BLACK_UI_COLOR;
        RECT(x1, y1, x2, y2);
    }
    else
    {
        DARK3_UI_COLOR;
        glBegin(GL_LINE_STRIP);
        glVertex2s(x1, y1 + 1);
        glVertex2s(x1, y2);
        glVertex2s(x2 + 1, y2);
        glEnd();
        DARK2_UI_COLOR;
        glBegin(GL_LINE_STRIP);
        glVertex2s(x1, y1);
        glVertex2s(x2, y1);
        glVertex2s(x2, y2);
        glEnd();
    }
}

void drawDownUIRegion(short x1, short y1, short x2, short y2)
{
    drawDownUIBorders(x1, y1, x2, y2);

    MAIN_UI_COLOR;
    x1 += UI_THICK;
    y1 += UI_THICK;
    x2 -= UI_THICK;
    y2 -= UI_THICK;
    RECT(x1, y1, x2, y2);
    x1++;
    y1++;
    x2--;
    y2--;
    RECT(x1, y1, x2, y2);
    x1++;
    y1++;
    x2--;
    y2--;

    drawDownUIBorders(x1, y1, x2, y2, TRUE);
}

void drawThumbUIRegion(short x1, short y1, short x2, short y2)
{
    short v[3][2];
    short x = (x1 + x2) / 2;

    v[0][0] = x1;
    v[0][1] = v[2][1] = y1;
    v[1][0] = x;
    v[1][1] = y2;
    v[2][0] = x2;

    MAIN_UI_COLOR;
    glBegin(GL_POLYGON);
    glVertex2sv(v[0]);
    glVertex2sv(v[1]);
    glVertex2sv(v[2]);
    glEnd();

    glBegin(GL_LINES);

    BLACK_UI_COLOR;
    glVertex2s(x1, y1);
    glVertex2s(x, y2);
    DARK3_UI_COLOR;
    glVertex2s(x, y2);
    glVertex2s(x2, y1);
    glVertex2s(x2, y1);
    glVertex2s(x1 + 1, y1);

    x1++;
    y1++;
    x2--;
    y2--;
    WHITE_UI_COLOR;
    glVertex2s(x1, y1);
    glVertex2s(x, y2);
    DARK2_UI_COLOR;
    glVertex2s(x, y2);
    glVertex2s(x2, y1);
    DARK1_UI_COLOR;
    glVertex2s(x2, y1);
    glVertex2s(x1 + 1, y1);

    x1++;
    y1++;
    x2--;
    y2--;
    WHITE_UI_COLOR;
    glVertex2s(x1, y1);
    glVertex2s(x, y2);
    DARK2_UI_COLOR;
    glVertex2s(x, y2);
    glVertex2s(x2, y1);

    glEnd();
}
