/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ------------------------------------------------------------------- 
 *
 *   plot.c:
 *
 *     This is part of the program ANIM. It provides some 
 *     subroutines to draw trajectories on plotters
 * 
 *     Date: Nov 95 peb
 *
 * ------------------------------------------------------------------- */

/* ------------------------------------------------------------------- */
/* Standard includes                                                   */
/* ------------------------------------------------------------------- */
#ifdef WIN32
#include <windows.h>
#endif
#include <sysdep/opengl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ------------------------------------------------------------------- */
/* Own includefiles                                                    */
/* ------------------------------------------------------------------- */
#include "anim.h"

/* ------------------------------------------------------------------- */
/* Prototypes                                                          */
/* ------------------------------------------------------------------- */
void anim_ini_plotter(void);
static void anim_ini_axis(anim_vector *, float, float, float, float);
void anim_ini_mat(float *, float, int, float, float, float);
void anim_ini_viewmat(float *);
void anim_draw_plotter(void);

/* defined in aux.c */
extern void matmult(anim_vector, float *, anim_vector);
extern int index_to_color_polygon(int);

/* ------------------------------------------------------------------- */
/* External defined global variables                                   */
/* ------------------------------------------------------------------- */
extern struct geometry geo;
extern struct animation str;
extern struct sensors sensor;
extern struct dyncolor dyn;
extern struct plotdata dat;
extern struct plotter plo;

extern float colorindex[MAXCOLORS + MAXNODYNCOLORS][4];
extern FILE *outfile;

/* ------------------------------------------------------------------- */
/* Subroutines                                                         */
/* ------------------------------------------------------------------- */
void anim_ini_plotter(void)
{
    int i, j;
    float a1, a2, a3, a4; /* dimensions of axis                       */
    float sc; /* factor for initializing edges of plotter */
    float height;

    /* free memory */
    if (plo.vertex != NULL)
    {
        OWN_FREE(plo.vertex);
    }
    for (i = 0; i < plo.nf; i++)
    {
        if (plo.face[i] != NULL)
        {
            OWN_FREE(plo.face[i]);
        }
    }
    if (plo.face != NULL)
    {
        OWN_FREE(plo.face);
    }
    for (i = 0; i < 2 * plo.nf; i++)
    {
        if (plo.axis[i] != NULL)
        {
            OWN_FREE(plo.axis[i]);
        }
    }
    if (plo.axis != NULL)
    {
        OWN_FREE(plo.axis);
    }
    if (plo.ky != NULL)
    {
        OWN_FREE(plo.ky);
    }
    if (plo.sy != NULL)
    {
        OWN_FREE(plo.sy);
    }

    /* initializing parameters */
    plo.nv = 8; /* ub hatte 9 ? */
    plo.nf = 6;
#ifdef VRANIM
    plo.h = 150;
    plo.b = 250;
    plo.t = 0;
#else
    plo.h = 2;
    plo.b = 3.2;
    plo.t = 2;
#endif

    /* memory allocation */
    OWN_CALLOC(plo.ky, float, plo.nf);
    OWN_CALLOC(plo.sy, float, plo.nf);
    OWN_CALLOC(plo.vertex, anim_vector, plo.nv);
    OWN_CALLOC(plo.face, int *, plo.nf);
    for (i = 0; i < plo.nf; i++)
    {
        OWN_CALLOC(plo.face[i], int, MAXVERTICES);
    }
    OWN_CALLOC(plo.axis, anim_vector *, plo.nf * 2);
    for (i = 0; i < plo.nf * 2; i++)
    {
        OWN_CALLOC(plo.axis[i], anim_vector, 10);
    }

    /* initializing  vertices */
    sc = 0.55f; /* edges of plotter 10% bigger  */
    plo.vertex[0][0] = -plo.b * sc; /* than drawing area            */
    plo.vertex[0][1] = -plo.h * sc;
    plo.vertex[0][2] = plo.t * sc;
    plo.vertex[1][0] = plo.b * sc;
    plo.vertex[1][1] = -plo.h * sc;
    plo.vertex[1][2] = plo.t * sc;
    plo.vertex[2][0] = plo.b * sc;
    plo.vertex[2][1] = plo.h * sc;
    plo.vertex[2][2] = plo.t * sc;
    plo.vertex[3][0] = -plo.b * sc;
    plo.vertex[3][1] = plo.h * sc;
    plo.vertex[3][2] = plo.t * sc;
    plo.vertex[4][0] = -plo.b * sc;
    plo.vertex[4][1] = -plo.h * sc;
    plo.vertex[4][2] = -plo.t * sc;
    plo.vertex[5][0] = -plo.b * sc;
    plo.vertex[5][1] = plo.h * sc;
    plo.vertex[5][2] = -plo.t * sc;
    plo.vertex[6][0] = plo.b * sc;
    plo.vertex[6][1] = plo.h * sc;
    plo.vertex[6][2] = -plo.t * sc;
    plo.vertex[7][0] = plo.b * sc;
    plo.vertex[7][1] = -plo.h * sc;
    plo.vertex[7][2] = -plo.t * sc;

    /*Initializing  faces */
    plo.face[0][0] = 1;
    plo.face[0][1] = 2;
    plo.face[0][2] = 3;
    plo.face[0][3] = 4;
    plo.face[0][4] = -93031;
    plo.face[1][0] = 1;
    plo.face[1][1] = 4;
    plo.face[1][2] = 6;
    plo.face[1][3] = 5;
    plo.face[1][4] = -93233;
    plo.face[2][0] = 1;
    plo.face[2][1] = 5;
    plo.face[2][2] = 8;
    plo.face[2][3] = 2;
    plo.face[2][4] = -93435;
    plo.face[3][0] = 7;
    plo.face[3][1] = 6;
    plo.face[3][2] = 4;
    plo.face[3][3] = 3;
    plo.face[3][4] = -93637;
    plo.face[4][0] = 7;
    plo.face[4][1] = 3;
    plo.face[4][2] = 2;
    plo.face[4][3] = 8;
    plo.face[4][4] = -93839;
    plo.face[5][0] = 7;
    plo.face[5][1] = 8;
    plo.face[5][2] = 5;
    plo.face[5][3] = 6;
    plo.face[5][4] = -94041;

    /* initializing axis */
    a1 = plo.b / 160.0;
    a4 = plo.h / 20.0;
    a3 = plo.b / 27.0;

    /* axis for face 0 */
    a2 = plo.h - a4;
    anim_ini_axis(plo.axis[0], a1, a2, a3, a4);
    a2 = plo.b - a4;
    anim_ini_axis(plo.axis[1], a1, a2, a3, a4);

    /* axis for face 1 */
    a2 = plo.t - a4;
    anim_ini_axis(plo.axis[2], a1, a2, a3, a4);
    a2 = plo.b - a4;
    anim_ini_axis(plo.axis[3], a1, a2, a3, a4);

    /* axis for face 2 */
    a2 = plo.h - a4;
    anim_ini_axis(plo.axis[4], a1, a2, a3, a4);
    a2 = plo.b - a4;
    anim_ini_axis(plo.axis[5], a1, a2, a3, a4);

    /* axis for face 3 */
    a2 = plo.t - a4;
    anim_ini_axis(plo.axis[6], a1, a2, a3, a4);
    a2 = plo.b - a4;
    anim_ini_axis(plo.axis[7], a1, a2, a3, a4);

    /* axis for face 4 */
    a2 = plo.h - a4;
    anim_ini_axis(plo.axis[8], a1, a2, a3, a4);
    a2 = plo.t - a4;
    anim_ini_axis(plo.axis[9], a1, a2, a3, a4);

    /* axis for face 5 */
    a2 = plo.h - a4;
    anim_ini_axis(plo.axis[10], a1, a2, a3, a4);
    a2 = plo.t - a4;
    anim_ini_axis(plo.axis[11], a1, a2, a3, a4);

    plo.drface[0] = 0; /*angle of rotation */
    plo.drface[1] = -1.5708f;
    plo.drface[2] = -3.1415f;
    plo.drface[3] = 1.5708f;
    plo.drface[4] = 1.5708f;
    plo.drface[5] = -1.5708f;
    plo.drax[0] = 1; /* axis of rotation */
    plo.drax[1] = 1;
    plo.drax[2] = 1;
    plo.drax[3] = 1;
    plo.drax[4] = 2;
    plo.drax[5] = 2;
    plo.drv[0][0] = -plo.b * 0.5; /* positional values */
    plo.drv[0][1] = -plo.h * 0.5;
    plo.drv[0][2] = plo.t * 0.56;

    plo.drv[1][0] = -plo.b * 0.5;
    plo.drv[1][1] = -plo.h * 0.56;
    plo.drv[1][2] = -plo.t * 0.5;

    plo.drv[2][0] = -plo.b * 0.5;
    plo.drv[2][1] = plo.h * 0.5;
    plo.drv[2][2] = -plo.t * 0.56;

    plo.drv[3][0] = -plo.b * 0.5;
    plo.drv[3][1] = plo.h * 0.56;
    plo.drv[3][2] = plo.t * 0.5;

    plo.drv[4][0] = plo.b * 0.56;
    plo.drv[4][1] = -plo.h * 0.5;
    plo.drv[4][2] = plo.t * 0.5;

    plo.drv[5][0] = -plo.b * 0.56;
    plo.drv[5][1] = -plo.h * 0.5;
    plo.drv[5][2] = -plo.t * 0.5;

    for (j = 0; j < dat.ndata; j++)
    {
        if ((j == 1) || (j == 3))
        {
            height = plo.t;
        }
        else
        {
            height = plo.h;
        }

        if ((dat.maxy[j] - dat.miny[j]) == 0)
        {
            fprintf(outfile, "Maximaler und minimaler Datenwert sind identisch!\n");
            /* return(ANIM_ERROR); gibt Fehler beim Kompilieren */
        }
        else
        {
            if ((dat.maxy[j] >= 0) && (dat.miny[j] <= 0))
            {
                plo.ky[j] = (-dat.miny[j] / (dat.maxy[j] - dat.miny[j])) * height;
                plo.sy[j] = height;
            }
            else
            {
                plo.ky[j] = 0;
                /* Hier muss noch ein Hinweis hinein, ob die X-Achse gezeichnet wird */
                plo.sy[j] = height;
            }
        }
    }
}

/* ------------------------------------------------------------------- */
static void anim_ini_axis(anim_vector *axis, float a1, float a2, float a3, float a4)
{
    axis[0][0] = -a1 * 0.5; /* points for bar of axis */
    axis[0][1] = 0;
    axis[0][2] = 0;
    axis[1][0] = a1 * 0.5;
    axis[1][1] = 0;
    axis[1][2] = 0;
    axis[2][0] = a1 * 0.5;
    axis[2][1] = a2;
    axis[2][2] = 0;
    axis[3][0] = -a1 * 0.5;
    axis[3][1] = a2;
    axis[3][2] = 0;

    axis[4][0] = a3 * 0.5; /* points for arrow of axis */
    axis[4][1] = a2;
    axis[4][2] = 0;
    axis[5][0] = 0;
    axis[5][1] = a2 + a4;
    axis[5][2] = 0;
    axis[6][0] = -a3 * 0.5;
    axis[6][1] = a2;
    axis[6][2] = 0;
}

/* ------------------------------------------------------------------- */
void anim_ini_mat(float *mat, float angle, int axis,
                  float rx, float ry, float rz)
{
    switch (axis)
    {

    case 0:
        mat[0] = 1;
        mat[1] = 0;
        mat[2] = 0;
        mat[3] = 0;
        mat[4] = 0;
        mat[5] = 1;
        mat[6] = 0;
        mat[7] = 0;
        mat[8] = 0;
        mat[9] = 0;
        mat[10] = 1;
        mat[11] = 0;
        mat[12] = rx;
        mat[13] = ry;
        mat[14] = rz;
        mat[15] = 1;
        break;
    case 1:
        mat[0] = 1;
        mat[1] = 0;
        mat[2] = 0;
        mat[3] = 0;
        mat[4] = 0;
        mat[5] = cos(angle);
        mat[6] = -sin(angle);
        mat[7] = 0;
        mat[8] = 0;
        mat[9] = sin(angle);
        mat[10] = cos(angle);
        mat[11] = 0;
        mat[12] = rx;
        mat[13] = ry;
        mat[14] = rz;
        mat[15] = 1;
        break;
    case 2:
        mat[0] = cos(angle);
        mat[1] = 0;
        mat[2] = -sin(angle);
        mat[3] = 0;
        mat[4] = 0;
        mat[5] = 1;
        mat[6] = 0;
        mat[7] = 0;
        mat[8] = sin(angle);
        mat[9] = 0;
        mat[10] = cos(angle);
        mat[11] = 0;
        mat[12] = rx;
        mat[13] = ry;
        mat[14] = rz;
        mat[15] = 1;
        break;
    case 3:
        mat[0] = cos(angle);
        mat[1] = -sin(angle);
        mat[2] = 0;
        mat[3] = 0;
        mat[4] = sin(angle);
        mat[5] = cos(angle);
        mat[6] = 0;
        mat[7] = 0;
        mat[8] = 0;
        mat[9] = 0;
        mat[10] = 1;
        mat[11] = 0;
        mat[12] = rx;
        mat[13] = ry;
        mat[14] = rz;
        mat[15] = 1;
        break;
    default:
        OWN_EXIT(ANIM_ERROR, "ERROR in anim_ini_mat");
    }
}

/* ------------------------------------------------------------------- */
void anim_ini_viewmat(float *plotmat)
{
    glPushMatrix();
    glLoadIdentity();

    /* initial orientation */
    glRotatef(PLOTTER_ROTX, 1, 0, 0);
    glRotatef(PLOTTER_ROTY, 0, 1, 0);
    glRotatef(PLOTTER_ROTZ, 0, 0, 1);

    /* initial position */
    glTranslatef(PLOTTER_MOVEX, PLOTTER_MOVEY, PLOTTER_MOVEZ);

    /* initial scale */
    glScalef(PLOTTER_SCALE, PLOTTER_SCALE, PLOTTER_SCALE);

    glGetFloatv(GL_MODELVIEW_MATRIX, plotmat);
    glPopMatrix();
}

/* ------------------------------------------------------------------------- */
void anim_draw_plotter(void) /* draw cube, axes and plots */
{
    int i, j, iax, index;
    float *adum = NULL, *bdum = NULL, *cdum = NULL;
    anim_vector vdum, v1dum, v2dum;
    float width;

    /* draw cube */
    for (i = 0; i < plo.nf; i++)
    {
        j = 0;
        while (plo.face[i][j] > 0)
        {
            j++;
        }
        glColor3fv(colorindex[index_to_color_polygon(plo.face[i][j])]);
        j = 0;
        glBegin(GL_POLYGON);
        while (plo.face[i][j] > 0)
        {
            glVertex3fv(plo.vertex[plo.face[i][j] - 1]);
            j++;
        }
        glEnd();
    }

    /* draw dat.ndata axes and plots */
    if (adum == NULL)
    {
        OWN_CALLOC(adum, float, 16);
    }
    if (bdum == NULL)
    {
        OWN_CALLOC(bdum, float, 16);
    }
    if (cdum == NULL)
    {
        OWN_CALLOC(cdum, float, 16);
    }

    iax = 0;
    index = -94848; /* set axis color */

    for (i = 0; i < dat.ndata; i++)
    {
        if (i >= 4)
        {
            width = plo.t;
        }
        else
        {
            width = plo.b;
        }
        anim_ini_mat(adum, plo.drface[i], plo.drax[i], 0.0, 0.0, 0.0);
        anim_ini_mat(bdum, 0.0, 0, plo.drv[i][0], plo.drv[i][1], plo.drv[i][2]);

        /* draw y - axis */
        glColor3fv(colorindex[index_to_color_polygon(index)]);

        /* draw axis bar */
        glBegin(GL_QUADS);
        for (j = 0; j < 4; j++)
        {
            matmult(vdum, adum, plo.axis[iax][j]);
            matmult(v2dum, bdum, vdum);
            glVertex3fv(v2dum);
        }
        glEnd();

        /* draw arrow */
        glBegin(GL_TRIANGLES);
        for (j = 4; j < 7; j++)
        {
            matmult(vdum, adum, plo.axis[iax][j]);
            matmult(v2dum, bdum, vdum);
            glVertex3fv(v2dum);
        }
        glEnd();

        iax++;

        /* draw x - axis */
        glColor3fv(colorindex[index_to_color_polygon(index)]);
        anim_ini_mat(cdum, 1.5708f, 3, 0.0, 0.0, 0.0);

        glBegin(GL_QUADS);
        for (j = 0; j < 4; j++)
        {
            matmult(v2dum, cdum, plo.axis[iax][j]);
            /* x-axis moved for negative y- values */
            v2dum[1] = v2dum[1] + plo.ky[i];
            matmult(vdum, adum, v2dum);
            matmult(v2dum, bdum, vdum);
            glVertex3fv(v2dum);
        }
        glEnd();

        glBegin(GL_TRIANGLES);
        for (j = 4; j < 7; j++)
        {
            matmult(v2dum, cdum, plo.axis[iax][j]);
            /* x-axis moved for negative y- values */
            v2dum[1] = v2dum[1] + plo.ky[i];
            matmult(vdum, adum, v2dum);
            matmult(v2dum, bdum, vdum);
            glVertex3fv(v2dum);
        }
        glEnd();

        iax++;

        /* plot data */
        glBegin(GL_LINE_STRIP);
        for (j = 0; j < str.act_step; j++)
        {

            /* First point of line to draw */
            v1dum[0] = j * width / str.timesteps;
            v1dum[1] = dat.data[i][j] * plo.sy[i];
            v1dum[2] = 0;

            matmult(vdum, adum, v1dum);
            matmult(v1dum, bdum, vdum);

            glVertex3fv(v1dum);
        }
        glEnd();
    }
}

/* ------------------------------------------------------------------- */
