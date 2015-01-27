/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ------------------------------------------------------------------- 
 *
 *   read.c:
 *
 *     This is part of the program ANIM. It provides some 
 *     subroutines to read files
 * 
 *     Date: Mar 96
 *
 * ------------------------------------------------------------------- */

/* ------------------------------------------------------------------- */
/* Standard includes                                                   */
/* ------------------------------------------------------------------- */
#ifdef WIN32
#include <windows.h>
#endif
#include <sysdep/opengl.h>
#ifndef VRANIM
#include <GL/glut.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------- */
/* Own includefiles                                                    */
/* ------------------------------------------------------------------- */
#include "anim.h"

/* ------------------------------------------------------------------- */
/* Prototypes                                                          */
/* ------------------------------------------------------------------- */
int anim_read_geo_file(char *);
int anim_read_iv_file(char *);
int anim_read_str_file(char *);
int anim_read_trmat_file(char *);
int anim_read_sns_file(char *);
int anim_read_cmp_file(char *);
int anim_read_lig_file(char *);
int anim_read_set_file(char *);
int anim_read_data_file(char *);
int anim_read_dyncolor_file(char *);
int anim_read_elgeo_file(char *);
int anim_read_lin_file(char *);

extern void delete_sensors(void);
extern void update_sensors(void);
extern void calc_lineel(void);
extern void find_norm(anim_vector *, int *, anim_vector *, int *, int);
extern float normize(anim_vector);
extern int fget_line(char *, FILE *);
extern int strcnt(char *, int);
extern void anim_ini_plotter(void);
extern void anim_ini_viewmat(float *);

#ifdef VRANIM
/* dummy routine */
void glutSetCursor(int);
#define GLUT_CURSOR_INHERIT 0
#define GLUT_CURSOR_WAIT 0
#endif

/* ------------------------------------------------------------------- */
/* External defined global variables                                   */
/* ------------------------------------------------------------------- */
extern struct geometry geo;
extern struct animation str;
extern struct sensors sensor;
extern struct plotdata dat;
extern struct elgeometry elgeo;
extern struct dyncolor dyn;
extern struct ivfiles iv;
extern struct lineelem lin;

extern float colorindex[MAXCOLORS + MAXNODYNCOLORS + 1][4];
extern FILE *outfile;
extern float plotmat[16];

/* ------------------------------------------------------------------- */
/* Subroutines                                                         */
/* ------------------------------------------------------------------- */
int anim_read_geo_file(char *geo_file)
{
    int *normindex = NULL, /* number of faces contacting */
        /* vertex i in open file      */
        i, j, jj;
    char *lne = NULL; /* line of text */
    FILE *fp, /* pointer on input file */
        *tmpfp = NULL; /* pointer on temporary file */

    /* read filename (if not given) */
    if (strlen(geo_file) == 0)
    {
        fprintf(outfile, "Enter the geometric file name: ");
        (void)system("ls *" STDGEO);
        (void)scanf("%s", geo_file);
    }

    /* open file */
    if ((fp = fopen(geo_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", geo_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    glutSetCursor(GLUT_CURSOR_WAIT);

    /* free memory allocation */
    if (geo.first != GL_TRUE)
    {
        for (i = 0; i < geo.nfiles; i++)
        {
            for (j = 0; j < geo.nf[i]; j++)
            {
                OWN_FREE(geo.face[i][j]);
            }
            OWN_FREE(geo.npoints[i]);
            OWN_FREE(geo.ecolor[i]);
            OWN_FREE(geo.fcolor[i]);
            OWN_FREE(geo.vertex[i]);
            OWN_FREE(geo.norm[i]);
            OWN_FREE(geo.face[i]);
            OWN_FREE(geo.name[i]);
        }
        OWN_FREE(geo.npoints);
        OWN_FREE(geo.ecolor);
        OWN_FREE(geo.fcolor);
        OWN_FREE(geo.name);
        OWN_FREE(geo.norm);
        OWN_FREE(geo.vertex);
        OWN_FREE(geo.face);
        OWN_FREE(geo.shading);
        OWN_FREE(geo.hide);
        OWN_FREE(geo.fixmotion);
        OWN_FREE(geo.fixtranslation);
        OWN_FREE(geo.nf);
        OWN_FREE(geo.nvertices);
        if (geo.nballs != 0)
        {
            OWN_FREE(geo.ballsradius);
            OWN_FREE(geo.ballscolor);
        }
        geo.first = GL_TRUE;
    }

    /* read geometric information */
    if ((i = fscanf(fp, "%d %d", &geo.nfiles, &geo.nballs)) == EOF)
    {
        fprintf(outfile,
                "... error while reading the object file %s (headerline)\n",
                geo_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    if (i == 1)
    { /* only one number in headerline -> no balls */
        geo.nballs = 0;
        fprintf(outfile, "... no balls must be read\n");
    }

    /* memory allocation (for each part) */
    OWN_CALLOC(geo.nf, int, geo.nfiles);
    OWN_CALLOC(geo.nvertices, int, geo.nfiles);
    OWN_CALLOC(geo.shading, int, geo.nfiles);
    OWN_CALLOC(geo.hide, int, geo.nfiles);
    OWN_CALLOC(geo.fixmotion, int, geo.nfiles);
    OWN_CALLOC(geo.fixtranslation, int, geo.nfiles);
    OWN_CALLOC(geo.face, int **, geo.nfiles);
    OWN_CALLOC(geo.vertex, anim_vector *, geo.nfiles);
    OWN_CALLOC(geo.norm, anim_vector *, geo.nfiles);
    OWN_CALLOC(geo.name, char *, geo.nfiles);
    OWN_CALLOC(geo.ecolor, int *, geo.nfiles);
    OWN_CALLOC(geo.fcolor, int *, geo.nfiles);
    OWN_CALLOC(geo.npoints, int *, geo.nfiles);

    if (geo.nballs != 0)
    {
        OWN_CALLOC(geo.ballsradius, double, geo.nballs);
        OWN_CALLOC(geo.ballscolor, int, geo.nballs);
    }

    for (i = 0; i < geo.nfiles; i++)
    {
        OWN_CALLOC(geo.name[i], char, MAXLENGTH);
    }

    /* read file names */
    for (i = 0; i < geo.nfiles; i++)
    {
        if (fscanf(fp, "%s", geo.name[i]) == EOF)
        {
            fprintf(outfile,
                    "... error while reading the object file %s (part %d)\n",
                    geo_file, i + 1);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }
    }
    if (geo.nballs != 0)
    {
        for (i = 0; i < geo.nballs; i++)
        {
            if (fscanf(fp, "%lf %d",
                       &geo.ballsradius[i], &geo.ballscolor[i]) == EOF)
            {
                fprintf(outfile,
                        "... error while reading the object file %s (ball %d)\n",
                        geo_file, i + 1);
                glutSetCursor(GLUT_CURSOR_INHERIT);
                return (ANIM_ERROR);
            }
        }
    }

    (void)fclose(fp);

    for (i = 0; i < geo.nfiles; i++)
    {
        geo.shading[i] = 1;
        /* geo.shading[i] = 0; diese Koerper werden immer als wire gezeichnet */
        geo.hide[i] = GL_FALSE;
        geo.fixmotion[i] = GL_FALSE;
        geo.fixtranslation[i] = GL_FALSE;
    }

    OWN_CALLOC(lne, char, MAXLENGTH);

    /* read the graphics-files */
    for (i = 0; i < geo.nfiles; i++)
    {
        if ((fp = fopen(geo.name[i], "r")) == NULL)
        {
            fprintf(outfile, "... cannot open file %s (%d of %d)\n",
                    geo.name[i], i, geo.nfiles);
            OWN_FREE(lne);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }

        glutSetCursor(GLUT_CURSOR_WAIT);

        fprintf(outfile, "        opened file %s (%d of %d)\n",
                geo.name[i], (i + 1), geo.nfiles);

        if (fget_line(lne, fp) == ANIM_ERROR)
        {
            fprintf(outfile, "... error while reading the file %s\n",
                    geo.name[i]);
            OWN_FREE(lne);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }
        if (sscanf(lne, "%d %d", &(geo.nvertices[i]), &(geo.nf[i])) != 2)
        {
            fprintf(outfile, "... error while reading the file %s\n",
                    geo.name[i]);
            OWN_FREE(lne);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }

        /* memory allocation (for vertices) */
        OWN_CALLOC(geo.vertex[i], anim_vector, geo.nvertices[i]);
        OWN_CALLOC(geo.norm[i], anim_vector, geo.nvertices[i]);

        /* read vertices */
        for (j = 0; j < geo.nvertices[i]; j++)
        {
            if (fget_line(lne, fp) == ANIM_ERROR)
            {
                fprintf(outfile,
                        "... error while reading the file %s , vertex %d\n",
                        geo.name[i], j);
                OWN_FREE(lne);
                glutSetCursor(GLUT_CURSOR_INHERIT);
                return (ANIM_ERROR);
            }
            if (sscanf(lne, "%f %f %f", &geo.vertex[i][j][0],
                       &geo.vertex[i][j][1],
                       &geo.vertex[i][j][2]) != 3)
            {
                fprintf(outfile,
                        "... error while reading the file %s , vertex %d\n",
                        geo.name[i], j);
                OWN_FREE(lne);
                glutSetCursor(GLUT_CURSOR_INHERIT);
                return (ANIM_ERROR);
            }
        }

/* produce temporary file containing face information without remarks */
#ifdef WIN32
        if ((tmpfp = fopen(_tempnam("c:\\", "vranim"), "w+")) == NULL)
        {
#else
        if ((tmpfp = tmpfile()) == NULL)
        {
#endif
            OWN_EXIT(ANIM_ERROR, "ERROR opening tmpfile");
        }
        while (fget_line(lne, fp) == ERRORFREE)
        {
            fprintf(tmpfp, "%s", lne);
        }
        rewind(tmpfp);
        (void)fclose(fp);

        /* memory allocation (for faces) */
        /* note: face is only allocated locally; therefore allocation needs
             not to be careful with memory occupation                    */
        OWN_CALLOC(normindex, int, geo.nvertices[i]);
        OWN_CALLOC(geo.face[i], int *, geo.nf[i]);
        OWN_CALLOC(geo.ecolor[i], int, geo.nf[i]);
        OWN_CALLOC(geo.fcolor[i], int, geo.nf[i]);
        OWN_CALLOC(geo.npoints[i], int, geo.nf[i]);

        /* read faces */
        for (j = 0; j < geo.nf[i]; j++)
        {
            if (fscanf(tmpfp, "%d %d %d",
                       &geo.npoints[i][j],
                       &geo.ecolor[i][j],
                       &geo.fcolor[i][j]) != 3)
            {
                fprintf(outfile,
                        "... error while reading file %s header face %d\n",
                        geo.name[i], j);
                OWN_FREE(lne);
                glutSetCursor(GLUT_CURSOR_INHERIT);
                return (ANIM_ERROR);
            }
            OWN_CALLOC(geo.face[i][j], int, geo.npoints[i][j]);
            for (jj = 0; jj < geo.npoints[i][j]; jj++)
            {
                if (fscanf(tmpfp, "%d", &geo.face[i][j][jj]) != 1)
                {
                    fprintf(outfile,
                            "... error while reading the file %s face %d vertex %d\n",
                            geo.name[i], j, jj);
                    OWN_FREE(lne);
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                if (geo.face[i][j][jj] > geo.nvertices[i])
                {
                    fprintf(outfile,
                            "... error while reading the file %s face %d\n",
                            geo.name[i], j);
                    fprintf(outfile, "   -> illegal vertex !\n");
                    OWN_FREE(lne);
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
            }
            find_norm(geo.vertex[i], normindex,
                      geo.norm[i], geo.face[i][j], geo.npoints[i][j]);
        }
        OWN_FREE(normindex);
        for (j = 0; j < geo.nvertices[i]; j++)
        {
            (void)normize(geo.norm[i][j]);
        }
        (void)fclose(tmpfp);
    }
    OWN_FREE(lne);

    fprintf(outfile, "... %d bodies and %d balls have been read from file %s\n",
            geo.nfiles, geo.nballs, geo_file);
    geo.first = GL_FALSE;

    /* write a warning if stripped data exists */
    if (str.timesteps > 0)
    {
        fprintf(outfile, "\nWarning: New geometry data was read, that may be\n");
        fprintf(outfile, "    inconsistent to currend stripped data!\n");
    }

    glutSetCursor(GLUT_CURSOR_INHERIT);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_iv_file(char *iv_file)
{
    int i;
    FILE *fp;

    /* read filename (if not given) */
    if (strlen(iv_file) == 0)
    {
        fprintf(outfile, "Enter the inventor file name: ");
        (void)system("ls *" STDIV);
        (void)scanf("%s", iv_file);
    }

    /* open file */
    if ((fp = fopen(iv_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", iv_file);
        return (ANIM_ERROR);
    }

    /* free memory allocation */
    if (iv.nfiles != 0)
    {
        for (i = 0; i < iv.nfiles; i++)
        {
            OWN_FREE(iv.name[i]);
        }
        OWN_FREE(iv.name);
    }

    /* read geometric information */
    if (fscanf(fp, "%d", &iv.nfiles) == EOF)
    {
        fprintf(outfile,
                "... error while reading the .ivall file %s (headerline)\n",
                iv_file);
        return (ANIM_ERROR);
    }

    /* read file names */
    OWN_CALLOC(iv.name, char *, iv.nfiles);
    for (i = 0; i < iv.nfiles; i++)
    {
        OWN_CALLOC(iv.name[i], char, MAXLENGTH);
        if (fscanf(fp, "%s", iv.name[i]) == EOF)
        {
            fprintf(outfile,
                    "... error while reading the .ivall file %s (name %d)\n",
                    iv_file, i + 1);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }
    }

    (void)fclose(fp);
    fprintf(outfile, "... %d names of .iv files have been read from file %s\n",
            iv.nfiles, iv_file);

    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_str_file(char *str_file)
{
    int i, j, k, type;
    char *lne = NULL;
    FILE *fp;

    /* read filename (if not given) */
    if (strlen(str_file) == 0)
    {
        fprintf(outfile, "Enter the stripped file name: ");
        (void)system("ls *" STDSTR);
        (void)scanf("%s", str_file);
    }

    /* free memory */
    if (str.first != GL_TRUE)
    {
        for (i = 0; i < str.timesteps; i++)
        {
            for (j = 0; j < geo.nfiles; j++)
            {
                OWN_FREE(str.a[i][j]);
            }
            for (j = 0; j < geo.nballs; j++)
            {
                OWN_FREE(str.a_ball[i][j]);
            }
            OWN_FREE(str.a[i]);
            OWN_FREE(str.a_ball[i]);
            OWN_FREE(str.balldyncolor[i]);
        }
        OWN_FREE(str.a);
        OWN_FREE(str.a_ball);
        OWN_FREE(str.balldyncolor);
    }

    glutSetCursor(GLUT_CURSOR_WAIT);

    /* open file */
    if ((fp = fopen(str_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", str_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    /* read the animation datafile */
    OWN_CALLOC(lne, char, MAXLENGTH);
    (void)fgets(lne, MAXLENGTH, fp);

    if (sscanf(lne, "%f %d %d", &(str.dt), &(str.timesteps), &type) != 3)
    {
        (void)fclose(fp);
        fprintf(outfile, "  error while reading header line of str-file: %s\n", lne);
        OWN_FREE(lne);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }
    OWN_FREE(lne);

    if (str.dt <= 0)
    {
        fprintf(outfile, "  wrong dt - time step width must be greater/equal 0");
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    if (str.timesteps <= 0)
    {
        fprintf(outfile, "  wrong number of time steps must be greater than 0: %d", str.timesteps);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    if (type != 12 && type != 16)
    {
        fprintf(outfile, "  wrong type in header line of str-file: %d\n", type);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    /* memory allocation */
    if (geo.nballs > 0)
    {
        OWN_CALLOC(str.balldyncolor, float *, str.timesteps);
        for (i = 0; i < str.timesteps; i++)
        {
            OWN_CALLOC(str.balldyncolor[i], float, geo.nballs);
        }
    }

    OWN_CALLOC(str.a, float **, str.timesteps);
    if (geo.nballs > 0)
    {
        OWN_CALLOC(str.a_ball, float **, str.timesteps);
    }
    str.act_step = 0;
    for (i = 0; i < str.timesteps; i++)
    {
        OWN_CALLOC(str.a[i], float *, geo.nfiles);
        for (j = 0; j < geo.nfiles; j++)
        {
            OWN_CALLOC(str.a[i][j], float, 16);
            if (type == 16)
            {
                for (k = 0; k < 16; k++)
                {
                    if (fscanf(fp, "%f", &str.a[i][j][k]) != 1)
                    {
                        goto endloop;
                    }
                }
            }
            else
            {
                if (fscanf(fp, "%f", &str.a[i][j][0]) != 1)
                {
                    goto endloop;
                }
                fscanf(fp, "%f", &str.a[i][j][1]);
                fscanf(fp, "%f", &str.a[i][j][2]);
                str.a[i][j][3] = 0;
                fscanf(fp, "%f", &str.a[i][j][4]);
                fscanf(fp, "%f", &str.a[i][j][5]);
                fscanf(fp, "%f", &str.a[i][j][6]);
                str.a[i][j][7] = 0;
                fscanf(fp, "%f", &str.a[i][j][8]);
                fscanf(fp, "%f", &str.a[i][j][9]);
                fscanf(fp, "%f", &str.a[i][j][10]);
                str.a[i][j][11] = 0;
                fscanf(fp, "%f", &str.a[i][j][12]);
                fscanf(fp, "%f", &str.a[i][j][13]);
                fscanf(fp, "%f", &str.a[i][j][14]);
                str.a[i][j][15] = 1;
            }
        }

        if (geo.nballs > 0)
        {
            OWN_CALLOC(str.a_ball[i], float *, geo.nballs);
            for (j = 0; j < geo.nballs; j++)
            {
                OWN_CALLOC(str.a_ball[i][j], float, 16);
                if (type == 16)
                {
                    for (k = 0; k < 16; k++)
                    {
                        if (fscanf(fp, "%f", &str.a_ball[i][j][k]) != 1)
                        {
                            goto endloop;
                        }
                    }
                    if (geo.ballscolor[j] < 0)
                    {
                        fscanf(fp, "%f", &str.balldyncolor[i][j]);
                    }
                }
                else
                {
                    if (fscanf(fp, "%f", &str.a_ball[i][j][0]) != 1)
                    {
                        goto endloop;
                    }
                    fscanf(fp, "%f", &str.a_ball[i][j][1]);
                    fscanf(fp, "%f", &str.a_ball[i][j][2]);
                    str.a_ball[i][j][3] = 0;
                    fscanf(fp, "%f", &str.a_ball[i][j][4]);
                    fscanf(fp, "%f", &str.a_ball[i][j][5]);
                    fscanf(fp, "%f", &str.a_ball[i][j][6]);
                    str.a_ball[i][j][7] = 0;
                    fscanf(fp, "%f", &str.a_ball[i][j][8]);
                    fscanf(fp, "%f", &str.a_ball[i][j][9]);
                    fscanf(fp, "%f", &str.a_ball[i][j][10]);
                    str.a_ball[i][j][11] = 0;
                    fscanf(fp, "%f", &str.a_ball[i][j][12]);
                    fscanf(fp, "%f", &str.a_ball[i][j][13]);
                    fscanf(fp, "%f", &str.a_ball[i][j][14]);
                    str.a_ball[i][j][15] = 1;
                    if (geo.ballscolor[j] < 0)
                    {
                        fscanf(fp, "%f", &str.balldyncolor[i][j]);
                    }
                }
            }
        }
        str.act_step++;
    }

/* completed reading */
endloop:
    (void)fclose(fp);

    if (str.timesteps != str.act_step)
    {
        fprintf(outfile, "\n  !!! WARNING !!!: wrong number of time steps in file: "
                         "requested=%d, read=%d\n\n",
                str.timesteps, str.act_step);
        /* lk why not just change number ? */
        str.timesteps = str.act_step;
        glutSetCursor(GLUT_CURSOR_INHERIT);
        /*      return(ANIM_ERROR); */
    }

    str.act_step = 0;
    fprintf(outfile, "... %d timesteps for %d bodies and %d balls have been read "
                     "from file %s (dt = %f)\n",
            str.timesteps, geo.nfiles, geo.nballs, str_file, str.dt);
    str.first = GL_FALSE;

    glutSetCursor(GLUT_CURSOR_INHERIT);

    update_sensors();

    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_trmat_file(char *trmat_file)
{
    int i; /* counter */
    FILE *fp; /* file pointer */
    float *trmat = NULL;
#ifdef VRANIM
    extern float globalmat[17];
#endif

#ifdef VRANIM
    strcpy(trmat_file, "viewing.mat");
#endif

    /* read filename (if not given) */
    if (strlen(trmat_file) == 0)
    {
        fprintf(outfile, "Enter the transformation file name: ");
        (void)system("ls *" STDTRANS);
        (void)scanf("%s", trmat_file);
    }

    /* open file */
    if ((fp = fopen(trmat_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", trmat_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    if (trmat == NULL)
    {
        OWN_CALLOC(trmat, float, 16);
    }

    glutSetCursor(GLUT_CURSOR_WAIT);

    /* read file */
    for (i = 0; i < 16; i++)
    {
        (void)fscanf(fp, "%f ", &trmat[i]);
    }

#ifdef VRANIM
    for (i = 0; i < 16; i++)
    {
        globalmat[i] = trmat[i];
    }
    /* scaling factor */
    (void)fscanf(fp, "%f ", &(globalmat[16]));
#else
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultMatrixf(trmat);
    glPushMatrix();
#endif

    OWN_FREE(trmat);

    /* completed reading */
    (void)fclose(fp);
    fprintf(outfile, "... transformation matrix has been read from file %s\n",
            trmat_file);
    glutSetCursor(GLUT_CURSOR_INHERIT);
    return (ERRORFREE);
}
/* ------------------------------------------------------------------- */
int anim_read_lin_file(char *lin_file)
{
    int bod1, bod2, type; /*input values*/
    anim_vector coo1, coo2; /*input values*/
    char *lne = NULL; /*line of text*/
    FILE *fp; /* file pointer */
    int i, n; /*Zaehlvariable*/

    /*check if timedata is available*/
    if (str.timesteps == 0)
    {
        fprintf(outfile, "   no line elements read because no animation data"
                         "seems to be available (str.timesteps=0)\n");
        return (ANIM_ERROR);
    }

    /* read filename (if not given) */
    if (strlen(lin_file) == 0)
    {
        fprintf(outfile, "Enter the line element file name: ");
        (void)system("ls *" STDLIN);
        (void)scanf("%s", lin_file);
    }
    if ((fp = fopen(lin_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", lin_file);
        return (ANIM_ERROR);
    }
    OWN_CALLOC(lne, char, MAXLENGTH);
    /* read number of elements*/
    if (fgets(lne, MAXLENGTH, fp) == NULL)
    {
        fprintf(outfile, "...error while reading header line");
        return (ANIM_ERROR);
    }
    if ((sscanf(lne, "%d", &lin.nr)) == EOF)
    {
        fprintf(outfile, "...error while reading header line");
        return (ANIM_ERROR);
    }

    /*memory allocation*/
    OWN_CALLOC(lin.color, int, lin.nr);
    OWN_CALLOC(lin.type, int, lin.nr);
    OWN_CALLOC(lin.bod1, int, lin.nr);
    OWN_CALLOC(lin.bod2, int, lin.nr);
    OWN_CALLOC(lin.coo1, anim_vector, lin.nr);
    OWN_CALLOC(lin.coo2, anim_vector, lin.nr);
    OWN_CALLOC(lin.pkt1, anim_vector *, lin.nr);
    OWN_CALLOC(lin.pkt2, anim_vector *, lin.nr);
    OWN_CALLOC(lin.dir, anim_vector *, lin.nr);
    OWN_CALLOC(lin.length, float *, lin.nr);
    OWN_CALLOC(lin.name, char *, lin.nr);
    for (i = 0; i < lin.nr; i++)
    {
        OWN_CALLOC(lin.name[i], char, MAXLENGTH);
    }

    /*read geometric information*/
    for (i = 0; i < lin.nr; i++)
    {
        if (fgets(lne, MAXLENGTH, fp) != NULL)
        {
            n = sscanf(lne, "%d %d %f %f %f %d %f %f %f", &type, &bod1, &coo1[0], &coo1[1], &coo1[2], &bod2, &coo2[0], &coo2[1], &coo2[2]);
            if (n == 9)
            {
                if ((bod1 < 1) || (bod1 > geo.nfiles + geo.nballs))
                {
                    fprintf(outfile, "  error while reading line element %d", i + 1);
                    fprintf(outfile, "  body %d does not exist\n", bod1);
                    return (ANIM_ERROR);
                }
                else if ((bod2 < 1) || (bod2 > geo.nfiles + geo.nballs))
                {
                    fprintf(outfile, "  error while reading line element %d", i + 1);
                    fprintf(outfile, "  body %d does not exist\n", bod2);
                    return (ANIM_ERROR);
                }
                else if ((type != 1) && (type != 2) && (type != 3))
                {
                    fprintf(outfile, "  error while reading line element %d", i + 1);
                    fprintf(outfile, "  invalid body type\n");
                }
                else
                {
                    lin.type[i] = type;
                    lin.bod1[i] = bod1 - 1;
                    lin.coo1[i][0] = coo1[0];
                    lin.coo1[i][1] = coo1[1];
                    lin.coo1[i][2] = coo1[2];
                    lin.bod2[i] = bod2 - 1;
                    lin.coo2[i][0] = coo2[0];
                    lin.coo2[i][1] = coo2[1];
                    lin.coo2[i][2] = coo2[2];
                }
            }
            else
            {
                fprintf(outfile, "  error while reading line element %d", i + 1);
                fprintf(outfile, "  incorrect format in line %d \n", i + 2);
                return (ANIM_ERROR);
            }
            if ((lin.type[i] == 1) || (lin.type[i] == 2))
            {
                if (fscanf(fp, "%d \n", &lin.color[i]) == EOF)
                {
                    fprintf(outfile, "...error while reading color information");
                    fprintf(outfile, "   of element %d \n", i + 1);
                }
            }
            else
            {
                if (fscanf(fp, "%s \n", lin.name[i]) == EOF)
                {
                    fprintf(outfile, "...error while reading the .iv file name");
                    fprintf(outfile, " of element %d \n", i + 1);
                }
                else
                {
                    lin.n_iv = lin.n_iv + 1;
                }
            }
        }
        else
        {
            fprintf(outfile, "  error while reading line elements");
            fprintf(outfile, "  number of elements does not correspond");
            fprintf(outfile, "  to the attachment information given\n");
        }
    }

    calc_lineel();
    (void)fclose(fp);
    OWN_FREE(lne);
    fprintf(outfile, "    %d line elements have been read\n", lin.nr);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_sns_file(char *sns_file)
{
    int bod, col, /* input values */
        n; /* counter */
    anim_vector coo; /* input value */
    char *lne = NULL; /* line of text */
    FILE *fp; /* file pointer */

    /* check if timedata is available */
    if (str.timesteps == 0)
    {
        fprintf(outfile, "    no sensors read because no animation data"
                         " seems to be available (str.timesteps=0)\n");
        return (ANIM_ERROR);
    }

    /* free memory */
    delete_sensors();

    /* read filename (if not given) */
    if (strlen(sns_file) == 0)
    {
        fprintf(outfile, "Enter the sensor file name: ");
        (void)system("ls *" STDSNS);
        (void)scanf("%s", sns_file);
    }

    /* determine number of sensors in file */
    if ((fp = fopen(sns_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", sns_file);
        return (ANIM_ERROR);
    }
    OWN_CALLOC(lne, char, MAXLENGTH);
    while (fgets(lne, MAXLENGTH, fp) != NULL)
    {
        if ((unsigned int)strlen(lne) > 1)
        { /* catch empty lines */
            sensor.nr++;
        }
    }
    (void)fclose(fp);
    if (sensor.nr == 0)
    {
        return (ANIM_ERROR);
    }

    /* open file again */
    if ((fp = fopen(sns_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", sns_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }
    glutSetCursor(GLUT_CURSOR_WAIT);

    /* memory allocation */
    OWN_CALLOC(sensor.bod, int, sensor.nr);
    OWN_CALLOC(sensor.col, int, sensor.nr);
    OWN_CALLOC(sensor.coo, anim_vector, sensor.nr);
    OWN_CALLOC(sensor.pkt, anim_vector *, sensor.nr);

    /* read sensor file */
    sensor.nr = 0;
    while (fgets(lne, MAXLENGTH, fp) != NULL)
    {
        if ((unsigned int)strlen(lne) > 1)
        { /* catch empty lines */
            n = sscanf(lne, "%d %f %f %f %d", &bod, &coo[0], &coo[1], &coo[2], &col);
            if (n >= 4)
            {
                if ((bod < 1) || (bod > geo.nfiles + geo.nballs))
                {
                    fprintf(outfile, "  unable to use sensor %d", (sensor.nr) + 1);
                    fprintf(outfile, "  -> sensor ignored\n");
                    fprintf(outfile, "  body %d does not exist\n", bod);
                }
                else
                {
                    sensor.bod[sensor.nr] = bod - 1;
                    sensor.coo[sensor.nr][0] = coo[0];
                    sensor.coo[sensor.nr][1] = coo[1];
                    sensor.coo[sensor.nr][2] = coo[2];
                    if (n == 4)
                    {
                        sensor.col[sensor.nr] = sensor.nr;
                    }
                    else
                    {
                        if (col > MAXCOLORS)
                        {
                            sensor.col[sensor.nr] = MAXCOLORS;
                        }
                        else
                        {
                            sensor.col[sensor.nr] = col;
                        }
                    }
                    sensor.nr++;
                }
            }
            else
            {
                fprintf(outfile, "... error in reading sensor data\n");
                fprintf(outfile, "   (not enough entries in line)\n");
            }
        }
    }

    (void)fclose(fp);
    OWN_FREE(lne);
    fprintf(outfile, "    %d sensors have been read\n", sensor.nr);

    /* compute sensor trajectories */
    update_sensors();

    glutSetCursor(GLUT_CURSOR_INHERIT);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_cmp_file(char *cmp_file)
{
    int col, i, nr = 0; /* count colors */
    float red, green, blue, alpha;
    char *lne = NULL; /* line of text */
    FILE *fp; /* file pointer */

    /* read filename (if not given) */
    if (strlen(cmp_file) == 0)
    {
        fprintf(outfile, "Enter the colormap file name: ");
        (void)system("ls *" STDCMP);
        (void)scanf("%s", cmp_file);
    }

    glutSetCursor(GLUT_CURSOR_WAIT);

    /* open file */
    if ((fp = fopen(cmp_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", cmp_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    /* read colormap-file */
    OWN_CALLOC(lne, char, MAXLENGTH);
    do
    {
        if (fget_line(lne, fp) != ANIM_ERROR)
        {
            i = sscanf(lne, "%d %f %f %f %f", &col, &red, &green, &blue, &alpha);
            if (i >= 4)
            {
                if (i == 4)
                {
                    alpha = 1;
                }
                if (red > 1 || green > 1 || blue > 1)
                {
                    fprintf(outfile, "please update your colormap file so that all color "
                                     "values are in the range 0<=val<=1.\n");
                    /* exit(ANIM_ERROR); */
                    fprintf(outfile, " For now all colors are scaled by 1/255, "
                                     "but in the future this will not work any longer\n");
                    red = red / 255;
                    green = green / 255;
                    blue = blue / 255;
                }
                nr++;
                if (col < MAXCOLORS)
                {
                    colorindex[col][0] = red;
                    colorindex[col][1] = green;
                    colorindex[col][2] = blue;
                    colorindex[col][3] = alpha;
                }
                else
                {
                    fprintf(outfile,
                            "... warning - legal colorindex range 0..%d, requested %d\n",
                            MAXCOLORS - 1, col);
                }
            }
            else
            {
                fprintf(outfile,
                        "... error reading color in file %s (line %d)\n",
                        cmp_file, nr + 1);
            }
        }
        else if (feof(fp) == GL_FALSE)
        {
            fprintf(outfile, "... error while reading the colormap file %s\n",
                    cmp_file);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }
    } while (feof(fp) == GL_FALSE);

    fprintf(outfile, "... %d colors have been read\n", nr);
    (void)fclose(fp);
    OWN_FREE(lne);
    glutSetCursor(GLUT_CURSOR_INHERIT);
    return (ERRORFREE);
}

/*------------------------------------------------------------------------*/
int anim_read_lig_file(char *lig_file)
{
    int property; /* previously read property */
    GLfloat lm[] = { 0.0, 0.0, 0.0, 0.0 };
    int n = 0, /* number of light sources */
        nr, /* number of light source (0 identifies lighting model) */
        index = 0, /* position in property vector */
        status, /* status of light source (define if GL_TRUE, 
                                     skip otherwise to keep compatibility */
        lightnumber;
    int maxnolights;
    FILE *fp; /* file pointer */

    /* read the filename (if not given) */
    if (strlen(lig_file) == 0)
    {
        fprintf(outfile, "Enter the light-file name: ");
        (void)system("ls *" STDLIG);
        (void)scanf("%s", lig_file);
    }

    /* open file */
    if ((fp = fopen(lig_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", lig_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    glutSetCursor(GLUT_CURSOR_WAIT);

    /* read light-file */
    while (feof(fp) != GL_TRUE)
    {
        if (fscanf(fp, "%d", &nr) == EOF)
        {
            break;
        }
        glGetIntegerv(GL_MAX_LIGHTS, &maxnolights);
        if (maxnolights < 8)
        {
            fprintf(outfile, "not enough lights available %d\n", maxnolights);
            exit(ANIM_ERROR);
        }
        if (nr > 7 || nr < 0)
        {
            fprintf(outfile, "Error, light number out of range: Number =  %d\n", nr);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }
        else
        {
            fprintf(outfile, "reading light number %d\n", nr);
            switch (nr)
            {

            case (0):
                lightnumber = GL_LIGHT0;
                break;

            case (1):
                lightnumber = GL_LIGHT1;
                break;

            case (2):
                lightnumber = GL_LIGHT2;
                break;

            case (3):
                lightnumber = GL_LIGHT3;
                break;

            case (4):
                lightnumber = GL_LIGHT4;
                break;

            case (5):
                lightnumber = GL_LIGHT5;
                break;

            case (6):
                lightnumber = GL_LIGHT6;
                break;

            case (7):
                lightnumber = GL_LIGHT7;
                break;

            } /* of switch */
        } /* of if_else */

        /* read properties of light sources */
        (void)fscanf(fp, "%d", &status);
        fprintf(outfile, "  status %d\n", status);
        do
        {
            if (fscanf(fp, "%d", &property) != 1)
            {
                fprintf(outfile, "... error in reading light file information\n");
                glutSetCursor(GLUT_CURSOR_INHERIT);
                return (ANIM_ERROR);
            }

            fprintf(outfile, "  property %d\n", property);

            switch (property)
            {

            case (GL_AMBIENT):
                if (fscanf(fp, "%f %f %f %f",
                           &lm[0], &lm[1], &lm[2], &lm[3]) != 4)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_AMBIENT, lm);
                break;

            case (GL_DIFFUSE):
                if (fscanf(fp, "%f %f %f %f",
                           &lm[0], &lm[1], &lm[2], &lm[3]) != 4)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_AMBIENT, lm);
                break;

            case (GL_SPECULAR):
                if (fscanf(fp, "%f %f %f %f",
                           &lm[0], &lm[1], &lm[2], &lm[3]) != 4)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_AMBIENT, lm);
                break;

            case (GL_POSITION):
                if (fscanf(fp, "%f %f %f %f",
                           &lm[0], &lm[1], &lm[2], &lm[3]) != 4)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_POSITION, lm);
                break;

            case (GL_SPOT_DIRECTION):
                if (fscanf(fp, "%f %f %f",
                           &lm[0], &lm[1], &lm[2]) != 3)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_SPOT_CUTOFF, lm);
                break;

            case (GL_SPOT_EXPONENT):
                if (fscanf(fp, "%f",
                           &lm[0]) != 1)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_SPOT_EXPONENT, lm);
                break;

            case (GL_SPOT_CUTOFF):
                if (fscanf(fp, "%f",
                           &lm[0]) != 1)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_SPOT_CUTOFF, lm);
                break;

            case (GL_CONSTANT_ATTENUATION):
                if (fscanf(fp, "%f",
                           &lm[0]) != 1)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_CONSTANT_ATTENUATION, lm);
                break;

            case (GL_LINEAR_ATTENUATION):
                if (fscanf(fp, "%f",
                           &lm[0]) != 1)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_LINEAR_ATTENUATION, lm);
                break;

            case (GL_QUADRATIC_ATTENUATION):
                if (fscanf(fp, "%f",
                           &lm[0]) != 1)
                {
                    fprintf(outfile, "..Error in reading light-file information\n");
                    glutSetCursor(GLUT_CURSOR_INHERIT);
                    return (ANIM_ERROR);
                }
                glLightfv(lightnumber, GL_QUADRATIC_ATTENUATION, lm);
                break;

            case ((int)0):
                break;

            default:
                fprintf(outfile, "..Error, wrong property: %i\n", property);
                index--;
                break;

            } /* of switch */

        } while (property != 0); /* end of do */

        if (status == 1)
        {
            glEnable(lightnumber);
        }
        else
        {
            glDisable(lightnumber);
        }

        n++;
    } /* of while (lights) */

    lm[0] = lm[1] = lm[2] = 0.9f;
    lm[3] = 1.0f;
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lm);

    lm[0] = lm[1] = lm[2] = lm[3] = 1.0;
    glMaterialfv(GL_FRONT, GL_SPECULAR, lm);
    lm[0] = 50.0;
    lm[1] = lm[2] = lm[3] = 0.0;
    glMaterialfv(GL_FRONT, GL_SHININESS, lm);

    glEnable(GL_LIGHTING);

    fprintf(outfile, "  %d light sources have been read\n", n);
    (void)fclose(fp);
    glutSetCursor(GLUT_CURSOR_INHERIT);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_set_file(char *comname)
/* read all files with common name */
{
    char *filename = NULL; /* actual filename */

    /* read common name (if not given) */
    if (strlen(comname) == 0)
    {
        fprintf(outfile,
                "Enter the common name of all data files WITHOUT extension: ");
        (void)scanf("%s", comname);
    }

    OWN_CALLOC(filename, char, MAXLENGTH);

    /* read geometric file  */
    strcpy(filename, comname);
    strcat(filename, STDGEO);
    if (anim_read_geo_file(filename) == ANIM_ERROR)
    {
        OWN_FREE(filename);
        /* return(ANIM_ERROR); */
    }
    if (filename == NULL)
        return (ANIM_ERROR);

    /* read stripped file */
    strcpy(filename, comname);
    strcat(filename, STDSTR);

    if (anim_read_str_file(filename) == ANIM_ERROR)
    {
        fprintf(outfile,
                "... no animation possible; read another stripped file\n");
        return (ANIM_ERROR);
    }

    /* read dynamic color file */
    strcpy(filename, comname);
    strcat(filename, STDDYN);
    (void)anim_read_dyncolor_file(filename);

    /* read sensor file */
    strcpy(filename, comname);
    strcat(filename, STDSNS);
    (void)anim_read_sns_file(filename);

    /* read line element file */
    strcpy(filename, comname);
    strcat(filename, STDLIN);
    (void)anim_read_lin_file(filename);

    /* read colormap file */
    strcpy(filename, comname);
    strcat(filename, STDCMP);
    (void)anim_read_cmp_file(filename);

    /* read light file */
    strcpy(filename, comname);
    strcat(filename, STDLIG);
    (void)anim_read_lig_file(filename);

    /* read transformation file */
    strcpy(filename, comname);
    strcat(filename, STDTRANS);
    (void)anim_read_trmat_file(filename);

    /* read elastic geometry file */
    strcpy(filename, comname);
    strcat(filename, STDELGEO);
    (void)anim_read_elgeo_file(filename);

    /* read inventor data file */
    strcpy(filename, comname);
    strcat(filename, STDIV);
    (void)anim_read_iv_file(filename);

    /* read data file */
    strcpy(filename, comname);
    strcat(filename, STDDAT);

    if (anim_read_data_file(filename) != ANIM_ERROR)
    {
        anim_ini_plotter();
        anim_ini_viewmat(plotmat);
    }

    OWN_FREE(filename);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_data_file(char *data_file)
{
    int i, j, col;
    FILE *fp;
    float deltay;

    /* read filename (if not given) */
    if (strlen(data_file) == 0)
    {
        fprintf(outfile, "Enter the  datafile name: ");
        (void)system("ls *" STDDAT);
        (void)scanf("%s", data_file);
    }

    /* open file */
    if ((fp = fopen(data_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", data_file);
        return (ANIM_ERROR);
    }

    glutSetCursor(GLUT_CURSOR_WAIT);
    /* free memory */
    if (dat.first != GL_TRUE)
    {
        for (i = 0; i < dat.ndata; i++)
        {
            OWN_FREE(dat.data[i]);
            OWN_FREE(dat.name[i]);
        }
        OWN_FREE(dat.data);
        OWN_FREE(dat.name);
        OWN_FREE(dat.minx);
        OWN_FREE(dat.miny);
        OWN_FREE(dat.maxx);
        OWN_FREE(dat.maxy);
        OWN_FREE(dat.col);
    }

    /* read file */
    (void)fscanf(fp, "%d%d", &(dat.timesteps), &(dat.ndata));

    if (dat.timesteps != str.timesteps)
    { /* then warnings */
        if ((dat.timesteps > str.timesteps) && (str.timesteps != 0))
        {
            fprintf(outfile,
                    "Warning : number of timesteps in %s\n is greater "
                    "than in animation data\n",
                    data_file);
        }
        else
        {
            fprintf(outfile,
                    "Warning : number of timesteps in %s\n is fewer "
                    "than in animation data\n",
                    data_file);
        }
    }

    /* memory allocation */
    OWN_CALLOC(dat.data, float *, dat.ndata);
    OWN_CALLOC(dat.name, char *, dat.ndata);
    OWN_CALLOC(dat.minx, float, dat.ndata);
    OWN_CALLOC(dat.miny, float, dat.ndata);
    OWN_CALLOC(dat.maxx, float, dat.ndata);
    OWN_CALLOC(dat.maxy, float, dat.ndata);
    OWN_CALLOC(dat.col, int, MAXPLOTS);
    for (j = 0; j < dat.ndata; j++)
    {
        OWN_CALLOC(dat.data[j], float, dat.timesteps);
        OWN_CALLOC(dat.name[j], char, MAXLENGTH);
    }

    /* read the datafile */
    for (j = 0; j < dat.ndata; j++)
    {
        (void)fscanf(fp, "%s%i", dat.name[j], &col);
        dat.col[j] = col;
        if (dat.timesteps > 0)
        {
            (void)fscanf(fp, "%f", &dat.data[j][0]);
            dat.miny[j] = dat.data[j][0];
            dat.maxy[j] = dat.data[j][0];
        }
        for (i = 1; i < dat.timesteps; i++)
        {
            (void)fscanf(fp, "%f", &dat.data[j][i]);
            if (dat.data[j][i] < dat.miny[j])
            {
                dat.miny[j] = dat.data[j][i];
            }
            if (dat.data[j][i] > dat.maxy[j])
            {
                dat.maxy[j] = dat.data[j][i];
            }
        }
        /* scale the x- and y-values to maxy-miny=1*/
        deltay = dat.maxy[j] - dat.miny[j];
        if (deltay != 0)
        {
            for (i = 0; i < dat.timesteps; i++)
            {
#ifndef VRANIM
                dat.data[j][i] = (dat.data[j][i] - dat.miny[j]) / deltay;
#endif
            }
        }
        else
        {
            fprintf(outfile, "Maximaler und minimaler Datenwert sind identisch!\n");
            return (ANIM_ERROR);
        }
    }

    /* completed reading */
    (void)fclose(fp);
    fprintf(outfile, "  %d timesteps of %d datasets have been read\n",
            dat.timesteps, dat.ndata);
    dat.first = GL_FALSE;
    glutSetCursor(GLUT_CURSOR_INHERIT);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_dyncolor_file(char *data_file)
{
    int i, j, number, timesteps, ndata, index;
    float d1, d2, d3, d4;
    FILE *fp;
    char *lne = NULL; /* line of text */

    /* read filename (if not given) */
    if (strlen(data_file) == 0)
    {
        fprintf(outfile, "Enter the  datafile name: ");
        (void)system("ls *" STDDYN);
        (void)scanf("%s", data_file);
    }

    /* open file */
    if ((fp = fopen(data_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n",
                data_file);
        return (ANIM_ERROR);
    }
    glutSetCursor(GLUT_CURSOR_WAIT);

    /* free memory */
    if (dyn.isset != NULL)
    {
        for (i = 0; i < MAXNODYNCOLORS; i++)
        {
            if (dyn.isset[i] == GL_TRUE)
            {
                for (j = 0; j < str.timesteps; j++)
                {
                    OWN_FREE(dyn.rgb[i][j]);
                }
                OWN_FREE(dyn.rgb[i]);
            }
        }
        OWN_FREE(dyn.rgb);
        OWN_FREE(dyn.isset);
    }

    /* read file */
    OWN_CALLOC(lne, char, MAXLENGTH);
    (void)fscanf(fp, "%d%d", &timesteps, &ndata);
    if (timesteps != str.timesteps)
    {
        fprintf(outfile,
                "Warning:  number of timesteps ( %d ) in %s"
                "is greater than in animation data ( %d )\n",
                timesteps, data_file, str.timesteps);
        OWN_EXIT(ANIM_ERROR, "ERROR reading dyn file");
    }

    /* memory allocation (part 1) */
    OWN_CALLOC(dyn.rgb, float **, MAXNODYNCOLORS);
    OWN_CALLOC(dyn.isset, int, MAXNODYNCOLORS);
    for (j = 0; j < MAXNODYNCOLORS; j++)
    {
        dyn.isset[j] = GL_FALSE;
    }

    /* read the dyncolorfile */
    for (j = 0; j < ndata; j++)
    {
        (void)fscanf(fp, "%d", &index);
        if ((index >= MAXCOLORS) && (index < MAXCOLORS + MAXNODYNCOLORS))
        {
            if (dyn.isset[index - MAXCOLORS] == GL_TRUE)
            {
                fprintf(outfile, "WARNING : dynamic color %d already defined,"
                                 "using new definition\n",
                        index);
            }
        }
        else
        {
            fprintf(outfile, "illegal index %d for dynamic color\n", index);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }

        index = index - MAXCOLORS;

        /* memory allocation (part 2) */
        OWN_CALLOC(dyn.rgb[index], float *, str.timesteps);
        for (i = 0; i < str.timesteps; i++)
        {
            OWN_CALLOC(dyn.rgb[index][i], float, (int)4);
        }

        for (i = 0; i < str.timesteps; i++)
        {
            fget_line(lne, fp);
            number = sscanf(lne, "%f%f%f%f", &d1, &d2, &d3, &d4);
            if (number == 3)
            {
                d4 = 1;
            }
            if (d1 > 1 || d2 > 1 || d3 > 1)
            {
                /* 	fprintf(outfile,"please update your dyn colormap file so that all color "  */
                /* 		"values are in the range 0<=val<=1.\n"); */
                /* 	/\* exit(ANIM_ERROR); *\/ */
                /* 	fprintf(outfile," For now all colors are scaled by 1/255, "  */
                /* 		"but in the future this will not work any longer\n"); */
                d1 = d1 / 255;
                d2 = d2 / 255;
                d3 = d3 / 255;
            }
            dyn.rgb[index][i][0] = d1;
            dyn.rgb[index][i][1] = d2;
            dyn.rgb[index][i][2] = d3;
            dyn.rgb[index][i][3] = d4;
        }
        dyn.isset[index] = GL_TRUE;
    }

    /* completed reading */
    (void)fclose(fp);
    fprintf(outfile,
            "  %d timesteps of %d dynamic colors have been read\n",
            str.timesteps, ndata);
    /*   for(j=0;j<MAXNODYNCOLORS;j++){ */
    /*     if(dyn.isset[j]==GL_TRUE){ */
    /*       fprintf(outfile,"dynamic color %d set\n",j); */
    /*     }else{ */
    /*      fprintf(outfile,"dynamic color %d not set\n",j); */
    /*     }       */
    /*   } */
    glutSetCursor(GLUT_CURSOR_INHERIT);
    OWN_FREE(lne);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
int anim_read_elgeo_file(char *elgeo_file)
{
    int i, j, jj, k, tmp;
    char *lne = NULL; /* line of text */
    FILE *fp, *tmpfp;

    /* read filename (if not given) */
    if (strlen(elgeo_file) == 0)
    {
        fprintf(outfile, "Enter the elast. geometric file name:   ");
        (void)system("ls *" STDELGEO);
        (void)scanf("%s", elgeo_file);
    }

    /* open file */
    if ((fp = fopen(elgeo_file, "r")) == NULL)
    {
        fprintf(outfile, "... unable to open the file %s\n", elgeo_file);
        return (ANIM_ERROR);
    }
    glutSetCursor(GLUT_CURSOR_WAIT);
    /* free memory */
    if (elgeo.first != GL_TRUE)
    {
        for (i = 0; i < elgeo.nfiles; i++)
        {
            for (j = 0; j < elgeo.nf[i]; j++)
            {
                OWN_FREE(elgeo.face[i][j]);
            }
            OWN_FREE(elgeo.face[i]);
            for (k = 0; k < elgeo.timesteps; k++)
            {
                OWN_FREE(elgeo.norm[i][k]);
                OWN_FREE(elgeo.vertex[i][k]);
            }
            OWN_FREE(elgeo.norm[i]);
            OWN_FREE(elgeo.vertex[i]);
            OWN_FREE(elgeo.name[i]);
            OWN_FREE(elgeo.npoints[i]);
            OWN_FREE(elgeo.ecolor[i]);
            OWN_FREE(elgeo.fcolor[i]);
        }
        OWN_FREE(elgeo.name);
        OWN_FREE(elgeo.norm);
        OWN_FREE(elgeo.vertex);
        OWN_FREE(elgeo.face);
        OWN_FREE(elgeo.nf);
        OWN_FREE(elgeo.npoints);
        OWN_FREE(elgeo.ecolor);
        OWN_FREE(elgeo.fcolor);
        OWN_FREE(elgeo.nvertices);
        elgeo.first = GL_TRUE;
        elgeo.nfiles = 0;
        elgeo.timesteps = 0;
    }

    /* read geometric information */
    if (fscanf(fp, "%d %d", &elgeo.nfiles, &elgeo.timesteps) == EOF)
    {
        fprintf(outfile,
                "... error while reading the file %s (headerline)\n",
                elgeo_file);
        glutSetCursor(GLUT_CURSOR_INHERIT);
        return (ANIM_ERROR);
    }

    /* read file names */
    OWN_CALLOC(elgeo.name, char *, elgeo.nfiles);
    for (i = 0; i < elgeo.nfiles; i++)
    {
        OWN_CALLOC(elgeo.name[i], char, MAXLENGTH);
        if (fscanf(fp, "%s", elgeo.name[i]) == EOF)
        {
            fprintf(outfile,
                    "... error while reading the file %s (el. part %d)\n",
                    elgeo_file, i + 1);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }
    }
    (void)fclose(fp);

    /* memory allocation (for each elastic part) */
    OWN_CALLOC(elgeo.nf, int, elgeo.nfiles);
    OWN_CALLOC(elgeo.nvertices, int, elgeo.nfiles);
    OWN_CALLOC(elgeo.face, int **, elgeo.nfiles);
    OWN_CALLOC(elgeo.vertex, anim_vector **, elgeo.nfiles);
    OWN_CALLOC(elgeo.norm, anim_vector **, elgeo.nfiles);
    OWN_CALLOC(elgeo.ecolor, int *, elgeo.nfiles);
    OWN_CALLOC(elgeo.fcolor, int *, elgeo.nfiles);
    OWN_CALLOC(elgeo.npoints, int *, elgeo.nfiles);

    OWN_CALLOC(lne, char, MAXLENGTH);

    /* read the graphics-files */
    for (i = 0; i < elgeo.nfiles; i++)
    {
        if ((fp = fopen(elgeo.name[i], "r")) == NULL)
        {
            fprintf(outfile, "... cannot open file %s (%d of %d)\n",
                    elgeo.name[i], i, elgeo.nfiles);
            OWN_FREE(lne);
            return (ANIM_ERROR);
        }
        glutSetCursor(GLUT_CURSOR_WAIT);
        fprintf(outfile, "    opened file %s (%d of %d)\n",
                elgeo.name[i], (i + 1), elgeo.nfiles);

        if (fget_line(lne, fp) == ANIM_ERROR)
        {
            fprintf(outfile, "... error while reading the file %s\n",
                    elgeo.name[i]);
            OWN_FREE(lne);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }
        if (sscanf(lne, "%d %d %d", &(elgeo.nvertices[i]), &(elgeo.nf[i]), &tmp) != 3)
        {
            fprintf(outfile, "... error while reading the file %s\n",
                    elgeo.name[i]);
            OWN_FREE(lne);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }

        if (tmp != elgeo.timesteps)
        {
            fprintf(outfile, "... error while reading the file %s (timesteps not consistent (%d vs. %d)\n",
                    elgeo.name[i], tmp, elgeo.timesteps);
            OWN_FREE(lne);
            glutSetCursor(GLUT_CURSOR_INHERIT);
            return (ANIM_ERROR);
        }

        /* memory allocation (for each timestep) */
        OWN_CALLOC(elgeo.vertex[i], anim_vector *, elgeo.timesteps);
        OWN_CALLOC(elgeo.norm[i], anim_vector *, elgeo.timesteps);
        for (j = 0; j < elgeo.timesteps; j++)
        {
            OWN_CALLOC(elgeo.vertex[i][j], anim_vector, elgeo.nvertices[i]);
            OWN_CALLOC(elgeo.norm[i][j], anim_vector, elgeo.nvertices[i]);
        }

        /* read vertices */
        for (k = 0; k < elgeo.timesteps; k++)
        {
            for (j = 0; j < elgeo.nvertices[i]; j++)
            {
                if (fget_line(lne, fp) == ANIM_ERROR)
                {
                    fprintf(outfile,
                            "... error while reading file %s, vertex %d, timestep %d\n",
                            elgeo.name[i], j, k);
                    return (ANIM_ERROR);
                }
                if (sscanf(lne, "%f %f %f", &elgeo.vertex[i][k][j][0],
                           &elgeo.vertex[i][k][j][1],
                           &elgeo.vertex[i][k][j][2]) != 3)
                {
                    fprintf(outfile,
                            "... error while reading file %s, vertex %d, timestep %d\n",
                            elgeo.name[i], j, k);
                    return (ANIM_ERROR);
                }
            }
        }

/* produce temporary file containing face information without remarks */
#ifdef WIN32
        if ((tmpfp = fopen(_tempnam("c:\\", "vranim"), "w+")) == NULL)
        {
#else
        if ((tmpfp = tmpfile()) == NULL)
        {
#endif
            OWN_EXIT(ANIM_ERROR, "ERROR opening tmpfile");
        }
        while (fget_line(lne, fp) == ERRORFREE)
        {
            fprintf(tmpfp, "%s", lne);
        }
        rewind(tmpfp);
        (void)fclose(fp);

        /* memory allocation (for faces) */
        OWN_CALLOC(elgeo.face[i], int *, elgeo.nf[i]);
        OWN_CALLOC(elgeo.ecolor[i], int, elgeo.nf[i]);
        OWN_CALLOC(elgeo.fcolor[i], int, elgeo.nf[i]);
        OWN_CALLOC(elgeo.npoints[i], int, elgeo.nf[i]);

        /* read faces */
        for (j = 0; j < elgeo.nf[i]; j++)
        {
            if (fscanf(tmpfp, "%d %d %d",
                       &elgeo.npoints[i][j],
                       &elgeo.ecolor[i][j],
                       &elgeo.fcolor[i][j]) != 3)
            {
                fprintf(outfile,
                        "... error while reading file %s header face %d\n",
                        elgeo.name[i], j);
                return (ANIM_ERROR);
            }
            OWN_CALLOC(elgeo.face[i][j], int, elgeo.npoints[i][j]);
            for (jj = 0; jj < elgeo.npoints[i][j]; jj++)
            {
                if (fscanf(tmpfp, "%d", &elgeo.face[i][j][jj]) != 1)
                {
                    fprintf(outfile,
                            "... error while reading the file %s face %d vertex %d\n",
                            elgeo.name[i], j, jj);
                    return (ANIM_ERROR);
                }
                if (elgeo.face[i][j][jj] > elgeo.nvertices[i])
                {
                    fprintf(outfile,
                            "... error while reading the file %s face %d\n",
                            elgeo.name[i], j);
                    fprintf(outfile, "   -> illegal vertex !\n");
                    return (ANIM_ERROR);
                }
            }
        }
        (void)fclose(tmpfp);
    }
    OWN_FREE(lne);

    fprintf(outfile,
            "%d timesteps of %d elastic bodies have been read\n",
            elgeo.timesteps, elgeo.nfiles);
    elgeo.first = GL_FALSE;
    glutSetCursor(GLUT_CURSOR_INHERIT);
    return (ERRORFREE);
}

/* ------------------------------------------------------------------- */
#ifdef VRANIM
/* dummy routine */
void glutSetCursor(int dummy)
{
    dummy = 1;
}
#endif

/* ------------------------------------------------------------------- */
