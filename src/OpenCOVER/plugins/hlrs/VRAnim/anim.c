/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ------------------------------------------------------------------ 
 *
 *  anim.c:      
 *
 *  Date: Mar 95
 *
 * ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Standard includes                                                  */
/* ------------------------------------------------------------------ */
#ifdef WIN32
> #include<windows.h>> #endif
    >> #include<GL / gl.h>> #ifndef WIN32
    > #include<GL / glut.h>> #include<unistd.h>> #endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h> //braucht man das?

/* ------------------------------------------------------------------ */
/* Own includefiles                                                   */
/* ------------------------------------------------------------------ */
#include "anim.h"

    /* ------------------------------------------------------------------ */
    /* Prototypes                                                         */
    /* ------------------------------------------------------------------ */
    /* functions defined in read.c */
    extern int
    anim_read_geo_file(char *);
extern int anim_read_str_file(char *);
extern int anim_read_trmat_file(char *);
extern int anim_read_sns_file(char *);
extern int anim_read_cmp_file(char *);
extern int anim_read_lig_file(char *);
extern int anim_read_set_file(char *);
extern int anim_read_data_file(char *);
extern int anim_read_dyncolor_file(char *);
extern int anim_read_elgeo_file(char *);
extern int anim_read_iv_file(char *);
extern int anim_read_lin_file(char *);

/* functions defined in menu.c */
extern void animCreateMenus(void);
extern void animUpdateMenus(void);
extern void animChangeMiscMenu(int);
extern void animChangeVideoMenu(int);

/* functions defined in plot.c */
extern void anim_ini_plotter(void);
extern void anim_ini_viewmat(float *);
extern void anim_draw_plotter(void);

/* functions defined in auxil.c */
extern int index_to_color_frame(int);
extern int index_to_color_polygon(int);
extern double gettime(void);
extern void minvert(float *, float *);
extern void transback(float *, float *);
extern void matmult(anim_vector, float *, anim_vector);
extern void mult(float *, float *, float *);
extern void vcopy(float *, float *);
extern void save_transmat(char *);
extern void save_frame(int, int);
extern void upk_transformation(void);
extern void writeileaffile(void);
extern void output_defines(FILE *file);
extern void calc_lineel(void);

/* functions defined in anim.c */
int main(int, char **);
static void dorot(float, char, int);
static void dotrans(float, char, int);
static void doscale(float, int);
static void animate(void);
static void calc_stride(void);
static void initialize(int, char **);
static void exit_anim(void);

void callback_display(void);
void animCallbackMenu(int);
void callback_mouse(int, int, int, int);
void callback_mousemotion(int, int);
void callback_keyboard(unsigned char, int, int);
void callback_spaceballtrans(int, int, int);
void callback_spaceballrot(int, int, int);
void callback_visibility(int);
void advanceSceneOneFrame(void);
void reset_timestep(void);
void draw_coordinate_systems(void);
void ballcolor(float *color, float fcolor);

/* ------------------------------------------------------------------ */
/* Definition of global variables                                     */
/* ------------------------------------------------------------------ */
/* geometry data */
struct geometry geo = { GL_TRUE, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                        NULL, NULL, 0, NULL, NULL };

/* inventor files for VR */
struct ivfiles iv = { 0, NULL };

/* animation data */
struct animation str = { GL_TRUE, 0, 1, 0.0, 0.0, 0, GL_FALSE, NULL, NULL, NULL, NULL };

/* sensor data */
struct sensors sensor = { 0, NULL, NULL, NULL, NULL };

/*line element data*/
struct lineelem lin = { 0, NULL, NULL, NULL, NULL };

/* plotter's appearence */
struct plotter plo = { 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL };

/* plotter's information */
struct plotdata dat = { GL_TRUE, 0, 1, 0.0, 0, 0, NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL };

/* elastic geometry data */
struct elgeometry elgeo = { GL_TRUE, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };

/* colors */
float colorindex[MAXCOLORS + MAXNODYNCOLORS][4];

/* dynamic color information */
struct dyncolor dyn = { NULL, NULL };

struct menuentries menus = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             NULL, NULL, NULL };

struct flags flag = {
    GL_FALSE, /* oitl_animation */
    GL_FALSE, /* simula */
    0, /* video */
    GL_FALSE, /* leftmbut */
    GL_FALSE /* middlembut */
};

struct modes mode = {
    ANIM_OFF, /* anim */
    ZOOM, /* mmb */
    SHADE_OFF, /* shade */
    SHADE_WIRE, /* shade_el */
    GL_TRUE, /* toggle: rigid */
    1, /* selected plotter */
    MVGEO, /* move geometry */
    GL_TRUE, /* toggle: display timestep */
    GL_FALSE, /* toggle: show coordinate sytems */
    1.0 /* scaling coordinate system */
};

FILE *outfile = NULL;

/* plotter viewing transformation */
float plotmat[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

/* mouse position */
static struct mousepos mpos = { 0, 0 }, oldmpos = { 0, 0 };

/* aspect ratio of graphics window */
static float scrnaspect;

/* graphics window size (in pixels) */
static int g_width, g_height;

/* graphics window size (in pixels) */
static int g_width, g_height;

/* graphics window borders in world coordinates */
static float left, right;
static float upper, lower;

/* ------------------------------------------------------------------ */
/* Subroutines                                                        */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    /* initialize main program */
    initialize(argc, argv);

    /* everything else is event driven */
    glutMainLoop();

    return (ERRORFREE);
}

/* ------------------------------------------------------------------ */
/* update scene                                                       */
/* ------------------------------------------------------------------ */
static void dorot(float angle, char axis, int mode) /* rotate model */
{
    float *trmat = NULL;

    /* is allocated once and never free'd */
    if (trmat == NULL)
    {
        OWN_CALLOC(trmat, float, 16);
    }

    if (mode == MVGEO)
    {
        /* get modelview matrix and store as trmat */
        glGetFloatv(GL_MODELVIEW_MATRIX, trmat);
    }
    else if (mode == MVPLOTTER)
    {
        glPushMatrix();
    }

    /* this quite compliceted procedure allows to rotate 
     around screen axis and not around body axis */
    glLoadIdentity();

    glTranslatef(-trmat[13], -trmat[14], -trmat[15]);

    if (axis == 'x')
    {
        glRotatef(angle, (float)1, (float)0, (float)0);
    }
    else if (axis == 'y')
    {
        glRotatef(angle, (float)0, (float)1, (float)0);
    }
    else if (axis == 'z')
    {
        glRotatef(angle, (float)0, (float)0, (float)1);
    }

    glTranslatef(trmat[13], trmat[14], trmat[15]);

    if (mode == MVGEO)
    {
        glMultMatrixf(trmat);
    }
    else if (mode == MVPLOTTER)
    {
        glMultMatrixf(plotmat);
        glGetFloatv(GL_MODELVIEW_MATRIX, plotmat);
        glPopMatrix();
    }
}

/* ------------------------------------------------------------------ */
static void dotrans(float dist, char dir, int mode) /* translate model */
{
    float *trmat = NULL;

    /* is allocated once and never free'd */
    if (trmat == NULL)
    {
        OWN_CALLOC(trmat, float, 16);
    }

    if (mode == MVGEO)
    {
        /* get modelview matrix and store as trmat */
        glGetFloatv(GL_MODELVIEW_MATRIX, trmat);
    }
    else if (mode == MVPLOTTER)
    {
        glPushMatrix();
    }

    glLoadIdentity();

    if (dir == 'x')
    {
        glTranslatef((right - left) * dist / (float)g_width, (float)0, (float)0);
    }
    else if (dir == 'y')
    {
        glTranslatef((float)0, (upper - lower) * dist / (float)g_height, (float)0);
    }
    else if (dir == 'z')
    {
        glTranslatef((float)0, (float)0, (right - left) * dist / (float)g_width);
    }

    if (mode == MVGEO)
    {
        glMultMatrixf(trmat);
    }
    else if (mode == MVPLOTTER)
    {
        glMultMatrixf(plotmat);
        glGetFloatv(GL_MODELVIEW_MATRIX, plotmat);
        glPopMatrix();
    }
}

/* ------------------------------------------------------------------ */
static void doscale(float scaling, int mode) /* perform zooming */
{
    if (mode == MVPLOTTER)
    {
        glPushMatrix();
        glLoadIdentity();
    }

    if (scaling > 0)
    {
        scaling = 1.0 - scaling / ZO_CONST;
    }
    else
    {
        scaling = ZO_CONST / (ZO_CONST + scaling);
    }
    glScalef(scaling, scaling, scaling);

    if (mode == MVPLOTTER)
    {
        glMultMatrixf(plotmat);
        glGetFloatv(GL_MODELVIEW_MATRIX, plotmat);
        glPopMatrix();
    }
}

/* ------------------------------------------------------------------ */
static void animate(void) /* execute animation step */
{
    float a_inv[16]; /* hold inverted matrix */
    float *colvec; /* color vector of actual color */
    float color[4]; /* explicit color vector */
    int col; /* color index of polygon/frame */
    int i, j, jj, k;
    static char *lne = NULL;
    static GLUquadricObj *qball = NULL;

    glPushMatrix();

    /* fix object by inverting its movements */
    for (j = 0; j < geo.nfiles; j++)
    {
        if (menus.fixed_file[j] == GL_TRUE)
        {
            minvert(str.a[str.act_step][j], a_inv);
            glMultMatrixf(a_inv);
            break;
        }

        /* fix object by inverting its translation */
        if (menus.fixtrans_file[j] == GL_TRUE)
        {
            transback(str.a[str.act_step][j], a_inv);
            glMultMatrixf(a_inv);
            break;
        }
    }

    /* draw body j ---------------------------------------------------- */
    for (j = 0; j < geo.nfiles; j++)
    {

        /* skip if hidden */
        if (menus.hide_file[j] == GL_TRUE)
            continue;

        glPushMatrix();
        glMultMatrixf(str.a[str.act_step][j]);

        /* draw polygon k */
        for (k = 0; k < geo.nf[j]; k++)
        {

            /* draw faces, if not in wire frame mode */
            if (geo.shading[j] != 0)
            {
                if (mode.shade == SHADE_OFF || mode.shade == SHADE_FLAT || mode.shade == SHADE_GOR)
                {

                    if (mode.shade == SHADE_OFF)
                    {

                        /* set drawing color for face */
                        col = geo.fcolor[j][k];
                        if (col >= MAXCOLORS)
                        {
                            glColor3fv(dyn.rgb[col - MAXCOLORS][str.act_step]);
                        }
                        else
                        {
                            glColor3fv(colorindex[col]);
                        }
                    }
                    else
                    {

                        /* define material, if shading is activated */
                        /* define color in reference material */
                        col = geo.fcolor[j][k];
                        if (col >= MAXCOLORS)
                        {
                            colvec = dyn.rgb[col - MAXCOLORS][str.act_step];
                        }
                        else
                        {
                            colvec = colorindex[col];
                        }
                        colvec[3] = 1.0;
                        glMaterialfv(GL_FRONT, GL_AMBIENT, colvec);
                        glMaterialfv(GL_FRONT, GL_DIFFUSE, colvec);
                    }

                    /* draw face */
                    glBegin(GL_POLYGON);
                    for (jj = 0; jj < geo.npoints[j][k]; jj++)
                    {
                        if (mode.shade != SHADE_OFF)
                        {
                            glNormal3fv(geo.norm[j][geo.face[j][k][jj] - 1]);
                        }
                        glVertex3fv(geo.vertex[j][geo.face[j][k][jj] - 1]);
                    }
                    glEnd();
                }
            }

            /* draw frame lines if shading is inactivated */
            if (mode.shade == SHADE_WIRE || mode.shade == SHADE_OFF)
            {

                /* set drawing color for frame line */
                col = geo.ecolor[j][k];
                if (col >= MAXCOLORS)
                {
                    glColor3fv(dyn.rgb[col - MAXCOLORS][str.act_step]);
                }
                else
                {
                    glColor3fv(colorindex[col]);
                }

                /* draw frame line */
                glBegin(GL_LINE_LOOP);
                for (jj = 0; jj < geo.npoints[j][k]; jj++)
                {
                    glVertex3fv(geo.vertex[j][geo.face[j][k][jj] - 1]);
                }
                glEnd();
            }
        }
        if (mode.coord_show_toggle == GL_TRUE)
        {
            draw_coordinate_systems();
        }
        glPopMatrix();
    }

    /* draw balls ---------------------------------------------- */
    if (geo.nballs != 0)
    {
        if (qball == NULL)
        {
            qball = gluNewQuadric();
        }

        for (j = 0; j < geo.nballs; j++)
        {

            glPushMatrix();
            glMultMatrixf(str.a_ball[str.act_step][j]);

            /* draw ball */
            col = geo.ballscolor[j];
            if (col < 0)
            {
                ballcolor(color, str.balldyncolor[str.act_step][j]);
                glColor3fv(color);
            }
            else
            {
                glColor3fv(colorindex[col]);
            }
            gluSphere(qball, geo.ballsradius[j], 8, 8);
            if (mode.coord_show_toggle == GL_TRUE)
            {
                draw_coordinate_systems();
            }
            glPopMatrix();
        }
    }

    /* draw elastic body j ---------------------------------------------- */
    for (j = 0; j < elgeo.nfiles; j++)
    {

        /* skip if hidden */
        if (menus.hide_file[j + geo.nfiles] == GL_TRUE)
            continue;

        /* draw polygon k */
        for (k = 0; k < elgeo.nf[j]; k++)
        {

            /* draw faces, if not in wire frame mode */
            if (mode.shade_el == SHADE_OFF || mode.shade_el == SHADE_FLAT || mode.shade_el == SHADE_GOR)
            {

                if (mode.shade_el == SHADE_OFF)
                {

                    /* set drawing color for face */
                    col = elgeo.fcolor[j][k];
                    if (col >= MAXCOLORS)
                    {
                        glColor3fv(dyn.rgb[col - MAXCOLORS][str.act_step]);
                    }
                    else
                    {
                        glColor3fv(colorindex[col]);
                    }
                }
                else
                {

                    /* define material, if shading is activated */
                    /* define color in reference material */
                    col = elgeo.fcolor[j][k];
                    if (col >= MAXCOLORS)
                    {
                        colvec = dyn.rgb[col - MAXCOLORS][str.act_step];
                    }
                    else
                    {
                        colvec = colorindex[col];
                    }
                    colvec[3] = 1.0;
                    glMaterialfv(GL_FRONT, GL_AMBIENT, colvec);
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, colvec);
                }

                /* draw face */
                glBegin(GL_POLYGON);
                for (jj = 0; jj < elgeo.npoints[j][k]; jj++)
                {
                    if (mode.shade != SHADE_OFF)
                    {
                        /*  glNormal3fv(elgeo.norm[j][str.act_step][elgeo.face[j][k][jj]-1]); */
                    }
                    glVertex3fv(elgeo.vertex[j][str.act_step][elgeo.face[j][k][jj] - 1]);
                }
                glEnd();
            }
        }

        /* draw frame lines if shading is inactivated */
        if (mode.shade_el == SHADE_WIRE || mode.shade_el == SHADE_OFF)
        {

            /* set drawing color for frame line */
            col = elgeo.ecolor[j][k];
            if (col >= MAXCOLORS)
            {
                glColor3fv(dyn.rgb[col - MAXCOLORS][str.act_step]);
            }
            else
            {
                glColor3fv(colorindex[col]);
            }

            /* draw frame line */
            glBegin(GL_LINE_LOOP);
            for (jj = 0; jj < elgeo.npoints[j][k]; jj++)
            {
                glVertex3fv(elgeo.vertex[j][str.act_step][elgeo.face[j][k][jj] - 1]);
            }
            glEnd();
        }
    }

    /* draw sensors ----------------------------------------------------- */
    glPushMatrix();
    for (j = 0; j < sensor.nr; j++)
    {
        glColor3fv(colorindex[sensor.col[j]]);
        glBegin(GL_LINE_STRIP);
        for (i = 0; i < str.act_step; i++)
        {
            glVertex3fv(sensor.pkt[j][i]);
        }
        glEnd();
    }
    glPopMatrix();

    /* print actual time ------------------------------------------------ */
    if (mode.displaytimestep == GL_TRUE)
    {
        color[0] = 1 - colorindex[0][0]; /* textcolor (inverse of background) */
        color[1] = 1 - colorindex[0][1];
        color[2] = 1 - colorindex[0][2];
        glColor3fv(color);

        /* do not draw the time string if multiple images are drawn in one */
        if (str.multimg == GL_FALSE)
        {
            /* determine actual time string */
            if (lne == NULL)
            {
                OWN_CALLOC(lne, char, MAXLENGTH);
            }
            (void)sprintf(lne, "time: %.3f (%d)", str.act_step * str.dt, str.act_step);

            /* position and print text (left lower corner) */
            glPushMatrix();
            glLoadIdentity();
            glRasterPos2f(scrnaspect * (O_LEFT + TIME_OFFSET_X * (O_RIGHT - (O_LEFT))),
                          O_LOWER + TIME_OFFSET_Y * (O_UPPER - (O_LOWER)));
            for (j = 0; lne[j] != '\0'; j++)
            {
                /*        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,(int)lne[j]); */
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, (int)lne[j]);
            }
            glPopMatrix();
        }
    }

    /* draw plotter ----------------------------------------------------- */
    if (dat.ndata > 0)
    {

        /* write plotter label (use inverted background color) */
        glColor3fv(color);
        (void)sprintf(lne, "plotter dataset %d: %s",
                      mode.plotsel, dat.name[mode.plotsel - 1]);
        glPushMatrix();
        glLoadIdentity();
        glRasterPos2f(scrnaspect * (O_LEFT + TIME_OFFSET_X * (O_RIGHT - (O_LEFT))),
                      O_LOWER + 4 * TIME_OFFSET_Y * (O_UPPER - (O_LOWER)));
        for (j = 0; lne[j] != '\0'; j++)
        {
            /*        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,(int)lne[j]); */
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, (int)lne[j]);
        }
        glPopMatrix();

        glPushMatrix();
        glLoadMatrixf(plotmat);

        /* select side of plotter cube */
        switch (mode.plotsel)
        {
        case 1:
            break;
        case 2:
            glRotatef((float)90, (float)1, (float)0, (float)0);
            break;
        case 3:
            glRotatef((float)180, (float)1, (float)0, (float)0);
            break;
        case 4:
            glRotatef((float)-90, (float)1, (float)0, (float)0);
            break;
        case 5:
            glRotatef((float)-90, (float)0, (float)1, (float)0);
            break;
        case 6:
            glRotatef((float)90, (float)0, (float)1, (float)0);
            break;
        }

        /* draw plotter cube, axis and plot data */
        anim_draw_plotter();

        glPopMatrix();
    }
    glPopMatrix();
}

/* ------------------------------------------------------------------ */
static void calc_stride(void)
/* calculate stride for realtime animation */
{
    int act_step, i; /* keep actual step */
    double start, finish, /* start and finish time of timestep */
        lapsed, /* time needed to build one scene */
        max_time = 0; /* max. time used to build one timestep */

    fprintf(outfile,
            "stride for realtime animation is being calculated\n"
            "based on 20 timesteps. Please wait... \n");

    act_step = str.act_step; /* save actual step */

    for (i = 1; i <= 5; i++)
    { /* do five 'warm up' loops */
        animate();
        str.act_step++;
        if ((str.act_step >= str.timesteps) || (str.act_step < 0))
        {
            reset_timestep();
        }
    }

    for (i = 1; i <= 25; i++)
    { /* measure drawing time in 20 loops */
        start = gettime();
        animate();
        str.act_step++;
        if ((str.act_step >= str.timesteps) || (str.act_step < 0))
        {
            reset_timestep();
        }
        finish = gettime();
        lapsed = finish - start;
        if (lapsed > max_time)
        {
            max_time = lapsed;
        }
    }

    /* calculate stride and time interval */
    if (max_time > str.dt)
    {
        str.stride = ((int)(max_time / str.dt)) + 1;
        str.timeint = str.dt * str.stride;
        if (str.stride >= str.timesteps)
        {
            fprintf(outfile, "Warning: stride would be too big (=%d)\n"
                             " -> no Realtime is possible (stride is set to one)\n"
                             "    (store more steps or accelerate the animation)\n",
                    str.stride);
            str.stride = 1;
            str.timeint = 0;
        }
    }
    else
    {
        str.stride = 1;
        str.timeint = str.dt * str.stride;
    }
    str.act_step = act_step;

    fprintf(outfile, "given or measured values:\n"
                     "  dt                  : %f seconds\n"
                     "  approx. drawing time: %f seconds\n"
                     "to run the animation in realtime following values are set:\n"
                     "  timeint             : %f seconds\n"
                     "  stride              : %d\n",
            str.dt, max_time, str.timeint, str.stride);
}

/* ------------------------------------------------------------------ */
/* initialize main program                                            */
/* ------------------------------------------------------------------ */
/* extern definitions (only for initialize) */
extern int optind;
extern char *optarg;

static void initialize(int argc, char **argv)
{
    int option; /* command line option */
    int s_width, s_height; /* screen size in pixels */
    char *optstring = NULL; /* define command line options */
    float winwidth = WINWIDTH, winheight = WINHEIGHT,
          winxpos = WINXPOS, winypos = WINYPOS;
    int gdtmp;
    int i, ii, iii, counter;

    /*   GLfloat mat_ambient[] = {0.7, 0.2, 0.2, 1.0}, */
    /* 	  mat_diffuse[] = {0.8, 0.8, 0.8, 1.0}, */
    /* 	  mat_specular[] = {1.0, 1.0, 1.0, 1.0}; */
    GLfloat mat_shininess[] = { 0.5 },
            /* light_ambient[] = {0.1, 0.1, 0.1, 1.0}; */
        /* heller: */
        /* mat_shininess[] = {7.0}, */
        light_ambient[] = { 0.8, 0.8, 0.8, 1.0 };

    if (outfile == NULL)
    {
        outfile = stdout;
    }

    /* 0 is the background color (default black) */
    /* other colors */
    counter = 0;
    for (i = 0; i < 4; i++)
    {
        for (ii = 0; ii < 4; ii++)
        {
            for (iii = 0; iii < 4; iii++)
            {
                colorindex[counter][0] = i * 0.3333;
                colorindex[counter][1] = ii * 0.3333;
                colorindex[counter][2] = iii * 0.3333;
                colorindex[counter][3] = 1;
                counter++;
            }
        }
    }

    /* dynamic colors (default red) */
    for (i = 0; i < MAXNODYNCOLORS; i++)
    {
        colorindex[MAXCOLORS + i][0] = 1;
        colorindex[MAXCOLORS + i][1] = 0;
        colorindex[MAXCOLORS + i][2] = 0;
        colorindex[MAXCOLORS + i][3] = 1;
    }

    glutInitWindowSize((int)(1000 * WINWIDTH), (int)(1000 * WINHEIGHT));
    glutInit(&argc, argv);

    OWN_CALLOC(optstring, char, 32);
    (void)strcpy(optstring, "hsg:k:c:d:i:jv:w:x:y:mt");

    fprintf(outfile, "\n\n\n********************************************\n");
    fprintf(outfile, "***                                      ***\n");
    fprintf(outfile, "***                 ANIM                 ***\n");
    fprintf(outfile, "***                ------                ***\n");
    fprintf(outfile, "***                                      ***\n");
    fprintf(outfile, "***  Version : " VERSION "                      ***\n");
    fprintf(outfile, "***                                      ***\n");
    fprintf(outfile, "***  by P.Eberhard                       ***\n");
    fprintf(outfile, "***     R.Sonthi     (      - V 1.0)     ***\n");
    fprintf(outfile, "***     S.Tumback    (V 1.0 - V 1.4)     ***\n");
    fprintf(outfile, "***     U.Blum       (V 1.4 - V 2.0)     ***\n");
    fprintf(outfile, "***     M.Spanninger (V 1.4 - V 2.0)     ***\n");
    fprintf(outfile, "***     S.Tumback    (V 2.1 - V 2.2)     ***\n");
    fprintf(outfile, "***     H.Claus      (V 3.0 - V 3.1)     ***\n");
    fprintf(outfile, "***     S.Mueller    (V 3.0        )     ***\n");
    fprintf(outfile, "***     F.Lippold    (V 3.1        )     ***\n");
    fprintf(outfile, "***     L.Kuebler    (V 3.3e -V 3.5)     ***\n");
    fprintf(outfile, "***     F.Fleissner  (V 3.6        )     ***\n");
    fprintf(outfile, "***                                      ***\n");
#ifdef CYGWIN
    fprintf(outfile, "***  ( 2button mouse -> left = middle )  ***\n");
#endif
    fprintf(outfile, "********************************************\n\n\n");

    /* test for options (several loops are performed to ensure a certain order) */

    /* test for option -h */
    while ((option = getopt(argc, argv, optstring)) != EOF)
    {
        switch (option)
        {
        case 'h':
            fprintf(outfile, "usage: anim [options]\n");
            fprintf(outfile, "-h:  command-line-options are shown\n");
            fprintf(outfile, "-m:  draw multiple images in one frame\n");
            fprintf(outfile, "-g filename:  geometric file will be loaded\n");
            fprintf(outfile, "-k filename:  stripped file will be loaded\n");
            fprintf(outfile, "-c filename:  colormap will be loaded\n");
            fprintf(outfile, "-d filename:  set of files will be loaded\n");
            fprintf(outfile, "-i stride:    set stride for animation\n");
            fprintf(outfile, "-j:           output predefined values\n");
            fprintf(outfile, "-t:           switch off timestep display\n");
            fprintf(outfile, "-v height:    height of window (default %f)\n",
                    (float)WINHEIGHT);
            fprintf(outfile, "-w width:     width of window (default %f)\n",
                    (float)WINWIDTH);
            fprintf(outfile, "-x posx:      position of left corners of window"
                             " (default %f)\n",
                    WINXPOS);
            fprintf(outfile, "-y posy:      position of lower corners of window"
                             " (default %f)\n",
                    WINYPOS);
            fprintf(outfile, "\n");
            exit(ERRORFREE);
        }
    }
    /* reset function getopt */
    optind = 1;

    /* test for program modes */
    while ((option = getopt(argc, argv, optstring)) != EOF)
    {
        switch (option)
        {

        case 'i':
            /* read stride from option's argument */
            if (sscanf(optarg, " %d", &(str.stride)) != 1)
            {
                fprintf(outfile, "...error in reading option -i arguments\n");
            }
            else
            {
                fprintf(outfile, "Option -i %d: using stride %d\n",
                        str.stride, str.stride);
            }
            break;

        case 'j':
            /* output define macro values */
            output_defines(outfile);
            break;

        case 'm':
            /* draw multiple images in one frame */
            str.multimg = GL_TRUE;
            fprintf(outfile, "Option -m: draw multiple images in one frame\n");
            break;

        case 't':
            /* switch off display of timesteps */
            mode.displaytimestep = GL_FALSE;
            fprintf(outfile, "Option -t: switch off timestep display (toggle with key '1')\n");
            break;

        case 'v':
            if (sscanf(optarg, " %f", &(winheight)) != 1)
            {
                fprintf(outfile, "...error in reading option -v arguments\n");
                winheight = WINHEIGHT;
                fprintf(outfile, "Option -v: (warning) using %f\n", winheight);
            }
            else
            {
                if (winheight <= 0 || winheight > 1)
                {
                    winheight = WINHEIGHT;
                    fprintf(outfile, "Option -v: (warning) using %f "
                                     "(using default value)\n",
                            winheight);
                }
            }
            break;

        case 'w':
            if (sscanf(optarg, " %f", &(winwidth)) != 1)
            {
                fprintf(outfile, "...error in reading option -w arguments\n");
                winwidth = WINWIDTH;
                fprintf(outfile, "Option -w: (warning) using %f\n", winwidth);
            }
            else
            {
                if (winwidth <= 0 || winwidth > 1)
                {
                    winwidth = WINWIDTH;
                    fprintf(outfile, "Option -w: (warning) using %f "
                                     "(using default value)\n",
                            winwidth);
                }
            }
            break;

        case 'x':
            if (sscanf(optarg, " %f", &(winxpos)) != 1)
            {
                fprintf(outfile, "...error in reading option -w arguments\n");
                winxpos = WINXPOS;
                fprintf(outfile, "Option -x: (warning) using %f\n", winxpos);
            }
            else
            {
                if (winxpos < 0 || winxpos > 1)
                {
                    winxpos = WINXPOS;
                    fprintf(outfile, "Option -x: (warning) using %f "
                                     "(using default value)\n",
                            winxpos);
                }
            }
            break;

        case 'y':
            if (sscanf(optarg, " %f", &(winypos)) != 1)
            {
                fprintf(outfile, "...error in reading option -w arguments\n");
                winypos = WINYPOS;
                fprintf(outfile, "Option -y: (warning) using  %f\n", winypos);
            }
            else
            {
                if (winypos < 0 || winypos > 1)
                {
                    winypos = WINYPOS;
                    fprintf(outfile, "Option -y: (warning) using %f "
                                     "(using default value)\n",
                            winypos);
                }
            }
            break;
        }
    }

    /* reset function getopt */
    optind = 1;

    glGetIntegerv(GL_DOUBLEBUFFER, &gdtmp);
    if (gdtmp == 0)
    {
        fprintf(outfile, "... %s won't work correctly on this machine,\n"
                         "... because double buffering is not supported.\n",
                argv[0]);
    }

    if (winypos + winheight > 1)
    {
        winypos = 0;
        fprintf(outfile, "Incompatible values for winypos and winheight\n"
                         " setting winypos=0\n");
    }
    if (winxpos + winwidth > 1)
    {
        winxpos = 0;
        fprintf(outfile, "Incompatible values for winxpos and winwidth\n"
                         " setting winxpos=0\n");
    }

    s_width = glutGet(GLUT_SCREEN_WIDTH);
    s_height = glutGet(GLUT_SCREEN_HEIGHT);

    /* correction required because of screen size */
    scrnaspect = winwidth / winheight * (float)s_width / (float)s_height;

    glutInitWindowSize(s_width * winwidth, s_height * winheight);
    /* GL uses lower left corner, OpenGL upper left corner! */
    glutInitWindowPosition(s_width * winxpos, s_height * (1 - winypos - winheight));

    /* open graphics window */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow(WINTITLE);
    if (s_width * winwidth == 1 && s_height * winheight == 1)
    {
        glutFullScreen();
    }

    glutDisplayFunc(callback_display);
    glutIdleFunc(advanceSceneOneFrame);
    glutKeyboardFunc(callback_keyboard);
    glutMouseFunc(callback_mouse);
    glutMotionFunc(callback_mousemotion);
    glutSpaceballMotionFunc(callback_spaceballtrans);
    glutSpaceballRotateFunc(callback_spaceballrot);
    glutVisibilityFunc(callback_visibility);

    /* enable Z buffer (if available) */
    glGetIntegerv(GL_DEPTH_BITS, &gdtmp);
    if (gdtmp > 0)
    {
        glEnable(GL_DEPTH_TEST);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
        fprintf(outfile, "... %s won't work correctly on this machine,\n",
                argv[0]);
        fprintf(outfile, "... because the zbuffer is not supported.\n");
        fprintf(outfile, "... You can use the program anyway, but the\n");
        fprintf(outfile, "... hidden surface algorithm will not work!\n\n");
    }

    /* enable elimination of backfacing polygons */
    glEnable(GL_CULL_FACE);

    /* set default shademodel */
    glShadeModel(GL_FLAT);

    /* set default material */
    /*   glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient); */
    /*   glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse); */
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    /*   glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular); */
    glEnable(GL_FRONT);

    /* enable lighting */
    glEnable(GL_LIGHT0);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_ambient);
    glDisable(GL_LIGHTING);

    /* perform automatic normalization */
    glEnable(GL_NORMALIZE);

    /* set initial viewing mode */
    left = O_LEFT * scrnaspect;
    right = O_RIGHT * scrnaspect;
    upper = O_UPPER;
    lower = O_LOWER;
    glMatrixMode(GL_PROJECTION);
    glOrtho(left, right, lower, upper, O_NEAR, O_FAR);
    glMatrixMode(GL_MODELVIEW);

    /* definition of real materials */
    /*     for(i=1;i<=MAXMATERIALS;i++) */
    /*     glMaterialf(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE, * mat[i-1]); */

    /* definition of default light source */
    /*     glMaterialf(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, * NULL); */

    /* enable lighting */
    /*   glLightModelfv(GL_LIGHT_MODEL_AMBIENT,light_ambient); */
    /*   glLightfv(GL_LIGHT0,GL_AMBIENT,light_ambient); */
    /*   glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse); */
    /*   glEnable(GL_LIGHT0); */
    /*   glEnable(GL_LIGHTING); */

    /* determine graphics window size and its ratio to screen size */
    g_width = glutGet(GLUT_WINDOW_WIDTH);
    g_height = glutGet(GLUT_WINDOW_HEIGHT);

    /* create menus */
    animCreateMenus();

    /* read files according to options */
    while ((option = getopt(argc, argv, optstring)) != EOF)
    {
        switch (option)
        {
        case 'g': /* read geometric file */
            fprintf(outfile, "Option -g %s: geometric data from %s\n\n",
                    optarg, optarg);
            anim_read_geo_file(optarg);
            break;

        case 'c': /* read colormap file */
            fprintf(outfile, "Option -c %s: color data from %s\n\n",
                    optarg, optarg);
            (void)anim_read_cmp_file(optarg);
            break;

        case 'd': /* read set of files */
            fprintf(outfile, "Option -d %s: load file set %s\n\n",
                    optarg, optarg);
            (void)anim_read_set_file(optarg);
            break;
        }
    }
    optind = 1;

    while ((option = getopt(argc, argv, optstring)) != EOF)
    {
        switch (option)
        {
        case 'k': /* read stripped file */
            if (geo.first == GL_TRUE)
            {
                fprintf(outfile, "Option -k %s: animation data can not be read\n"
                                 "              without geometric data\n",
                        optarg);
            }
            else
            {
                (void)anim_read_str_file(optarg);
                fprintf(outfile, "Option -k %s: animation data from %s\n",
                        optarg, optarg);
            }
            break;
        }
    }
    OWN_FREE(optstring);

    /* update menus */
    animUpdateMenus();

    /* clear screen and Z buffer */
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glClearColor(colorindex[0][0], colorindex[0][1],
                 colorindex[0][2], (float)1);

    glutSwapBuffers();
}

/* ------------------------------------------------------------------ */
/* exit animation program                                             */
/* ------------------------------------------------------------------ */
static void exit_anim(void)
{
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glutDestroyWindow(glutGetWindow());

    exit(EXIT_SUCCESS);
}

/* ------------------------------------------------------------------ */
void callback_display(void)
{
    static double start; /* start time of animation step */
    double finish, /* finish time of animation step */
        lapsed; /* lapsed time between start and finish */

    if (str.multimg == GL_FALSE)
    {
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glClearColor(colorindex[0][0], colorindex[0][1],
                     colorindex[0][2], (float)1);
        glDepthFunc(GL_LEQUAL);
    }
    else
    {
        /* hier kann man das loeschen ausschalten, falls mehrere
       Bilder in  eines gezeichnet werden sollen, z.B.:  */
        if (str.act_step == 0)
        {
            glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
            glClearColor(colorindex[0][0], colorindex[0][1],
                         colorindex[0][2], (float)1);
            /* damit erst ab Schritt 100 gezeichnet wird: */
            /* str.act_step=100; */
        }
        else
        {
            glClearDepth(0.9);
            glClear(GL_DEPTH_BUFFER_BIT);
            glDepthFunc(GL_LEQUAL);
        }
    }

    /* if geo and str file are read */
    if (str.timesteps != 0)
    {

        /* get start time */
        start = gettime();

        /* build scene in invisible buffer */
        animate();

        /* time synchronisation */
        if ((str.timeint > 0) && (mode.anim == ANIM_AUTO))
        {
            finish = gettime(); /* get finish time */
            lapsed = finish - start; /* calculate drawing time */
            if (lapsed > str.timeint)
            {
                fprintf(outfile,
                        "WARNING: Drawing time longer than realtime interval!\n"
                        "...drawing time:      %f seconds\n"
                        "...realtime interval: %f seconds\n",
                        lapsed, str.timeint);
            }
            else
            {
                do
                { /* wait until realtime passed */
                    finish = gettime();
                    lapsed = finish - start;
                } while (lapsed < str.timeint);
            }
            /* new start time */
            start = finish;
        }

        /* grep frame hardcopy and save it */
        if ((mode.anim != ANIM_OFF) && (flag.video == 1))
        {
            /* Save: onoff = 1 */
            save_frame((int)1, str.act_step);
        }
    }

    glutSwapBuffers();
}

/* ------------------------------------------------------------------ */
void animCallbackMenu(int value)
{
    char *name = NULL;
    int tmp;

    OWN_CALLOC(name, char, MAXLENGTH);
    switch (value)
    {

    case ANIM_EXIT:
        exit_anim();
        break;

    case INPUT_GEO:
        mode.anim = ANIM_OFF;
        name[0] = '\0';
        (void)anim_read_geo_file(name);
        animUpdateMenus();
        break;

    case INPUT_STR:
        mode.anim = ANIM_OFF;
        name[0] = '\0';
        tmp = str.timesteps;
        (void)anim_read_str_file(name);
        if (tmp == 0)
        {
            /* we only have to update the menus when no animation data existed before */
            animUpdateMenus();
        }
        break;

    case INPUT_SNS:
        name[0] = '\0';
        (void)anim_read_sns_file(name);
        break;

    case INPUT_CMP:
        name[0] = '\0';
        (void)anim_read_cmp_file(name);
        break;

    case INPUT_LIG:
        name[0] = '\0';
        (void)anim_read_lig_file(name);
        break;

    case INPUT_SET:
        mode.anim = ANIM_OFF;
        name[0] = '\0';
        (void)anim_read_set_file(name);
        animUpdateMenus();
        break;

    case INPUT_DAT:
        name[0] = '\0';
        if (anim_read_data_file(name) != ERROR)
        {
            anim_ini_plotter();
            anim_ini_viewmat(plotmat);
            animUpdateMenus();
        }
        break;

    case INPUT_TRMAT:
        name[0] = '\0';
        if (anim_read_trmat_file(name) == ERROR)
        {
            fprintf(outfile, "Warning: error reading transformation matrix\n");
        }
        break;

    case INPUT_DYNCOL:
        name[0] = '\0';
        if (anim_read_dyncolor_file(name) == ERROR)
        {
            fprintf(outfile, "Warning: error reading dynamic colors\n");
        }
        break;

    case INPUT_ELGEO:
        name[0] = '\0';
        (void)anim_read_elgeo_file(name);
        break;

    case ANIM_AUTO:
        mode.anim = ANIM_AUTO;
        break;

    case ANIM_OFF:
        mode.anim = ANIM_OFF;
        break;

    case ANIM_STEP:
        mode.anim = ANIM_STEP;
        break;

    case ANIM_RESET:
        reset_timestep();
        mode.anim = ANIM_OFF;
        break;

    case INPUT_STRIDE:
        fprintf(outfile, "old stride:       %d\nenter new stride: ",
                str.stride);
        (void)scanf("%d", &str.stride);
        break;

    case INPUT_INT:
        fprintf(outfile, "old time interval between timesteps:   "
                         "%f second\nenter new time intervall (in seconds): ",
                str.timeint);
        (void)scanf("%f", &str.timeint);
        break;

    case CALC_STRIDE:
        calc_stride();
        break;

    case ZOOM:
    case TRANSLATE:
    case ROTX:
    case ROTY:
    case ROTZ:
        mode.mmb = value;
        break;

    case PERSPECTIVE:
        upper = (float)tan((double)(M_PI * P_FOVY / 360)) * (P_NEAR + P_FAR) / 2;
        lower = -upper;
        left = upper * scrnaspect;
        right = -left;
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(P_FOVY, scrnaspect, P_NEAR, P_FAR);
        gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        glMatrixMode(GL_MODELVIEW);
        break;

    case ORTHOGRAPHIC:
        left = O_LEFT * scrnaspect;
        right = O_RIGHT * scrnaspect;
        upper = O_UPPER;
        lower = O_LOWER;
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(left, right, lower, upper, O_NEAR, O_FAR);
        glMatrixMode(GL_MODELVIEW);
        break;

    case SHADE_TOGGLE: /* toggle shade mode flex/rigid*/
        if (mode.shade_toggle_rigid == GL_TRUE)
        {
            mode.shade_toggle_rigid = GL_FALSE;
        }
        else
        {
            mode.shade_toggle_rigid = GL_TRUE;
        }
        break;

    case SHADE_OFF:
    case SHADE_WIRE:
        if (mode.shade_toggle_rigid == GL_TRUE)
        {
            mode.shade = value;
        }
        else
        {
            mode.shade_el = value;
        }
        glShadeModel(GL_FLAT);
        glDisable(GL_LIGHTING);
        glDisable(GL_FRONT);
        break;

    case SHADE_FLAT:
        if (mode.shade_toggle_rigid == GL_TRUE)
        {
            mode.shade = value;
        }
        else
        {
            mode.shade_el = value;
        }
        glShadeModel(GL_FLAT);
        glEnable(GL_LIGHTING);
        glEnable(GL_FRONT);
        break;

    case SHADE_GOR:
        if (mode.shade_toggle_rigid == GL_TRUE)
        {
            mode.shade = value;
        }
        else
        {
            mode.shade_el = value;
        }
        glShadeModel(GL_SMOOTH);
        glEnable(GL_LIGHTING);
        glEnable(GL_FRONT);
        break;

    case SAVE_TRMAT:
        save_transmat("viewing.mat");
        break;

    case RESET:
        glLoadIdentity();
        anim_ini_viewmat(plotmat);
        break;

    case MVGEO:
    case MVPLOTTER:
        mode.move = value;
        animChangeMiscMenu(value);
        break;

    case WRITEILEAF:
        writeileaffile();
        break;

    case VIDEO_ON:
        animChangeVideoMenu(value);
        flag.video = 1;
        str.timeint = 0;
        /* Reset and Read filename: onoff=0 */
        save_frame((int)0, str.act_step);
        fprintf(outfile,
                "ANIM grabs and saves display frames, therefore\n"
                "the time syncronization is switched off\n");
        break;

    case VIDEO_OFF:
        flag.video = 2;
        /* Save window of last timestep: onoff=1 */
        save_frame((int)1, str.act_step);
        fprintf(outfile, "video mode finished\n");
        animChangeVideoMenu(value);
        break;

    case VIDEO_CREATE:
        flag.video = 2;
        /* Create video file: onoff=2 */
        save_frame((int)2, str.act_step);
        break;

    case MULT_ON:
        str.multimg = GL_TRUE;
        animChangeMiscMenu(value);
        break;

    case MULT_OFF:
        str.multimg = GL_FALSE;
        animChangeMiscMenu(value);
        break;

    case INPUT_COORD_SCALING:
        fprintf(outfile, "old scaling factor for coordinate systems:  %f\nenter new scaling factor: ",
                mode.coord_scaling);
        (void)scanf("%lf", &mode.coord_scaling);
        glutPostRedisplay();
        break;

    case TOGGLE_COORD:
        if (mode.coord_show_toggle == GL_TRUE)
        {
            mode.coord_show_toggle = GL_FALSE;
        }
        else
        {
            mode.coord_show_toggle = GL_TRUE;
        }
        animUpdateMenus();
        break;
    }

    OWN_FREE(name);

    glutPostRedisplay();
}

/* ------------------------------------------------------------------ */
void callback_mouse(int button, int state, int xpos, int ypos)
{
    switch (button)
    {

    case (GLUT_LEFT_BUTTON):
        if (state == GLUT_UP)
        {
            glutSetCursor(GLUT_CURSOR_INHERIT);
#ifdef CYGWIN
            flag.midmbut = GL_FALSE;
#else
            flag.leftmbut = GL_FALSE;
#endif
        }
        else
        {
            glutSetCursor(GLUT_CURSOR_INFO);
#ifdef CYGWIN
            flag.midmbut = GL_TRUE;
#else
            flag.leftmbut = GL_TRUE;
#endif
            mpos.x = xpos;
            mpos.y = ypos;
        }
        break;

    case (GLUT_MIDDLE_BUTTON):
        if (state == GLUT_UP)
        {
            glutSetCursor(GLUT_CURSOR_INHERIT);
            flag.midmbut = GL_FALSE;
        }
        else
        {
            glutSetCursor(GLUT_CURSOR_INFO);
            flag.midmbut = GL_TRUE;
            mpos.x = xpos;
            mpos.y = ypos;
        }
        break;

    case (GLUT_RIGHT_BUTTON):
        if (state == GLUT_UP)
        {
            fprintf(outfile, "RM up\n");
        }
        else
        {
            fprintf(outfile, "RM down\n");
        }
        break;
    }
}

/* ------------------------------------------------------------------ */
void callback_mousemotion(int xpos, int ypos)
{
    if (flag.leftmbut == GL_TRUE || flag.midmbut == GL_TRUE)
    {
        oldmpos.x = mpos.x;
        mpos.x = xpos;
        oldmpos.y = mpos.y;
        mpos.y = ypos;
    }
    if (flag.leftmbut == GL_TRUE)
    {
        dorot((float)(mpos.x - oldmpos.x), 'y', mode.move);
        dorot((float)(mpos.y - oldmpos.y), 'x', mode.move);
        glutPostRedisplay();
    }
    if (flag.midmbut == GL_TRUE)
    {
        if (mode.mmb == ROTX)
        {
            dorot((float)(-mpos.y + oldmpos.y), 'x', mode.move);
        }
        else if (mode.mmb == ROTY)
        {
            dorot((float)(-mpos.y + oldmpos.y), 'y', mode.move);
        }
        else if (mode.mmb == ROTZ)
        {
            dorot((float)(-mpos.y + oldmpos.y), 'z', mode.move);
        }
        else if (mode.mmb == TRANSLATE)
        {
            dotrans((float)(mpos.x - oldmpos.x), 'x', mode.move);
            dotrans((float)(-mpos.y + oldmpos.y), 'y', mode.move);
        }
        else
        {
            doscale((float)(mpos.y - oldmpos.y), mode.move);
        }
        glutPostRedisplay();
    }
}

/* ------------------------------------------------------------------ */
void callback_spaceballtrans(int xpos, int ypos, int zpos)
{
    dotrans((float)xpos, 'x', mode.move);
    dotrans((float)ypos, 'y', mode.move);
    dotrans((float)zpos, 'z', mode.move);

    glutPostRedisplay();
}

/* ------------------------------------------------------------------ */
void callback_spaceballrot(int xpos, int ypos, int zpos)
{
    dorot((float)(SPACE_ROT * xpos), 'x', mode.move);
    dorot((float)(SPACE_ROT * ypos), 'y', mode.move);
    dorot((float)(SPACE_ROT * zpos), 'z', mode.move);

    glutPostRedisplay();
}

/* ------------------------------------------------------------------ */
void callback_keyboard(unsigned char key, int xpos, int ypos)
{
    switch (key)
    {

    case (27): /* esc key */
    case ('q'):
        exit_anim();
        break;

    case ('r'): /* reset animation */
        reset_timestep();
        mode.anim = ANIM_OFF;
        glutPostRedisplay();
        break;

    case ('s'):
        mode.anim = ANIM_OFF;
        break;

    case ('g'):
        mode.anim = ANIM_AUTO;
        break;

    case ('i'):
        str.stride *= -1;
        break;

    case ('d'):
        mode.anim = ANIM_STEP;
        break;

    case ('t'):
        mode.mmb = TRANSLATE;
        break;

    case ('m'): /* toggle shade mode flex/rigid*/
        if (mode.shade_toggle_rigid == GL_TRUE)
        {
            mode.shade_toggle_rigid = GL_FALSE;
        }
        else
        {
            mode.shade_toggle_rigid = GL_TRUE;
        }
        break;

    case ('w'):
        if (mode.shade_toggle_rigid == GL_TRUE)
        {
            mode.shade = SHADE_WIRE;
        }
        else
        {
            mode.shade_el = SHADE_WIRE;
        }
        glShadeModel(GL_FLAT);
        glDisable(GL_LIGHTING);
        glDisable(GL_FRONT);
        glutPostRedisplay();
        break;

    case ('1'):
        if (mode.displaytimestep == GL_TRUE)
        {
            mode.displaytimestep = GL_FALSE;
        }
        else
        {
            mode.displaytimestep = GL_TRUE;
        }
        glutPostRedisplay();
        break;

    case ('o'):
        if (mode.shade_toggle_rigid == GL_TRUE)
        {
            mode.shade = SHADE_OFF;
        }
        else
        {
            mode.shade_el = SHADE_OFF;
        }
        glShadeModel(GL_FLAT);
        glDisable(GL_LIGHTING);
        glDisable(GL_FRONT);
        glutPostRedisplay();
        break;

    case ('k'):
        if (mode.coord_show_toggle == GL_TRUE)
        {
            mode.coord_show_toggle = GL_FALSE;
        }
        else
        {
            mode.coord_show_toggle = GL_TRUE;
        }
        animUpdateMenus();
        glutPostRedisplay();
        break;

    case ('z'):
        mode.mmb = ZOOM;
        break;

    case (32): /* space key */
        if (mode.anim == ANIM_STEP)
        {
            str.act_step = str.act_step + str.stride;
            if ((str.act_step >= str.timesteps) || (str.act_step < 0))
            {
                reset_timestep();
            }
            /* no step is omitted but bad interactivity: callback_display(); */
            /* some steps are simply ignored because several Redisplay
         events are combined to one: glutPostRedisplay() */
            glutPostRedisplay();
        }
        break;
    }
}

/* ------------------------------------------------------------------ */
void callback_visibility(int status)
{
    if (status == GLUT_VISIBLE)
    {
        glutIdleFunc(advanceSceneOneFrame);
    }
    else
    {
        glutIdleFunc(NULL);
    }
}

/* ------------------------------------------------------------------ */
void advanceSceneOneFrame(void)
{
    /* go to next timestep */

    if ((flag.leftmbut == GL_FALSE)
        && (flag.midmbut == GL_FALSE)
        && (mode.anim == ANIM_AUTO))
    {
        str.act_step = str.act_step + str.stride;
        if ((str.act_step >= str.timesteps) || (str.act_step < 0))
        {
            if (flag.video == 1)
            {
                mode.anim = ANIM_OFF;
                str.act_step = str.timesteps - 1;
            }
            else
            {
                reset_timestep();
            }
        }
        glutPostRedisplay();
    }
}

/* ------------------------------------------------------------------ */
void reset_timestep(void)
{
    /* reset animation */
    if (str.stride < 0)
    {
        str.act_step = str.timesteps - 1;
    }
    else
    {
        str.act_step = 0;
    }
}

/* ------------------------------------------------------------------ */
void ballcolor(float *color, float fcolor)
{
    color[0] = fcolor;
    color[1] = 0;
    color[2] = 1 - fcolor;
}

/* ------------------------------------------------------------------ */
void draw_coordinate_systems(void)
{

#define COORD_LENGTH 0.1
#define COORD_WIDTH 0.05 * COORD_LENGTH

/* taking advantage of then symmetry of the bodies */
#define ROT_XY(X, Y, Z) \
    {                   \
        Z, X, Y         \
    } /* rotation around the xy-bisector */
#define ROT_XZ(X, Y, Z) \
    {                   \
        Y, Z, X         \
    } /* rotation around the xz-bisector */
#define ROT_XX(X, Y, Z) \
    {                   \
        X, Y, Z         \
    } /* dummy (no rotation) */

/* define first body and create the others by rotation */
#define INITIALIZE_CUBE(ROT)                                                           \
    ROT(0.5 * COORD_WIDTH, -0.5 * COORD_WIDTH, -0.5 * COORD_WIDTH),                    \
        ROT(0.5 * COORD_WIDTH + COORD_LENGTH, -0.5 * COORD_WIDTH, -0.5 * COORD_WIDTH), \
        ROT(0.5 * COORD_WIDTH + COORD_LENGTH, 0.5 * COORD_WIDTH, -0.5 * COORD_WIDTH),  \
        ROT(0.5 * COORD_WIDTH, 0.5 * COORD_WIDTH, -0.5 * COORD_WIDTH),                 \
        ROT(0.5 * COORD_WIDTH, -0.5 * COORD_WIDTH, 0.5 * COORD_WIDTH),                 \
        ROT(0.5 * COORD_WIDTH + COORD_LENGTH, -0.5 * COORD_WIDTH, 0.5 * COORD_WIDTH),  \
        ROT(0.5 * COORD_WIDTH + COORD_LENGTH, 0.5 * COORD_WIDTH, 0.5 * COORD_WIDTH),   \
        ROT(0.5 * COORD_WIDTH, 0.5 * COORD_WIDTH, 0.5 * COORD_WIDTH)

    /* vertex coordinates of the rods of the coordinate arrows */
    double coordRodX[8][3] = {
        INITIALIZE_CUBE(ROT_XX)
    };
    double coordRodY[8][3] = {
        INITIALIZE_CUBE(ROT_XY)
    };
    double coordRodZ[8][3] = {
        INITIALIZE_CUBE(ROT_XZ)
    };

#define INITIALIZE_PYRAMID(ROT)                                                       \
    ROT(0.5 * COORD_WIDTH + COORD_LENGTH, -1.5 * COORD_WIDTH, -1.5 * COORD_WIDTH),    \
        ROT(0.5 * COORD_WIDTH + COORD_LENGTH, 1.5 * COORD_WIDTH, -1.5 * COORD_WIDTH), \
        ROT(0.5 * COORD_WIDTH + COORD_LENGTH, 1.5 * COORD_WIDTH, 1.5 * COORD_WIDTH),  \
        ROT(0.5 * COORD_WIDTH + COORD_LENGTH, -1.5 * COORD_WIDTH, 1.5 * COORD_WIDTH), \
        ROT(4.5 * COORD_WIDTH + COORD_LENGTH, 0, 0)

    /* vertex coordinates of the pikes of the coordinate arrows */
    double coordPikeX[5][3] = {
        INITIALIZE_PYRAMID(ROT_XX)
    };
    double coordPikeY[5][3] = {
        INITIALIZE_PYRAMID(ROT_XY)
    };
    double coordPikeZ[5][3] = {
        INITIALIZE_PYRAMID(ROT_XZ)
    };

/* makro routines for automization of drawing primitives */
#define QUAD_FACE(BODY, A, B, C, D)                 \
    glVertex3f(BODY[A][0], BODY[A][1], BODY[A][2]); \
    glVertex3f(BODY[B][0], BODY[B][1], BODY[B][2]); \
    glVertex3f(BODY[C][0], BODY[C][1], BODY[C][2]); \
    glVertex3f(BODY[D][0], BODY[D][1], BODY[D][2]);

#define TRI_FACE(BODY, A, B, C)                     \
    glVertex3f(BODY[A][0], BODY[A][1], BODY[A][2]); \
    glVertex3f(BODY[B][0], BODY[B][1], BODY[B][2]); \
    glVertex3f(BODY[C][0], BODY[C][1], BODY[C][2]);

#define LINE(BODY, A, B)                            \
    glVertex3f(BODY[A][0], BODY[A][1], BODY[A][2]); \
    glVertex3f(BODY[B][0], BODY[B][1], BODY[B][2]);

#define DRAW_CUBE(BODY)         \
    glBegin(GL_QUADS);          \
    QUAD_FACE(BODY, 4, 5, 1, 0) \
    QUAD_FACE(BODY, 5, 6, 2, 1) \
    QUAD_FACE(BODY, 6, 7, 3, 2) \
    QUAD_FACE(BODY, 7, 4, 0, 3) \
    QUAD_FACE(BODY, 7, 6, 5, 4) \
    QUAD_FACE(BODY, 0, 1, 2, 3) \
    glEnd();

#define DRAW_CUBE_CONTOUR(BODY) \
    glBegin(GL_LINES);          \
    LINE(BODY, 0, 1)            \
    LINE(BODY, 1, 2)            \
    LINE(BODY, 2, 3)            \
    LINE(BODY, 3, 0)            \
    LINE(BODY, 4, 5)            \
    LINE(BODY, 5, 6)            \
    LINE(BODY, 6, 7)            \
    LINE(BODY, 7, 4)            \
    LINE(BODY, 0, 4)            \
    LINE(BODY, 1, 5)            \
    LINE(BODY, 2, 6)            \
    LINE(BODY, 3, 7)            \
    glEnd();

#define DRAW_PIKE(BODY)                  \
    glBegin(GL_QUADS);                   \
    QUAD_FACE(BODY, 0, 1, 2, 3) glEnd(); \
    glBegin(GL_TRIANGLES);               \
    TRI_FACE(BODY, 0, 4, 1)              \
    TRI_FACE(BODY, 4, 2, 1)              \
    TRI_FACE(BODY, 4, 3, 2)              \
    TRI_FACE(BODY, 3, 4, 0)              \
    glEnd();

#define DRAW_PIKE_CONTOUR(BODY) \
    glBegin(GL_LINES);          \
    LINE(BODY, 0, 4)            \
    LINE(BODY, 3, 4)            \
    LINE(BODY, 2, 4)            \
    LINE(BODY, 1, 4)            \
    LINE(BODY, 0, 3)            \
    LINE(BODY, 3, 2)            \
    LINE(BODY, 2, 1)            \
    LINE(BODY, 1, 0)            \
    glEnd();

    glPushMatrix();
    glScalef(mode.coord_scaling, mode.coord_scaling, mode.coord_scaling);

    /* draw red x-coordinate arrow */
    glColor3f(1, 0, 0); /* red */
    DRAW_CUBE(coordRodX)
    DRAW_PIKE(coordPikeX)
    glColor3f(0.8, 0, 0); /* dark red contour */
    DRAW_CUBE_CONTOUR(coordRodX)
    DRAW_PIKE_CONTOUR(coordPikeX)

    /* draw green x-coordinate arrow */
    glColor3f(0, 1, 0); /* green */
    DRAW_CUBE(coordRodY)
    DRAW_PIKE(coordPikeY)
    glColor3f(0, 0.8, 0); /* dark green contour */
    DRAW_CUBE_CONTOUR(coordRodY)
    DRAW_PIKE_CONTOUR(coordPikeY)

    /* draw blue x-coordinate arrow */
    glColor3f(0, 0, 1); /* blue */
    DRAW_CUBE(coordRodZ)
    DRAW_PIKE(coordPikeZ)
    glColor3f(0, 0, 0.8); /* dark blue contour */
    DRAW_CUBE_CONTOUR(coordRodZ)
    DRAW_PIKE_CONTOUR(coordPikeZ)

    glPopMatrix();
}

/* ------------------------------------------------------------------ */
