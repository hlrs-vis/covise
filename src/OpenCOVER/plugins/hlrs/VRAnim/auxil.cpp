/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ------------------------------------------------------------------- 
 *
 *   auxil.c: 
 *
 *     This is part of the program ANIM. It provides some 
 *     subroutines for different purposes.
 * 
 *     Date: Mar 96
 *
 * ------------------------------------------------------------------- */

/* ------------------------------------------------------------------- */
/* Standard includes                                                   */
/* ------------------------------------------------------------------- */
#ifdef _WIN32
#include <windows.h>
#endif
#include <sysdep/opengl.h>

#ifndef VRANIM
#include <GL/glut.h>
#endif

#ifdef HAVE_LIBTIFF
#include <tiffio.h> /* Sam Leffler's libtiff library. */
/* liegt in /usr/local/include/ */
#endif

#include <stdio.h>
#include <stdlib.h>

#ifdef HPPAVERSION
struct timeval
{
    unsigned long tv_sec; /* seconds since Jan. 1, 1970 */
    long tv_usec; /* and microseconds */
};
struct timezone
{
    int tz_minuteswest; /* of UTC */
    int tz_dsttime; /* type of DST correction to apply */
};
#else
#ifndef _MSC_VER
#include <sys/time.h>
#else
#include <sys/timeb.h>
#endif
#endif

#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------- */
/* Own includefiles                                                    */
/* ------------------------------------------------------------------- */
#include "anim.h"

/* ------------------------------------------------------------------- */
/* Prototypes                                                          */
/* ------------------------------------------------------------------- */
int index_to_color_frame(int);
int index_to_color_polygon(int);
double gettime(void);
void calc_lineel(void);
void update_sensors(void);
void delete_sensors(void);
void find_norm(anim_vector *, int *, anim_vector *, int *, int);
float normize(anim_vector);
void minvert(float *, float *);
void transback(float *, float *);
void matmult(anim_vector, float *, anim_vector);
void vcopy(float *, float *);
int fget_line(char *, FILE *);
int strcnt(char *, int);
void save_transmat(char *);
void save_frame(int, int);
void writeileaffile(void);
static void writeileafprefix(FILE *);
static void writeileafpostfix(FILE *);
void output_defines(FILE *file);

#ifdef HAVE_LIBTIFF
void writetiff(char *, char *, int, int, int, int, int);
#endif

#ifndef WIN32
#ifndef SGIVERSION
#ifndef SUNVERSION
/*
extern int gettimeofday       (struct timeval *,struct timezone *);
*/
#include <sys/time.h>
#endif
#endif
#endif

#ifdef VRANIM
/* dummy routine */
int glutGet(int);
#define GLUT_WINDOW_X 0
#define GLUT_WINDOW_Y 0
#define GLUT_WINDOW_WIDTH 0
#define GLUT_WINDOW_HEIGHT 0
#endif

/* ------------------------------------------------------------------- */
/* External defined global variables                                   */
/* ------------------------------------------------------------------- */
extern struct geometry geo;
extern struct animation str;
extern struct sensors sensor;
extern struct dyncolor dyn;
extern struct menuentries menus;
extern struct lineelem lin;

extern FILE *outfile;
extern float colorindex[MAXCOLORS + MAXNODYNCOLORS][4];

/* ------------------------------------------------------------------- */
/* Subroutines                                                         */
/* ------------------------------------------------------------------- */
int index_to_color_frame(int index)
{
    int i;

    i = -index;
    if (i > 90000 && i < 100000)
    {
        i = i - 90000;
        i = i / 100;
    }
    else if (i > 9000000 && i < 10000000)
    {
        i = i - 9000000;
        i = i / 1000;
    }
    else
    {
        fprintf(outfile, "error in index_to_color_frame: index=%d\n", i);
    }
    return (i);
}

/* ------------------------------------------------------------------ */
int index_to_color_polygon(int index)
{
    int i;

    i = -index;
    if (i > 90000 && i < 100000)
    {
        i = i - 90000;
        i = i % 100;
    }
    else if (i > 9000000 && i < 10000000)
    {
        i = i - 9000000;
        i = i % 1000;
    }
    else
    {
        fprintf(outfile, "error in index_to_color_polygon: index=%d\n", i);
    }
    return (i);
}

/* ------------------------------------------------------------------ */
double gettime(void) /* get actual time in seconds */
{
#ifdef _MSC_VER
    struct __timeb64 currentTime;
#if _MSC_VER < 1400
    _ftime64(&currentTime);
#else
    _ftime64_s(&currentTime);
#endif
    return (currentTime.time + (double)currentTime.millitm / 1000.0);
#else
#ifndef SGIVERSION
    double acttimesec; /* actual time in seconds */
    struct timeval acttime; /* actual time in sec. and usec */
    static struct timezone tzp; /* actual timezone
                                            (no definition necessary) */
    (void)gettimeofday(&acttime, &tzp);
    acttimesec = ((double)(acttime.tv_sec))
                 + (((double)(acttime.tv_usec)) / 1000000);
    return (acttimesec);
#endif
#endif
}
/* ------------------------------------------------------------------ */
void calc_lineel(void)
{
    int i, ts, j; /*counter*/
    float l_2 = 0;

    if (lin.nr == 0)
    {
        return;
    }

    /* allocate absolute coordinates of line elements*/
    for (i = 0; i < lin.nr; i++)
    {
        if (lin.pkt1[i] != NULL)
        {
            OWN_FREE(lin.pkt1[i]);
        }
        if (lin.pkt2[i] != NULL)
        {
            OWN_FREE(lin.pkt2[i]);
        }
        OWN_CALLOC(lin.pkt1[i], anim_vector, str.timesteps);
        OWN_CALLOC(lin.pkt2[i], anim_vector, str.timesteps);
        OWN_CALLOC(lin.dir[i], anim_vector, str.timesteps);
        OWN_CALLOC(lin.length[i], float, str.timesteps);
    }

    /*compute absolute coordinates of line elements*/
    for (ts = 0; ts < str.timesteps; ts++)
    {
        for (i = 0; i < lin.nr; i++)
        {
            matmult(lin.pkt1[i][ts], str.a[ts][lin.bod1[i]], lin.coo1[i]);
            matmult(lin.pkt2[i][ts], str.a[ts][lin.bod2[i]], lin.coo2[i]);
            for (j = 0; j < 3; j++)
            {
                lin.dir[i][ts][j] = lin.pkt2[i][ts][j] - lin.pkt1[i][ts][j];
            }
            l_2 = 0;
            for (j = 0; j < 3; j++)
            {
                l_2 = l_2 + lin.dir[i][ts][j] * lin.dir[i][ts][j];
            }
            lin.length[i][ts] = sqrt(l_2); /* pruefen */
        }
    }
}

/* ------------------------------------------------------------------ */
void update_sensors(void)
{
    int i, ts;

    if (sensor.nr == 0)
    {
        return;
    }

    /* allocate sensor trajectories */
    for (i = 0; i < sensor.nr; i++)
    {
        if (sensor.pkt[i] != NULL)
        {
            OWN_FREE(sensor.pkt[i]);
        }
        OWN_CALLOC(sensor.pkt[i], anim_vector, str.timesteps);
    }

    /* compute sensor trajectories */
    for (ts = 0; ts < str.timesteps; ts++)
    {
        for (i = 0; i < sensor.nr; i++)
        {
            matmult(sensor.pkt[i][ts],
                    str.a[ts][sensor.bod[i]],
                    sensor.coo[i]);
        }
    }
}

/* ------------------------------------------------------------------ */
void delete_sensors(void) /* delete sensors and free memory */
{
    int i;

    if (sensor.nr != 0)
    {
        for (i = 0; i < sensor.nr; i++)
        {
            OWN_FREE(sensor.pkt[i]);
        }
        OWN_FREE(sensor.pkt);
        OWN_FREE(sensor.bod);
        OWN_FREE(sensor.col);
        OWN_FREE(sensor.coo);
        sensor.nr = 0;
    }
}

/* ------------------------------------------------------------------- */
void find_norm(anim_vector *vert, int *nface, anim_vector *norm, int *pol, int nvert)
/* Find norm of the surface of given polygon
                 vert[j][k]   coordinate k of vertex j
                 nface[j]     number of faces contacting vertex j
                 norm[j][k]   coordinate k of normal in vertex j     
                 pol[i]       vertex i on given polygon
                 nvert        number of vertices of given poygon
              */
{
    int i, k;
    anim_vector n;

    n[0] = n[1] = n[2] = 0;
    for (i = 0; i < nvert; i++)
    {
        n[0] += (vert[pol[i % nvert] - 1][2] + vert[pol[(i + 1) % nvert] - 1][2]) * (vert[pol[i % nvert] - 1][1] - vert[pol[(i + 1) % nvert] - 1][1]);
        n[1] += (vert[pol[i % nvert] - 1][0] + vert[pol[(i + 1) % nvert] - 1][0]) * (vert[pol[i % nvert] - 1][2] - vert[pol[(i + 1) % nvert] - 1][2]);
        n[2] += (vert[pol[i % nvert] - 1][1] + vert[pol[(i + 1) % nvert] - 1][1]) * (vert[pol[i % nvert] - 1][0] - vert[pol[(i + 1) % nvert] - 1][0]);
    }

    /* normalize the norm */
    if (normize(n) == 0)
    {
        n[0] = 1;
        n[1] = 0;
        n[2] = 0;
    }

    /* update the vertex norms */
    for (i = 0; i < nvert; i++)
    {
        for (k = 0; k < 3; k++)
        {
            norm[pol[i % nvert] - 1][k] = (norm[pol[i % nvert] - 1][k] * nface[pol[i % nvert] - 1] + n[k]) / (nface[pol[i % nvert] - 1] + 1);
        }
        nface[pol[i % nvert] - 1]++;
    }
}

/* ------------------------------------------------------------------- */
float normize(anim_vector v) /* normize vector v and return its length */
{
    float length;

    length = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (length < EPS)
    {
        length = 0;
    }
    else
    {
        length = (float)sqrt((double)length);
        v[0] = v[0] / length;
        v[1] = v[1] / length;
        v[2] = v[2] / length;
    }
    return (length);
}

/* ------------------------------------------------------------------ */
void minvert(float *input, float result[16]) /* invert Matrix
                                inverse rotation done by
                                - transposing rotation matrix
                                - changing translation's direction
                                  (caution: translation is to be rotated!)*/
{
    int i, j;

    for (i = 0; i < 3; i++)
    {
        result[12 + i] = 0;
        for (j = 0; j < 3; j++)
        {
            result[j * 4 + i] = input[i * 4 + j];
            result[12 + i] = result[12 + i] - result[j * 4 + i] * input[12 + j];
        }
    }

    /* add last row */
    result[3] = 0;
    result[7] = 0;
    result[11] = 0;
    result[15] = 1;
}

/* ------------------------------------------------------------------ */
void transback(float *input, float *result)
/* change translations direction */
{
    /* copy identity matrix */
    result[1] = result[2] = result[3] = result[4] = result[6] = result[7] = result[8] = result[9] = result[11] = 0;
    result[0] = result[5] = result[10] = result[15] = 1;

    /* change sign of translation elements */
    result[12] = -input[12];
    result[13] = -input[13];
    result[14] = -input[14];
}

/* ------------------------------------------------------------------ */
void matmult(anim_vector a, float *input, anim_vector b)
/* derives translation and rotation of [3x1]-vector b by 
               matrix input solution is given back in [3x1]-vector a  */
{
    int i, j;

    for (i = 0; i < 3; i++)
    {
        a[i] = input[12 + i];
        for (j = 0; j < 3; j++)
        {
            a[i] = a[i] + (input[j * 4 + i] * b[j]);
        }
    }
}

/* ------------------------------------------------------------------ */
void vcopy(float *input, float *result)
{
    int i;

    for (i = 0; i < 16; i++)
    {
        result[i] = input[i];
    }
}

/* ------------------------------------------------------------------ */
void save_transmat(char *filename)
/* save actual viewing transformation */
{
    FILE *fp;
    int i;
    float *trmat = NULL;

    if ((fp = fopen(filename, "w")) == NULL)
    {
        fprintf(outfile, "error opening file %s,\n", filename);
        fprintf(outfile, "  Transformation matrix not saved\n");
    }

    if (trmat == NULL)
    {
        OWN_CALLOC(trmat, float, 16);
    }

    /* get modelview matrix and store as trmat */
    glGetFloatv(GL_MODELVIEW_MATRIX, trmat);

    for (i = 0; i < 16; i++)
    {
        fprintf(fp, "%f ", trmat[i]);
    }

    OWN_FREE(trmat);
    (void)fclose(fp);
    fprintf(outfile, "  Transformation matrix saved in file %s\n", filename);
}

/* ------------------------------------------------------------------ */
void save_frame(int onoff, int actstep) /* onoff = 0 ... Reset + filename
                                               = 1 ... (Start) save
                                             = 2 ... create MPEG-File */
{
    static char video_file[MAXLENGTH]; /* name of video file */
    char *com_line = NULL; /* command line to perform hardcopy */
#ifdef HAVE_LIBTIFF
    char *fullvideoname = NULL;
    char *convertcommand = NULL;
#endif
    static int copy_nr = 0, /* number of actual hardcopy */
        laststep = -1; /* number of latest copied step */
    int x_ll, y_ll, /* graphics window's position   */
        heigth, width; /* and size                     */
    /* (origin at lower left corner */
    FILE *fp_enc;

    /* If Save because window event, but actual step already saved */
    if ((laststep == actstep) && (onoff == 1))
    {
        return;
    }

    /* If Reset+filename called, Reset is automatically included */
    if (onoff == 0)
    {
        copy_nr = 0;
        laststep = -1;
        video_file[0] = '\0';
        /* read filename for hardcopies */
        fprintf(outfile, "Please enter a filename, where ANIM should\n"
                         "place the frame files\n"
                         " (an increasing number is automatically appended): ");
        (void)scanf("%s", video_file);
        return;
    }

    /* If Save then prepare the hardcopy */
    if (onoff == 1)
    {
        /* Count up the copy number */
        copy_nr++;

        /* Opengl screen counts from left top to right bottom */
        OWN_CALLOC(com_line, char, MAXLENGTH);
        x_ll = glutGet(GLUT_WINDOW_X);
        y_ll = glutGet(GLUT_WINDOW_Y);
        width = glutGet(GLUT_WINDOW_WIDTH);
        heigth = glutGet(GLUT_WINDOW_HEIGHT);

#ifdef SGIVERSION
        /* SGI screen counts from left bottom to right top, (1280x1024) */
        sprintf(com_line, "scrsave %s%i.rgb %i %i %i %i",
                video_file, copy_nr, x_ll, x_ll + width, 1024 - y_ll, 1024 - (y_ll + heigth));

        /* do the hardcopy */
        fprintf(outfile, "writing file %s%i.rgb -> ", video_file, copy_nr);
        fflush(outfile);
        (void)system(com_line);
        fprintf(outfile, "done\n");
        fflush(outfile);

#else
#ifdef HAVE_LIBTIFF
        OWN_CALLOC(fullvideoname, char, MAXLENGTH);
        OWN_CALLOC(convertcommand, char, MAXLENGTH);
        sprintf(fullvideoname, "%s%i.tiff", video_file, copy_nr);

        fprintf(outfile, "writing file %s -> ", fullvideoname);
        fflush(outfile);
        /* do the hardcopy */
        writetiff(fullvideoname, video_file, (int)0, (int)0,
                  width, heigth, COMPRESSION_NONE);
        fprintf(outfile, "done\n");

        /* Lars: try to convert tiff directly to pnm */
        sprintf(convertcommand, "convert %s %s%i.pnm", fullvideoname, video_file, copy_nr);
        (void)system(convertcommand);

        /* for windows tool bmp2avi */
        if (copy_nr >= 100000)
        {
            sprintf(convertcommand, "convert %s %s%6d.bmp", fullvideoname, video_file, copy_nr);
        }
        else if (copy_nr >= 10000)
        {
            sprintf(convertcommand, "convert %s %s0%5d.bmp", fullvideoname, video_file, copy_nr);
        }
        else if (copy_nr >= 1000)
        {
            sprintf(convertcommand, "convert %s %s00%4d.bmp", fullvideoname, video_file, copy_nr);
        }
        else if (copy_nr >= 100)
        {
            sprintf(convertcommand, "convert %s %s000%3d.bmp", fullvideoname, video_file, copy_nr);
        }
        else if (copy_nr >= 10)
        {
            sprintf(convertcommand, "convert %s %s0000%2d.bmp", fullvideoname, video_file, copy_nr);
        }
        else
        {
            sprintf(convertcommand, "convert %s %s00000%1d.bmp", fullvideoname, video_file, copy_nr);
        }
        (void)system(convertcommand);

        fflush(outfile);
        OWN_FREE(fullvideoname);
        OWN_FREE(convertcommand);
#else
        fprintf(outfile, "---> don't HAVE_LIBTIFF\n");
        fprintf(outfile, "tiff-files cannot be created, because\n");
        fprintf(outfile, "libtiff is required!\n");
#endif
#endif

        laststep = actstep;
        OWN_FREE(com_line);

        return;
    }

    /* If create MPEG-File */
    if (onoff == 2)
    {

#ifdef SGIVERSION
        OWN_CALLOC(com_line, char, MAXLENGTH);
        sprintf(com_line, "rgb2mpg %s_out %s %i &\n",
                video_file, video_file, copy_nr);
        fprintf(outfile, "Create MPEG - video file: %s\n", com_line);
        (void)system(com_line);
        fprintf(outfile, "MPEG - video file created\n");

        OWN_FREE(com_line);
        return;
#else
        OWN_CALLOC(com_line, char, MAXLENGTH);
        sprintf(com_line, "%s.enc", video_file);

        fp_enc = fopen(com_line, "w");
        fprintf(fp_enc, "PATTERN          I\n"
                        "OUTPUT           %s.mpg\n"
                        "BASE_FILE_FORMAT PNM\n"
                        "GOP_SIZE         30\n"
                        "SLICES_PER_FRAME 1\n"
                        "PIXEL            HALF\n"
                        "RANGE            10\n"
                        "PSEARCH_ALG      LOGARITHMIC\n"
                        "BSEARCH_ALG      CROSS2\n"
                        "IQSCALE          8\n"
                        "PQSCALE          10\n"
                        "BQSCALE          25\n"
                        "REFERENCE_FRAME  DECODED\n"
                        "PNM_SIZE         500x500\n"
                        /*  !?! followin entry didn't work properly 18/11/02 Lars  */
                        /*  "INPUT_CONVERT    cat * | convert TIFF:- PNM:-\n"      */
                        /*  This worked fine, but PowerPoint wasn't able to read mpeg */
                        /*  "INPUT_CONVERT    tifftopnm *\n"  */
                        "INPUT_CONVERT    *\n"
                        "INPUT_DIR        .\n"
                        "INPUT            \n"
                        "%s*.pnm          [1-%i+1]\n"
                        /* "%s*.tiff   [1-%i+1]\n" */
                        "END_INPUT        \n",
                video_file, video_file, copy_nr);
        fclose(fp_enc);

        sprintf(com_line, "(mpeg_encode %s.enc; "
                          "echo MPEG - video file created;"
                          "xanim %s.mpg;"
                          "echo MPEG - video playing )&\n",
                video_file, video_file);
        (void)system(com_line);

        OWN_FREE(com_line);
        return;
#endif
    }
}

/* ------------------------------------------------------------------ */
int fget_line(char *lne, FILE *fp) /* read line from file
                                               free of remarks        */
{
    char *rempos; /* position of remark     */

    do
    {

        /* read line of text out of file, NULL=FALSE */
        if (fgets(lne, MAXLENGTH, fp) == NULL)
        {
            return (ANIM_ERROR);
        }

        /* ignore remarks and empty lines */
        rempos = strchr(lne, '#');
        if (rempos != NULL)
        {
            *rempos = '\0';
        }
    } while (strlen(lne) == strspn(lne, " \f\n\r\t\v"));

    return (ERRORFREE);
}

/* ------------------------------------------------------------------ */
int strcnt(char *lne, int symbol) /* count number of occurences
                                           of symbol in line 'lne'    */
{
    int counter; /* counter for symbol */

    counter = 0;
    while ((lne = strchr(lne, symbol)) != NULL)
    {
        lne++;
        counter++;
    }
    return (counter);
}

/* ------------------------------------------------------------------ */
void writeileaffile(void)
{
    float a_inv[16]; /* hold inverted matrix */
    int i, j, k, l;
    int polygon_color, frame_color;
    char *filename = NULL;
    FILE *fp;
    float z, zmin, viewingmat[16];
    anim_vector v1, v2, v3;

    fprintf(outfile, "writeileaffile called\n");

    OWN_CALLOC(filename, char, MAXLENGTH);
    fprintf(outfile, "Input ileaf filename: ");
    scanf("%s", filename);
    if ((fp = fopen(filename, "w")) == NULL)
    {
        fprintf(outfile, "error opening file %s,\n", filename);
    }

    writeileafprefix(fp);

    /* get minimum  ---------------------------------------------------- */
    zmin = 0;
    for (j = 0; j < geo.nfiles; j++)
    {

        if (menus.hide_file[j] == GL_TRUE)
            continue;

        /* get matrix */
        glPushMatrix();
        /* fix object by inverting its movements */
        for (i = 0; i < geo.nfiles; i++)
        {
            if (menus.fixed_file[i] == GL_TRUE)
            {
                minvert(str.a[str.act_step][i], a_inv);
                glMultMatrixf(a_inv);
                break;
            }
            /* fix object by inverting its translation */
            if (menus.fixtrans_file[i] == GL_TRUE)
            {
                transback(str.a[str.act_step][i], a_inv);
                glMultMatrixf(a_inv);
                break;
            }
        }
        glMultMatrixf(str.a[str.act_step][j]);
        glGetFloatv(GL_MODELVIEW_MATRIX, viewingmat);
        glPopMatrix();

        for (k = 0; k < geo.nf[j]; k++)
        {

            l = 0;
            while (geo.face[j][k][l] > 0)
            {
                matmult(v1, viewingmat, geo.vertex[j][geo.face[j][k][l] - 1]);
                if (v1[2] < zmin)
                {
                    zmin = v1[2];
                }
                l++;
            }
        }
    }

    /* draw body j ---------------------------------------------------- */
    for (j = 0; j < geo.nfiles; j++)
    {

        /* skip if hidden */
        if (menus.hide_file[j] == GL_TRUE)
            continue;

        /* get matrix */
        glPushMatrix();
        /* fix object by inverting its movements */
        for (i = 0; i < geo.nfiles; i++)
        {
            if (menus.fixed_file[i] == GL_TRUE)
            {
                minvert(str.a[str.act_step][i], a_inv);
                glMultMatrixf(a_inv);
                break;
            }
            /* fix object by inverting its translation */
            if (menus.fixtrans_file[i] == GL_TRUE)
            {
                transback(str.a[str.act_step][i], a_inv);
                glMultMatrixf(a_inv);
                break;
            }
        }
        glMultMatrixf(str.a[str.act_step][j]);
        glGetFloatv(GL_MODELVIEW_MATRIX, viewingmat);
        glPopMatrix();

        fprintf(fp, "(g9,2,0,\n");

        /* draw polygon k */
        for (k = 0; k < geo.nf[j]; k++)
        {

            /* eliminate backfacing polygons */
            matmult(v1, viewingmat, geo.vertex[j][geo.face[j][k][0] - 1]);
            matmult(v2, viewingmat, geo.vertex[j][geo.face[j][k][1] - 1]);
            matmult(v3, viewingmat, geo.vertex[j][geo.face[j][k][2] - 1]);
            if ((v2[0] - v1[0]) * (v3[1] - v2[1]) - (v2[1] - v1[1]) * (v3[0] - v2[0]) >= 0)
            {

                l = 0;
                matmult(v1, viewingmat, geo.vertex[j][geo.face[j][k][0] - 1]);
                z = v1[2];
                while (geo.face[j][k][l] > 0)
                {
                    matmult(v1, viewingmat, geo.vertex[j][geo.face[j][k][l] - 1]);
                    if (v1[2] < z)
                    {
                        z = v1[2];
                    }
                    l++;
                }

                z = z - zmin + 1;

                /* get face and vertices color */
                l = 0;
                while (geo.face[j][k][l] > 0)
                {
                    l++;
                }
                polygon_color = index_to_color_polygon(geo.face[j][k][l]);
                frame_color = index_to_color_frame(geo.face[j][k][l]);

                /* offset the color by 10 to avoid overwriting of default grey colors */
                polygon_color = polygon_color + 10;
                frame_color = frame_color + 10;

                fprintf(fp, "  (p8,%f,0,,5,%d,0\n", z, polygon_color);
                fprintf(fp, "    (g9,%f,0,\n", z);
                fprintf(fp, "      (g9,%f,0,\n", z);

                /* draw face */
                l = 0;
                while (geo.face[j][k][l] > 0)
                {

                    if (geo.face[j][k][l + 1] <= 0)
                    {
                        matmult(v1, viewingmat, geo.vertex[j][geo.face[j][k][l] - 1]);
                        matmult(v2, viewingmat, geo.vertex[j][geo.face[j][k][0] - 1]);
                        fprintf(fp, "        (v7,%f,65536,,%f,%f,%f,%f,%d,0,1,0)\n",
                                z, v1[0], -v1[1], v2[0], -v2[1], frame_color);
                    }
                    else
                    {
                        matmult(v1, viewingmat, geo.vertex[j][geo.face[j][k][l] - 1]);
                        matmult(v2, viewingmat, geo.vertex[j][geo.face[j][k][l + 1] - 1]);
                        fprintf(fp, "        (v7,%f,65536,,%f,%f,%f,%f,%d,0,1,0)\n",
                                z, v1[0], -v1[1], v2[0], -v2[1], frame_color);
                    }
                    l++;
                }
                fprintf(fp, "      )\n");
                fprintf(fp, "    )\n");
                fprintf(fp, "  )\n");
            }
        }
        fprintf(fp, ")\n");
    }

    writeileafpostfix(fp);

    (void)fclose(fp);
    fprintf(outfile, "  Ileaffile %s saved\n", filename);
    OWN_FREE(filename);
}

/* ------------------------------------------------------------------ */
static void writeileafpostfix(FILE *fp)
{
    fprintf(fp, "(E16,0,0,,5,1,1,0.0533333,1,15,0,0,1,0,0,0,1,5,127,"
                "7,0,0,7,0,1,1,0.0666667,0.0666667,6,6,0,0.0666667,6))>\n");
}

/* ------------------------------------------------------------------ */
static void writeileafprefix(FILE *fp)
{
    int col;
    float cyan, margenta, yellow, alpha;

    fprintf(fp, "<!OPS, Version = 8.0>\n");
    fprintf(fp, "<!Class, \"para\">\n");
    fprintf(fp, "<!Page,\n");
    fprintf(fp, "  Height = 8.5 Inches,\n");
    fprintf(fp, "  Width = 11.0 Inches>\n");
    fprintf(fp, "<!Color Definitions,\n");

    /* print color definitions */
    for (col = 0; col < MAXCOLORS; col++)
    {

        cyan = (1 - colorindex[col][0]) * 100;
        margenta = (1 - colorindex[col][1]) * 100;
        yellow = (1 - colorindex[col][2]) * 100;
        alpha = colorindex[col][3];
        fprintf(fp, "C%d = %5.1f, %5.1f, %5.1f, %3.1f,\n",
                (10 + col), cyan, margenta, yellow, alpha);
    }

    fprintf(fp, ">\n");
    fprintf(fp, "\n");
    fprintf(fp, "<\"para\">\n");
    fprintf(fp, "<Frame,\n");
    fprintf(fp, "        Placement = Overlay,\n");
    fprintf(fp, "        Width =     11.0 Inches,\n");
    fprintf(fp, "        Height =    8.5 Inches,\n");
    fprintf(fp, "        Diagram =\n");
    fprintf(fp, "V11,\n");
    fprintf(fp, "(g9,1,0,\n");
}

/* ------------------------------------------------------------------ */
#ifdef HAVE_LIBTIFF
void writetiff(char *filename, char *description,
               int x, int y, int width, int height, int compression)
{
    TIFF *file;
    GLubyte *image = NULL, *p;
    int i;

    file = TIFFOpen(filename, "w");
    if (file == NULL)
    {
        exit(ANIM_ERROR);
    }
    /*pe image = (GLubyte *) malloc(width * height * sizeof(GLubyte) * 3); */
    OWN_CALLOC(image, GLubyte, width * height * 3);

    /* OpenGL's default 4 byte pack alignment would leave extra bytes at the
     end of each image row so that each full row contained a number of bytes
     divisible by 4.  Ie, an RGB row with 3 pixels and 8-bit componets would
     be laid out like "RGBRGBRGBxxx" where the last three "xxx" bytes exist
     just to pad the row out to 12 bytes (12 is divisible by 4). To make sure
     the rows are packed as tight as possible (no row padding), set the pack
     alignment to 1. */
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, image);
    TIFFSetField(file, TIFFTAG_IMAGEWIDTH, (uint32)width);
    TIFFSetField(file, TIFFTAG_IMAGELENGTH, (uint32)height);
    TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(file, TIFFTAG_COMPRESSION, compression);
    TIFFSetField(file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(file, TIFFTAG_ROWSPERSTRIP, 1);
    TIFFSetField(file, TIFFTAG_IMAGEDESCRIPTION, description);
    p = image;
    for (i = height - 1; i >= 0; i--)
    {
        if (TIFFWriteScanline(file, p, i, 0) < 0)
        {
            OWN_FREE(image);
            TIFFClose(file);
            exit(ANIM_ERROR);
        }
        p += width * sizeof(GLubyte) * 3;
    }
    TIFFClose(file);
    OWN_FREE(image);
}
#endif

/* ------------------------------------------------------------------ */
void output_defines(FILE *file)
{
/* formated output */
#define PRINT(WHAT, TYPE) \
    fprintf(file, "\t%-18s%" #TYPE "\n", #WHAT, WHAT)

/* empty line */
#define EMPTY fprintf(file, "\n");

    fprintf(file, "Predefined values:\n\n");

    PRINT(VERSION, s);
    EMPTY;
    PRINT(MAXVERTICES, d);
    PRINT(MAXLENGTH, d);
    PRINT(MAXCOLORS, d);
    PRINT(MAXMATERIALS, d);
    PRINT(MAXNODYNCOLORS, d);
    PRINT(MAXPLOTS, d);
    PRINT(ANIM_ERROR, d);
    PRINT(ERRORFREE, d);
    PRINT(EPS, f);
    EMPTY;
    PRINT(STDGEO, s);
    PRINT(STDSTR, s);
    PRINT(STDSNS, s);
    PRINT(STDCMP, s);
    PRINT(STDLIG, s);
    PRINT(STDELGEO, s);
    PRINT(STDTRANS, s);
    PRINT(STDDAT, s);
    PRINT(STDDYN, s);
    EMPTY;
    PRINT(SPACE_ROT, f);
    PRINT(ZO_CONST, f);
    EMPTY;
    PRINT(O_LEFT, f);
    PRINT(O_RIGHT, f);
    PRINT(O_LOWER, f);
    PRINT(O_UPPER, f);
    PRINT(O_NEAR, f);
    PRINT(O_FAR, f);
    EMPTY;
    PRINT(P_FOVY, f);
    PRINT(P_NEAR, f);
    PRINT(P_FAR, f);
    EMPTY;
    PRINT(TIME_OFFSET_X, f);
    PRINT(TIME_OFFSET_Y, f);
    EMPTY;
    PRINT(PLOTTER_ROTX, f);
    PRINT(PLOTTER_ROTY, f);
    PRINT(PLOTTER_ROTZ, f);
    PRINT(PLOTTER_MOVEX, f);
    PRINT(PLOTTER_MOVEY, f);
    PRINT(PLOTTER_MOVEZ, f);
    PRINT(PLOTTER_SCALE, f);
    EMPTY;

#undef PRINT
#undef EMPTY
}

/* ------------------------------------------------------------------ */
#ifdef VRANIM
/* dummy routine */
int glutGet(int dummy)
{
    (void)dummy;
    return (0);
}
#endif

/* ------------------------------------------------------------------- */
