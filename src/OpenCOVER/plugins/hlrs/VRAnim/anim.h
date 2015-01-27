/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -------------------------------------------------------------------
   anim.h Headerfile for ANIM
   ------------------------------------------------------------------- */

/* version number */
#define VERSION "4.0" /* 1.0          original from Nikravesh                    \
                         1.1            - " -  with bug fixes                    \
                         1.2 01.05.94 before PVM-coupling and                    \
                                      stereo-modus                               \
                         1.3 17.06.94 including stereo-modus                     \
                         1.4 09.08.94 including PVM                              \
                                      with ANIM                                  \
                         1.41 3.11.94 merge mit Martin Spanninger                \
                         2.0  10.5.95 merge mit Uli Blum                         \
                                               (el. Koerper, ...                 \
                         2.1  11.95   peb: diverse Aenderungen                   \
                         2.2  3.96/10.97 include GUI, Anpassung auf              \
                                               alle drei Plattformen             \
                         2.3          Uebergang auf OpenGL                       \
                         3.0          Anpassen des Plotters, ileaf, ...          \
                         3.5 01/04   peb: Uebernahme der Erlanger Erweiterungen  \
                         3.6 05/05   ff: Info-Routine zur Ausgabe der define-    \
                              Werte hinzugefuegt                                 \
                         3.7 7/05 PEB: Kugeln                                    \
                         3.8 11/05 PEB VR-Anpassungen                            \
                         4.0 03/10 work with opencover (based on OpenSceneGraph) \
                                   and the tabletUI                              \
                      */

/* ------------------------------------------------------------------- */
/* macro definitions */
#define WINTITLE "ANIM " VERSION " by P.Eberhard, ..."
/* title of graphics window */
#define MAXVERTICES 1000 /* maximal number of vertices on one face */
#define MAXLENGTH 400 /* maximal length of strings */
#define MAXCOLORS 64 /* maximal number of colors */
#define MAXMATERIALS 21 /* maximal number of materials */
#define MAXNODYNCOLORS 20 /* maximal number of colorscales */
#define MAXPLOTS 20 /* maximal number of plots */
#define ANIM_ERROR EXIT_FAILURE /* error return value */
#define ERRORFREE EXIT_SUCCESS /* return value when on error occured */
#define EPS 1.5e-15 /* minimal floating point value */
#define STDGEO ".geoall" /* standard extension for geometric file */
#define STDSTR ".str" /* standard extension for stripped file */
#define STDSNS ".sensor" /* standard extension for sensor file */
#define STDCMP ".cmp" /* standard extension for colormap file */
#define STDLIG ".lig" /* standard extension for lights file */
#define STDELGEO ".elgeoall"
#define STDTRANS ".mat"
#define STDDAT ".dat"
#define STDDYN ".dyn"
#define STDIV ".ivall"
#define STDLIN ".lin"

/* -------------------------------------------------------------------- */
/* Values returned from popup menu */
#define INPUT_GEO 0
#define INPUT_STR 1
#define INPUT_SNS 2
#define INPUT_CMP 3
#define INPUT_SET 4
#define INPUT_LIG 5
#define INPUT_DAT 6
#define INPUT_ELGEO 7
#define INPUT_TRMAT 8
#define INPUT_DYNCOL 9

#define ANIM_AUTO 11
#define ANIM_OFF 12
#define ANIM_STEP 13
#define ANIM_RESET 14
#define MULT_ON 15
#define MULT_OFF 16

#define INPUT_STRIDE 20
#define INPUT_INT 21
#define CALC_STRIDE 22

#define PERSPECTIVE 30
#define ORTHOGRAPHIC 31

#define ZOOM 40
#define TRANSLATE 41
#define ROTX 42
#define ROTY 43
#define ROTZ 44

#define SHADE_HIDE 55
#define SHADE_FLAT 50
#define SHADE_OFF 51
#define SHADE_GOR 52
#define SHADE_WIRE 53
#define SHADE_TOGGLE 54

#define SAVE_TRMAT 60
#define SAVE_TRANS 61
#define RESET 62
#define STARTSIM 63
#define MVGEO 64
#define MVPLOTTER 65
#define VIDEO_ON 66
#define VIDEO_OFF 67
#define VIDEO_CREATE 68
#define WRITEILEAF 69

#define TOGGLE_COORD 70
#define INPUT_COORD_SCALING 71

#define ANIM_EXIT 80

#define CREATE_GEO 90
#define CREATE_WIRE 91
#define CREATE_CS 92

/* ------------------------------------------------------------------- */
/* movement factors */
#define SPACE_ROT (float)0.1 /* spaceball's rotation */
#define ZO_CONST (float)100

/* ------------------------------------------------------------------- */
/* viewing area defaults (orthogonal viewing mode) */
#define O_LEFT ((float)-10) /* coordinates of window edges */
#define O_RIGHT ((float)10)
#define O_LOWER ((float)-10)
#define O_UPPER ((float)10)
#define O_NEAR ((float)-100)
#define O_FAR ((float)100)
/* viewing area defaults (perspective viewing mode) */
#define P_FOVY ((float)90.0) /* viewing angle */
#define P_NEAR ((float)0.01) /* distance to 1st viewing plane */
#define P_FAR ((float)200.0) /* distance to last viewing plane */

/* default window positions and dimensions (may be changed by 
   commandline options */
#define WINWIDTH 0.5
#define WINHEIGHT 0.5
#define WINXPOS 0.25
#define WINYPOS 0.25

/* ------------------------------------------------------------------- */
/* time string default */
#define TIME_OFFSET_X 0.03 /* offset of time string to window borders */
#define TIME_OFFSET_Y 0.08

/* ------------------------------------------------------------------- */
/* PI definition */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------- */
/* plotter defaults */
#define PLOTTER_ROTX 5.0 /* initial rotation angles in degrees */
#define PLOTTER_ROTY 5.0
#define PLOTTER_ROTZ 0.0
#define PLOTTER_MOVEX -9.0 /* initial translation */
#define PLOTTER_MOVEY 7.5 /* (world coordinates) */
#define PLOTTER_MOVEZ 0.0
#define PLOTTER_SCALE 1.6f /* scale of plotter cube */

/* ------------------------------------------------------------------- */
/* message types */
#define FIRST_MESSAGE_SIM 950 /* messages of simulation program */
#define START_SIMULATION 951
#define TRANSFORMATION 952
#define END_SIMULATION 953

#define STOP_PROCESS 999 /* messages to all processes */

/* message senders */
#define NO_SENDER 0
#define SIMULATION 2
#define SEND 3 /* send message slave process */

/* ------------------------------------------------------------------- */
/* type definitions                                                    */
/* ------------------------------------------------------------------- */
typedef float anim_vector[3];

/* ------------------------------------------------------------------- */
/* structure definitions                                               */
/* ------------------------------------------------------------------- */
struct plotter
{
    int nf; /* number of faces */
    int nv; /* number of vertices */
    float h;
    float b;
    float t; /* edges of plotter */
    float *ky; /* origin on face of plotter */
    float *sy; /* origin on face of plotter */
    anim_vector *vertex;
    int **face; /* matrix of faces */
    anim_vector **axis;
    float drface[6];
    float drv[6][3];
    int drax[6];
};

/* ------------------------------------------------------------------- */
struct plotdata
{
    int first;
    int timesteps;
    int stride;
    float timeint;
    int act_step;
    int ndata;
    char **name;
    float **data;
    float *maxx;
    float *minx;
    float *maxy;
    float *miny;
    int *col;
};

/* ------------------------------------------------------------------- */
/* Note: order of indices is i, j, k, l */
struct elgeometry
{
    int first; /* is true, when file is read the first time */
    int nfiles; /* total number of files (bodies)            */
    int timesteps; /* number of timesteps                       */
    anim_vector ***vertex; /* coordinate l of vertex k of body i at
                               timestep j                                 */
    anim_vector ***norm; /* coordinate l of normal in vertex k of 
                               body i at timestep j                       */
    int *nf; /* total number of faces in body i           */
    int ***face; /* vertex k/index of face j in body i        */
    char **name; /* name of file i                            */
    int *nvertices; /* total number of vertices in body i     */
    int **ecolor; /* edge color                                */
    int **fcolor; /* face color                                */
    int **npoints; /* number of vertices in face j of body i    */
};

/* ------------------------------------------------------------------- */
struct dyncolor
{
    float ***rgb;
    int *isset;
};

/* ------------------------------------------------------------------- */
/* Note: order of indicees is i, j, k */
struct geometry
{
    int first; /* is true, when file is read the first time */
    int nfiles; /* total number of files (bodies)            */
    int *shading; /* shading of body i                         */
    anim_vector **vertex; /* coordinate k of vertex j of body i        */
    anim_vector **norm; /* coord. k of normal in vertex j of body i  */
    int *nf; /* total number of faces in body i           */
    int *nvertices; /* total number of vertices in body i     */
    int ***face; /* vertex k/index of face j in body i        */
    char **name; /* name of file i                            */
    int **ecolor; /* edge color                                */
    int **fcolor; /* face color                                */
    int **npoints; /* number of vertices in face j of body i    */
    int nballs; /* total number of files (bodies)            */
    double *ballsradius;
    int *ballscolor; /* edge color                             */
    int *hide;
    int *fixmotion;
    int *fixtranslation;
};

/* ------------------------------------------------------------------- */
struct ivfiles
{
    int nfiles; /* total number of files (bodies)            */
    char **name; /* name of file i                            */
};

/* ------------------------------------------------------------------- */
struct animation
{
    int first; /* is true, when file is read the first time */
    int timesteps; /* total number of timesteps                 */
    int stride; /* stride for timesteps                      */
    float timeint; /* realtime interval between timesteps in sec*/
    float dt; /* time difference between timesteps         */
    int act_step; /* actuell timestep                          */
    int multimg; /* draw multiple images in one frame         */
    float ***a; /* animation matrix for body j and timestep i*/
    float ***a_ball; /* animation matrix for ball j and timestep i*/
    float ***a_iv; /* animation matrix for ivfile j and timestep i*/
    float **balldyncolor; /* dynamic color for ball j and timestep i*/
};
/*----------------------------------------------------------------------*/
struct lineelem
{
    int nr; /* total number of line elements*/
    int n_iv; /*number of line elements as iv-Files*/
    int *color; /*color of line element*/
    int *type; /*type of line element*/
    int *bod1; /* number of body 1 element i is attached to*/
    int *bod2; /* number of body 2 element i is attached to*/
    anim_vector *coo1; /*coordinates of attachment at body 1 of element i*/
    anim_vector *coo2; /*coordinates of attachment at body 2 of element i*/
    char **name; /*name of file i*/
    anim_vector **pkt1; /*absolute coordinate of attachment 1 of element i 
                               at time j*/
    anim_vector **pkt2; /*absolute coordinate of attachment 2 of element i
                                at time j*/
    anim_vector **dir; /*direction of element i at time j*/
    float **length; /*length of element i at time j*/
};
/* ------------------------------------------------------------------- */
struct sensors
{
    int nr; /* total number of sensors                   */
    int *bod; /* number of body sensor i is fixed to       */
    int *col; /* number of color to draw sensor i          */
    anim_vector *coo; /* coordinate j of sensor i in body's system */
    anim_vector **pkt; /* coordinate k of last/last but one (j=1/0) */
}; /*   point of sensor i                       */

/* ------------------------------------------------------------------- */
struct mousepos
{
    int x;
    int y;
};

/* ------------------------------------------------------------------- */
struct menuentries
{
    int mainmenu;
    int anim;
    int mauto;
    int time;
    int proj;
    int mode;
    int shadem;
    int shade;
    int misc;
    int hide;
    int input;
    int fixed;
    int fixtrans;
    int plot;
    int move;
    int video;
    int *hide_file; /* field for hide-information */
    int *fixed_file; /* field for fixed-information */
    /* (fix totaly) */
    int *fixtrans_file; /* field for fixed-information */
}; /* (fix only translation) */

/* ------------------------------------------------------------------- */
struct flags
{
    int oitl_animation; /* operator-in-the-loop animation enabled */
    int simula; /* simulation program active */
    int video; /* save video of animation */
    int leftmbut; /* left mouse button pressed */
    int midmbut; /* middle mouse button pressed */
};

/* ------------------------------------------------------------------- */
struct modes
{
    int anim; /* animation mode */
    int mmb; /* middle mouse button */
    int shade; /* shading model */
    int shade_el; /* shading model for flexible bodies */
    int shade_toggle_rigid; /* toggle between rigid and flexible */
    int plotsel; /* plotter selection */
    int move; /* move objects or plotter */
    int displaytimestep; /* display time step or not */
    int coord_show_toggle; /* toggle between showing or hiding coordinate systems */
    double coord_scaling; /* scaling factor for coordinate systems */
};

/* ------------------------------------------------------------------- */
/* casts the calloc, free and realloc routines to allow error checking */
#define OWN_CALLOC(ptr, type, num)                                                                                     \
    {                                                                                                                  \
        if ((ptr) != NULL)                                                                                             \
        {                                                                                                              \
            printf("Warning: Try to allocate memory, but\n");                                                          \
            /*LINTED*/ printf("         the initial ptr is not NULL(0) " " (file %s, line %d)\n", __FILE__, __LINE__); \
            printf("         (Maybe an initialization error?)\n");                                                     \
            (void) fflush(stdout);                                                                                     \
            OWN_EXIT(ANIM_ERROR, "OWN_CALLOC");                                                                        \
            /* usually this is too strict but we want to find this problems */                                         \
        }                                                                                                              \
        (ptr) = (type *)calloc((size_t)(num), sizeof(type));                                                           \
        if ((ptr) == NULL)                                                                                             \
        {                                                                                                              \
            printf("Error: Could not allocate memory (file %s, line %d)\n", __FILE__, __LINE__);                       \
            (void) fflush(stdout);                                                                                     \
            OWN_EXIT(ANIM_ERROR, "OWN_CALLOC");                                                                        \
        }                                                                                                              \
    }

#define OWN_REALLOC(ptr, type, num)                                                                \
    {                                                                                              \
        (ptr) = (type *)realloc((ptr), (num) * sizeof(type));                                      \
        if ((ptr) == NULL)                                                                         \
        {                                                                                          \
            printf("Error: Could not reallocate memory (file %s, line %d)\n", __FILE__, __LINE__); \
            (void) fflush(stdout);                                                                 \
            OWN_EXIT(ANIM_ERROR, "OWN_REALLOC");                                                   \
        }                                                                                          \
    }

#define OWN_FREE(ptr)                                                                                     \
    {                                                                                                     \
        if ((ptr) != NULL)                                                                                \
        {                                                                                                 \
            free((ptr));                                                                                  \
            (ptr) = NULL;                                                                                 \
        }                                                                                                 \
        else                                                                                              \
        {                                                                                                 \
            /*LINTED*/ printf("Warning: Try to free NULL ptr? (file %s, line %d)\n", __FILE__, __LINE__); \
            (void) fflush(stdout);                                                                        \
            OWN_EXIT(ANIM_ERROR, "OWN_FREE");                                                             \
        }                                                                                                 \
    }

#define OWN_EXIT(value, msg)                                                                 \
    {                                                                                        \
        printf("Exit in file %s, line %d: value=%d (" msg ")\n", __FILE__, __LINE__, value); \
        (void) fflush(stdout);                                                               \
        exit(value);                                                                         \
    }

/* ------------------------------------------------------------------- */
