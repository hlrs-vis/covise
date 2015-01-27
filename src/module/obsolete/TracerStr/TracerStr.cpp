/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************
 *                                      *
 *  P A R T I C L E  -  T R A C E R     *
 *                                      *
 ****************************************
 * Rechenzentrum                        *
 * Universitaet Stuttgart               *
 ****************************************
 * Uwe Woessner                         *
 ****************************************/
// Sven Kufer 30.01.2000: added check_angle
//-------------------------------------------------------------------------------------------------------------
#include <util/coviseCompat.h>
#include "TracerStr.h"
#include <stdlib.h>

#ifndef _WIN32
#include <sys/time.h>
#endif

coDoLines *one_trace(int numblocks, coDoStructuredGrid **grid, coDoVec3 **vel, char *Lines, char *Magnitude, char *Velocity, float xpos, float ypos, float zpos, float, int direction, int whatout, float crit_angle);

int ExpandSetList(coDistributedObject *const *objects, int h_many, const char *type,
                  ia<coDistributedObject *> &out_array);

float dout(const float vel[3], int whatout);

char check_angle(float, float, float, float, float, float, float, float, float, int cr, float);
// check angle between parts of the line to stop trace at real walls
char *color[10] = {
    "yellow", "green", "blue", "red", "violet", "chocolat",
    "linen", "pink", "crimson", "indigo"
};

coDoVec3 *velo_obj;
coDoFloat *scal_obj;
int trace_num;

int is_ok;
// ----------------------
// Interpolationsfunktion
// ----------------------

#ifndef MAX
#define MAX(v1, v2) ((v1) > (v2) ? (v1) : (v2))
#endif
#ifndef MIN
#define MIN(v1, v2) ((v1) < (v2) ? (v1) : (v2))
#endif
#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif
#define SIGN(a, b) ((b) < 0.0 ? -(ABS((a))) : ABS((a)))
#define SQR(a) ((a) * (a))

static void cell3(int idim, int jdim, int kdim,
                  float *x_in, float *y_in, float *z_in,
                  int *i, int *j, int *k,
                  float *a, float *b, float *g,
                  float x[3], float amat[3][3], float bmat[3][3],
                  int *status);

static void intp3(
    int idim, int jdim, int kdim,
    float *u_in, float *v_in, float *w_in,
    int i, int j, int k,
    float a, float b, float g,
    float *fi);

static void metr3(int idim, int jdim, int kdim,
                  float *x_in, float *y_in, float *z_in,
                  int i, int j, int k,
                  float a, float b, float g,
                  float amat[3][3], float bmat[3][3],
                  int *idegen, int *status);

static void padv3(int *first, float cellfr, int direction,
                  int idim, int jdim, int kdim,
                  float *x_in, float *y_in, float *z_in,
                  float *u_in, float *v_in, float *w_in,
                  int *i, int *j, int *k,
                  float *a, float *b, float *g, float x[4],
                  float min_velo, int *status, float *ovel, float *nvel);

static void ssvdc(float *x,
                  int n, int p, float *s, float *e,
                  float *u, float *v, float *work,
                  int job, int *info);

static void srot(int n, float *sx, int incx, float *sy,
                 int incy, float c, float s);
static void srotg(float sa, float sb, float c, float s);
static void sscal(int n, float sa, float *sx, int incx);
static void sswap(int n, float *sx, int incx, float *sy, int incy);
static void saxpy(int n, float sa, float *sx, int incx, float *sy, int incy);
static float sdot(int n, float *sx, int incx, float *sy, int incy);
static float snrm2(int n, float *sx, int incx);

/*++ ptran3
 *****************************************************************************
 * PURPOSE: Transform a vector from physical coordinates to computational coordinates
 *          or vice-versa.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

static void ptran3(float amat[3][3], float v[3], float vv[3])
{
    vv[0] = v[0] * amat[0][0] + v[1] * amat[1][0] + v[2] * amat[2][0];
    vv[1] = v[0] * amat[0][1] + v[1] * amat[1][1] + v[2] * amat[2][1];
    vv[2] = v[0] * amat[0][2] + v[1] * amat[1][2] + v[2] * amat[2][2];
    return;
}

static void inv3x3(float a[3][3], float ainv[3][3], int *status);

//... Data in shared memory

coDoStructuredGrid *grid;
coDoVec3 *vel;

// ----------------------
// The Module Description
// ----------------------

int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();

    return 0;
}

// =======================================================================
// START WORK HERE (main block)
// =======================================================================
void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //

    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//------------------------------------------------------------------------------
/*
 * The module description is very important for the definition of the
 * message structure corresponding to that module. It follows this
 * description for the tracer module
 *
 *	MODULE	    tracer
 *	CATEGORY    mapper
 *	DESCRIPTION "particle tracer"
 *	INPUT  grid	coDoStructuredGrid "The cartesian mesh"	req
 *	INPUT  velocity coDoVec3	"The velocity"  req
 *	OUTPUT geometry	coDoGeometry "Particle paths"		req
 *	PARIN	xst Slider  "x_start"	0
 *	PARIN	yst Slider  "y_start"	0
 *	PARIN	zst Slider  "z_start"	0
 *	PARIN   rkk Slider  "RungeKutta Konstante"  1
 */
//-------------------------------------------------------------------------------------------------------------

// -------------
// Hauptprogramm
// -------------

//-------------------------------------------------------------------------------------------------------------

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================

//
// Remark: The hack to store data in a temp. DO so that it can be used in later execution
// cycles has been removed to make TracerStr usable in nets containing more
// TracerStr modules and to enhance the overall stability of the module. (RM 22.02.2001)
//
//

void Application::compute(void *)
{
    int i, j, k;
    int direction = 1, xs, ys, zs;
    int whatout;
    static int xss = 0, yss = 0, zss = 0;
    // int xss=0,yss=0,zss=0;
    long numstart_min, numstart_max, numstart = 1;
    int numblocks = 1, numblocks_d = 1;
    float startp1[3];
    float startp2[3];
    float startp[3];
    float cellfr = 2.0;
    float *tmpx, *tmpy, *tmpz;
    float *tmpxs, *tmpys, *tmpzs;
    char buf[500], buf1[500], buf2[500], *gtype, *dtype;

    int destroyIt = 0;
    coDoStructuredGrid **grids = NULL;
    coDoRectilinearGrid *rgrid = NULL;
    coDoUniformGrid *ugrid = NULL;
    coDoVec3 **datas = NULL;
    coDoSet *set_in = NULL;
    coDistributedObject *data_obj, *const *data_objs, *tmp_obj = NULL, *tmp_obj2 = NULL;

#if defined(_PROFILE_)
    WristWatch _ww;
    _ww.start();
#endif

    is_ok = 1;

    //get number of startpoints
    Covise::get_slider_param("no_startp", &numstart_min, &numstart_max, &numstart);

    //get startpoints
    for (i = 0; i < 3; i++)
    {
        // 1st start point
        Covise::get_vector_param("startpoint1", i, &startp1[i]);
        // 2nd start point
        Covise::get_vector_param("startpoint2", i, &startp2[i]);
        startp[i] = startp1[i];
    }

    Covise::get_choice_param("direction", &direction);
    Covise::get_choice_param("whatout", &whatout);

    Covise::get_scalar_param("cellfr", &cellfr);
    Covise::get_scalar_param("crit_angle", &crit_angle);

    //      retrieve grid object from shared memory
    char *GridIn = Covise::get_object_name("meshIn");
    char *DataIn = Covise::get_object_name("dataIn");
    char *Lines = Covise::get_object_name("lines");
    char *Magnitude = Covise::get_object_name("dataOut");
    char *Velocity = Covise::get_object_name("velocOut");

    // create a unique name for temp. DO_.. obj.
    runCnt_++;
    runCnt_ = runCnt_ % 365;
    sprintf(buf, "TMP%s%s%d", Covise::get_module(), Covise::get_instance(), runCnt_);
    char *tmpObjName = new char[1 + strlen(buf)];
    strcpy(tmpObjName, buf);

    if (GridIn == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'meshIn'");
        return;
    }

    if (DataIn == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'dataIn'");
        return;
    }

    if (Lines == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for lines");
        return;
    }

    tmp_obj = new coDistributedObject(GridIn);
    data_obj = tmp_obj->createUnknown();

    if (data_obj != 0L)
    {
        gtype = data_obj->getType();
        if (strcmp(gtype, "UNIGRD") == 0)
        {

            grids = new coDoStructuredGrid *[1];
            ugrid = (coDoUniformGrid *)data_obj;
            ugrid->getGridSize(&xs, &ys, &zs);
            float u_x_min, u_x_max;
            float u_y_min, u_y_max;
            float u_z_min, u_z_max;
            float u_dx, u_dy, u_dz;
            ugrid->getMinMax(&u_x_min, &u_x_max, &u_y_min, &u_y_max,
                             &u_z_min, &u_z_max);
            ugrid->getDelta(&u_dx, &u_dy, &u_dz);

            if ((xss == 0) && (yss == 0) && (zss == 0))
            {
                xss = xs;
                yss = ys;
                zss = zs;

                grids[0] = new coDoStructuredGrid(tmpObjName, xs, ys, zs);
                destroyIt = 1;

                grids[0]->getAddresses(&tmpxs, &tmpys, &tmpzs);

                float dist_x, dist_y, dist_z;
                for (i = 0, dist_x = 0.0; i < xs; i++)
                {
                    for (j = 0, dist_y = 0.0; j < ys; j++)
                    {
                        for (k = 0, dist_z = 0.0; k < zs; k++)
                        {
                            tmpxs[i * ys * zs + j * zs + k] = u_x_min + dist_x;
                            tmpys[i * ys * zs + j * zs + k] = u_y_min + dist_y;
                            tmpzs[i * ys * zs + j * zs + k] = u_z_min + dist_z;
                            dist_z += u_dz;
                        }
                        dist_y += u_dy;
                    }
                    dist_x += u_dx;
                }
            }
            else
            {
                grids[0] = new coDoStructuredGrid(tmpObjName);
                destroyIt = 1;

                // even if the grid is the same size as the previous
                // one, it might be a different grid
                if (1 /* || (xss!=xs)||(yss!=ys)||(zss!=zs) */)
                {
                    xss = xs;
                    yss = ys;
                    zss = zs;
                    grids[0]->destroy();
                    delete grids[0];
                    grids[0] = new coDoStructuredGrid(tmpObjName, xs, ys, zs);
                    destroyIt = 1;

                    grids[0]->getAddresses(&tmpxs, &tmpys, &tmpzs);
                    float dist_x, dist_y, dist_z;
                    for (i = 0, dist_x = 0.0; i < xs; i++)
                    {
                        for (j = 0, dist_y = 0.0; j < ys; j++)
                        {
                            for (k = 0, dist_z = 0.0; k < zs; k++)
                            {
                                tmpxs[i * ys * zs + j * zs + k] = u_x_min + dist_x;
                                tmpys[i * ys * zs + j * zs + k] = u_y_min + dist_y;
                                tmpzs[i * ys * zs + j * zs + k] = u_z_min + dist_z;
                                dist_z += u_dz;
                            }
                            dist_y += u_dy;
                        }
                        dist_x += u_dx;
                    }
                }
            }
            numblocks = 1;
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {

            grids = new coDoStructuredGrid *[1];
            rgrid = (coDoRectilinearGrid *)data_obj;
            rgrid->getGridSize(&xs, &ys, &zs);
            rgrid->getAddresses(&tmpx, &tmpy, &tmpz);
            if ((xss == 0) && (yss == 0) && (zss == 0))
            {
                xss = xs;
                yss = ys;
                zss = zs;

                grids[0] = new coDoStructuredGrid(tmpObjName, xs, ys, zs);
                destroyIt = 1;

                grids[0]->getAddresses(&tmpxs, &tmpys, &tmpzs);
                for (i = 0; i < xs; i++)
                    for (j = 0; j < ys; j++)
                        for (k = 0; k < zs; k++)
                        {
                            tmpxs[i * ys * zs + j * zs + k] = tmpx[i];
                            tmpys[i * ys * zs + j * zs + k] = tmpy[j];
                            tmpzs[i * ys * zs + j * zs + k] = tmpz[k];
                        }
            }
            else
            {
                grids[0] = new coDoStructuredGrid(tmpObjName);
                destroyIt = 1;

                // even if the grid is the same size as the previous
                // one, it could be a different grid
                if (1 /* (xss!=xs)||(yss!=ys)||(zss!=zs) */)
                {
                    xss = xs;
                    yss = ys;
                    zss = zs;
                    grids[0]->destroy();
                    delete grids[0];
                    grids[0] = new coDoStructuredGrid(tmpObjName, xs, ys, zs);
                    destroyIt = 1;

                    grids[0]->getAddresses(&tmpxs, &tmpys, &tmpzs);
                    for (i = 0; i < xs; i++)
                        for (j = 0; j < ys; j++)
                            for (k = 0; k < zs; k++)
                            {
                                tmpxs[i * ys * zs + j * zs + k] = tmpx[i];
                                tmpys[i * ys * zs + j * zs + k] = tmpy[j];
                                tmpzs[i * ys * zs + j * zs + k] = tmpz[k];
                            }
                }
            }

            numblocks = 1;
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            grids = new coDoStructuredGrid *[1];
            grids[0] = (coDoStructuredGrid *)data_obj;
            numblocks = 1;
        }
        else if (strcmp(gtype, "SETELE") == 0)
        {
            ia<coDistributedObject *> ia_data_objs;
            if (data_obj->getAttribute("TIMESTEPS"))
            {
                Covise::sendError("TIMESTEPS handling is not implemented yet");
                return;
            }
            set_in = (coDoSet *)data_obj;
            data_objs = set_in->getAllElements(&numblocks);
            numblocks = ExpandSetList(data_objs, numblocks, "STRGRD", ia_data_objs);
            /******************************************************************************************
                     if( data_objs[0]->isType("SETELE") )
                      {
                      Covise::sendError("Handling of grids which are set of sets is not implemented yet.");
                      return;
                      }
         *******************************************************************************************/
            grids = new coDoStructuredGrid *[numblocks];
            for (i = 0; i < numblocks; i++)
            {
                grids[i] = (coDoStructuredGrid *)ia_data_objs[i];
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return;
    }

    tmp_obj2 = new coDistributedObject(DataIn);
    data_obj = tmp_obj2->createUnknown();
    if (data_obj != 0L)
    {
        dtype = data_obj->getType();
        if (strcmp(dtype, "STRVDT") == 0)
        {
            datas = new coDoVec3 *[1];
            datas[0] = (coDoVec3 *)data_obj;
            numblocks = 1;

            if (strcmp(gtype, "SETELE") == 0)
            {
                Covise::sendError("The grid is a set and the vector field is not");
                return;
            }
        }
        else if (strcmp(dtype, "SETELE") == 0)
        {

            if (data_obj->getAttribute("TIMESTEPS"))
            {
                Covise::sendError("TIMESTEPS handling is not implemented yet");
                return;
            }

            if (strcmp(gtype, "SETELE") != 0)
            {
                Covise::sendError("The vector field is a set and the grid is not");
                return;
            }

            ia<coDistributedObject *> ia_data_objs;
            set_in = (coDoSet *)data_obj;
            data_objs = set_in->getAllElements(&numblocks_d);
            numblocks_d = ExpandSetList(data_objs, numblocks_d, "STRVDT", ia_data_objs);
            if (numblocks != numblocks_d)
            {
                Covise::sendError("The set of grids is not consistent with the set of vector field objects");
                return;
            }
            datas = new coDoVec3 *[numblocks];
            for (i = 0; i < numblocks; i++)
            {
                datas[i] = (coDoVec3 *)ia_data_objs[i];
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return;
    }

    coDoLines **lines = new coDoLines *[numstart * 2 + 1];
    coDoVec3 **velo_objs = new coDoVec3 *[numstart * 2 + 1];
    coDoFloat **scal_objs = new coDoFloat *[numstart * 2 + 1];
    int nl = 0;
    float *x1, *y1, *z1, *x2, *y2, *z2;
    int *dummy;
    for (i = 0; i < numstart; i++)
    {
        trace_num = i;
        sprintf(buf, "%s_%d", Lines, nl);
        sprintf(buf1, "%s_%d", Magnitude, nl);
        sprintf(buf2, "%s_%d", Velocity, nl);
        if (numstart == 1)
        {
            startp[0] = startp1[0];
            startp[1] = startp1[1];
            startp[2] = startp1[2];
        }
        else
        {
            startp[0] = startp1[0] + i * ((startp2[0] - startp1[0]) / (numstart - 1));
            startp[1] = startp1[1] + i * ((startp2[1] - startp1[1]) / (numstart - 1));
            startp[2] = startp1[2] + i * ((startp2[2] - startp1[2]) / (numstart - 1));
        }
        if ((direction == 1) || (direction == 2))
        {
            lines[nl] = one_trace(numblocks, grids, datas, buf, buf1, buf2, startp[0],
                                  startp[1], startp[2], cellfr, direction, whatout, crit_angle);
            velo_objs[nl] = velo_obj;
            scal_objs[nl] = scal_obj;

            if (lines[nl])
            {
                lines[nl]->addAttribute("COLOR", color[i % 10]);
                nl++;
            }
        }
        else
        {
            lines[nl] = one_trace(numblocks, grids, datas, buf, buf1, buf2,
                                  startp[0], startp[1], startp[2], cellfr, 1, whatout, crit_angle);

            velo_objs[nl] = velo_obj;
            scal_objs[nl] = scal_obj;
            if (lines[nl])
            {
                lines[nl]->addAttribute("COLOR", color[i % 10]);
                nl++;
            }
            sprintf(buf, "%s_%d", Lines, nl);
            sprintf(buf1, "%s_%d", Magnitude, nl);
            sprintf(buf2, "%s_%d", Velocity, nl);
            lines[nl] = one_trace(numblocks, grids, datas, buf, buf1, buf2,
                                  startp[0], startp[1], startp[2], cellfr, 2, whatout, crit_angle);
            velo_objs[nl] = velo_obj;
            scal_objs[nl] = scal_obj;
            if (lines[nl])
            {
                // CHECK IF LINES ARE UNCONTINUOUS
                if (nl > 0)
                {
                    lines[nl]->getAddresses(&x2, &y2, &z2, &dummy, &dummy);
                    lines[nl - 1]->getAddresses(&x1, &y1, &z1, &dummy, &dummy);
                    if (!check_angle(
                            x2[1], y2[1], z2[1],
                            x2[0], y2[0], z2[0],
                            x1[1], y1[1], z1[1], 3, crit_angle))
                    { // delete backward line
                        sprintf(buf2, "%s_del", buf);
                        lines[nl] = new coDoLines(buf2, 0, 0, 0);
                    }
                }
                lines[nl]->addAttribute("COLOR", color[i % 10]);
                nl++;
            }
        }
    }

    if (destroyIt)
    {
        grids[0]->destroy();
    }

    lines[nl] = NULL;
    velo_objs[nl] = NULL;
    scal_objs[nl] = NULL;
    coDoSet setout(Lines, (coDistributedObject **)lines);
    coDoSet setout1(Magnitude, (coDistributedObject **)scal_objs);
    coDoSet setout2(Velocity, (coDistributedObject **)velo_objs);
    sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    setout.addAttribute("FEEDBACK", buf);
    for (i = 0; i < nl; i++)
        delete lines[i];
    delete[] lines;
    for (i = 0; i < nl; i++)
    {
        delete velo_objs[i];
        delete scal_objs[i];
    }
    delete[] velo_objs;
    delete[] scal_objs;
    for (i = 0; i < numblocks; i++)
    {
        delete grids[i];
        delete datas[i];
    }

    delete[] tmpObjName;
    delete[] grids;
    delete[] datas;
    delete tmp_obj;
    delete tmp_obj2;

#if defined(_PROFILE_)
    _ww.stop("complete run");
#endif
}

coDoLines *one_trace(int numblocks, coDoStructuredGrid **grids, coDoVec3 **datas, char *Lines, char *Magnitude, char *Velocity, float xpos, float ypos, float zpos, float cellfr, int direction, int whatout, float crit_angle)
{
    // static
    int currblock = 0;
    int x_dim, y_dim, z_dim, status = 0, first = 1;
    int xv_dim, yv_dim, zv_dim;
    int i = 1, j = 1, k = 1, t = 1, n = 0, m = 0;
    int retry = 0;
    int thisblock;
    int told;
    //float s[3];
    float x[4];
    float coord[3][100000]; // Koordinate des Partikels [richtung0=x,1=y,2=z][Zeitschritt]
    float velo[3][100000];
    float scal[100000];

    float *u_in;
    float *v_in;
    float *w_in;
    float *x_in;
    float *y_in;
    float *z_in;
    float a = 0.5, b = 0.5, g = 0.50;
    float nvel[3] = { 1.0, 0.0, 0.0 };
    float ovel[3] = { 1.0, 0.0, 0.0 };
    int neighbor_number[200];
    char *neighbors;
    // lines
    coDoLines *lines;
    int *v_list;
    int *l_list;
    float *xc, *yc, *zc, *v[3], *s;
    int cr;

    t = 1;
    for (n = 0; n < numblocks; n++)
    {
        datas[currblock]->getAddresses(&u_in, &v_in, &w_in);
        datas[currblock]->getGridSize(&xv_dim, &yv_dim, &zv_dim);

        grids[currblock]->getGridSize(&x_dim, &y_dim, &z_dim);
        if (xv_dim != x_dim || xv_dim != x_dim || xv_dim != x_dim)
        {
            if (is_ok)
            {
                is_ok = 0;
                Covise::sendError("Inconsistent sizes of grid and vector field objects");
            }
            velo_obj = new coDoVec3(Velocity, 0);
            scal_obj = new coDoFloat(Magnitude, 0);
            return new coDoLines(Lines, 0, 0, 0);
            ;
        }
        grids[currblock]->getAddresses(&x_in, &y_in, &z_in);
        x[0] = coord[0][0] = xpos;
        x[1] = coord[1][0] = ypos;
        x[2] = coord[2][0] = zpos;
        x[3] = 0;
        while (((!status) || (retry < 2 && status != 1)) && (t < 100000))
        {
            if (first)
            {
                if (t == 1)
                {
                    x[0] = coord[0][0] = xpos;
                    x[1] = coord[1][0] = ypos;
                    x[2] = coord[2][0] = zpos;
                    x[3] = 0;
                }
                if (retry % 2 && x_dim > 3 && y_dim > 3 && z_dim > 3)
                {
                    i = x_dim - 2;
                    j = y_dim - 2;
                    k = z_dim - 2;
                }
                else
                {
                    i = 1;
                    j = 1;
                    k = 1;
                }
                padv3(&first, cellfr, direction,
                      x_dim, y_dim, z_dim,
                      x_in, y_in, z_in,
                      u_in, v_in, w_in,
                      &i, &j, &k,
                      &a, &b, &g, x,
                      0.001, &status, ovel, nvel);
                // sl: 0.001, &status,&velo[0],&velo[1]);
                //     update velo[0] or velo[1] if status==0
                if (status == 0)
                {
                    if (t == 1)
                    {
                        velo[0][t - 1] = ovel[0];
                        velo[1][t - 1] = ovel[1];
                        velo[2][t - 1] = ovel[2];
                        if (whatout == Application::TIME)
                        {
                            scal[t - 1] = 0.0;
                        }
                        else
                        {
                            scal[t - 1] = dout(ovel, whatout);
                        }
                    }
                    velo[0][t] = nvel[0];
                    velo[1][t] = nvel[1];
                    velo[2][t] = nvel[2];
                    if (whatout == Application::TIME)
                    {
                        scal[t] = x[3];
                    }
                    else
                    {
                        scal[t] = dout(nvel, whatout);
                    }
                }
            }
            else
            {
                padv3(&first, cellfr, direction,
                      x_dim, y_dim, z_dim,
                      x_in, y_in, z_in,
                      u_in, v_in, w_in,
                      &i, &j, &k,
                      &a, &b, &g, x,
                      0.001, &status, ovel, nvel);
            }
            if (!status)
            {
                //               velo[t]=nvel;
                //	       coord[0][t]=x[0];
                //	       coord[1][t]=x[1];
                //	       coord[2][t]=x[2];
                //	       t++;
                //	       retry = 0;
                if (check_angle(coord[0][t - 2], coord[1][t - 2], coord[2][t - 2],
                                coord[0][t - 1], coord[1][t - 1], coord[2][t - 1],
                                x[0], x[1], x[2], t, crit_angle))
                {
                    velo[0][t] = nvel[0];
                    velo[1][t] = nvel[1];
                    velo[2][t] = nvel[2];
                    coord[0][t] = x[0];
                    coord[1][t] = x[1];
                    coord[2][t] = x[2];
                    if (whatout == Application::TIME)
                    {
                        scal[t] = x[3];
                    }
                    else
                    {
                        scal[t] = dout(nvel, whatout);
                    }
                    t++;
                    retry = 0;
                }
                else
                {
                    status = 1;
                    retry = 2;
                }
            }
            else
            {
                if (t == 1)
                {
                    first = 1;
                    retry++;
                }
                else
                    retry = 2;
            }
        }
        if (t > 1 && status != 1) // we found the starting point but left the block, now try neighbors
        {
            float xold, yold, zold, kold;
            thisblock = currblock;
            told = 0;
            while (t != told)
            {
                xold = x[0];
                yold = x[1];
                zold = x[2];
                kold = x[3];
                told = t;
                neighbors = grids[thisblock]->getAttribute("NEIGHBOUR");
                if (neighbors)
                {
                    sscanf(neighbors, "%d %d %d %d %d %d", neighbor_number, neighbor_number + 1, neighbor_number + 2, neighbor_number + 3, neighbor_number + 4, neighbor_number + 5);
                    for (m = 0; m < 6; m++)
                    {
                        if (neighbor_number[m] > 0)
                        {
                            retry = 0;
                            status = 0;
                            first = 1;
                            thisblock = neighbor_number[m];
                            datas[thisblock]->getAddresses(&u_in, &v_in, &w_in);

                            grids[thisblock]->getGridSize(&x_dim, &y_dim, &z_dim);
                            grids[thisblock]->getAddresses(&x_in, &y_in, &z_in);
                            while (((!status) || (retry < 2 && status != 1)) && (t < 100000))
                            {
                                if (first)
                                {

                                    if (retry % 2 && x_dim > 3 && y_dim > 3 && z_dim > 3)
                                    {
                                        i = x_dim - 2;
                                        j = y_dim - 2;
                                        k = z_dim - 2;
                                    }
                                    else
                                    {
                                        i = 1;
                                        j = 1;
                                        k = 1;
                                    }
                                }
                                padv3(&first, cellfr, direction,
                                      x_dim, y_dim, z_dim,
                                      x_in, y_in, z_in,
                                      u_in, v_in, w_in,
                                      &i, &j, &k,
                                      &a, &b, &g, x,
                                      0.001, &status, ovel, nvel);

                                if (!status)
                                {

                                    if (check_angle(
                                            coord[0][t - 2], coord[1][t - 2], coord[2][t - 2],
                                            coord[0][t - 1], coord[1][t - 1], coord[2][t - 1],
                                            x[0], x[1], x[2], t, crit_angle))
                                    {
                                        velo[0][t] = nvel[0];
                                        velo[1][t] = nvel[1];
                                        velo[2][t] = nvel[2];
                                        coord[0][t] = x[0];
                                        coord[1][t] = x[1];
                                        coord[2][t] = x[2];
                                        if (whatout == Application::TIME)
                                        {
                                            scal[t] = x[3];
                                        }
                                        else
                                        {
                                            scal[t] = dout(nvel, whatout);
                                        }
                                        t++;
                                        retry = 0;
                                    }
                                    else
                                    { // wall detected
                                        status = 1;
                                        retry = 2;
                                    }
                                }
                                else
                                {
                                    if (t == told)
                                    {
                                        x[0] = xold;
                                        x[1] = yold;
                                        x[2] = zold;
                                        x[3] = kold;
                                        first = 1;
                                        retry++;
                                    }
                                    else
                                        retry = 2;
                                }
                            }
                            if (t > told) // we found the the neighborblock
                            {
                                break;
                            }
                            retry = 0;
                            status = 0;
                        }
                    }
                }
                else
                {
                    for (m = 0; m < numblocks; m++)
                    {
                        retry = 0;
                        status = 0;
                        first = 1;
                        thisblock = m;
                        datas[thisblock]->getAddresses(&u_in, &v_in, &w_in);

                        grids[thisblock]->getGridSize(&x_dim, &y_dim, &z_dim);
                        grids[thisblock]->getAddresses(&x_in, &y_in, &z_in);
                        while (((!status) || (retry < 2 && status != 1)) && (t < 100000))
                        {
                            if (first)
                            {

                                if (retry % 2 && x_dim > 3 && y_dim > 3 && z_dim > 3)
                                {
                                    i = x_dim - 2;
                                    j = y_dim - 2;
                                    k = z_dim - 2;
                                }
                                else
                                {
                                    i = 1;
                                    j = 1;
                                    k = 1;
                                }
                            }
                            padv3(&first, cellfr, direction,
                                  x_dim, y_dim, z_dim,
                                  x_in, y_in, z_in,
                                  u_in, v_in, w_in,
                                  &i, &j, &k,
                                  &a, &b, &g, x,
                                  0.001, &status, ovel, nvel);

                            if (!status)
                            {

                                if (check_angle(
                                        coord[0][t - 2], coord[1][t - 2], coord[2][t - 2],
                                        coord[0][t - 1], coord[1][t - 1], coord[2][t - 1],
                                        x[0], x[1], x[2], t, crit_angle))
                                {
                                    velo[0][t] = nvel[0];
                                    velo[1][t] = nvel[1];
                                    velo[2][t] = nvel[2];
                                    coord[0][t] = x[0];
                                    coord[1][t] = x[1];
                                    coord[2][t] = x[2];
                                    if (whatout == Application::TIME)
                                    {
                                        scal[t] = x[3];
                                    }
                                    else
                                    {
                                        scal[t] = dout(nvel, whatout);
                                    }
                                    t++;
                                    retry = 0;
                                }
                                else // wall detected
                                {
                                    status = 1;
                                    retry = 2;
                                }
                            }
                            else
                            {
                                if (t == told)
                                {
                                    x[0] = xold;
                                    x[1] = yold;
                                    x[2] = zold;
                                    x[3] = kold;
                                    first = 1;
                                    retry++;
                                }
                                else
                                    retry = 2;
                            }
                        }
                        if (t > told) // we found the the neighborblock
                        {
                            break;
                        }
                        retry = 0;
                        status = 0;
                    }
                }
            }
            break; // stop tracing
        }
        currblock++;
        if (currblock >= numblocks)
            currblock = 0;
        retry = 0;
        status = 0;
    }

    //-------------------------------------------------------------------------------
    // Linien erzeugen
    //-------------------------------------------------------------------------------

    //... first write lines, then do geometry

    if (t > 1)
    {

        lines = new coDoLines(Lines, t, t, 1);
        lines->getAddresses(&xc, &yc, &zc, &v_list, &l_list);
        for (cr = 0; cr < t; cr++)
        {
            v_list[cr] = cr;
            xc[cr] = coord[0][cr];
            yc[cr] = coord[1][cr];
            zc[cr] = coord[2][cr];
        }
        l_list[0] = 0;

        velo_obj = new coDoVec3(Velocity, t);
        scal_obj = new coDoFloat(Magnitude, t);
        lines->addAttribute("TRACER_DATA_OBJECT", Magnitude);
        lines->addAttribute("TRACER_VEL_OBJECT", Velocity);
        velo_obj->getAddresses(&v[0], &v[1], &v[2]);
        scal_obj->getAddress(&s);
        memcpy(v[0], velo[0], t * sizeof(float));
        memcpy(v[1], velo[1], t * sizeof(float));
        memcpy(v[2], velo[2], t * sizeof(float));
        memcpy(s, scal, t * sizeof(float));
        return lines;
    }
    else
    {
        lines = new coDoLines(Lines, 0, 0, 0);
        velo_obj = new coDoVec3(Velocity, 0);
        scal_obj = new coDoFloat(Magnitude, 0);
        return lines;
    }
}

char check_angle(float coord0x, float coord0y, float coord0z,
                 float coord1x, float coord1y, float coord1z,
                 float coord2x, float coord2y, float coord2z, int cr, float crit_angle)
{
    double a[3], b[3];
    if (cr > 2)
    {
        a[0] = coord0x - coord1x;
        a[1] = coord0y - coord1y;
        a[2] = coord0z - coord1z;
        b[0] = coord2x - coord1x;
        b[1] = coord2y - coord1y;
        b[2] = coord2z - coord1z;

        if (SQR(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / ((SQR(a[0]) + SQR(a[1]) + SQR(a[2])) * (SQR(b[0]) + SQR(b[1]) + SQR(b[2]))) >= cos(crit_angle * M_PI / 180.0) && a[0] * b[0] + a[1] * b[1] + a[2] * b[2] < 0.0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    return 1;
}

/*++ padv3
 *****************************************************************************
 * PURPOSE: Advance a particle through a vector field V, roughly CELLFR fraction of a
 *          computational cell.  Use 2-step Runge-Kutta time-advance:
 *          x(*)   = x(n) + dt*f[x(n)]
 *          x(n+1) = x(n) + dt*(1/2)*{f[x(n)]+f[x(*)]}
 *          = x(*) + dt*(1/2)*{f[x(*)]-f[x(n)]}
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: STATUS=1 - Particle has come to rest.
 *          STATUS=2 - Particle has left the (valid) computational domain.
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

static void padv3(int *first, float cellfr, int direction,
                  int idim, int jdim, int kdim,
                  float *x_in, float *y_in, float *z_in,
                  float *u_in, float *v_in, float *w_in,
                  int *i, int *j, int *k,
                  float *a, float *b, float *g, float x[4],
                  float min_velo, int *status, float *ovel, float *nvel)
{
    //float asav, bsav, gsav;
    float u[3], uu[3], ustar[3], uustar[3], xstar[3];
    float uumax, dt;
    static float amat[3][3], bmat[3][3];
    int idegen, istat;

    *status = 0;

    /*  First time through, compute the metric transformation matrices.	*/

    if (*first)
    {
        metr3(idim, jdim, kdim, x_in, y_in, z_in, *i, *j, *k, *a, *b, *g, amat, bmat,
              &idegen, &istat);
        xstar[0] = x[0];
        xstar[1] = x[1];
        xstar[2] = x[2];
        cell3(idim, jdim, kdim, x_in, y_in, z_in, i, j, k, a, b, g, xstar, amat, bmat, &istat);
        x[0] = xstar[0];
        x[1] = xstar[1];
        x[2] = xstar[2];
        metr3(idim, jdim, kdim, x_in, y_in, z_in, *i, *j, *k, *a, *b, *g, amat, bmat,
              &idegen, &istat);
    }

    /*  Interpolate for the velocity.					*/

    intp3(idim, jdim, kdim, u_in, v_in, w_in, *i, *j, *k, *a, *b, *g, u);
    if (*first)
    {
        // *ovel=sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        ovel[0] = u[0];
        ovel[1] = u[1];
        ovel[2] = u[2];
        *first = 0;
    }

    /*  Transform velocity to computational space.				*/

    ptran3(bmat, u, uu);

    /*  Limit the timestep so we move only CELLFR fraction of a cell (at 	*/
    /*  least in the predictor step).					*/

    uumax = MAX(ABS(uu[0]), ABS(uu[1]));
    uumax = MAX(uumax, ABS(uu[2]));
    if (uumax <= min_velo)
    {
        *status = 1;
        return;
    }

    do
    {
        dt = cellfr / uumax;
        if (direction == 2)
            dt = -dt;

        /*  Predictor step.							*/

        xstar[0] = x[0] + dt * u[0];
        xstar[1] = x[1] + dt * u[1];
        xstar[2] = x[2] + dt * u[2];
        // x[3]    = x[3] + dt;   not here!!!....

        /*  Find the new point in computational space.				*/

        cell3(idim, jdim, kdim, x_in, y_in, z_in, i, j, k, a, b, g, xstar, amat, bmat, &istat);
        if (istat == 1)
        {
            *status = 2;
            x[0] = xstar[0];
            x[1] = xstar[1];
            x[2] = xstar[2];
            x[3] = x[3] + dt; // but here, or....
            return;
        }

        /*  Interpolate for velocity at the new point.				*/

        intp3(idim, jdim, kdim, u_in, v_in, w_in, *i, *j, *k, *a, *b, *g, ustar);

        /*  Transform velocity to computational space.				*/

        ptran3(bmat, ustar, uustar);

        /*  Check that our timestep is still reasonable.			*/

        uu[0] = .5 * (uu[0] + uustar[0]);
        uu[1] = .5 * (uu[1] + uustar[1]);
        uu[2] = .5 * (uu[2] + uustar[2]);
        uumax = MAX(ABS(uu[0]), ABS(uu[1]));
        uumax = MAX(uumax, ABS(uu[2]));
        if (uumax <= min_velo)
        {
            *status = 1;
            break;
        }
    } while (ABS(dt * uumax) > 1.5 * (cellfr));

    x[3] = x[3] + dt; // here.

    /*  Corrector step.							*/
    // *nvel=sqrt(ustar[0]*ustar[0]+ustar[1]*ustar[1]+ustar[2]*ustar[2]);
    nvel[0] = ustar[0];
    nvel[1] = ustar[1];
    nvel[2] = ustar[2];

    x[0] = x[0] + dt * .5 * (u[0] + ustar[0]);
    x[1] = x[1] + dt * .5 * (u[1] + ustar[1]);
    x[2] = x[2] + dt * .5 * (u[2] + ustar[2]);

    /*  Find the corresponding point in computational space.		*/

    cell3(idim, jdim, kdim, x_in, y_in, z_in, i, j, k, a, b, g, x, amat, bmat, &istat);
    if (istat == 1)
        *status = 2;

    return;
}

/*++ cell3
 *****************************************************************************
 * PURPOSE: Find the point (x,y,z) in the grid XYZ and return its (i,j,k) cell number.
 *          We ASSUME that the given (i,j,k), (a,b,g), and subset are valid.  These
 *          can be checked with STRTxx.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: status=1 - Unable to find the point without going out of the
 *          computational domain or active subet.  The computational
 *          point returned indicates the direction to look...
 * RETURNS: None
 *          PROCEDURES CALLED IN THIS ROUTINE :
 *          None
 * NOTES:   None
 *****************************************************************************
--*/

static void cell3(int idim, int jdim, int kdim,
                  float *x_in, float *y_in, float *z_in,
                  int *i, int *j, int *k,
                  float *a, float *b, float *g,
                  float x[3], float amat[3][3], float bmat[3][3],
                  int *status)
{
#define MITER 5
/*
      defines for readability
   */
#ifdef PI
#undef PI
#endif

#define PA (*a)
#define PB (*b)
#define PG (*g)
#define PI (*i)
#define PJ (*j)
#define PK (*k)

    float x0, x1, x2, x3, x4, x5, x6, x7, x8,
        xa, xb, xg, xag, xbg, xabg, xab,

        /* y0 and y1 cannot conflict in math.h */
        Y0, Y1, y2, y3, y4, y5, y6, y7, y8,
        ya, yb, yg, yag, ybg, yabg, yab,

        z0, z1, z2, z3, z4, z5, z6, z7, z8,
        za, zb, zg, zag, zbg, zabg, zab;

    int nstep, msteps;
    int iter;
    float dx, dy, dz, da, db, dg, as, bs, gs;
    int i1, J1, k1, in, jN, kn;
    float xh, yh, zh;
    int dialog = 0, istat;
    float err2;
    int di, dj, dk;

    int isav1 = 0, jsav1 = 0, ksav1 = 0,
        isav2 = 0, jsav2 = 0, ksav2 = 0;
    float pab, pbg, pag, pabg;

    *status = 0;

    /*
       Reset saved points used for checking if we're stuck on the same point (or
       ping-ponging back and forth) when searching for the right cell.  This
       would happen while following a point outside the computational domain.
   */

    isav1 = 0;
    isav2 = 0;
    di = 1;
    dj = 1;
    dk = 1;

    /*
     Maximum number of steps before we'll give up is maximum of (idim,jdim,kdim).
   */

    msteps = MAX(idim, jdim);
    msteps = MAX(msteps, kdim);
    msteps *= 50;
    nstep = 0;

    /*
      beginning of loop
   */
    while (1)
    {
        nstep++;
        i1 = PI - 1;
        J1 = PJ - 1;
        k1 = PK - 1;

        x1 = x_in[i1 * jdim * kdim + J1 * kdim + k1];
        Y1 = y_in[i1 * jdim * kdim + J1 * kdim + k1];
        z1 = z_in[i1 * jdim * kdim + J1 * kdim + k1];
        x2 = x_in[PI * jdim * kdim + J1 * kdim + k1];
        y2 = y_in[PI * jdim * kdim + J1 * kdim + k1];
        z2 = z_in[PI * jdim * kdim + J1 * kdim + k1];
        x3 = x_in[i1 * jdim * kdim + PJ * kdim + k1];
        y3 = y_in[i1 * jdim * kdim + PJ * kdim + k1];
        z3 = z_in[i1 * jdim * kdim + PJ * kdim + k1];
        x4 = x_in[PI * jdim * kdim + PJ * kdim + k1];
        y4 = y_in[PI * jdim * kdim + PJ * kdim + k1];
        z4 = z_in[PI * jdim * kdim + PJ * kdim + k1];
        x5 = x_in[i1 * jdim * kdim + J1 * kdim + PK];
        y5 = y_in[i1 * jdim * kdim + J1 * kdim + PK];
        z5 = z_in[i1 * jdim * kdim + J1 * kdim + PK];
        x6 = x_in[PI * jdim * kdim + J1 * kdim + PK];
        y6 = y_in[PI * jdim * kdim + J1 * kdim + PK];
        z6 = z_in[PI * jdim * kdim + J1 * kdim + PK];
        x7 = x_in[i1 * jdim * kdim + PJ * kdim + PK];
        y7 = y_in[i1 * jdim * kdim + PJ * kdim + PK];
        z7 = z_in[i1 * jdim * kdim + PJ * kdim + PK];
        x8 = x_in[PI * jdim * kdim + PJ * kdim + PK];
        y8 = y_in[PI * jdim * kdim + PJ * kdim + PK];
        z8 = z_in[PI * jdim * kdim + PJ * kdim + PK];

        x0 = x1;
        xa = x2 - x1;
        xb = x3 - x1;
        xg = x5 - x1;
        xab = x4 - x3 - xa;
        xag = x6 - x5 - xa;
        xbg = x7 - x5 - xb;
        xabg = x8 - x7 - x6 + x5 - x4 + x3 + xa;

        Y0 = Y1;
        ya = y2 - Y1;
        yb = y3 - Y1;
        yg = y5 - Y1;
        yab = y4 - y3 - ya;
        yag = y6 - y5 - ya;
        ybg = y7 - y5 - yb;
        yabg = y8 - y7 - y6 + y5 - y4 + y3 + ya;

        z0 = z1;
        za = z2 - z1;
        zb = z3 - z1;
        zg = z5 - z1;
        zab = z4 - z3 - za;
        zag = z6 - z5 - za;
        zbg = z7 - z5 - zb;
        zabg = z8 - z7 - z6 + z5 - z4 + z3 + za;

        PA = .5;
        PB = .5;
        PG = .5;

        iter = 0;
        while (1)
        {
            iter++;
            /*
         These next 4 lines of code reduce the number of
         multiplications performed in the succeding 12 lines
         of code.
         */
            pab = PA * PB;
            pag = PA * PG;
            pbg = PB * PG;
            pabg = pab * PG;

            xh = x0 + xa * PA + xb * PB + xg * PG + xab * pab + xag * pag + xbg * pbg + xabg * pabg;
            yh = Y0 + ya * PA + yb * PB + yg * PG + yab * pab + yag * pag + ybg * pbg + yabg * pabg;
            zh = z0 + za * PA + zb * PB + zg * PG + zab * pab + zag * pag + zbg * pbg + zabg * pabg;

            amat[0][0] = xa + xab * PB + xag * PG + xabg * pbg;
            amat[0][1] = ya + yab * PB + yag * PG + yabg * pbg;
            amat[0][2] = za + zab * PB + zag * PG + zabg * pbg;

            amat[1][0] = xb + xab * PA + xbg * PG + xabg * pag;
            amat[1][1] = yb + yab * PA + ybg * PG + yabg * pag;
            amat[1][2] = zb + zab * PA + zbg * PG + zabg * pag;

            amat[2][0] = xg + xag * PA + xbg * PB + xabg * pab;
            amat[2][1] = yg + yag * PA + ybg * PB + yabg * pab;
            amat[2][2] = zg + zag * PA + zbg * PB + zabg * pab;

            inv3x3(amat, bmat, &istat);
            if (istat)
            {
                if (dialog)
                    printf("Degenerate volume at index %d %d %d\n", PI, PJ, PK);
                /*
              See if we're at the edge of the cell.
            If so, move away from the (possibly degenerate)
            edge and recompute the matrix.
            */
                as = PA;
                bs = PB;
                gs = PG;

                if (PA == 0.)
                    PA = .01;
                if (PA == 1.)
                    PA = .99;
                if (PB == 0.)
                    PB = .01;
                if (PB == 1.)
                    PB = .99;
                if (PG == 0.)
                    PG = .01;
                if (PG == 1.)
                    PG = .99;
                if (PA != as || PB != bs || PG != gs)
                    continue;
                else
                {
                    /*
                 We're inside a cell and the transformation
                 matrix is singular.  Move to the next cell and try again.
               */
                    PA = di + .5;
                    PB = dj + .5;
                    PG = dk + .5;
                    break;
                }
            }
            dx = x[0] - xh;
            dy = x[1] - yh;
            dz = x[2] - zh;
            da = dx * bmat[0][0] + dy * bmat[1][0] + dz * bmat[2][0];
            db = dx * bmat[0][1] + dy * bmat[1][1] + dz * bmat[2][1];
            dg = dx * bmat[0][2] + dy * bmat[1][2] + dz * bmat[2][2];
            PA += da;
            PB += db;
            PG += dg;

            /*
           If we're WAY off, don't bother with the error test.
           In fact, go ahead and try another cell.
         */

            if (ABS(PA - .5) > 3.)
                break;
            else if (ABS(PB - .5) > 3.)
                break;
            else if (ABS(PG - .5) > 3.)
                break;
            else
            {
                err2 = da * da + db * db + dg * dg;
                /*
              Check iteration error and branch out if it's small enough.
            */
                if (err2 <= 1.e-4)
                    break;
            }
            if (iter >= MITER)
                break;

        } /* end of edge while */

        /*
        The point is in this cell.
      */

        if (ABS(PA - .5) <= .50005 && ABS(PB - .5) <= .50005 && ABS(PG - .5) <= .50005)
        {
            if (dialog)
                printf("match\n");
            return;
        }
        /*
         We've taken more steps then we're willing to wait...
      */
        else if (nstep > msteps)
        {
            *status = 1;
            if (dialog)
                printf("more than %d steps\n", msteps);
            /*
           Update our (i,j,k) guess, keeping it inbounds.
         */
        }
        else
        {
            in = PI;
            jN = PJ;
            kn = PK;
            if (PA < 0.)
                in = MAX(in - 1, 1);
            if (PA > 1.)
                in = MIN(in + 1, idim - 1);
            if (PB < 0.)
                jN = MAX(jN - 1, 1);
            if (PB > 1.)
                jN = MIN(jN + 1, jdim - 1);
            if (PG < 0.)
                kn = MAX(kn - 1, 1);
            if (PG > 1.)
                kn = MIN(kn + 1, kdim - 1);
            if (dialog)
                printf("try cell index %d %d %d\n", in, jN, kn);
            /*
           Not repeating a previous point.  Use the
           new (i,j,k) and try again.
         */
            if ((in != isav1 || jN != jsav1 || kn != ksav1)
                && (in != isav2 || jN != jsav2 || kn != ksav2))
            {
                isav2 = isav1;
                jsav2 = jsav1;
                ksav2 = ksav1;
                isav1 = in;
                jsav1 = jN;
                ksav1 = kn;
                di = (int)SIGN(1, in - PI);
                dj = (int)SIGN(1, jN - PJ);
                dk = (int)SIGN(1, kn - PK);
                PI = in;
                PJ = jN;
                PK = kn;
                continue;
            }
            else
            {
                /*
              It seems to be outside the domain.
            We would have to extrapolate to find it.
            */
                *status = 1;
                if (dialog)
                    printf("extrapolate 2\n");
                return;
            }
        }
        /* exit */
        break;

    } /* end main while */
}

/*++ intp3
 *****************************************************************************
 * PURPOSE: Interpolate to find the value of F at point (I+A,J+B,K+G).  Use trilinear
 *          interpolation.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   F4 is a series of pointers that takes advantage of the way
 *          C dereferences pointers.  It allows the referencing
 *          of the array "f" as an array of dimension 4, but speeds
 *          array reference calculation as the addresses are stored
 *          in the pointer array.  It also allows array dimensions to
 *          be changed during runtime.  The pointer addresses are
 *          In this particular routine it is more slightly more efficient
 *          to calculate array references than to use the multiple
 *          indirect method with F4, so f is assigned the value of the
 *          address for the first element of the array.
 *          The variable "skip" has been added to allow the routine to
 *          ignore the first value in the solution array.  this routine
 *          for the solution array.
 *****************************************************************************
--*/

static void intp3(int /*idim*/, int jdim, int kdim,
                  float *u_in, float *v_in, float *w_in,
                  int i, int j, int k,
                  float a, float b, float g,
                  float *fi)

{
#define NDIM 5

    int i1, J1, k1; //, nx, itmp;
    float a1, b1, g1, c1, c2, c3, c4, c5, c6, c7, c8;

    a1 = 1. - a;
    b1 = 1. - b;
    g1 = 1. - g;

    c1 = a1 * b1 * g1;
    c2 = a * b1 * g1;
    c3 = a1 * b * g1;
    c4 = a * b * g1;
    c5 = a1 * b1 * g;
    c6 = a * b1 * g;
    c7 = a1 * b * g;
    c8 = a * b * g;

    /*   get subscripts for current array element (original code started
        numbering from 1, whereas here we start from 0 */

    i1 = i - 1;
    J1 = j - 1;
    k1 = k - 1;

    /*------
       the computation of fi[nx] consumed more time than any other in the
       program.  Array reference calculation is a significant part of this.
       The array reference calculations are removed here so that we can
       use manually optimized code.  Repetitive parts of the array reference
       are computed outside of the loop, and the offset for the first subsrcipt
       is calculated ONCE.  The number of multiplies is reduced substantially,
       and execution time is improved by ~30%.  SHPhilipson	--------*/

    fi[0] = c1 * u_in[i1 * jdim * kdim + J1 * kdim + k1] + c2 * u_in[i * jdim * kdim + J1 * kdim + k1] + c3 * u_in[i1 * jdim * kdim + j * kdim + k1] + c4 * u_in[i * jdim * kdim + j * kdim + k1] + c5 * u_in[i1 * jdim * kdim + J1 * kdim + k] + c6 * u_in[i * jdim * kdim + J1 * kdim + k] + c7 * u_in[i1 * jdim * kdim + j * kdim + k] + c8 * u_in[i * jdim * kdim + j * kdim + k];
    fi[1] = c1 * v_in[i1 * jdim * kdim + J1 * kdim + k1] + c2 * v_in[i * jdim * kdim + J1 * kdim + k1] + c3 * v_in[i1 * jdim * kdim + j * kdim + k1] + c4 * v_in[i * jdim * kdim + j * kdim + k1] + c5 * v_in[i1 * jdim * kdim + J1 * kdim + k] + c6 * v_in[i * jdim * kdim + J1 * kdim + k] + c7 * v_in[i1 * jdim * kdim + j * kdim + k] + c8 * v_in[i * jdim * kdim + j * kdim + k];
    fi[2] = c1 * w_in[i1 * jdim * kdim + J1 * kdim + k1] + c2 * w_in[i * jdim * kdim + J1 * kdim + k1] + c3 * w_in[i1 * jdim * kdim + j * kdim + k1] + c4 * w_in[i * jdim * kdim + j * kdim + k1] + c5 * w_in[i1 * jdim * kdim + J1 * kdim + k] + c6 * w_in[i * jdim * kdim + J1 * kdim + k] + c7 * w_in[i1 * jdim * kdim + j * kdim + k] + c8 * w_in[i * jdim * kdim + j * kdim + k];
    return;
}

/*++ metr3
 *****************************************************************************
 * PURPOSE: Compute the metric transformations for the point (I+A,J+B,K+G).  If the
 *          cell is degenerate, return transformations for a "nearby" point (if a cell
 *          edge or face is collapsed), or return the singular transformation and its
 *          pseudo-inverse (if an entire side is collapsed). Indicate the degenerate
 *          coordinate direction(s) in IDEGEN.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: STATUS=1 - The cell is degenerate.  Nearby transformations are
 *          returned.
 *          STATUS=2 - The cell is degenerate.  Singular trasformations are
 *          returned and IDEGEN indicates the degenerate direction(s).
 *          IDEGEN=1 - The cell is degenerate in the i-direction.
 *          IDEGEN=2 - The cell is degenerate in the j-direction.
 *          IDEGEN=3 - The cell is degenerate in the i- and j-directions.
 *          IDEGEN=4 - The cell is degenerate in the k-direction.
 *          IDEGEN=5 - The cell is degenerate in the i- and k-directions.
 *          IDEGEN=6 - The cell is degenerate in the j- and k-directions.
 *          IDEGEN=7 - The cell is degenerate in the i-, j-, and k-directions.
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

static void metr3(int /*idim*/, int jdim, int kdim,
                  float *x_in, float *y_in, float *z_in,
                  int i, int j, int k,
                  float a, float b, float g,
                  float amat[3][3], float bmat[3][3],
                  int *idegen, int *status)
{
    /* local variables						*/

    int i1, J1, k1, istat;
    float aa, bb, gg, as, bs, gs;
    float x1, x2, x3, x4, x5, x6, x7, x8;
    float Y1, y2, y3, y4, y5, y6, y7, y8;
    float z1, z2, z3, z4, z5, z6, z7, z8;
    float xa, xb, xg, xab, xag, xbg, xabg;
    float ya, yb, yg, yab, yag, ybg, yabg;
    float za, zb, zg, zab, zag, zbg, zabg;

    *status = 0;
    *idegen = 0;

    aa = a;
    bb = b;
    gg = g;

    i1 = i - 1;
    J1 = j - 1;
    k1 = k - 1;

    x1 = x_in[i1 * jdim * kdim + J1 * kdim + k1];
    x2 = x_in[i * jdim * kdim + J1 * kdim + k1];
    x3 = x_in[i1 * jdim * kdim + j * kdim + k1];
    x4 = x_in[i * jdim * kdim + j * kdim + k1];
    x5 = x_in[i1 * jdim * kdim + J1 * kdim + k];
    x6 = x_in[i * jdim * kdim + J1 * kdim + k];
    x7 = x_in[i1 * jdim * kdim + j * kdim + k];
    x8 = x_in[i * jdim * kdim + j * kdim + k];
    Y1 = y_in[i1 * jdim * kdim + J1 * kdim + k1];
    y2 = y_in[i * jdim * kdim + J1 * kdim + k1];
    y3 = y_in[i1 * jdim * kdim + j * kdim + k1];
    y4 = y_in[i * jdim * kdim + j * kdim + k1];
    y5 = y_in[i1 * jdim * kdim + J1 * kdim + k];
    y6 = y_in[i * jdim * kdim + J1 * kdim + k];
    y7 = y_in[i1 * jdim * kdim + j * kdim + k];
    y8 = y_in[i * jdim * kdim + j * kdim + k];
    z1 = z_in[i1 * jdim * kdim + J1 * kdim + k1];
    z2 = z_in[i * jdim * kdim + J1 * kdim + k1];
    z3 = z_in[i1 * jdim * kdim + j * kdim + k1];
    z4 = z_in[i * jdim * kdim + j * kdim + k1];
    z5 = z_in[i1 * jdim * kdim + J1 * kdim + k];
    z6 = z_in[i * jdim * kdim + J1 * kdim + k];
    z7 = z_in[i1 * jdim * kdim + j * kdim + k];
    z8 = z_in[i * jdim * kdim + j * kdim + k];

    xa = x2 - x1;
    xb = x3 - x1;
    xg = x5 - x1;
    xab = x4 - x3 - x2 + x1;
    xag = x6 - x5 - x2 + x1;
    xbg = x7 - x5 - x3 + x1;
    xabg = x8 - x7 - x6 + x5 - x4 + x3 + x2 - x1;

    ya = y2 - Y1;
    yb = y3 - Y1;
    yg = y5 - Y1;
    yab = y4 - y3 - y2 + Y1;
    yag = y6 - y5 - y2 + Y1;
    ybg = y7 - y5 - y3 + Y1;
    yabg = y8 - y7 - y6 + y5 - y4 + y3 + y2 - Y1;

    za = z2 - z1;
    zb = z3 - z1;
    zg = z5 - z1;
    zab = z4 - z3 - z2 + z1;
    zag = z6 - z5 - z2 + z1;
    zbg = z7 - z5 - z3 + z1;
    zabg = z8 - z7 - z6 + z5 - z4 + z3 + z2 - z1;

    while (1)
    {
        amat[0][0] = xa + xab * bb + xag * gg + xabg * (bb * gg);
        amat[0][1] = ya + yab * bb + yag * gg + yabg * (bb * gg);
        amat[0][2] = za + zab * bb + zag * gg + zabg * (bb * gg);

        amat[1][0] = xb + xab * aa + xbg * gg + xabg * (aa * gg);
        amat[1][1] = yb + yab * aa + ybg * gg + yabg * (aa * gg);
        amat[1][2] = zb + zab * aa + zbg * gg + zabg * (aa * gg);

        amat[2][0] = xg + xag * aa + xbg * bb + xabg * (aa * bb);
        amat[2][1] = yg + yag * aa + ybg * bb + yabg * (aa * bb);
        amat[2][2] = zg + zag * aa + zbg * bb + zabg * (aa * bb);

        inv3x3(amat, bmat, &istat);

        if (istat >= 1)

        /*  See if we're at the edge of the cell.  If so, move away from the 	*/
        /*  (possibly degenerate) edge and recompute the matrix.		*/

        {
            as = aa;
            bs = bb;
            gs = gg;
            if (aa == 0.)
                aa = .01;
            if (aa == 1.)
                aa = .99;
            if (bb == 0.)
                bb = .01;
            if (bb == 1.)
                bb = .99;
            if (gg == 0.)
                gg = .01;
            if (gg == 1.)
                gg = .99;
            if (aa != as || bb != bs || gg != gs)
            {
                *status = 1;
                continue;
            }

            /*   We're inside a cell and the transformation matrix is singular.	*/
            /*      Determine which directions are degenerate.			*/

            else
            {
                *status = 2;
                if (amat[0][0] == 0. && amat[0][1] == 0.
                    && amat[0][2] == 0.)
                    *idegen = *idegen + 1;
                if (amat[1][0] == 0. && amat[1][1] == 0.
                    && amat[1][2] == 0.)
                    *idegen = *idegen + 2;
                if (amat[2][0] == 0. && amat[2][1] == 0.
                    && amat[2][2] == 0.)
                    *idegen = *idegen + 4;
            } /* endif */
        } /* endif */

        return;

    } /* endwhile */
}

/*++ inv3x3
 *****************************************************************************
 * PURPOSE: Invert the 3x3 matrix A.  If A is singular, do our best to find the
 *          pseudo-inverse.
 * AUTHORS: 9/89 Written in C by Steven H. Philipson, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: STATUS = 1   A has one dependent column.
 *          = 2   A has two dependent columns.
 *          = 3   A is zero.
 * RETURNS: None
 * NOTES:   there may be problems in the port of this code to c, as both
 *          array subscripts and indexing had to be changed.  In
 *          particular, the value of "info" is suspect, as it is set in
 *          the routine "ssvdc" to a loop limit value.  It is then
 *          checked in this routine.  The values may be off by 1.  It
 *          used to report the location of singularities.  In FORTRAN,
 *          the value of zero was used to report singularities.  In C,
 *          since the first element of an array is element zero, we
 *          may be precluded from reporting this element as a singularity.
 *          Verification of this problem requires analysis of the use of
 *          this variable.  Initial analysis seems to indicate that the
 *          value contains an array bound and is only used for checking
 *          limits, and it is NOT used as an index.  Thus the code should
 *          work as orignially desired.
 *          SHP  9/7/89
 *****************************************************************************
--*/

static void inv3x3(float a[3][3], float ainv[3][3], int *status)
{
    float tmp[3][3], work[3], s[3], e[3], u[3][3], v[3][3], siu[3][3];
    float det;
    int info;

    *status = 0;

    ainv[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    ainv[1][0] = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    ainv[2][0] = a[1][0] * a[2][1] - a[1][1] * a[2][0];

    ainv[0][1] = a[0][2] * a[2][1] - a[0][1] * a[2][2];
    ainv[1][1] = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    ainv[2][1] = a[0][1] * a[2][0] - a[0][0] * a[2][1];

    ainv[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    ainv[1][2] = a[0][2] * a[1][0] - a[0][0] * a[1][2];
    ainv[2][2] = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    det = a[0][0] * ainv[0][0] + a[0][1] * ainv[1][0] + a[0][2] * ainv[2][0];

    /*   Matrix is nonsingular.  Finish up AINV.		*/

    if (det != 0.0)
    {
        det = 1. / det;
        ainv[0][0] = ainv[0][0] * det;
        ainv[0][1] = ainv[0][1] * det;
        ainv[0][2] = ainv[0][2] * det;
        ainv[1][0] = ainv[1][0] * det;
        ainv[1][1] = ainv[1][1] * det;
        ainv[1][2] = ainv[1][2] * det;
        ainv[2][0] = ainv[2][0] * det;
        ainv[2][1] = ainv[2][1] * det;
        ainv[2][2] = ainv[2][2] * det;
    }

    /*   Matrix is singular.  Do a singular value decomposition to construct*/
    /*   the pseudo-inverse.  Use LINPACK routine SSVDC.			*/

    else
    {
        memcpy((char *)tmp, (char *)a, sizeof(tmp));
        ssvdc(&tmp[0][0], 3, 3, s, e, &u[0][0], &v[0][0], work, 11, &info);
        if (s[0] == 0.0)
        {
            *status = 3;
            memset((char *)ainv, 0, sizeof(tmp));
            return;
        }

        /*              -1 T			*/
        /*   Compute V S  U .			*/

        s[0] = 1. / s[0];
        if (s[2] * s[0] < 1.e-5)
        {
            *status = 1;
            s[2] = 0.;
        }
        else
            s[2] = 1. / s[2];

        if (s[1] * s[0] < 1.e-5)
        {
            *status = 2;
            s[1] = 0.;
        }
        else
            s[1] = 1. / s[1];

        /*   Start out assuming S is a diagonal matrix.	*/

        siu[0][0] = s[0] * u[0][0];
        siu[0][1] = s[1] * u[1][0];
        siu[0][2] = s[2] * u[2][0];

        siu[1][0] = s[0] * u[0][1];
        siu[1][1] = s[1] * u[1][1];
        siu[1][2] = s[2] * u[2][1];

        siu[2][0] = s[0] * u[0][2];
        siu[2][1] = s[1] * u[1][2];
        siu[2][2] = s[2] * u[2][2];

        /*   S is upper bidiagonal, with E as the super diagonal.	*/

        if (info >= 1)
        {
            siu[0][0] = siu[0][0] - (e[0] * s[0] * s[1]) * u[0][1];
            siu[1][0] = siu[1][0] - (e[0] * s[0] * s[1]) * u[1][1];
            siu[2][0] = siu[2][0] - (e[0] * s[0] * s[1]) * u[2][1];
        }
        if (info >= 2)
        {
            siu[0][0] = siu[0][0] + (e[0] * e[1] * s[0] * pow((double)s[1], 2.) * s[2]) * u[0][2];
            siu[1][0] = siu[1][0] + (e[0] * e[1] * s[0] * pow((double)s[1], 2.) * s[2]) * u[1][2];
            siu[2][0] = siu[2][0] + (e[0] * e[1] * s[0] * pow((double)s[1], 2.) * s[2]) * u[2][2];

            siu[0][1] = siu[0][1] - (e[1] * s[1] * s[2]) * u[0][2];
            siu[1][1] = siu[1][1] - (e[1] * s[1] * s[2]) * u[1][2];
            siu[2][1] = siu[2][1] - (e[1] * s[1] * s[2]) * u[2][2];
        }

        /*               +       -1 T			*/
        /*   Finish up  A   = V S  U .			*/

        ainv[0][0] = v[0][0] * siu[0][0] + v[1][0] * siu[0][1] + v[2][0] * siu[0][2];
        ainv[0][1] = v[0][1] * siu[0][0] + v[1][1] * siu[0][1] + v[2][1] * siu[0][2];
        ainv[0][2] = v[0][2] * siu[0][0] + v[1][2] * siu[0][1] + v[2][2] * siu[0][2];

        ainv[1][0] = v[0][0] * siu[1][0] + v[1][0] * siu[1][1] + v[2][0] * siu[1][2];
        ainv[1][1] = v[0][1] * siu[1][0] + v[1][1] * siu[1][1] + v[2][1] * siu[1][2];
        ainv[1][2] = v[0][2] * siu[1][0] + v[1][2] * siu[1][1] + v[2][2] * siu[1][2];

        ainv[2][0] = v[0][0] * siu[2][0] + v[1][0] * siu[2][1] + v[2][0] * siu[2][2];
        ainv[2][1] = v[0][1] * siu[2][0] + v[1][1] * siu[2][1] + v[2][1] * siu[2][2];
        ainv[2][2] = v[0][2] * siu[2][0] + v[1][2] * siu[2][1] + v[2][2] * siu[2][2];
    }

    return;
}

/*++ ssvdc
 *****************************************************************************
 * PURPOSE: ssvdc is a subroutine to reduce a real nxp matrix x by
 *          orthogonal transformations u and v to diagonal form.  the
 *          diagonal elements s[i] are the singular values of x.  the
 *          columns of u are the corresponding left singular vectors,
 *          and the columns of v the right singular vectors.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  on entry
 *          x        *float.
 *          x contains the matrix whose singular value
 *          decomposition is to be computed.  x is
 *          destroyed by ssvdc.
 *          n        int .
 *          n is the number of rows of the matrix x.
 *          p        int .
 *          p is the number of columns of the matrix x.
 *          work     float(n).
 *          work is a scratch array.
 *          job      int .
 *          job controls the computation of the singular
 *          vectors.  it has the decimal expansion ab
 *          with the following meaning
 *          a = 0    do not compute the left singular
 *          vectors.
 *          a = 1    return the n left singular vectors
 *          in u.
 *          a > 2    return the first min(n,p) singular
 *          vectors in u.
 *          b = 0    do not compute the right singular
 *          vectors.
 *          b = 1    return the right singular vectors
 * OUTPUTS: on return
 *          s         float(mm), where mm=min(n+1,p).
 *          the first min(n,p) entries of s contain the
 *          singular values of x arranged in descending
 *          order of magnitude.
 *          e         float(p).
 *          e ordinarily contains zeros.  however see the
 *          discussion of info for exceptions.
 *          u         float.  if joba = 1 then
 *          k = n, if joba > 2 then
 *          k = min(n,p).
 *          u contains the matrix of left singular vectors.
 *          u is not referenced if joba = 0.  if n < p
 *          or if joba = 2, then u may be identified with x
 *          in the subroutine call.
 *          v         float..
 *          v contains the matrix of right singular vectors.
 *          v is not referenced if job = 0.  if p < n,
 *          then v may be identified with x in the
 *          subroutine call.
 *          info      int .
 *          the singular values (and their corresponding
 *          singular vectors) s(info+1),s(info+2),...,s(m)
 *          are correct (here m=min(n,p)).  thus if
 *          info = 0, all the singular values and their
 *          vectors are correct.  in any event, the matrix
 *          b = trans(u)*x*v is the bidiagonal matrix
 *          with the elements of s on its diagonal and the
 *          elements of e on its super-diagonal (trans(u)
 *          is the transpose of u).  thus the singular
 *          values of x and b are the same.
 * RETURNS: None
 *          PROCEDURES CALLED FROM THIS ROUTINE :
 *          srot, saxpy,sdot,sscal,sswap,snrm2,srotg
 * NOTES:   None
 *****************************************************************************
--*/

#define X_ARR(a, b) (*(x + ((a)*n) + (b)))
#define U_ARR(a, b) (*(u + ((a)*p) + (b)))
#define V_ARR(a, b) (*(v + ((a)*p) + (b)))

static void ssvdc(float *x,
                  int n, int p, float *s, float *e,
                  float *u, float *v, float *work,
                  int job, int *info)
{
    /*
      LOCAL VARIABLES
   */
    //TODO: is ls=0 the right initialization?
    int i, iter, j, jobu, k, kase, kk, l, ll, lls, lm1, lp1, ls = 0, lu, m, maxit,
                                                             mm, mm1, mp1, nct, nctp1, ncu, nrt, nrtp1;
    float b, c, cs = 0.0, el, emm1, f, g, scale_val, shift, sl, sm, sn = 0.0, smm1, t1, test,
                t, ztest;
    int wantu = 0, wantv = 0;

    /*
     set the maximum number of iterations.
   */
    maxit = 30;

    /*
     determine what is to be computed.
   */

    jobu = (job % 100) / 10;
    ncu = n;
    if (jobu > 1)
        ncu = MIN(n, p);
    if (jobu != 0)
        wantu = 1;
    if ((job % 10) != 0)
        wantv = 1;

    /*
     reduce x to bidiagonal form, storing the diagonal elements
     in s and the super-diagonal elements in e.
   */

    *info = 0;
    nct = MIN(n - 1, p);
    nrt = MAX(0, MIN(p - 2, n));
    lu = MAX(nct, nrt);
    if (lu >= 1)
    {
        for (l = 1; l <= lu; l++)
        {
            lp1 = l + 1;
            if (l <= nct)
            {
                /*
                   compute the transformation for the l-th column and
                   place the l-th diagonal in s[l].
            */
                s[l - 1] = snrm2(n - l + 1, &X_ARR(l - 1, l - 1), 1);
                if (s[l - 1] != 0.0e0)
                {
                    if (X_ARR(l - 1, l - 1) != 0.0e0)
                        s[l - 1] = SIGN(s[l - 1], X_ARR(l - 1, l - 1));
                    /* !alert +1 to +2 */
                    sscal(n - l + 1, (1.0 / s[l - 1]), &X_ARR(l - 1, l - 1), 1);
                    X_ARR(l - 1, l - 1) = 1.0e0 + X_ARR(l - 1, l - 1);
                }
                s[l - 1] = -s[l - 1];
            }
            if (p >= lp1)
            {
                for (j = lp1; j <= p; j++)
                {
                    if (l <= nct)
                    {
                        if (s[l - 1] != 0.0e0)
                        {
                            /*
                            apply the transformation.
                     */
                            t = -sdot(n - l + 1, &X_ARR(l - 1, l - 1), 1,
                                      &X_ARR(j - 1, l - 1), 1) / X_ARR(l - 1, l - 1);
                            saxpy(n - l + 1, t, &X_ARR(l - 1, l - 1), 1, &X_ARR(j - 1, l - 1), 1);
                        }
                    }
                    /*
                         place the l-th row of x into  e for the
                         subsequent calculation of the row transformation.
               */
                    e[j - 1] = X_ARR(j - 1, l - 1);
                }
            }
            if (wantu && l <= nct)
            {
                /*
                   place the transformation in u for subsequent back
                   multiplication.
            */
                for (i = 1; i <= n; i++)
                    U_ARR(l - 1, i - 1) = X_ARR(l - 1, i - 1);
            }
            if (l <= nrt)
            {
                /*
                   compute the l-th row transformation and place the
                   l-th super-diagonal in e[l].
            */
                e[l - 1] = snrm2(p - l, &e[lp1 - 1], 1);
                if (e[l - 1] != 0.0e0)
                {
                    if (e[lp1 - 1] != 0.0e0)
                        e[l - 1] = SIGN(e[l - 1], e[lp1 - 1]);
                    sscal(p - l, 1.0e0 / e[l - 1], &e[lp1 - 1], 1);
                    e[lp1 - 1] = 1.0e0 + e[lp1 - 1];
                }
                e[l - 1] = -e[l - 1];
                if (lp1 <= n && e[l - 1] != 0.0e0)
                {
                    /*
                        apply the transformation.
                     */
                    for (i = lp1; i <= n; i++)
                        work[i - 1] = 0.0e0;
                    for (j = lp1; j <= p; j++)
                        saxpy(n - 1, e[j - 1], &X_ARR(j - 1, lp1 - 1), 1, &work[lp1 - 1], 1);
                    for (j = lp1; j <= p; j++)
                        saxpy(n - l, -e[j - 1] / e[lp1 - 1],
                              &work[lp1 - 1], 1, &X_ARR(j - 1, lp1 - 1), 1);
                }
                if (wantv)
                {
                    /*
                      place the transformation in v for subsequent
                      back multiplication.
               */
                    for (i = lp1; i <= p; i++)
                        V_ARR(l - 1, i - 1) = e[i - 1];
                }
            }
        } /* end of 'for loop' */
    } /* end of 'if lu' */

    /*
     set up the final bidiagonal matrix or order m.
   */

    m = MIN(p, n + 1);
    nctp1 = nct + 1;
    nrtp1 = nrt + 1;
    if (nct < p)
        s[nctp1 - 1] = X_ARR(nctp1 - 1, nctp1 - 1);
    if (n < m)
        s[m - 1] = 0.0e0;
    if (nrtp1 < m)
        e[nrtp1 - 1] = X_ARR(m - 1, nrtp1 - 1);
    e[m - 1] = 0.0e0;

    /*
     if required, generate u.
   */

    if (wantu)
    {
        if (ncu >= nctp1)
        {
            for (j = nctp1; j < ncu; j++)
            {
                for (i = 1; i <= n; i++)
                    U_ARR(j - 1, i - 1) = 0.0e0;
                U_ARR(j - 1, j - 1) = 1.0e0;
            }
        }
        if (nct >= 1)
        {
            for (ll = 1; ll <= nct; ll++)
            {
                l = nct - ll + 1;
                if (s[l - 1] != 0.0e0)
                {
                    lp1 = l + 1;
                    if (ncu >= lp1)
                    {
                        for (j = lp1; j <= ncu; j++)
                        {
                            t = -sdot(n - l + 1, &U_ARR(l - 1, l - 1), 1,
                                      &U_ARR(j - 1, l - 1), 1) / U_ARR(l - 1, l - 1);
                            saxpy(n - l + 1, t, &U_ARR(l - 1, l - 1), 1, &U_ARR(j - 1, l - 1), 1);
                        }
                    }
                    sscal(n - l + 1, -1.0e0, &U_ARR(l - 1, l - 1), 1);
                    U_ARR(l - 1, l - 1) = 1.0e0 + U_ARR(l - 1, l - 1);
                    lm1 = l - 1;
                    if (lm1 >= 1)
                    {
                        for (i = 1; i <= lm1; i++)
                            U_ARR(l - 1, i - 1) = 0.0e0;
                    }
                    continue;
                }
                for (i = 1; i <= n; i++)
                    U_ARR(l - 1, i - 1) = 0.0e0;
                U_ARR(l - 1, l - 1) = 1.0e0;
            }
        }

        /*
        if it is required, generate v.
      */

        if (wantv)
        {
            for (ll = 1; ll <= p; ll++)
            {
                l = p - ll + 1;
                lp1 = l + 1;
                if (l <= nrt)
                {
                    if (e[l - 1] != 0.0e0)
                    {
                        for (j = lp1; j <= p; j++)
                        {
                            t = -sdot(p - l, &V_ARR(l - 1, lp1 - 1), 1,
                                      &V_ARR(j - 1, lp1 - 1), 1) / V_ARR(l - 1, lp1 - 1);
                            saxpy(p - l, t, &V_ARR(l - 1, lp1 - 1), 1, &V_ARR(j - 1, lp1 - 1), 1);
                        }
                    }
                }
                for (i = 1; i <= p; i++)
                    V_ARR(l - 1, i - 1) = 0.0e0;
                V_ARR(l - 1, l - 1) = 1.0e0;
            }
        }

        /*
        main iteration loop for the singular values.
      */

        mm = m;
        iter = 0;
        while (1)
        {
            /*
                quit if all the singular values have been found.
         */
            if (m == 0)
                break;

            /*
                if too many iterations have been performed, set
                flag and return.
         */

            if (iter >= maxit)
            {
                *info = m;
                /*
            exit
            */
                break;
            }
            /*
                this section of the program inspects for
                negligible elements in the s and e arrays.  on
                completion the variables kase and l are set as follows.

                   kase = 1     if s[m) and e(l-1) are negligible and l.lt.m
                   kase = 2     if s[l) is negligible and l.lt.m
                   kase = 3     if e(l-1) is negligible, l.lt.m, and
                                s[l), ..., s[m) are not negligible (qr step).
                   kase = 4     if e(m-1) is negligible (convergence).
         */

            //TODO: since n and p are both 3, m is set to 3. Thus the following loop is
            //executed at least once and l is initialized there. The initialization of l
            //before the loop is only for gcc.
            //But: What if n and p have other values?

            assert(m > 1);
            l = m - 1;
            for (ll = 1; ll < m; ll++)
            {
                l = m - ll;
                test = ABS(s[l - 1]) + ABS(s[l + 1 - 1]);
                ztest = test + ABS(e[l - 1]);
                if (ztest == test)
                {
                    e[l - 1] = 0.0e0;
                    /*
                 exit
               */
                    break;
                }
            }
            while (1)
            {
                if (l == m - 1)
                {
                    kase = 4;
                    break;
                }
                lp1 = l + 1;
                mp1 = m + 1;
                for (lls = lp1; lls <= mp1; lls++)
                {
                    ls = m - lls + lp1;
                    /*
                      exit
               */
                    if (ls == l)
                        break;
                    test = 0.0e0;
                    if (ls != m)
                        test = test + ABS(e[ls - 1]);
                    if (ls != l + 1)
                        test += ABS(e[ls - 2]);
                    ztest = test + ABS(s[ls - 1]);
                    if (ztest == test)
                    {
                        s[ls - 1] = 0.0e0;
                        /*
                         exit
                  */
                        break;
                    }
                }
                if (ls == l)
                {
                    kase = 3;
                    break;
                }
                if (ls == m)
                {
                    kase = 1;
                    break;
                }
                kase = 2;
                l = ls;
                break;
            } /* end of while */
            l = l + 1;

            /*
                perform the task indicated by kase.
         */

            switch (kase)
            {

            /*
                   deflate negligible s[m].
            */
            case 1:
                mm1 = m - 1;
                f = e[m - 2];
                e[m - 2] = 0.0e0;
                for (kk = l; kk <= mm1; kk++)
                {
                    k = mm1 - kk + l;
                    t1 = s[k - 1];
                    srotg(t1, f, cs, sn);
                    s[k - 1] = t1;
                    if (k != l)
                    {
                        f = -sn * e[k - 2];
                        e[k - 2] = cs * e[k - 2];
                    }
                    if (wantv)
                        srot(p, &V_ARR(k - 1, 0), 1, &V_ARR(m - 1, 0), 1, cs, sn);
                }
                continue;
            //break;

            /*
                      split at negligible s[l].
               */

            case 2:
                f = e[l - 2];
                e[l - 2] = 0.0e0;
                for (k = l; k <= m; k++)
                {
                    t1 = s[k - 1];
                    srotg(t1, f, cs, sn);
                    s[k - 1] = t1;
                    f = -sn * e[k - 1];
                    e[k - 1] = cs * e[k - 1];
                    if (wantu)
                        srot(n, &U_ARR(k - 1, 0), 1, &U_ARR(l - 2, 0), 1, cs, sn);
                }
                continue;
            //break;

            /*
                      perform one qr step.
               */
            case 3:

                /*
                      calculate the shift.
               */
                scale_val = MAX(ABS(s[m - 1]), ABS(s[m - 2]));
                scale_val = MAX(scale_val, ABS(e[m - 2]));
                scale_val = MAX(scale_val, ABS(s[l - 1]));
                scale_val = MAX(scale_val, ABS(e[l - 1]));

                sm = s[m - 1] / scale_val;
                smm1 = s[m - 2] / scale_val;
                emm1 = e[m - 2] / scale_val;
                sl = s[l - 1] / scale_val;
                el = e[l - 1] / scale_val;
                b = ((smm1 + sm) * (smm1 - sm) + pow(emm1, 2)) / 2.0e0;
                c = pow(sm * emm1, 2);
                shift = 0.0e0;
                if (b != 0.0e0 || c != 0.0e0)
                {
                    shift = sqrt(pow(b, 2) + c);
                    if (b < 0.0e0)
                        shift = -shift;
                    shift = c / (b + shift);
                }
                f = (sl + sm) * (sl - sm) + shift;
                g = sl * el;

                /*
                      chase zeros.
               */

                mm1 = m - 1;
                for (k = l; k <= mm1; k++)
                {
                    srotg(f, g, cs, sn);
                    if (k != l)
                        e[k - 2] = f;
                    f = cs * s[k - 1] + sn * e[k - 1];
                    e[k - 1] = cs * e[k - 1] - sn * s[k - 1];
                    g = sn * s[k + 1 - 1];
                    s[k + 1 - 1] = cs * s[k + 1 - 1];
                    if (wantv)
                        srot(p, &V_ARR(k - 1, 0), 1, &V_ARR(k + 1 - 1, 0), 1, cs, sn);
                    srotg(f, g, cs, sn);
                    s[k - 1] = f;
                    f = cs * e[k - 1] + sn * s[k + 1 - 1];
                    s[k + 1 - 1] = -sn * e[k - 1] + cs * s[k + 1 - 1];
                    g = sn * e[k + 1 - 1];
                    e[k + 1 - 1] = cs * e[k + 1 - 1];
                    if (wantu && k < n)
                        srot(n, &U_ARR(k - 1, 0), 1, &U_ARR(k + 1 - 1, 0), 1, cs, sn);
                }
                e[m - 2] = f;
                iter++;
                continue;
            //break;

            /*
                      convergence.
               */
            case 4:
                /*
                      make the singular value  positive.
               */
                if (s[l - 1] >= 0.0e0)
                {
                    s[l - 1] = -s[l - 1];
                    if (wantv)
                        sscal(p, -1.0e0, &V_ARR(l - 1, 0), 1);
                }

                /*
                      order the singular value.
               */
                while (l != mm)
                {
                    /*
                              exit
                  */
                    if (s[l - 1] >= s[l + 1 - 1])
                        break;
                    t = s[l - 1];
                    s[l - 1] = s[l + 1 - 1];
                    s[l + 1 - 1] = t;
                    if (wantv && l < p)
                        sswap(p, &V_ARR(l - 1, 0), 1, &V_ARR(l + 1 - 1, 0), 1);
                    if (wantu && l < n)
                        sswap(n, &U_ARR(l - 1, 0), 1, &U_ARR(l + 1 - 1, 0), 1);
                    l++;
                }
                iter = 0;
                m--;
            } /* end of switch */
        }
    }
}

/*++ srot
 *****************************************************************************
 * PURPOSE: this routine applies a plane rotation.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:
 *****************************************************************************
 --*/

static void srot(int n, float *sx, int incx, float *sy,
                 int incy, float c, float s)
{
    float stemp;
    int i, ix, iy;
    if (n < 0)
        return;

    if (incx == 1 && incy == 1)
    {
        /*
            code for both increments equal to 1
      */
        for (i = 0; i < n; i++)
        {
            stemp = c * sx[i] + s * sy[i];
            sy[i] = c * sy[i] - s * sx[i];
            sx[i] = stemp;
        }
        return;
    }
    /*
     code for unequal increments or equal increments not equal to 1
   */
    ix = 1;
    iy = 1;
    if (incx < 0)
        ix = (-n + 1) * incx + 1;
    if (incy < 0)
        iy = (-n + 1) * incy + 1;
    for (i = 0; i < n; i++)
    {
        stemp = c * sx[ix] + s * sy[iy];
        sy[iy] = c * sy[iy] - s * sx[ix];
        sx[ix] = stemp;
        ix = ix + incx;
        iy = iy + incy;
    }
}

/*++ srotg
 *****************************************************************************
 * PURPOSE: This routine constructs given plane rotation.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   Uses unrolled loops for increment equal to 1.
 *****************************************************************************
 --*/

static void srotg(float sa, float sb, float c, float s)
{
    float roe, scale_val, r, z;

    roe = sb;
    if (ABS(sa) > ABS(sb))
        roe = sa;
    scale_val = ABS(sa) + ABS(sb);
    if (scale_val != 0.0)
    {
        r = scale_val * sqrt(pow((double)(sa / scale_val), 2.0) + pow((double)(sb / scale_val), 2.0));
        r = SIGN(1.0, roe) * r;
        c = sa / r;
        s = sb / r;
    }
    else
    {
        c = 1.0;
        s = 0.0;
        r = 0.0;
    }
    z = 1.0;
    if (ABS(sa) > ABS(sb))
        z = s;
    if (ABS(sb) >= ABS(sa) && c != 0.0)
        z = 1.0 / c;
    sa = r;
    sb = z;
}

/*++ sscal
 *****************************************************************************
 * PURPOSE: sscal scales a vector by a constant.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   Uses unrolled loops for increment equal to 1.
 *****************************************************************************
 --*/

static void sscal(int n, float sa, float *sx, int incx)
{
    int i, m, mp1, nincx;
    if (n < 0)
        return;
    if (incx == 1)
    {
        /*
             increment equal to 1
      */
        m = n % 5;
        if (m == 0)
            ;
        else
        {
            for (i = 0; i < m; i++)
                sx[i] = sa * sx[i];
            if (n < 5)
                return;
        }

        mp1 = m + 1;
        for (i = mp1; i < n; i += 5)
        {
            sx[i] = sa * sx[i];
            sx[i + 1] = sa * sx[i + 1];
            sx[i + 2] = sa * sx[i + 2];
            sx[i + 3] = sa * sx[i + 3];
            sx[i + 4] = sa * sx[i + 4];
        }
    }
    else
    {
        /*
             code for increment not equal to 1
      */
        nincx = n * incx;
        for (i = 0; i < nincx; i += incx)
            sx[i] = sa * sx[i];
    }
}

/*++ sswap
 *****************************************************************************
 * PURPOSE: Interchanges two vectors.
 *          uses unrolled loops for increments equal to 1.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

static void sswap(int n, float *sx, int incx, float *sy, int incy)
{
    float stemp;
    int i, ix, iy, m, mp1;
    if (n < 0)
        return;
    if (incx == 1 && incy == 1)
        ;
    else
    {
        /*
        code for unequal increments or equal increments not equal to 1
      */
        ix = 1;
        iy = 1;
        if (incx < 0)
            ix = (-n + 1) * incx + 1;
        if (incy < 0)
            iy = (-n + 1) * incy + 1;
        for (i = 0; i < n; i++)
        {
            stemp = sx[ix];
            sx[ix] = sy[iy];
            sy[iy] = stemp;
            ix = ix + incx;
            iy = iy + incy;
        }
        return;
    }
    /*
     code for both increments equal to 1
     clean-up loop
   */
    m = n % 3;
    if (m == 0)
        ;
    else
    {
        for (i = 0; i < m; i++)
        {
            stemp = sx[i];
            sx[i] = sy[i];
            sy[i] = stemp;
        }
        if (n < 3)
            return;
    }
    mp1 = m + 1;
    for (i = mp1; i < n; i += 3)
    {
        stemp = sx[i];
        sx[i] = sy[i];
        sy[i] = stemp;
        stemp = sx[i + 1];
        sx[i + 1] = sy[i + 1];
        sy[i + 1] = stemp;
        stemp = sx[i + 2];
        sx[i + 2] = sy[i + 2];
        sy[i + 2] = stemp;
    }
}

/*++ saxpy
 *****************************************************************************
 * PURPOSE: Constant times a vector plus a vector.
 *          uses unrolled loop for increments equal to one.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   None
 *****************************************************************************
--*/

static void saxpy(int n, float sa, float *sx, int incx, float *sy, int incy)
{
    int i, ix, iy, m, mp1;
    if (n < 0)
        return;
    if (sa == 0.0)
        return;
    if (incx == 1 && incy == 1)
        ;
    else
    {
        /*
         code for unequal increments or equal increments
         not equal to 1
      */
        ix = 1;
        iy = 1;
        if (incx < 0)
            ix = (-n + 1) * incx + 1;
        if (incy < 0)
            iy = (-n + 1) * incy + 1;
        for (i = 0; i < n; i++)
        {
            sy[iy] = sy[iy] + sa * sx[ix];
            ix = ix + incx;
            iy = iy + incy;
        }
        return;
    }
    /*
      code for both increments equal to 1
      clean-up loop
   */
    m = n % 4;
    if (m == 0)
        ;
    else
    {
        for (i = 0; i < m; i++)
            sy[i] = sy[i] + sa * sx[i];
        if (n < 4)
            return;
    }
    mp1 = m + 1;
    for (i = mp1; i < n; i += 4)
    {
        sy[i] = sy[i] + sa * sx[i];
        sy[i + 1] = sy[i + 1] + sa * sx[i + 1];
        sy[i + 2] = sy[i + 2] + sa * sx[i + 2];
        sy[i + 3] = sy[i + 3] + sa * sx[i + 3];
    }
}

/*++ sdot
 *****************************************************************************
 * PURPOSE: Forms the dot product of two vectors.
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   uses unrolled loops for increments equal to one.
 *****************************************************************************
 --*/

static float sdot(int n, float *sx, int incx, float *sy, int incy)
{
    float stemp = 0.0;
    int i, ix, iy, m, mp1;
    if (n < 0)
        return (0.0);
    if (incx != 1 || incy != 1)
    {
        /*
         code for unequal increments or equal increments not equal to 1
      */
        ix = 1;
        iy = 1;
        if (incx < 0)
            ix = (-n + 1) * incx + 1;
        if (incy < 0)
            iy = (-n + 1) * incy + 1;
        for (i = 0; i < n; i++)
        {
            stemp += sx[ix] * sy[iy];
            ix = ix + incx;
            iy = iy + incy;
        }
        return (stemp);
    }
    /*
      code for both increments equal to 1
      clean-up loop
   */
    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            stemp = stemp + sx[i] * sy[i];
        if (n < 5)
        {
            return (stemp);
        }
    }
    mp1 = m + 1;
    for (i = mp1; i < n; i += 5)
    {
        stemp += sx[i] * sy[i] + sx[i + 1] * sy[i + 1] + sx[i + 2] * sy[i + 2] + sx[i + 3] * sy[i + 3] + sx[i + 4] * sy[i + 4];
    }
    return (stemp);
}

/*++ snrm2
 *****************************************************************************
 * AUTHORS: 9/89 Written in C by John M. Semans, NASA Ames/Sterling Software
 * HISTORY: None
 * INPUTS:  None
 * OUTPUTS: None
 * RETURNS: None
 * NOTES:   euclidean norm of the n-vector stored in sx[] with storage increment incx .
 *          if    n .le. 0 return with result = 0.
 *          if n .ge. 1 then incx must be .ge. 1
 *          four phase method     using two built-in constants that are
 *          hopefully applicable to all machines.
 *          cutlo = maximum of  sqrt(u/eps)  over all known machines.
 *          cuthi = minimum of  sqrt(v)      over all known machines.
 *          where
 *          eps = smallest no. such that eps + 1. .gt. 1.
 *          u   = smallest positive no.   (underflow limit)
 *          v   = largest  no.            (overflow  limit)
 *          brief outline of algorithm..
 *          phase 1    scans zero components.
 *          move to phase 2 when a component is nonzero and .le. cutlo
 *          move to phase 3 when a component is .gt. cutlo
 *          move to phase 4 when a component is .ge. cuthi/m
 *          where m = n for x() real and m = 2*n for complex.
 *          values for cutlo and cuthi..
 *          from the environmental parameters listed in the imsl converter
 *          document the limiting values are as follows..
 *          cutlo, s.p.   u/eps = 2**(-102) for  honeywell.  close seconds are
 *          univac and dec at 2**(-103)
 *          thus cutlo = 2**(-51) = 4.44089e-16
 *          cuthi, s.p.   v = 2**127 for univac, honeywell, and dec.
 *          thus cuthi = 2**(63.5) = 1.30438e19
 *          cutlo, d.p.   u/eps = 2**(-67) for honeywell and dec.
 *          thus cutlo = 2**(-33.5) = 8.23181d-11
 *          cuthi, d.p.   same as s.p.  cuthi = 1.30438d19
 *          data cutlo, cuthi / 8.232d-11,  1.304d19 /
 *          data cutlo, cuthi / 4.441e-16,  1.304e19 /
 *          data cutlo, cuthi / 4.441e-16,  1.304e19 /
 * NOTES:
 *****************************************************************************
--*/

static float snrm2(int n, float *sx, int incx)
{
    int next, nn;
    float cutlo = 4.441e-16, cuthi = 1.304e19, hitest, xmax = 0;
    float sum;
    float zero = 0.0, one = 1.0;
    register int i, j;

    if (n <= 0)
        return (zero);

    next = 30;
    sum = zero;
    nn = n * incx;

    /*
     begin main loop
   */

    i = 0;
    while (i < nn)
    {
        switch (next)
        {
        case 30:
            if (ABS(sx[i]) > cutlo)
            {
                hitest = cuthi / (float)(n);

                /*
                      phase 3.  sum is mid-range.  no scaling.
               */

                for (j = i; j < nn; j += incx)
                {
                    if (ABS(sx[j]) >= hitest)
                    {
                        i = j;
                        next = 110;
                        sum = (sum / sx[i]) / sx[i];
                        xmax = ABS(sx[i]);
                        sum += pow(sx[i] / xmax, 2);
                        i += incx;
                        continue;
                    }
                    sum += pow((double)sx[j], 2.);
                }
#ifdef __sgi
                return (fsqrt(sum));
#else
                return (sqrt(sum));
#endif
            }
            next = 50;
            xmax = zero;
            break;

        case 50:
            /*
              phase 1.  sum is zero
            */

            if (sx[i] == zero)
            {
                i += incx;
                continue;
            }
            if (ABS(sx[i]) > cutlo)
            {
                hitest = cuthi / (float)(n);

                /*
                      phase 3.  sum is mid-range.  no scaling.
               */

                for (j = i; j < nn; j += incx)
                {
                    if (ABS(sx[j]) >= hitest)
                    {
                        i = j;
                        next = 110;
                        sum = (sum / sx[i]) / sx[i];
                        xmax = ABS(sx[i]);
                        sum += pow((sx[i] / xmax), 2);
                        i += incx;
                        continue;
                    }
                    sum += pow((double)sx[j], 2.);
                }
                return sqrt(sum);
            }

            /*
              prepare for phase 2.
            */

            next = 70;
            xmax = ABS(sx[i]);
            sum += pow((sx[i] / xmax), 2);
            i += incx;
            break;

        case 70:

            /*
              phase 2.  sum is small.
              scale to avoid destructive underflow.
            */

            if (ABS(sx[i]) > cutlo)
            {

                /*
                 prepare for phase 3
               */

                sum = (sum * xmax) * xmax;

                /*
                 for real or d.p. set hitest = cuthi/n
                 for complex      set hitest = cuthi/(2*n)
               */

                hitest = cuthi / (float)(n);

                /*
                 phase 3.  sum is mid-range.  no scaling.
               */

                for (j = i; j < nn; j += incx)
                {
                    if (ABS(sx[j]) >= hitest)
                    {
                        i = j;
                        next = 110;
                        sum = (sum / sx[i]) / sx[i];
                        xmax = ABS(sx[i]);
                        sum += pow((sx[i] / xmax), 2);
                        i += incx;
                        continue;
                    }
                    sum += pow(sx[j], 2);
                }
                return (sqrt(sum));
            }

            next = 110;
            break;

        case 110:

            /*
                        common code for phases 2 and 4.
                        in phase 4 sum is large.  scale to avoid overflow.
            */

            if (ABS(sx[i]) > xmax)
            {
                sum = one + sum * pow(xmax / sx[i], 2);
                xmax = ABS(sx[i]);
                i += incx;
                continue;
            }
            sum += pow(sx[i] / xmax, 2);
            i += incx;
            break;
        }
    } /* end of while */

    /*
        compute square root and adjust for scaling.
   */
    return (xmax * sqrt(sum));
}

int ExpandSetList(coDistributedObject *const *objects, int h_many, const char *type,
                  ia<coDistributedObject *> &out_array)
{
    int i, j;
    int count = 0;
    const char *obj_type;
    for (i = 0; i < h_many; ++i)
    {
        obj_type = objects[i]->getType();
        if (strcmp(obj_type, type) == 0)
        {
            out_array[count++] = objects[i];
        }
        else if (strcmp(obj_type, "SETELE") == 0)
        {
            ia<coDistributedObject *> partial;
            coDistributedObject *const *in_set_elements;
            int in_set_len;
            in_set_elements = ((coDoSet *)(objects[i]))->getAllElements(&in_set_len);
            int add = ExpandSetList(in_set_elements, in_set_len, type, partial);
            for (j = 0; j < add; ++j)
            {
                out_array[count++] = partial[j];
            }
        }
    }
    return count;
}

float dout(const float vel[3], int whatout)
{
    switch (whatout)
    {
    case Application::V_MAG:
        return sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
    case Application::V_X:
        return vel[0];
    case Application::V_Y:
        return vel[1];
    case Application::V_Z:
        return vel[2];
    case Application::NUMBER:
        return trace_num;
    default:
        Covise::sendInfo("Incorrect option for parameter whatout: using number");
        return trace_num;
    }
}

//===============================================================================================
//===============================================================================================
//===============================================================================================
//===============================================================================================

WristWatch::WristWatch()
{
}

WristWatch::~WristWatch()
{
}

void WristWatch::start()
{
#ifndef _WIN32
    gettimeofday(&myClock, NULL);
#endif
    return;
}

void WristWatch::stop(char *s)
{
#ifndef _WIN32
    timeval now;
    float dur;

    gettimeofday(&now, NULL);
    dur = ((float)(now.tv_sec - myClock.tv_sec)) + ((float)(now.tv_usec - myClock.tv_usec)) / 1000000.0;
    // sk: 11.04.2001
    char info_buf[200];
    // fprintf( stderr, "%s took %6.3f seconds\n", s, dur );
    sprintf(info_buf, "%s took %6.3f seconds.", s, dur);
    Covise::sendInfo(info_buf);
#endif
    return;
}
