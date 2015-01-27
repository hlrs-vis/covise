/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE Tracer application module                         **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "Tracer.h"

//

extern "C" {
float gb_(int *, int *, int *, int *);
float vb_(int *, int *, int *, int *);
int lb_(void);
void geob_(int *, int *, int *, int *, int *, int *);
void varb_(int *, int *, int *, int *, int *, int *);
void limb_(int *, int *, int *, int *);
void slinm_(int *, int *, int *, float[][3], float[][3], int[][3],
            int[], int[], int *, int *, float *, int *, float *, int *, int *, int *);
}

void tracer();

int istart, istep, npmax, dim[3];
float start1[3], start2[3], step;
float dl, delta;
float *u, *v, *w, *x, *y, *z;
char *color;
coDoVec3 *velobj;

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

//
// static stub callback functions calling the real class
// member functions
//

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

//
//
//..........................................................................
//
//

// Description file :MODULE Tracer
//CATEGORY Mapper
//DESCRIPTION "Trace particles in flow fields"
//INPUT mesh "coDoRectilinearGrid|coDoStructuredGrid" "mesh" req
//INPUT velocity coDoVec3 "velocity" req
//OUTPUT lines coDoLines "particle lines" default
//PARIN istart Choice "starting mode" "1 IJK XYZ"
//PARIN istep Choice "step type" "2 space time"
//PARIN dl Scalar "step value (it depends on istep)" .02
//PARIN delta Scalar "angular tolerance (used if > 0)" .1
//PARIN npmax Scalar "max no. of points to be generated" 1000
//PARIN start1 Vector "starting position" "3 8 11"
//PARIN start2 Vector "starting position" "4 25 12"
//PARIN step Scalar "step between starting positions" 1

//
//
//..........................................................................
//
//

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

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
void Application::compute(void *)
{
    int i, j, k, np, ip;
    int x_1, x_2, y_1, y_2, z_1, z_2;
    float *xr, *yr, *zr;
    char *objtype;
    char *Mesh, *Veloc;

    coDistributedObject *tmp_obj, *data_obj;
    coDoRectilinearGrid *rect;
    coDoUniformGrid *unigrd;
    coDoStructuredGrid *structur;

    //	get input data object names

    Mesh = Covise::get_object_name("mesh");
    if (Mesh == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'mesh'");
        return;
    }
    Veloc = Covise::get_object_name("velocity");
    if (Veloc == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'velovity'");
        return;
    }

    // 	read input parameters

    Covise::get_choice_param("istart", &istart);
    Covise::get_choice_param("istep", &istep);
    Covise::get_scalar_param("dl", &dl);
    Covise::get_scalar_param("delta", &delta);
    Covise::get_scalar_param("istep", &istep);
    Covise::get_scalar_param("npmax", &npmax);
    Covise::get_scalar_param("step", &step);
    for (i = 0; i < 3; i++)
        Covise::get_vector_param("start1", i, &start1[i]);
    for (i = 0; i < 3; i++)
        Covise::get_vector_param("start2", i, &start2[i]);
    Covise::get_string_param("color", &color);

    //	retrieve velocity object from shared memeory

    velobj = new coDoVec3(Veloc);
    if (!velobj->objectOk())
    {
        Covise::sendError("ERROR: Data object 'velocity' can't be accessed in shared memory");
        return;
    }
    velobj->getGridSize(&x_2, &y_2, &z_2);
    velobj->getAddresses(&u, &v, &w);

    //	retrieve mesh object from shared memeory

    tmp_obj = new coDistributedObject(Mesh);
    data_obj = tmp_obj->createUnknown();
    if (data_obj != 0L)
    {
        objtype = data_obj->getType();
        if (strcmp(objtype, "UNIGRD") == 0)
        {
            unigrd = (coDoUniformGrid *)data_obj;
            unigrd->getGridSize(&x_1, &y_1, &z_1);
            if ((x_1 != x_2) || (y_1 != y_2) || (z_1 != z_2))
            {
                Covise::sendError("ERROR: Objects have different dimensions");
                return;
            }
            x = xr = new float[x_1 * y_1 * z_1];
            y = yr = new float[x_1 * y_1 * z_1];
            z = zr = new float[x_1 * y_1 * z_1];
            if (x == NULL || y == NULL || z == NULL)
            {
                Covise::sendError("ERROR: out of memory in conversion uniform -> structured grid");
                return;
            }
            for (i = 0; i < x_1; i++)
                for (j = 0; j < y_1; j++)
                    for (k = 0; k < z_1; k++)
                        unigrd->getPointCoordinates(i, xr++, j, yr++, k, zr++);
        }
        else if (strcmp(objtype, "RCTGRD") == 0)
        {
            rect = (coDoRectilinearGrid *)data_obj;
            rect->getGridSize(&x_1, &y_1, &z_1);
            if ((x_1 != x_2) || (y_1 != y_2) || (z_1 != z_2))
            {
                Covise::sendError("ERROR: Objects have different dimensions");
                return;
            }
            rect->getAddresses(&xr, &yr, &zr);
            np = x_1 * y_1 * z_1;
            x = new float[np];
            y = new float[np];
            z = new float[np];
            ip = 0;
            for (i = 0; i < x_1; i++)
            {
                for (j = 0; j < y_1; j++)
                {
                    for (k = 0; k < z_1; k++)
                    {
                        x[ip] = xr[i];
                        y[ip] = yr[j];
                        z[ip] = zr[k];
                        ip++;
                    }
                }
            }
        }
        else if (strcmp(objtype, "STRGRD") == 0)
        {
            structur = (coDoStructuredGrid *)data_obj;
            structur->getGridSize(&x_1, &y_1, &z_1);
            if ((x_1 != x_2) || (y_1 != y_2) || (z_1 != z_2))
            {
                Covise::sendError("ERROR: Objects have different dimensions");
                return;
            }
            structur->getAddresses(&x, &y, &z);
        }

        else
        {
            Covise::sendError("ERROR: Data object 'mesh' has wrong data type");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'mesh' can't be accessed in shared memory");
        return;
    }

    //	start the tracer

    dim[0] = x_2;
    dim[1] = y_2;
    dim[2] = z_2;
    tracer();

    if ((strcmp(objtype, "RCTGRD") == 0) || (strcmp(objtype, "UNIGRD") == 0))
    {
        delete[] x;
        delete[] y;
        delete[] z;
    }
}

//  ===========================================================================
//
//  Here the particle tracer
//
//  ===========================================================================

void tracer()
{

    coDoLines *lineobj;
    char *Line;
    char buf[300];
    char *COLOR = "COLOR";

    const int START_IJK = 1;
    const int START_XYZ = 2;
    const int MAXDX = 3;
    const int MAXDXC = MAXDX;
    const int MAXPTS = 3000;
    const int MAXNIS = 99;

    int ic[MAXPTS][MAXDXC], ibp[MAXPTS];
    float xp[MAXPTS][MAXDX], xc[MAXPTS][MAXDXC];
    int nismax = MAXNIS, ndx = MAXDX, ndxc = MAXDXC;
    int ivel[MAXDX] = { 1, 2, 3 };
    int np, irtn = 0, iboc = 0, nptot = 0, npoints = 0;
    register int i, j;

    //	get output data object	names
    Line = Covise::get_object_name("lines");
    if (Line == NULL)
    {
        Covise::sendError("Object name not correct for 'lines'");
        return;
    }

    ibp[0] = 1;
    /* force the PT to start the particle
            from a computational point in the
            first block */

    int npi = (int)((start2[0] - start1[0]) / step) + 1;
    int npj = (int)((start2[1] - start1[1]) / step) + 1;
    int npk = (int)((start2[2] - start1[2]) / step) + 1;
    nptot = npj * npi * npk;

    float **xptmp = (float **)malloc(nptot * sizeof(float *));
    float **yptmp = (float **)malloc(nptot * sizeof(float *));
    float **zptmp = (float **)malloc(nptot * sizeof(float *));
    int *nptmp = new int[nptot];
    int currp = 0;

    for (float fk = start1[2]; fk <= start2[2]; fk += step)
        for (float fj = start1[1]; fj <= start2[1]; fj += step)
            for (float fi = start1[0]; fi <= start2[0]; fi += step)
            {
                switch (istart)
                {
                case START_IJK:
                    ic[0][0] = (int)fi;
                    ic[0][1] = (int)fj;
                    ic[0][2] = (int)fk;
                    xc[0][0] = fi - (float)ic[0][0];
                    xc[0][1] = fj - (float)ic[0][1];
                    xc[0][2] = fk - (float)ic[0][2];
                    break;
                case START_XYZ:
                    xp[0][0] = fk;
                    xp[0][1] = fj;
                    xp[0][2] = fi;
                    break;
                }

                slinm_(&ndx, &ndxc, &istart,
                       xp, xc, ic, ibp,
                       ivel, &iboc,
                       &istep, &dl, &npmax, &delta, &nismax, &np, &irtn);
                if (irtn != 0)
                {
                    sprintf(buf, "WARNING: Error no. %d in particle tracer \n", irtn);
                    Covise::sendWarning(buf);
                }

                else
                {
                    xptmp[currp] = new float[np];
                    yptmp[currp] = new float[np];
                    zptmp[currp] = new float[np];
                    for (i = 0; i < np; i++)
                    {
                        xptmp[currp][i] = xp[i][0];
                        yptmp[currp][i] = xp[i][1];
                        zptmp[currp][i] = xp[i][2];
                    }
                    nptmp[currp] = np;
                    npoints += np;
                    currp++;
                }
            }

    //	Prepare data to fill a coDoLines object
    sprintf(buf, "INFORMATION: No. of lines : %d \n", nptot);
    Covise::sendWarning(buf);
    sprintf(buf, "INFORMATION: No. of points : %d \n", npoints);
    Covise::sendWarning(buf);

    if (currp == 0)
    {
        Covise::sendWarning(" No particle lines found,  Block may be empty");
        lineobj = new coDoLines(Line, 0, 0, 0);
    }

    else
    {
        int *vlist = new int[npoints];
        for (i = 0; i < npoints; i++)
        {
            vlist[i] = i;
        }
        int *llist = new int[currp];
        float *xv = new float[npoints];
        float *yv = new float[npoints];
        float *zv = new float[npoints];
        int ind = 0;
        for (i = 0; i < currp; i++)
        {
            llist[i] = (i == 0 ? 0 : llist[i - 1] + nptmp[i - 1]);
            for (j = 0; j < nptmp[i]; j++)
            {
                xv[ind] = xptmp[i][j];
                yv[ind] = yptmp[i][j];
                zv[ind] = zptmp[i][j];
                ind++;
            }
        }

        lineobj = new coDoLines(Line, npoints, xv, yv, zv, npoints, vlist, currp, llist);
        if (!lineobj->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'lines' failed");
            return;
        }
        else
            lineobj->addAttribute(COLOR, color);

        //	free space
        delete[] xv;
        delete[] yv;
        delete[] zv;
        delete[] vlist;
        delete[] llist;
        for (i = 0; i < currp; i++)
        {
            delete[] xptmp[i];
            delete[] yptmp[i];
            delete[] zptmp[i];
        }
        delete[] nptmp;
    }
}

//  ===========================================================================
//
//  Some Auxiliary functions
//
//  ===========================================================================

float gb_(int *comp, int *i, int *j, int *k)
{
    static float tmp[3];
    int pos;

    pos = (*i - 1) * dim[1] * dim[2] + (*j - 1) * dim[2] + (*k - 1);
    tmp[0] = x[pos];
    tmp[1] = y[pos];
    tmp[2] = z[pos];

    return (tmp[*comp - 1]);
}

float vb_(int *comp, int *i, int *j, int *k)
{
    static float tmp[3];
    int pos;

    pos = (*i - 1) * dim[1] * dim[2] + (*j - 1) * dim[2] + (*k - 1);
    tmp[0] = u[pos];
    tmp[1] = v[pos];
    tmp[2] = w[pos];

    return (tmp[*comp - 1]);
}

int lb_(void /* in the current implementation */)
{
    return ((int)0);
}

void geob_(int *block, int *ndx, int *ndxc, int *ngb, int *, int *irtn)
{
    if (*block < 1)
    {
        *irtn = 1;
        return;
    }
    ngb[0] = dim[0];
    ngb[1] = dim[1];
    ngb[2] = dim[2];

    *ndxc = 3;
    if (ngb[2] == 1)
        *ndxc = *ndxc - 1;
    if (ngb[1] == 1)
        *ndxc = *ndxc - 1;
    if (ngb[0] == 1)
        *ndxc = *ndxc - 1;
    *ndx = 3;
    *irtn = 0;
}

void varb_(int *block, int *, int *, int *ngb, int *, int *irtn)
{
    if (*block < 1)
    {
        *irtn = 1;
        return;
    }
    velobj->getGridSize(&ngb[0], &ngb[1], &ngb[2]);
    *irtn = 0;
}

void limb_(int *block, int *, int *, int *irtn)
{
    if (*block < 1)
    {
        *irtn = 1;
        return;
    }
    *irtn = 0;
}
