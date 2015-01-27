/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:   COVISE ReadIv application module                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author: D.Rantzau, U. Woessner                                         **
 **                                                                        **
 **                                                                        **
 ** Date:  14.02.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadBAW.h"

#include <sys/types.h>
#include <stdio.h>
#include <sys/stat.h>

//
// static stub callback functions calling the real class
// member functions
//

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

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
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void Application::compute(void *)
{
    //
    // ...... do work here ........
    //

    // read input parameters and data object name
    int ofp;
    char buf[300];
    char *adress;
    int length;
    float dummy;
    struct stat My_stat_buf;
    int i;

    err_status = Covise::get_browser_param("path", &filename);
    char *Gridname = Covise::get_object_name("grid");
    char *Vname = Covise::get_object_name("data1");
    char *Sname = Covise::get_object_name("data2");

    if (!openFile())
        return;

    for (i = 0; i < 20; i++)
        fgets(buf, 300, fp);

    cerr << buf << endl;

    int num_elem = 398994;
    int num_coord = 203413;
    int num_conn = 3 * num_elem;
    float *x = new float[num_coord];
    float *y = new float[num_coord];
    float *z = new float[num_coord];
    float *u = new float[num_coord];
    float *v = new float[num_coord];
    float *w = new float[num_coord];
    float *d = new float[num_coord];
    float *s = new float[num_coord];
    float *b = new float[num_coord];
    float *sv = new float[num_coord];
    float *cn = new float[num_coord];
    int *elem_list = new int[num_elem];
    int *conn_list = new int[num_conn];
    int *type_list = new int[num_elem];
    for (i = 0; i < num_coord; i++)
        fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
               //             &Rechtswert[i], &Hochwert[i], &Sohle[i], &VELOCITY_U[i], &VELOCITY_V[i],
               //             &WATER_DEPTH[i], &FREE_SURFACE[i], &BOTTOM[i],
               //             &SCALAR_VELOCITY[i], &COURANT_NUMBER);
               &x[i], &y[i], &z[i], &u[i], &v[i], &d[i], &s[i], &b[i], &sv[i], &cn[i],
               &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy);
    w[i] = 0.0;
    for (i = 0; i < num_elem; i++)
    {
        elem_list[i] = 3 * i;
        type_list[i] = TYPE_TRIANGLE;
        fscanf(fp, "%d %d %d\n", &conn_list[3 * i + 0],
               &conn_list[3 * i + 1], &conn_list[3 * i + 2]);
        conn_list[3 * i + 0]--;
        conn_list[3 * i + 1]--;
        conn_list[3 * i + 2]--;
    }

    coDoUnstructuredGrid *grid = new coDoUnstructuredGrid(Gridname,
                                                          num_elem, num_conn, num_coord,
                                                          elem_list, conn_list, x, y, z, type_list);

    coDoVec3 *data1 = new coDoVec3(Vname, num_coord, u, v, w);
    coDoFloat *data2 = new coDoFloat(Sname, num_coord, sv);

    delete grid;
    delete data1;
    delete data2;

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] u;
    delete[] v;
    delete[] w;
    delete[] d;
    delete[] s;
    delete[] b;
    delete[] sv;
    delete[] cn;
    delete[] elem_list;
    delete[] conn_list;
    delete[] type_list;
}

int Application::openFile()
{
    char line[1000]; // error message

    if ((fp = Covise::fopen(filename, "r")) == NULL)
    {
        strcpy(line, "ERROR: Can't open file >> '");
        strcat(line, filename);
        strcat(line, "'");
        Covise::sendError(line);
        return (0);
    }
    else
    {
        return (1);
    }
}
