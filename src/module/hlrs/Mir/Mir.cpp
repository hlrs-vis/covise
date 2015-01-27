/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:   Michael Resch simulation  module                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  D.Rantzau, U. Woessner                                        **
 **                                                                        **
 **                                                                        **
 ** Date:  04.03.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "Mir.h"
int main(int argc, char *argv[])
{
    Application *application = new Application(argc, argv);
    application->run();

    return 0;
}

Application::~Application()
{
}

Application::Application(int argc, char *argv[])
{
    Covise::set_module_description("MIR simulation");
    Covise::add_port(OUTPUT_PORT, "mesh", "Polygons", "Mesh");
    Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "Vector field");
    Covise::add_port(OUTPUT_PORT, "press", "Float", "Scalar field");
    Covise::add_port(OUTPUT_PORT, "eps", "Float", "Scalar field");
    Covise::add_port(PARIN, "grid", "Browser", "Filename");
    Covise::add_port(PARIN, "num_timesteps", "IntScalar", "Number of Timesteps");
    Covise::set_port_default("grid", "src/application/hlrs/Mir/test.mesh *.mesh");
    Covise::set_port_default("num_timesteps", "2000");
    Covise::init(argc, argv);
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);
}

void Application::run()
{
    Covise::sendInfo("MIR computation starting up");
    // send EXECUTE if finished
    char buf[200];
    sprintf(buf, "E%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    Covise::set_feedback_info(buf);
    while (1)
    {
        while (Covise::check_and_handle_event())
        {

            // no COVISE message arrived, nothing to do
        }
        if (computeit == 1)
        {
            visco_test_(filename, &num_timesteps);
            Covise::sendInfo("MIR computation finished");
            Covise::send_feedback_message("EXEC", "");
            computeit = 2;
        }
    }
    Covise::main_loop();
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
// locally global :-)
//
static int current_step = 1;
char message[200];

//..........................................................................
//
void Application::quit(void *callbackData)
{
    //
    // ...... delete your data here .....

    sprintf(message, "MIR computation finished and exiting");
    Covise::sendInfo(message);
}

//
// C++b accessible FORTRAN variables/arrays
//
int computeit = 0;
char *filename;
int num_timesteps;

void Application::compute(void *callbackData)
{
    //
    // ...... do work here ........
    //
    int i, *v_l, *el, n_elem, n_coord;
    float *u, *v, *w;
    float *xc, *yc, *zc;
    char *fname;
    double *x_coord, *y_coord, *ud, *vd;
    int *vl;
    int num_conn, num_c;
    coDoVec3 *Veloc;
    coDoPolygons *Polys;

    // read output data object name
    //
    MeshName = Covise::get_object_name("mesh");
    VelocName = Covise::get_object_name("velocity");
    KName = Covise::get_object_name("press");

    Covise::get_browser_param("grid", &fname);
    long ts_tmp;
    Covise::get_scalar_param("num_timesteps", &ts_tmp);
    num_timesteps = ts_tmp;
    //strcpy(filename,fname);
    filename = fname;
    if (computeit == 0)
    {
        computeit = 1;
        return;
    }
    else if (computeit == 2)
    {
        computeit = 0;
    }
    sprintf(message, "Visualizing MIR iteration step no. %d", current_step);
    Covise::sendInfo(message);
    x_coord = new double[num_cells];
    y_coord = new double[num_cells];
    ud = new double[num_cells];
    vd = new double[num_cells];
    vl = new int[num_cells * 4];
    visco_get_vectors_(x_coord, y_coord, ud, vd, vl, &num_c, &num_conn);
    n_elem = num_conn / 4;
    n_coord = num_cells;
    // create objects in shared memory from computed iteration step
    if (MeshName != 0L)
    {
        Polys = new coDoPolygons(MeshName, n_coord, num_conn, n_elem);
        Polys->getAddresses(&xc, &yc, &zc, &v_l, &el);
        for (i = 0; i < n_coord; i++)
        {
            xc[i] = x_coord[i];
            yc[i] = y_coord[i];
            zc[i] = 0.0;
        }
        for (i = 0; i < num_conn; i++)
        {
            v_l[i] = vl[i];
        }
        for (i = 0; i < n_elem; i++)
        {
            el[i] = i * 4;
        }
    }

    if (VelocName != 0L)
    {
        Veloc = new coDoVec3(VelocName, n_coord);
        Veloc->getAddresses(&u, &v, &w);
        for (i = 0; i < n_coord; i++)
        {
            u[i] = ud[i];
            v[i] = vd[i];
            w[i] = 0.0;
        }
    }
    /*

     if(KName != 0L) {
       K = new coDoFloat(KName, n_coord+1);
       K->getAddress(&p);
     }

     if(EpsName != 0L) {
       Eps = new coDoFloat(EpsName, n_coord+1);
       Eps->getAddress(&p);
   }    */

    //
    // delete COVISE objects
    //
    delete Polys;
    delete Veloc;
    delete[] ud;
    delete[] vd;
    delete[] vl;
    delete[] x_coord;
    delete[] y_coord;
    /* delete K;
    delete Eps;*/
}

//==============================================================================
//
// Routine called once per iteration from the MIR simulation code.
// Handles checking for incoming COVISE messages and let handle those
// in the callback routines
//
//
//
//==============================================================================

void covise_update_(
    int *timestep, int *dim)
{

    current_step = *timestep;
    //if(numsim == 4)
    //    Covise::send_feedback_message("EXEC","");
    //numsim++;
    num_cells = *dim;
    // Tell COVSIE that we have finished another iteration
    //
    sprintf(message, "Computed MIR iteration step no. %d", current_step);
    Covise::sendInfo(message);

    // check for COVISE requests
    //
    while (Covise::check_and_handle_event())
    {

        // no COVISE message arrived, nothing to do
    }
}
