/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:   COVISE ReadParticles application module                 **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  28.09.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadSand.h"

int main(int argc, char *argv[])
{
    Application *application = new Application(argc, argv);
    application->run();

    return 0;
}

//
// member functions
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
    FILE *fp;
    char buf[600];
    char *Path;
    char *Points;
    char *Grid;
    char *Velocity;
    char *Radius;
    int timestep, numpoints, i;
    float time, x0, x1, y0, y1, z0, z1, *x_c, *y_c, *z_c, *x_v, *y_v, *z_v, *r, *x_c2 = NULL, *y_c2 = NULL, *z_c2 = NULL;

    Covise::get_browser_param("path", &Path);

    Points = Covise::get_object_name("points");
    Grid = Covise::get_object_name("grid");
    Velocity = Covise::get_object_name("velocity");
    Radius = Covise::get_object_name("radius");
    if (Points == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for Points");
        return;
    }
    if (Grid == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for Points");
        return;
    }
    if (Velocity == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for Points");
        return;
    }
    if (Radius == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for Points");
        return;
    }
    if ((fp = Covise::fopen(Path, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s", Path);
        return;
    }
    timestep = 0;

    p_set = new coDoSet(Points, SET_CREATE);
    if (!p_set->objectOk())
    {
        Covise::sendError("ERROR: creation of set 'Points' failed");
        return;
    }
    g_set = new coDoSet(Grid, SET_CREATE);
    if (!g_set->objectOk())
    {
        Covise::sendError("ERROR: creation of set 'Grids' failed");
        return;
    }

    v_set = new coDoSet(Velocity, SET_CREATE);
    if (!v_set->objectOk())
    {
        Covise::sendError("ERROR: creation of set 'velocity' failed");
        return;
    }
    r_set = new coDoSet(Radius, SET_CREATE);
    if (!r_set->objectOk())
    {
        Covise::sendError("ERROR: creation of set 'radius' failed");
        return;
    }
    while (!feof(fp))
    {

        if (fgets(buf, 600, fp) == NULL)
            break;
        if (feof(fp))
            break;

        if (sscanf(buf, "%d %f %f %f %f %f %f %f", &numpoints, &time, &x0, &y0, &z0, &x1, &y1, &z1) != 8)
        {
            cerr << "ReadSand::compute: sscanf failed" << endl;
        }
        sprintf(buf, "%s_%d", Points, timestep);
        points = new coDoPoints(buf, numpoints);
        if (!points->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'Points' failed");
            return;
        }
        sprintf(buf, "%s_%d", Grid, timestep);
        if ((((int)sqrt((float)numpoints)) * ((int)sqrt((float)numpoints))) == numpoints)
        {
            s_grid = new coDoStructuredGrid(buf, (int)sqrt((float)numpoints), (int)sqrt((float)numpoints), 1);
            if (!s_grid->objectOk())
            {
                Covise::sendError("ERROR: creation of data object 'Grid' failed");
                return;
            }
        }
        else
            s_grid = NULL;
        sprintf(buf, "%s_%d", Velocity, timestep);
        velocity = new coDoVec3(buf, numpoints);
        if (!velocity->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'velocity' failed");
            return;
        }
        sprintf(buf, "%s_%d", Radius, timestep);
        radius = new coDoFloat(buf, numpoints);
        if (!radius->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'radius' failed");
            return;
        }
        points->getAddresses(&x_c, &y_c, &z_c);
        if (s_grid)
            s_grid->getAddresses(&x_c2, &y_c2, &z_c2);
        velocity->getAddresses(&x_v, &y_v, &z_v);
        radius->getAddress(&r);
        for (i = 0; i < numpoints; i++)
        {
            if ((fgets(buf, 600, fp) == NULL) || (feof(fp)))
            {
                Covise::sendError("ERROR: unexpected end of file");
                break;
            }
            if (sscanf(buf, "%f %f %f %f %f %f %f", x_c, y_c, z_c, x_v, y_v, z_v, r) != 7)
            {
                cerr << "ReadSand::compute: sscanf failed" << endl;
            }
            if (s_grid)
            {
                *x_c2 = *x_c;
                *y_c2 = *y_c;
                *z_c2 = *z_c;
            }
            x_c++;
            y_c++;
            z_c++;
            x_c2++;
            y_c2++;
            z_c2++;
            x_v++;
            y_v++;
            z_v++;
            r++;
        }

        p_set->addElement(points);
        if (s_grid)
            g_set->addElement(s_grid);
        r_set->addElement(radius);
        v_set->addElement(velocity);
        delete radius;
        delete velocity;
        delete points;
        timestep++;
    }
    if (timestep > 1)
    {
        p_set->addAttribute("TIMESTEP", "0 100 0");
        if (s_grid)
            g_set->addAttribute("TIMESTEP", "0 100 0");
    }
    delete p_set;
    delete g_set;
    delete r_set;
    delete v_set;
}
