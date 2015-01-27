/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include <do/coDoSet.h>
#include "ReadIVD.h"
int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
    return 0;
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
    FILE *fp, *data_fp = NULL;
    int i;
    char buf[600];
    int dummy1, dummy2, t, currt, endt;
    long numt;
    char dp[600];
    char dpend[600];
    mesh = NULL;
    veloc = NULL;
    temp = NULL;

    Covise::get_browser_param("grid_path", &grid_Path);
    Covise::get_browser_param("data_path", &data_Path);
    Covise::get_scalar_param("numt", &numt);
    strcpy(dp, data_Path);
    i = strlen(dp) - 1;
    while ((dp[i] < '0') || (dp[i] > '9'))
        i--;
    // dp[i] ist jetzt die letzte Ziffer, alles danach ist Endung
    strcpy(dpend, dp + i + 1); // dpend= Endung;
    dp[i + 1] = 0;
    while ((dp[i] >= '0') && (dp[i] <= '9'))
        i--;
    sscanf(dp + i + 1, "%d", &currt); //currt = Aktueller Zeitschritt
    endt = currt + numt;
    dp[i + 1] = 0; // dp = basename

    coDistributedObject **Veloc_sets = new coDistributedObject *[numt + 1];
    coDistributedObject **Temp_sets = new coDistributedObject *[numt + 1];
    coDistributedObject **Mesh_sets = new coDistributedObject *[numt + 1];

    Veloc_sets[0] = NULL;
    Temp_sets[0] = NULL;
    Mesh_sets[0] = NULL;

    Mesh = Covise::get_object_name("mesh");
    Velocity = Covise::get_object_name("velocity");
    Temperature = Covise::get_object_name("temperature");

    if ((fp = Covise::fopen(grid_Path, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s", grid_Path);
        return;
    }

    fread(&dummy1, sizeof(int), 1, fp);
    fread(&n_u, sizeof(int), 1, fp);
    fread(&n_v, sizeof(int), 1, fp);
    fread(&n_w, sizeof(int), 1, fp);
    fread(&dummy2, sizeof(int), 1, fp);
    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker");
        return;
    }

    if (Mesh != NULL)
    {
        if (numt > 1)
            sprintf(buf, "%s_mesh", Mesh);
        else
            strcpy(buf, Mesh);
        mesh = new coDoRectilinearGrid(buf, n_u, n_v, n_w);
        if (mesh->objectOk())
        {
            mesh->getAddresses(&x_coord, &y_coord, &z_coord);

            fread(&dummy1, sizeof(int), 1, fp);
            fread(x_coord, sizeof(int), (n_u), fp);
            fread(&dummy2, sizeof(int), 1, fp);
            if (dummy1 != dummy2)
            {
                Covise::sendError("ERROR wrong FORTRAN block marker reading header");
                return;
            }
            fread(&dummy1, sizeof(int), 1, fp);
            fread(y_coord, sizeof(int), (n_v), fp);
            fread(&dummy2, sizeof(int), 1, fp);
            if (dummy1 != dummy2)
            {
                Covise::sendError("ERROR wrong FORTRAN block marker reading header");
                return;
            }
            fread(&dummy1, sizeof(int), 1, fp);
            fread(z_coord, sizeof(int), (n_w), fp);
            fread(&dummy2, sizeof(int), 1, fp);
            if (dummy1 != dummy2)
            {
                Covise::sendError("ERROR wrong FORTRAN block marker reading header");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'mesh' failed");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'mesh'");
        return;
    }
    Mesh_sets[0] = mesh;
    Mesh_sets[1] = NULL;
    //values = new float[n_u*n_v*n_w];
    fclose(fp);
    for (t = currt; t < endt; t++)
    {
        if (numt > 1)
            sprintf(buf, "%s%04d%s", dp, t, dpend);
        else
            strcpy(buf, data_Path);

        if ((data_fp = Covise::fopen(buf, "r")) == NULL)
        {
            if (t == currt)
            {
                Covise::sendError("ERROR: Can't open file >> %s", buf);
                return;
            }
            else
            {
                break;
            }
        }

        Covise::sendInfo("Reading data for timestep %d\n", t);
        if (numt > 1)
            sprintf(buf, "%s_%d", Velocity, t);
        else
            strcpy(buf, Velocity);
        if (Velocity != 0)
        {
            veloc = new coDoVec3(buf, n_u * n_v * n_w);
            if (!veloc->objectOk())
            {
                Covise::sendError("ERROR: creation of data object 'Velocity' failed");
                return;
            }
            veloc->getAddresses(&vx, &vy, &vz);
        }
        else
        {
            Covise::sendError("ERROR: Object name not correct for 'Velocity'");
            return;
        }
        if (numt > 1)
            sprintf(buf, "%s_%d", Temperature, t);
        else
            strcpy(buf, Temperature);
        if (Temperature != 0)
        {
            temp = new coDoFloat(buf, n_u * n_v * n_w);
            if (!temp->objectOk())
            {
                Covise::sendError("ERROR: creation of data object 'Temperature' failed");
                return;
            }
            temp->getAddress(&tem);
        }
        else
        {
            Covise::sendError("ERROR: Object name not correct for 'Temperature'");
            return;
        }

        /*  fread(&dummy1, sizeof(int), 1, fp) ;
        fread(&n_u2, sizeof(int), 1, fp) ;
        fread(&n_v2, sizeof(int), 1, fp) ;
        fread(&n_w2, sizeof(int), 1, fp) ;
        fread(&dummy2, sizeof(int), 1, fp) ;

      fread(&dummy1, sizeof(int), 1, data_fp) ;
      fread(vx, sizeof(char), (dummy1), data_fp) ;
      fread(&dummy2, sizeof(int), 1, data_fp) ;*/

        fread(&dummy1, sizeof(int), 1, data_fp);
        fread(vx, sizeof(float), (n_u * n_v * n_w), data_fp);
        fread(&dummy2, sizeof(int), 1, data_fp);
        if (dummy1 != dummy2)
        {
            Covise::sendError("ERROR wrong FORTRAN block marker reading u");
            return;
        }
        fread(&dummy1, sizeof(int), 1, data_fp);
        fread(vy, sizeof(float), (n_u * n_v * n_w), data_fp);
        fread(&dummy2, sizeof(int), 1, data_fp);
        if (dummy1 != dummy2)
        {
            Covise::sendError("ERROR wrong FORTRAN block marker reading u");
            return;
        }
        fread(&dummy1, sizeof(int), 1, data_fp);
        fread(vz, sizeof(float), (n_u * n_v * n_w), data_fp);
        fread(&dummy2, sizeof(int), 1, data_fp);
        if (dummy1 != dummy2)
        {
            Covise::sendError("ERROR wrong FORTRAN block marker reading u");
            return;
        }
        fread(&dummy1, sizeof(int), 1, data_fp);
        fread(tem, sizeof(float), (n_u * n_v * n_w), data_fp);
        fread(&dummy2, sizeof(int), 1, data_fp);
        if (dummy1 != dummy2)
        {
            Covise::sendError("ERROR wrong FORTRAN block marker reading u");
            return;
        }

        if (t != currt)
        {
            for (i = 0; Mesh_sets[i]; i++)
                ;
            Mesh_sets[i] = mesh;
            mesh->incRefCount();
            Mesh_sets[i + 1] = NULL;
        }
        for (i = 0; Veloc_sets[i]; i++)
            ;
        Veloc_sets[i] = veloc;
        Veloc_sets[i + 1] = NULL;
        for (i = 0; Temp_sets[i]; i++)
            ;
        Temp_sets[i] = temp;
        Temp_sets[i + 1] = NULL;
        fclose(data_fp);
    }

    fclose(data_fp);

    if (numt > 1)
    {
        coDoSet *Veloc_set = new coDoSet(Velocity, Veloc_sets);
        coDoSet *Temp_set = new coDoSet(Temperature, Temp_sets);
        coDoSet *Mesh_set = new coDoSet(Mesh, Mesh_sets);
        Mesh_set->addAttribute("TIMESTEP", "1 200");
        delete Mesh_set;
        delete Temp_set;
        delete Veloc_set;
    }

    //delete[] values;
    delete Mesh_sets[0];
    delete[] Mesh_sets;
    for (i = 0; Veloc_sets[i]; i++)
        delete Veloc_sets[i];
    delete[] Veloc_sets;
    for (i = 0; Temp_sets[i]; i++)
        delete Temp_sets[i];
    delete[] Temp_sets;
}
