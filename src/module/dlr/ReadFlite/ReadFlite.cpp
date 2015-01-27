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
#include "ReadFlite.h"
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
    int i;
    char buf[300];
    int dummy1, dummy2;
    mesh = NULL;
    surface = NULL;
    rho = NULL;
    rhou = NULL;
    rhov = NULL;
    rhow = NULL;
    rhoe = NULL;

    Covise::get_browser_param("grid_path", &grid_Path);

    Mesh = Covise::get_object_name("mesh");
    Surface = Covise::get_object_name("surface");
    Rho = Covise::get_object_name("rho");
    Rhou = Covise::get_object_name("rhou");
    Rhov = Covise::get_object_name("rhov");
    Rhow = Covise::get_object_name("rhow");
    Rhoe = Covise::get_object_name("rhoe");

    if ((fp = Covise::fopen(grid_Path, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, grid_Path);
        Covise::sendError(buf);
        return;
    }

    fread(&dummy1, sizeof(int), 1, fp);
    fread(&n_elem, sizeof(int), 1, fp);
    fread(&n_coord, sizeof(int), 1, fp);
    fread(&n_triang, sizeof(int), 1, fp);
    fread(&dummy2, sizeof(int), 1, fp);
    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker");
        return;
    }

    if (Mesh != NULL)
    {
        mesh = new coDoUnstructuredGrid(Mesh, n_elem, n_elem * 4, n_coord, 1);
        if (mesh->objectOk())
        {
            mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
            mesh->getTypeList(&tl);

            int *tmpind = new int[n_elem * 4];
            fread(&dummy1, sizeof(int), 1, fp);
            fread(tmpind, sizeof(int), (4 * n_elem), fp);
            fread(&dummy2, sizeof(int), 1, fp);
            if (dummy1 != dummy2)
            {
                Covise::sendError("ERROR wrong FORTRAN block marker");
                return;
            }

            for (i = 0; i < n_elem; i++)
            {
                *el = i * 4;
                vl[*el] = tmpind[i] - 1;
                vl[*el + 1] = tmpind[i + n_elem] - 1;
                vl[*el + 2] = tmpind[i + n_elem * 2] - 1;
                vl[*el + 3] = tmpind[i + n_elem * 3] - 1;
                el++;
                *tl = TYPE_TETRAHEDER;
                tl++;
            }
            delete[] tmpind;

            double *tmpd = new double[n_coord];
            fread(&dummy1, sizeof(int), 1, fp);
            fread(tmpd, sizeof(double), (n_coord), fp);
            for (i = 0; i < n_coord; i++)
                x_coord[i] = (float)tmpd[i];
            fread(tmpd, sizeof(double), (n_coord), fp);
            for (i = 0; i < n_coord; i++)
                y_coord[i] = (float)tmpd[i];
            fread(tmpd, sizeof(double), (n_coord), fp);
            for (i = 0; i < n_coord; i++)
                z_coord[i] = (float)tmpd[i];
            fread(&dummy2, sizeof(int), 1, fp);

            delete[] tmpd;
            if (dummy1 != dummy2)
            {
                Covise::sendError("ERROR wrong FORTRAN block marker");
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

    if (Rho != 0)
    {
        rho = new coDoFloat(Rho, n_coord);
        if (!rho->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'rho' failed");
            return;
        }
        rho->getAddress(&r);
    }
    else
    {
        Covise::sendError("ERROR: Object name not correct for 'rho'");
        return;
    }
    if (Rhou != 0)
    {
        rhou = new coDoFloat(Rhou, n_coord);
        if (!rhou->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'rhou' failed");
            return;
        }
        rhou->getAddress(&ru);
    }
    else
    {
        Covise::sendError("ERROR: Object name not correct for 'rhou'");
        return;
    }
    if (Rhov != 0)
    {
        rhov = new coDoFloat(Rhov, n_coord);
        if (!rhov->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'rhov' failed");
            return;
        }
        rhov->getAddress(&rv);
    }
    else
    {
        Covise::sendError("ERROR: Object name not correct for 'rhov'");
        return;
    }
    if (Rhow != 0)
    {
        rhow = new coDoFloat(Rhow, n_coord);
        if (!rhow->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'rhow' failed");
            return;
        }
        rhow->getAddress(&rw);
    }
    else
    {
        Covise::sendError("ERROR: Object name not correct for 'rhow'");
        return;
    }
    if (Rhoe != 0)
    {
        rhoe = new coDoFloat(Rhoe, n_coord);
        if (!rhoe->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'rhoe' failed");
            return;
        }
        rhoe->getAddress(&re);
    }
    else
    {
        Covise::sendError("ERROR: Object name not correct for 'rhoe'");
        return;
    }

    double *tmpd = new double[n_coord];
    fread(&dummy1, sizeof(int), 1, fp);
    fread(tmpd, sizeof(double), (n_coord), fp);
    for (i = 0; i < n_coord; i++)
        r[i] = (float)tmpd[i];
    fread(tmpd, sizeof(double), (n_coord), fp);
    for (i = 0; i < n_coord; i++)
        ru[i] = (float)tmpd[i];
    fread(tmpd, sizeof(double), (n_coord), fp);
    for (i = 0; i < n_coord; i++)
        rv[i] = (float)tmpd[i];
    fread(tmpd, sizeof(double), (n_coord), fp);
    for (i = 0; i < n_coord; i++)
        rw[i] = (float)tmpd[i];
    fread(tmpd, sizeof(double), (n_coord), fp);
    for (i = 0; i < n_coord; i++)
        re[i] = (float)tmpd[i];
    fread(&dummy2, sizeof(int), 1, fp);
    delete[] tmpd;
    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker");
        return;
    }

    if (Surface != NULL)
    {
        surface = new coDoPolygons(Surface, n_coord, n_triang * 3, n_triang);
        if (surface->objectOk())
        {
            surface->getAddresses(&x_coord2, &y_coord2, &z_coord2, &vl, &el);

            int *tmpind = new int[n_triang * 3];
            fread(&dummy1, sizeof(int), 1, fp);
            fread(tmpind, sizeof(int), (3 * n_triang), fp);
            fread(&dummy2, sizeof(int), 1, fp);
            //if(dummy1!=dummy2)
            //{
            //			Covise::sendError("ERROR wrong FORTRAN block marker");
            //			return;
            // }

            for (i = 0; i < n_triang; i++)
            {
                *el = i * 3;
                vl[*el] = tmpind[i] - 1;
                vl[*el + 1] = tmpind[i + n_triang] - 1;
                vl[*el + 2] = tmpind[i + n_triang * 2] - 1;
                el++;
            }
            delete[] tmpind;

            memcpy(x_coord2, x_coord, n_coord * sizeof(float));
            memcpy(y_coord2, y_coord, n_coord * sizeof(float));
            memcpy(z_coord2, z_coord, n_coord * sizeof(float));
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

    delete mesh;
    delete surface;
    delete rho;
    delete rhou;
    delete rhov;
    delete rhow;
    delete rhoe;
}
