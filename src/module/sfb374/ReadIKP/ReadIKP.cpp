/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Diablo Read         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                           Christoph Kunz                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadIKP.h"
#include "string.h"
#include "math.h"
#include <stdlib.h>

const int Application::maxtimesteps = MAXT;

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

    if (x_nodes != NULL)
    {
        free(x_nodes);
        free(y_nodes);
        free(z_nodes);
        free(elsav);
        free(tlsav);
        free(vlsav);
    }
}

// eigene Routinen
void fsearch(FILE *fp, char *str)
{
    char buf[300];

    // Sucht String in einem File
    fgets(buf, 300, fp);
    while (strcmp(buf, str) != 0)
    {
        fgets(buf, 300, fp);
    }
}

void Application::init_module()
{
    char buf[300];

    if (x_nodes != NULL)
    {
        free(x_nodes);
        free(y_nodes);
        free(z_nodes);
        free(elsav);
        free(tlsav);
        free(vlsav);
    }

    // Mesh-Header einlesen
    if (get_meshheader() != 0)
    {
        strcpy(buf, "FEHLER: Falsches Dateiformat");
        strcat(buf, data_path);
        Covise::sendError(buf);
        return;
    }

    // Slider einstellen
    if (Covise::get_slider_param("TimeStep", &min, &max, &timestep) != 0)
    {
        if ((timestepped == 0) && (timestep != 0))
            timestep = 0;
        Covise::update_slider_param("TimeStep", 0, n_timesteps, timestep);
    }

    // Netzdaten einlesen
    get_mesh();
    Covise::sendInfo("Netzdaten eingelesen");
    get_nodes();
    Covise::sendInfo("Modul initialisiert");
}

int Application::get_meshheader()
{
    int i, dummy;
    char buf[300];

    n_groups = 0;
    fgets(buf, 300, data_fp);
    if (fscanf(data_fp, "%d%d%d%d%d%d\n", &INUM, &n_coord, &n_elem, &n_dimension,
               &NSTRES, &dummy) == EOF)
    {
        return (-1);
    }
    if (fscanf(data_fp, "%d%d%d%d%d%d\n", &dummy, &dummy, &dummy, &NNODMX, &dummy,
               &dummy) == EOF)
    {
        return (-1);
    }
    if (fscanf(data_fp, "%d%d%d%d%d%d\n", &dummy, &dummy, &NDISTL, &NSET, &NSPRNG,
               &NDIE) == EOF)
    {
        return (-1);
    }
    for (i = 0; i < (INUM + 2); i++)
        fgets(buf, 300, data_fp);

    // Anzahl der Connections ausrechnen
    switch (n_dimension)
    {
    case 2:
        n_conn = (n_elem)*4;
        break;
    case 3:
        n_conn = (n_elem)*8;
        break;
    }
    // Anzahl der timesteps
    n_timesteps = 0;
    while (fgets(buf, 300, data_fp) != NULL)
    {
        if (strcmp(buf, "****\n") == 0)
        {
            timepos[n_timesteps] = ftell(data_fp);
            n_timesteps++;
        }
    }
    return (0);
}

int Application::get_nodes()
{
    int i, dummy;
    char buf[300];
    float *x, *y, *z;

    x_nodes = (float *)malloc(n_coord * sizeof(float));
    y_nodes = (float *)malloc(n_coord * sizeof(float));
    z_nodes = (float *)malloc(n_coord * sizeof(float));

    x = x_nodes;
    y = y_nodes;
    z = z_nodes;

    // Knoten einlesen
    for (i = 0; i < n_coord; i++)
    {
        fgets(buf, 300, data_fp);
        switch (n_dimension)
        {
        case 2:
            sscanf(buf, "%d%e%e\n", &dummy, x, y);
            *z = 0;
            break;
        case 3:
            sscanf(buf, "%d%e%e%e\n", &dummy, x, y, z);
            break;
        }
        x++;
        y++;
        z++;
    }
    return (0);
}

void Application::get_mesh()
{
    int i, p1, p2, p3, p4, p5, p6, p7, p8, dummy;
    int geometr;
    int vlcount;
    int *els, *tls, *vls;
    char buf[300];

    rewind(data_fp);
    for (i = 0; i < (INUM + 6); i++)
    {
        fgets(buf, 300, data_fp);
    }

    elsav = (int *)malloc(n_elem * sizeof(int));
    tlsav = (int *)malloc(n_elem * sizeof(int));
    vlsav = (int *)malloc(n_conn * sizeof(int));

    els = elsav;
    tls = tlsav;
    vls = vlsav;

    // Element Gruppen einlesen
    for (i = 0; i < n_elem; i++)
    {
        switch (n_dimension)
        {
        case 2:
            fscanf(data_fp, "%d%d%d%d%d%d%d\n", &vlcount, &geometr, &dummy,
                   &p1, &p2, &p3, &p4);

            // 4 Punkte

            *vls = p4 - 1;
            vls++;
            *vls = p1 - 1;
            vls++;
            *vls = p2 - 1;
            vls++;
            *vls = p3 - 1;
            vls++;
            *els = (vlcount - 1) * 4;
            break;

        case 3:
            fscanf(data_fp, "%d%d%d%d%d%d%d%d%d%d%d\n", &vlcount, &geometr,
                   &dummy, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8);

            // 8 Punkte

            *vls = p4 - 1;
            vls++;
            *vls = p1 - 1;
            vls++;
            *vls = p2 - 1;
            vls++;
            *vls = p3 - 1;
            vls++;
            *vls = p8 - 1;
            vls++;
            *vls = p5 - 1;
            vls++;
            *vls = p6 - 1;
            vls++;
            *vls = p7 - 1;
            vls++;
            *els = (vlcount - 1) * 8;
            break;
        }
        switch (geometr)
        {
        case 11:
            *tls = TYPE_QUAD;
            break;
        case 7:
            *tls = TYPE_HEXAEDER;
            break;
        }
        els++;
        tls++;
    }
}

int Application::get_timestep()
{
    int i;
    char buf[300];
    float *x, *y, *z, *dat, *xn, *yn, *zn;
    int *tls, *els, *vls;

    // Marc file format variables
    // d -> displacement xf -> extaernal forces r -> reaction forces
    int NEWCC, INC, SUBINC, JANTYP, KNOD, IDMY1;
    float d_x, d_y, d_z, xf_x, xf_y, xf_z, r_x, r_y, r_z;

    // Koordinaten-Pointer initialisieren
    x = x_coord_start;
    y = y_coord_start;
    z = z_coord_start;
    dat = dat_start;
    xn = x_nodes;
    yn = y_nodes;
    zn = z_nodes;
    tls = tlsav;
    vls = vlsav;
    els = elsav;

    for (i = 0; i < n_elem; i++)
    {
        *tl = *tls;
        *el = *els;
        tls++;
        els++;
        tl++;
        el++;
    }
    for (i = 0; i < n_conn; i++)
    {
        *vl = *vls;
        vls++;
        vl++;
    }

    if (timestep == 0)
    {
        for (i = 0; i < n_coord; i++)
        {
            *x = *xn;
            *y = *yn;
            *z = *zn;
            *dat = 0;

            x++;
            xn++;
            y++;
            yn++;
            z++;
            zn++;
            dat++;
        }
    }
    else
    {
        if (fseek(data_fp, timepos[timestep - 1], SEEK_SET) != 0)
        {
            Covise::sendError("ERROR: fseek-error on datafile");
            return (-1);
        }

        // Block 11
        fscanf(data_fp, "%d%d%d%d%d%d\n", &NEWCC, &INC, &SUBINC, &JANTYP, &KNOD, &IDMY1);

        // Block 12
        fgets(buf, 300, data_fp);

        // Block 14
        if (NDISTL != 0)
        {
            for (i = 0; i < ((NDISTL - 1) / 6 + 1); i++)
                fgets(buf, 300, data_fp);
        }

        // Block 17
        for (i = 0; i < (((INUM - 1) / 6 + 1) * NSTRES * n_elem); i++)
            fgets(buf, 300, data_fp);

        // Block 18
        for (i = 0; i < n_coord; i++)
        {
            switch (n_dimension)
            {
            case 2:
                fscanf(data_fp, "%f%f%f%f%f%f\n", &d_x, &d_y, &xf_x, &xf_y,
                       &r_x, &r_y);
                d_z = 0;
                xf_z = 0;
                r_z = 0;
                break;
            case 3:
                fscanf(data_fp, "%f%f%f%f%f%f%f%f%f\n", &d_x, &d_y, &d_z, &xf_x, &xf_y,
                       &xf_z, &r_x, &r_y, &r_z);

                break;
            }
            *x = *xn + d_x;
            *y = *yn + d_y;
            *z = *zn + d_z;

#ifdef __sgi
            if (Forces_choice == 1)
                *dat = fsqrt(powf(xf_x, 2) + powf(xf_y, 2) + powf(xf_z, 2));
            else
                *dat = fsqrt(powf(r_x, 2) + powf(r_y, 2) + powf(r_z, 2));
#else
            if (Forces_choice == 1)
                *dat = sqrt(powf(xf_x, 2) + powf(xf_y, 2) + powf(xf_z, 2));
            else
                *dat = sqrt(powf(r_x, 2) + powf(r_y, 2) + powf(r_z, 2));
#endif
            x++;
            xn++;
            y++;
            yn++;
            z++;
            zn++;
            dat++;
        }
    }
    return (0);
}

void Application::compute(void *)
{
    char buf[300];

    // read input parameters and data object name
    Covise::get_browser_param("data_path", &data_path);
    Covise::get_slider_param("TimeStep", &min, &max, &timestep);
    Covise::get_choice_param("Forces", &Forces_choice);

    // Dateien oeffnen
    if ((data_fp = Covise::fopen(data_path, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, data_path);
        Covise::sendError(buf);
        return;
    }

    if ((lastname[0] == '0') || (strcmp(data_path, lastname) != 0))
    {
        init_module();
        strcpy(lastname, data_path);
    }

    Mesh = Covise::get_object_name("mesh");
    Data = Covise::get_object_name("data");

    // Speicher fuer Daten reservieren
    if (Data != NULL)
    {
        data = new coDoFloat(Data, n_coord);
        if (data->objectOk())
        {
            data->getAddress(&dat_start);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'data' failed");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: Object name not correct for 'data'");
        return;
    }

    if (Mesh != NULL)
    {
        mesh = new coDoUnstructuredGrid(Mesh, n_elem, n_conn, n_coord, 1);
        if (mesh->objectOk())
        {
            mesh->getAddresses(&el, &vl, &x_coord_start, &y_coord_start, &z_coord_start);
            mesh->getTypeList(&tl);
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

    // Zeitschritt einlesen
    if (get_timestep() != 0)
    {
        strcpy(buf, "FEHLER: Falsches Dateiformat");
        strcat(buf, data_path);
        Covise::sendError(buf);
        return;
    }

    fclose(data_fp);

    delete mesh;
    delete data;
}
