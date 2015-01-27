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
#include "ReadDiablo.h"
#include "string.h"

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

int get_meshheader(FILE *fp, int *n_coord, int *n_elem, int *ngroups, int *n_conn)
{
    int i, j, dummy;
    int anzelem, anznode, geometr;
    char buf[300];

    // Header lesen
    for (i = 0; i < 5; i++)
    {
        fgets(buf, 300, fp);
    }
    if (fscanf(fp, "%d%d%d%d%d\n", n_coord, n_elem, ngroups, &dummy, &dummy) == EOF)
    {
        return (-1);
    }
    for (i = 0; i < 7; i++)
    {
        fgets(buf, 300, fp);
    }

    // File nach Anzahl der Connections durchsuchen
    *n_conn = 0;
    for (i = 0; i < *ngroups; i++)
    {
        fscanf(fp, "%s%d%s%d%s%d%s%d%s%d\n", &buf, &dummy, &buf, &anzelem, &buf,
               &anznode, &buf, &geometr, &buf, &dummy);
        fgets(buf, 300, fp);
        for (j = 0; j < anzelem; j++)
        {
            fgets(buf, 300, fp);
            switch (anznode)
            {
            case 6:
                *n_conn += 6;
                break;
            case 8:
                *n_conn += 8;
                break;
            }
        }
    }
    // wieder Anfang der Datei
    rewind(fp);
    for (i = 0; i < 13; i++)
    {
        fgets(buf, 300, fp);
    }
    return (0);
}

int get_haerte(FILE *h_fp, int npunkte, float *h)
{
    int temphaerte, tempnode, i;
    char buf[300];
    float fdummy;

    // Header einlesen
    fgets(buf, 300, h_fp);

    if (fscanf(h_fp, "%d%f%f%f%d\n", &tempnode, &fdummy, &fdummy, &fdummy, &temphaerte) == EOF)
    {
        return (-1);
    }
    for (i = 0; i < npunkte; i++)
    {
        if (i == tempnode)
        {
            *h = (float)temphaerte;
            if (fscanf(h_fp, "%d%f%f%f%d\n", &tempnode, &fdummy, &fdummy, &fdummy, &temphaerte) == EOF)
            {
                tempnode = 1;
            }
        }
        else
        {
            *h = 100.0;
        }
        h++;
    }
    return (0);
}

int get_nodes(FILE *n_fp, int npunkte, float *x, float *y, float *z)
{
    int i, dummy;

    for (i = 0; i < npunkte; i++)
    {
        if (fscanf(n_fp, "%d%f%f%f\n", &dummy, x, y, z) == EOF)
        {
            return (-1);
        }
        x++;
        y++;
        z++;
    }
    return (0);
}

void get_mesh(FILE *fp, int ngroups, int *vlist, int *elist, int *tlist)
{
    char buf[300];
    int i, j, p1, p2, p3, p4, p5, p6, p7, p8, dummy;
    int anzelem, anznode, geometr;
    int vlcount;

    // Element Gruppen suchen
    fsearch(fp, "ELEMENT GROUPS\n");

    vlcount = 0;
    for (i = 0; i < ngroups; i++)
    {
        fscanf(fp, "%s%d%s%d%s%d%s%d%s%d\n", &buf, &dummy, &buf, &anzelem, &buf,
               &anznode, &buf, &geometr, &buf, &dummy);
        fgets(buf, 300, fp);
        for (j = 0; j < anzelem; j++)
        {
            *elist = vlcount;
            switch (anznode)
            {
            case 6:
                fscanf(fp, "%d%d%d%d%d%d%d\n", &dummy, &p1, &p2, &p3, &p4,
                       &p5, &p6);
                *vlist = p1 - 1;
                vlist++;
                *vlist = p2 - 1;
                vlist++;
                *vlist = p3 - 1;
                vlist++;
                *vlist = p4 - 1;
                vlist++;
                *vlist = p5 - 1;
                vlist++;
                *vlist = p6 - 1;
                vlist++;
                vlcount += 6;
                break;
            case 8:
                fscanf(fp, "%d%d%d%d%d%d%d%d%d\n", &dummy, &p1, &p2, &p3, &p4,
                       &p5, &p6, &p7, &p8);
                *vlist = p1 - 1;
                vlist++;
                *vlist = p2 - 1;
                vlist++;
                *vlist = p4 - 1;
                vlist++;
                *vlist = p3 - 1;
                vlist++;
                *vlist = p5 - 1;
                vlist++;
                *vlist = p6 - 1;
                vlist++;
                *vlist = p8 - 1;
                vlist++;
                *vlist = p7 - 1;
                vlist++;
                vlcount += 8;
                break;
            }
            switch (geometr)
            {
            case 3:
                *tlist = TYPE_HEXAGON;
                break;
            case 4:
                *tlist = TYPE_PRISM;
                break;
            }
            elist++;
            tlist++;
        }
    }
}

void Application::compute(void *)
{
    int ngroups, n_conn;
    char buf[300];
    FILE *mesh_fp, *haerte_fp;

    // read input parameters and data object name
    Covise::get_browser_param("mesh_path", &mesh_path);
    Covise::get_browser_param("haerte_path", &haerte_path);

    Mesh = Covise::get_object_name("mesh");
    Data = Covise::get_object_name("data");

    // Dateien oeffnen
    if ((mesh_fp = Covise::fopen(mesh_path, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, mesh_path);
        Covise::sendError(buf);
        return;
    }
    if ((haerte_fp = Covise::fopen(haerte_path, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, haerte_path);
        Covise::sendError(buf);
        return;
    }

    // Mesh-Header einlesen
    if (get_meshheader(mesh_fp, &n_coord, &n_elem, &ngroups, &n_conn) != 0)
    {
        strcpy(buf, "FEHLER: Falsches Dateiformat");
        strcat(buf, mesh_path);
        Covise::sendError(buf);
        return;
    }

    // Speicher fuer Daten reservieren
    if (Data != 0)
    {
        data = new coDoFloat(Data, n_coord);
        if (data->objectOk())
        {
            data->getAddress(&dat);
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
        mesh = new coDoUnstructuredGrid(Mesh, n_elem, n_elem * 8, n_coord, 1);
        if (mesh->objectOk())
        {
            mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
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

    // Knoten einlesen
    if (get_nodes(mesh_fp, n_coord, x_coord, y_coord, z_coord) != 0)
    {
        strcpy(buf, "FEHLER: Falsches Dateiformat");
        strcat(buf, mesh_path);
        Covise::sendError(buf);
        return;
    }

    // Netzdaten einlesen
    get_mesh(mesh_fp, ngroups, vl, el, tl);

    // Haerte einlesen
    if (get_haerte(haerte_fp, n_coord, dat) != 0)
    {
        strcpy(buf, "FEHLER: Falsches Dateiformat");
        strcat(buf, haerte_path);
        Covise::sendError(buf);
        return;
    }

    fclose(mesh_fp);
    fclose(haerte_fp);

    delete mesh;
    delete data;
}
