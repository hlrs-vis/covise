/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for ITE data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                                                                        **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  Wed May 28  1997                                                **
\**************************************************************************/

#include "Read_ITE.h"

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
    FILE *grid_fp, *data_fp, *gridelem_fp;

    int i, k, tmpi;
    int count;

    char buf[300];
    char buf1[300];

    int t;
    int elemgruppe;
    enum
    {
        Q20 = 27,
        Q8 = 64,
        Q20_2 = 132,
        Q8_2 = 564
    };

    int index[20];
    int *neuindex;
    int *altindex;
    float X, Y, Z, UU, VV, WW, PP;

    // File.COR
    Covise::get_browser_param("grid_path", &grid_Path);
    // File.INI
    Covise::get_browser_param("data_path", &data_Path);
    // File.ELE
    Covise::get_browser_param("gridelem_path", &gridelem_Path);
    //Covise::get_scalar_param("numt", &numt);

    Mesh = Covise::get_object_name("mesh");
    Veloc = Covise::get_object_name("velocity");
    Press = Covise::get_object_name("pressure");

    // File.COR oeffnen
    // Lesen der Anzahl von Koordinaten

    if ((grid_fp = Covise::fopen(grid_Path, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, grid_Path);
        Covise::sendError(buf);
        return;
    }

    fscanf(grid_fp, "%5d", &n_coord);

    fclose(grid_fp); // File.COR schliessen

    // File.ELE offnen
    // Lesen der Anzahl von Elementen

    if ((gridelem_fp = Covise::fopen(gridelem_Path, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, gridelem_Path);
        Covise::sendError(buf);
        return;
    }

    fscanf(gridelem_fp, "%5d", &n_elem);

    fgets(buf, 300, gridelem_fp); //Die erste Zeile eliminieren

    // Lesen der Hexaeder-Elemente, Ueberspringen aller anderen Elemente

    altindex = new int[8 * n_elem];
    neuindex = new int[n_coord + 1];
    n_el = 0;
    n_co = 0;

    //Initialisieren der Markierungsfelder
    for (i = 0; i < 8 * (n_elem + 1); i++)
        altindex[i] = 0;

    for (i = 0; i < n_coord + 1; i++)
        neuindex[i] = -2;

    // Lesen der Hexaeder-Elemente
    for (i = 1; i <= n_elem; i++)
    {
        fgets(buf, 300, gridelem_fp);
        sscanf(buf, "%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d", &tmpi,
               &tmpi, &tmpi, &elemgruppe, &tmpi, &tmpi, index, index + 1, index + 2, index + 3, index + 4, index + 5, index + 6, index + 7,
               index + 8, index + 9);

        //Q20 Elemente lesen und 8 Knoten daraus waehlen
        if ((elemgruppe == Q20) || (elemgruppe == Q20_2))
        {
            fgets(buf1, 300, gridelem_fp);

            sscanf(buf1, "%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d",
                   index + 10, index + 11, index + 12, index + 13, index + 14, index + 15, index + 16, index + 17, index + 18,
                   index + 19, &tmpi);

            //elemcount++;
            neuindex[index[14]] = -1;
            altindex[8 * n_el] = index[14];

            neuindex[index[12]] = -1;
            altindex[8 * n_el + 1] = index[12];

            neuindex[index[18]] = -1;
            altindex[8 * n_el + 2] = index[18];

            neuindex[index[16]] = -1;
            altindex[8 * n_el + 3] = index[16];

            neuindex[index[2]] = -1;
            altindex[8 * n_el + 4] = index[2];

            neuindex[index[0]] = -1;
            altindex[8 * n_el + 5] = index[0];

            neuindex[index[6]] = -1;
            altindex[8 * n_el + 6] = index[6];

            neuindex[index[4]] = -1;
            altindex[8 * n_el + 7] = index[4];

            n_el++;
        }

        //alle anderen Elemente Ueberspringen

    } //End der For-Schleife

    fclose(gridelem_fp); //File.ELE schliessen

    // benoetigte Vertices zaehlen, neue Indizierung einfuehren
    count = 0;
    for (i = 1; i <= n_coord; i++)
        if (neuindex[i] == -1)
        {
            neuindex[i] = count;
            count++;
        }
    n_co = count;

    //File.INI lesen

    /*    el = new int[n_el];
       vl = new int[n_el * 8];
       x_coord = new float[n_co];
       y_coord = new float[n_co];
       z_coord = new float[n_co];
       u = new float[n_co];
       v = new float[n_co];
       w = new float[n_co];
       p = new float[n_co];
   */
    // COR File oeffnen
    // Koordinaten und Daten der benoetigten Vertices lesen

    if ((data_fp = Covise::fopen(data_Path, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, data_Path);
        Covise::sendError(buf);
        return;
    }

    fgets(buf, 300, data_fp); //Die erste Zeile eliminieren

    for (t = 0; t < 1; t++)
    {
        if (Mesh != NULL)
        {
            mesh = new coDoUnstructuredGrid(Mesh, n_el, n_el * 8, n_co, 0);
            if (mesh->objectOk())
            {
                mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
                //mesh->getTypeList(&tl);
                if (Veloc != 0)
                {
                    veloc = new coDoVec3(Veloc, n_co);
                    if (veloc->objectOk())
                    {
                        veloc->getAddresses(&u, &v, &w);
                        if (Press != 0)
                        {
                            press = new coDoFloat(Press, n_co);
                            if (press->objectOk())
                            {
                                press->getAddress(&p);
                                for (i = 1; i <= n_coord; i++)
                                {
                                    fgets(buf, 300, data_fp);
                                    sscanf(buf, "%5d%12f%12f%12f%12f%12f%12f%12f",
                                           &tmpi, &X, &Y, &Z, &UU, &VV, &WW, &PP);

                                    if (neuindex[i] != -2)
                                    {
                                        x_coord[neuindex[i]] = X;
                                        y_coord[neuindex[i]] = Y;
                                        z_coord[neuindex[i]] = Z;
                                        u[neuindex[i]] = UU;
                                        v[neuindex[i]] = VV;
                                        w[neuindex[i]] = WW;
                                        p[neuindex[i]] = PP;
                                    }
                                    // alle anderen Zeilen ueberspringen
                                } //End der FOR_Schleife

                                // vl-Liste erzeugen
                                for (i = 0; i < 8 * n_el; i++)
                                    if (neuindex[altindex[i]] != -2)
                                        vl[i] = neuindex[altindex[i]];

                                //el-Liste erzeugen

                                for (k = 0; k < n_el; k++)
                                    el[k] = 8 * k;
                            }
                            else
                            {
                                Covise::sendError("ERROR: creation of data object 'pressure' failed");
                                return;
                            }
                        }
                        else
                        {
                            Covise::sendError("ERROR: Object name not correct for 'pressure'");
                            return;
                        }
                    }
                    else
                    {
                        Covise::sendError("ERROR: creation of data object 'velocity' failed");
                        return;
                    }
                }
                else
                {
                    Covise::sendError("ERROR: Object name not correct for 'velocity'");
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

        delete[] altindex;
        delete[] neuindex;
        /*
         delete [] x_coord;
         delete [] y_coord;
         delete [] z_coord;
         delete [] u;
         delete [] v;
         delete [] w;
         delete [] p;
         delete [] el;
         delete [] vl;
      */
        fclose(data_fp); // File.INI schliessen

    } // for (t=0; ...)
}
