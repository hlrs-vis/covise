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
#include "ReadFidap.h"
//#define DEBUGMODE 1

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
    FILE *NeutralFile;
    int i, tmpi;
    char buf[600];
    int *tb, *tbt;
    int *tb2;
    int n_group;
    int *v_list, *e_list, *t_list;
    float *xc, *yc, *zc;

    mesh = NULL;
    veloc = NULL;
    press = NULL;
    K = NULL;
    EPS = NULL;
    B_U = NULL;
    STR = NULL;

    Covise::get_browser_param("grid_path", &grid_Path);
    //Covise::get_browser_param("data_path", &data_Path);

    Mesh = Covise::get_object_name("mesh");
    Veloc = Covise::get_object_name("velocity");
    Press = Covise::get_object_name("pressure");
    K_name = Covise::get_object_name("K");
    EPS_name = Covise::get_object_name("Dissipation");
    B_U_name = Covise::get_object_name("B_U");
    STR_name = Covise::get_object_name("NUt");

    if ((NeutralFile = Covise::fopen(grid_Path, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s", grid_Path);
        return;
    }

    if (fgets(buf, 600, NeutralFile) == NULL)
        printf("fgets_1 failed in ReadFidap.cpp");
    if (strncmp(buf, "** FIDAP NEUTRAL FILE", 21) != 0)
    {
        Covise::sendError("No Valid FIDAP neutral Header");
        return;
    }
    for (i = 0; i < 3; i++)
    {
        if (fgets(buf, 600, NeutralFile) == NULL)
            printf("fgets_2 failed in ReadFidap.cpp");
#ifdef DEBUGMODE
        Covise::sendInfo(buf);
#endif
    }
    if (fgets(buf, 600, NeutralFile) == NULL)
        printf("fgets_3 failed in ReadFidap.cpp"); //NO. OF NODES   NO. ELEMENTS NO. ELT GROUPS
    if (fgets(buf, 600, NeutralFile) == NULL)
        printf("fgets_4 failed in ReadFidap.cpp");
    // now read in Dimensions
    sscanf(buf, "%d%d%d", &n_coord, &n_elem, &n_group);

    for (i = 0; i < 6; i++)
    {
        if (fgets(buf, 600, NeutralFile) == NULL)
            printf("fgets_5 failed in ReadFidap.cpp");
    }
    if (fgets(buf, 600, NeutralFile) == NULL)
        printf("fgets_6 failed in ReadFidap.cpp");
    if (strncmp(buf, "NODAL COORDINATES", 17) != 0)
    {
        Covise::sendError("No Coordinate Header");
        return;
    }

    xc = x_coord = new float[n_coord];
    yc = y_coord = new float[n_coord];
    zc = z_coord = new float[n_coord];
    el = e_list = new int[n_elem];
    tl = t_list = new int[n_elem];
    vl = v_list = new int[n_elem * 8]; // maximum of eight vertices per element (Hexaeder)
    tbt = tb = new int[n_coord];

    for (i = 0; i < n_coord; i++)
    {
        if (fgets(buf, 600, NeutralFile) == NULL)
            printf("fgets_7 failed in ReadFidap.cpp");
        if (feof(NeutralFile))
        {
            Covise::sendError("ERROR: unexpected end of file");
            return;
        }
        sscanf(buf, "%d%f%f%f\n", tbt, x_coord, y_coord, z_coord);
        x_coord++;
        y_coord++;
        z_coord++;
        tbt++;
    }
    tmpi = 0;
    for (i = 0; i < n_coord; i++)
        if (tb[i] > tmpi)
            tmpi = tb[i];
    tb2 = new int[tmpi + 1];
    tbt = tb;
    for (i = 0; i < n_coord; i++)
    {
        tb2[*tbt] = i;
        tbt++;
    }
    delete[] tb;

    if (fgets(buf, 600, NeutralFile) == NULL)
        printf("fgets_8 failed in ReadFidap.cpp");
    if (strncmp(buf, "BOUNDARY CONDITIONS", 19) == 0)
    {
#ifdef DEBUGMODE
        Covise::sendInfo("BOUNDARY CONDITIONS");
#endif
        while (1)
        {
            if (fgets(buf, 600, NeutralFile) == NULL)
                printf("fgets_9 failed in ReadFidap.cpp");
            if (feof(NeutralFile))
            {
                Covise::sendError("ERROR: unexpected end of file");
                return;
            }
            if (buf[0] != ' ')
            {
                if (strncmp(buf, "ELEMENT", 7) == 0)
                {
                    break;
                }
            }
        }
    }

    int n_vert = 0, gnum, n;
    // Read Element Groups;
    if (strncmp(buf, "ELEMENT GROUPS", 14) != 0)
    {
        Covise::sendError("No Element Group Header");
        return;
    }
    for (gnum = 0; gnum < n_group; gnum++)
    {
        int ne, type, nn;
        if (fgets(buf, 600, NeutralFile) == NULL)
            printf("fgets_10 failed in ReadFidap.cpp");
        if (strncmp(buf, "GROUP", 5) != 0)
        {
            Covise::sendError("No Element Group Header");
            return;
        }
        sscanf(buf, "GROUP: %d ELEMENTS: %d NODES: %d GEOMETRY: %d TYPE: %d", &tmpi, &ne, &nn, &type, &tmpi);

        if (fgets(buf, 600, NeutralFile) == NULL)
            printf("fgets_11 failed in ReadFidap.cpp");
#ifdef DEBUGMODE
        Covise::sendInfo(buf);
#endif
        for (i = 0; i < ne; i++)
        {
            if (fgets(buf, 300, NeutralFile) == NULL)
                printf("fgets_12 failed in ReadFidap.cpp");
            if (feof(NeutralFile))
            {
                Covise::sendError("ERROR: unexpected end of file");
                return;
            }
            if (nn == 1)
                sscanf(buf, "%d%d", &tmpi, vl);
            else if (nn == 2)
                sscanf(buf, "%d%d%dn", &tmpi, vl, vl + 1);
            else if (nn == 3)
                sscanf(buf, "%d%d%d%d", &tmpi, vl, vl + 1, vl + 2);
            else if (nn == 4)
                sscanf(buf, "%d%d%d%d%d", &tmpi, vl, vl + 1, vl + 2, vl + 3);
            else if (nn == 5)
                sscanf(buf, "%d%d%d%d%d%d", &tmpi, vl, vl + 1, vl + 2, vl + 3, vl + 4);
            else if (nn == 6)
                sscanf(buf, "%d%d%d%d%d%d%d", &tmpi, vl, vl + 1, vl + 2, vl + 3, vl + 4, vl + 5);
            else if (nn == 7)
                sscanf(buf, "%d%d%d%d%d%d%d%d", &tmpi, vl, vl + 1, vl + 2, vl + 3, vl + 4, vl + 5, vl + 6);
            else if (nn == 8)
                sscanf(buf, "%d%d%d%d%d%d%d%d%d", &tmpi, vl, vl + 1, vl + 3, vl + 2, vl + 4, vl + 5, vl + 7, vl + 6);

            for (n = 0; n < nn; n++)
            {
                *vl = tb2[*vl];
                vl++;
            }
            *el++ = n_vert;
            n_vert += nn;

            switch (type)
            {
            case 0:
                *tl++ = TYPE_BAR;
                break;
            case 1:
                *tl++ = TYPE_QUAD;
                break;
            case 2:
                *tl++ = TYPE_TRIANGLE;
                break;
            case 3:
                *tl++ = TYPE_HEXAEDER;
                break;
            case 4:
                *tl++ = TYPE_PRISM;
                break;
            case 5:
                *tl++ = TYPE_TETRAHEDER;
                break;
            }
        }
    }

    if (Mesh != NULL)
    {
        tbt = tb = new int[n_coord];
        mesh = new coDoUnstructuredGrid(Mesh, n_elem, n_vert, n_coord, e_list, v_list, xc, yc, zc, t_list);
        if (mesh->objectOk())
        {
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
    int nt = 0;
    while (!feof(NeutralFile))
    {
        if (fgets(buf, 600, NeutralFile) == NULL)
            printf("fgets_13 failed in ReadFidap.cpp");
        if (strncmp(buf, "ENDOFTIMESTEP", 13) == 0)
        {
            if (fgets(buf, 600, NeutralFile) == NULL)
                printf("fgets_14 failed in ReadFidap.cpp");
        }
        if (strncmp(buf, "TIMESTEP", 8) == 0)
        {
            if (nt)
            {
                Covise::sendError("ERROR: timesteps not supported");
                return;
            }
            nt++;
            if (fgets(buf, 600, NeutralFile) == NULL)
                printf("fgets_15 failed in ReadFidap.cpp");
        }
        if (strncmp(buf, "VELOCITY", 8) == 0)
        {
            veloc = new coDoVec3(Veloc, n_coord);
            if (veloc->objectOk())
            {
                int nr;
                n = 0;
                veloc->getAddresses(&u, &v, &w);
                for (nr = 0; nr < (n_coord * 3); nr += 5)
                {
                    n = nr / 3;
                    if (fgets(buf, 600, NeutralFile) == NULL)
                        printf("fgets_16 failed in ReadFidap.cpp");
                    if (nr + 5 <= (n_coord * 3))
                    {
                        if (nr % 3 == 0)
                            sscanf(buf, "%f%f%f%f%f", u + n, v + n, w + n, u + n + 1, v + n + 1);
                        else if (nr % 3 == 1)
                            sscanf(buf, "%f%f%f%f%f", v + n, w + n, u + n + 1, v + n + 1, w + n + 1);
                        else if (nr % 3 == 2)
                            sscanf(buf, "%f%f%f%f%f", w + n, u + n + 1, v + n + 1, w + n + 1, u + n + 2);
                    }
                    else if (nr + 4 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f%f", w + n, u + n + 1, v + n + 1, w + n + 1);
                    }
                    else if (nr + 3 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f", u + n, v + n, w + n);
                    }
                    else if (nr + 2 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f", v + n, w + n);
                    }
                    else if (nr + 1 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f", w + n);
                    }
                }
            }
        }
        if (strncmp(buf, "PRESSURE", 8) == 0)
        {
            press = new coDoFloat(Press, n_coord);
            if (press->objectOk())
            {
                int nr = 0;
                n = 0;
                press->getAddress(&u);
                for (n = 0; n < n_coord; n += 5)
                {
                    if (fgets(buf, 600, NeutralFile) == NULL)
                        printf("fgets_17 failed in ReadFidap.cpp");
                    if (nr + 5 <= n_coord)
                    {
                        sscanf(buf, "%f%f%f%f%f", u + n, u + n + 1, u + n + 2, u + n + 3, u + n + 4);
                    }
                    else if (nr + 4 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f%f", u + n, u + n + 1, u + n + 2, u + n + 3);
                    }
                    else if (nr + 3 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f", u + n, u + n + 1, u + n + 2);
                    }
                    else if (nr + 2 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f", u + n, u + n + 1);
                    }
                    else if (nr + 1 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f", u + n);
                    }
                }
            }
        }
        /*if(strncmp(buf,"TEMPERATURE",11)==0)
        {
        press = new coDoFloat(Temp, n_coord);
        if (press->objectOk())
        {
        int nr;
        n=0;
        press->getAddress(&u);
        for(n=0;n<n_coord;n+=5)
        {
        if (fgets(buf,600,NeutralFile)==NULL)
        printf("fgets_18 failed in ReadFidap.cpp");
        if(nr+5<=n_coord)
        {
        sscanf(buf,"%f%f%f%f%f",u+n,u+n+1,u+n+2,u+n+3,u+n+4);
        }
        else if(nr+4<=(n_coord*3))
        {
        sscanf(buf,"%f%f%f%f",u+n,u+n+1,u+n+2,u+n+3);
        }
        else if(nr+3<=(n_coord*3))
        {
        sscanf(buf,"%f%f%f",u+n,u+n+1,u+n+2);
        }
        else if(nr+2<=(n_coord*3))
        {
        sscanf(buf,"%f%f%",u+n,u+n+1);
        }
        else if(nr+1<=(n_coord*3))
        {
        sscanf(buf,"%f",u+n);
        }
        }
        }
        }*/
        if (strncmp(buf, "TURBULENT K.E.", 14) == 0)
        {
            K = new coDoFloat(K_name, n_coord);
            if (K->objectOk())
            {
                int nr = 0;
                n = 0;
                K->getAddress(&u);
                for (n = 0; n < n_coord; n += 5)
                {
                    if (fgets(buf, 600, NeutralFile) == NULL)
                        printf("fgets_19 failed in ReadFidap.cpp");
                    if (nr + 5 <= n_coord)
                    {
                        sscanf(buf, "%f%f%f%f%f", u + n, u + n + 1, u + n + 2, u + n + 3, u + n + 4);
                    }
                    else if (nr + 4 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f%f", u + n, u + n + 1, u + n + 2, u + n + 3);
                    }
                    else if (nr + 3 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f", u + n, u + n + 1, u + n + 2);
                    }
                    else if (nr + 2 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f", u + n, u + n + 1);
                    }
                    else if (nr + 1 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f", u + n);
                    }
                }
            }
        }
        if (strncmp(buf, "TURBULENT DISSIPATION", 21) == 0)
        {
            EPS = new coDoFloat(EPS_name, n_coord);
            if (EPS->objectOk())
            {
                int nr = 0;
                n = 0;
                EPS->getAddress(&u);
                for (n = 0; n < n_coord; n += 5)
                {
                    if (fgets(buf, 600, NeutralFile) == NULL)
                        printf("fgets_20 failed in ReadFidap.cpp");
                    if (nr + 5 <= n_coord)
                    {
                        sscanf(buf, "%f%f%f%f%f", u + n, u + n + 1, u + n + 2, u + n + 3, u + n + 4);
                    }
                    else if (nr + 4 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f%f", u + n, u + n + 1, u + n + 2, u + n + 3);
                    }
                    else if (nr + 3 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f%f", u + n, u + n + 1, u + n + 2);
                    }
                    else if (nr + 2 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f%f", u + n, u + n + 1);
                    }
                    else if (nr + 1 <= (n_coord * 3))
                    {
                        sscanf(buf, "%f", u + n);
                    }
                }
            }
        }
    }

    fclose(NeutralFile);

    delete[] xc;
    delete[] yc;
    delete[] zc;
    delete[] e_list;
    delete[] v_list;
    delete[] tb2;

    delete mesh;
    delete veloc;
    delete press;
    delete K;
    delete EPS;
    delete B_U;
    delete STR;
}
