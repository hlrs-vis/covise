/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:DIABLO calculation module         	                  **
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
#include "Diablo.h"
#include "string.h"
#include "Diablo_util.h"
#include <stdio.h>

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

//..........................................................................
//
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

int Application::start_FIDAP()
{
    FILE *ptr;
    char buf[300];
    int err;
    // FIDAP ueber script starten

    Covise::sendInfo("FIDAP wird gestartet ...");

    sprintf(buf, "cp %s ~/covise/data/sfb374/fidap/FDREAD.temp\n", prjname);
    if (system(buf) == -1)
    {
        sprintf(buf, "Fehler >> Geometriefile %s existiert nicht", prjname);
        Covise::sendInfo(buf);
        return (-1);
    }

    // Skript ueber systemaufruf starten
    ptr = popen("~/covise/data/sfb374/fidap/calc", "r");
    if (ptr == NULL)
        return (-1);

    while (fgets(buf, 300, ptr) != NULL)
        Covise::sendInfo(buf);

    err = pclose(ptr);

    sprintf(buf, "mv ~/covise/data/sfb374/fidap/temp.FIOUT %s\n", DATAPATH);
    if (system(buf) == -1)
    {
        Covise::sendInfo("Fehler >> Fehler in Skript");
        return (-1);
    }
    sprintf(buf, "mv ~/covise/data/sfb374/fidap/temp.FDNEUT %s\n", DATAPATH);
    if (system(buf) == -1)
    {
        Covise::sendInfo("Fehler >> Neutralfile konnte nicht erstellt werden");
        return (-1);
    }

    sprintf(buf, "mv ~/covise/data/sfb374/fidap/temp.FIOUT.nodes %s\n", DATAPATH);
    system(buf);

    Covise::sendInfo("FIDAP beendet.");
    return (0);
}

// COMMON Blocks

struct S1
{
    float MS[5500], MF[5500], DTKRIT[5500];
    int HMART[5500], HAUST[5500];
} MART_;

struct S2
{
    int HZWI[5500][5500];
} ZWISCH_;

struct S3
{
    double NTAUZT[100], NCCZTA[100], TBEZZT[100];

    int TAUZTA[100], CCZTA[100], AC3ZTA[100][100];
} ZTA_;

void Application::compute(void *)
{
    // Diablo FORTRAN Variablen
    char ZTANAME[120], MARTNAME[120], ZWNAME[120], DIFFNAME[120], FIOUT1[120],
        FIOUT2[120];
    double CCNODE, LKORN, FMIKRO, CCMAX, TSCHM, TX;
    int FMESH, NSTEPS, CCRESM, CCRESZ, DTRESZ, EXEMOD;
    int HDEF0, HDEFWG, HDEFTS, NDNMAX, MVNMAX;

    // Diablo FORTRAN interne Variablen
    double NDX[5500], NDY[5500], NDZ[5500], MVTIME[500], TIME1,
        TIME2, TEMP1, TAUA3X, TEMPERATURES[500][5500], DTMAX;
    int NDID[5500], NDN, NDACT, NDHARD, MVN, ERRFLG;

    // Diablo C Variablen
    float tmpf;
    int node;
    FILE *f;
    char *tmpstr, buf[300];
    float *dat;

    // read input parameters and data object name
    Covise::get_boolean_param("Launch_FIDAP", &FIDAP_launch);
    Covise::get_browser_param("DataPath", &tmpstr);
    strcpy(DATAPATH, tmpstr);
    Covise::get_string_param("Projektname", &tmpstr);
    strcpy(prjname, DATAPATH);
    strcat(prjname, tmpstr);
    Covise::get_choice_param("Execution-Mode", &EXEMOD);
    Covise::get_scalar_param("TSchmelz", &tmpf);
    TSCHM = double(tmpf);
    Covise::get_scalar_param("HDefault", &HDEF0);
    Covise::get_scalar_param("HWeichgl", &HDEFWG);
    Covise::get_scalar_param("HSchmelz", &HDEFTS);
    Covise::get_scalar_param("CGehalt", &tmpf);
    CCNODE = double(tmpf);
    Covise::get_scalar_param("cCMax", &tmpf);
    CCMAX = double(tmpf);
    Covise::get_scalar_param("lKorn", &tmpf);
    LKORN = double(tmpf);
    Covise::get_scalar_param("fMikro", &tmpf);
    FMIKRO = double(tmpf);
    Covise::get_scalar_param("fMesh", &FMESH);
    Covise::get_scalar_param("nSteps", &NSTEPS);
    Covise::get_scalar_param("TAbschreck", &tmpf);
    TX = double(tmpf);
    Covise::get_scalar_param("cCResMart", &CCRESM);
    Covise::get_scalar_param("cCResZwi", &CCRESZ);
    Covise::get_scalar_param("dTResZwi", &DTRESZ);
    strcpy(ZTANAME, DATAPATH);
    strcat(ZTANAME, "ZTA");
    strcpy(MARTNAME, DATAPATH);
    strcat(MARTNAME, "Martensitdaten");
    strcpy(ZWNAME, DATAPATH);
    strcat(ZWNAME, "Zwischengefuege");
    strcpy(DIFFNAME, DATAPATH);
    strcat(DIFFNAME, "Diffusionskoeff");
    strcpy(geofile, DATAPATH);
    strcat(geofile, "temp.FDNEUT");

    Data = Covise::get_object_name("data");
    Mesh = Covise::get_object_name("mesh");

    Covise::sendInfo("*****           D I A B L O wird gestartet  ....");

    if (FIDAP_launch == 1)
    {
        // FIDAP starten
        if (start_FIDAP() != 0)
        {
            Covise::sendError("FIDAP-skript konnt nicht gestartet werden");
            return;
        }
    }

    // Daten reservieren --- algemeiner Init
    if ((geo_fp = fopen(geofile, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, geofile);
        Covise::sendError(buf);
        return;
    }
    if (get_meshheader(geo_fp, &n_coord, &n_elem, &n_groups, &n_conn) != 0)
    {
        strcpy(buf, "FEHLER: Falsches Dateiformat");
        strcat(buf, geofile);
        Covise::sendError(buf);
        return;
    }
    if (Mesh != NULL)
    {
        mesh = new coDoUnstructuredGrid(Mesh, n_elem, n_conn, n_coord, 1);
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
    // Geoemetrie einlesen
    if (get_geometrie(geo_fp, n_coord, n_groups, x_coord, y_coord, z_coord,
                      vl, el, tl) != 0)
    {
        strcpy(buf, "ERROR:  Error in file >> ");
        strcat(buf, geofile);
        Covise::sendError(buf);
        return;
    }

    // Diablo code
    MVNMAX = 500;
    NDNMAX = 5500;
    ERRFLG = 0;

#ifdef _SGI
    rdmart_(MARTNAME, &CCMAX, &CCRESM, &TSCHM, &DTMAX, &ERRFLG, 120);
#endif
#ifdef _CRAY
    RDMART(MARTNAME, &CCMAX, &CCRESM, &TSCHM, &DTMAX, &ERRFLG, 120);
#endif
    if (ERRFLG != 0)
    {
        strcpy(buf, "Error in File ");
        strcat(buf, MARTNAME);
        Covise::sendError(buf);
        return;
    }
    else
    {
        Covise::sendInfo("Martensitdaten eingelesen ...");
    }
#ifdef _SGI
    rdzw_(ZWNAME, &CCMAX, &CCRESZ, &DTMAX, &DTRESZ, &ERRFLG, 120);
#endif
#ifdef _CRAY
    RDZW(ZWNAME, &CCMAX, &CCRESZ, &DTMAX, &DTRESZ, &ERRFLG, 120);
#endif
    if (ERRFLG != 0)
    {
        strcpy(buf, "Error in File ");
        strcat(buf, ZWNAME);
        Covise::sendError(buf);
        return;
    }
    else
    {
        Covise::sendInfo("Zwischengefuegedaten eingelesen ...");
    }
#ifdef _SGI
    rdzta_(ZTANAME, &TSCHM, &ERRFLG, 120);
#endif
#ifdef _CRAY
    RDZTA(ZTANAME, &TSCHM, &ERRFLG, 120);
#endif
    if (ERRFLG != 0)
    {
        strcpy(buf, "Error in File ");
        strcat(buf, ZTANAME);
        Covise::sendError(buf);
        return;
    }
    else
    {
        Covise::sendInfo("ZTAdaten eingelesen ...");
    }
    strcpy(FIOUT1, DATAPATH);
    strcat(FIOUT1, "temp.FIOUT.nodes");
    strcpy(FIOUT2, DATAPATH);
    strcat(FIOUT2, "temp.FIOUT");
#ifdef _SGI
    prep_(FIOUT1, FIOUT2, NDID, NDX, NDY, NDZ, &NDN, &NDNMAX, MVTIME, &MVN, &MVNMAX,
          &TEMPERATURES[0], &ERRFLG, 120, 120);
#endif
#ifdef _CRAY
    PREP(FIOUT1, FIOUT2, NDID, NDX, NDY, NDZ, &NDN, &NDNMAX, MVTIME, &MVN, &MVNMAX,
         &TEMPERATURES[0], &ERRFLG, 120, 120);
#endif
    if (ERRFLG != 0)
    {
        strcpy(buf, "Error in File ");
        strcat(buf, FIOUT1);
        Covise::sendError(buf);
        return;
    }
    else
    {
        Covise::sendInfo("FIOUT aufbereitet ...");
    }

    // Main Loop
    strcpy(buf, DATAPATH);
    strcat(buf, "HAERTE");
    f = fopen(buf, "w");
    NDACT = 1;
    NDHARD = 0;
    for (node = 1; node < n_coord + 1; node++)
    {
        if (NDID[NDACT - 1] == node) /* -1 wegen fortran */
        {
            sprintf(buf, "Calculating node %d of %d", NDACT, NDN);
            Covise::sendInfo(buf);
#ifdef _SGI
            wrtver_(&NDACT, &NDN, MVTIME, &MVN, &CCNODE, &CCMAX, &CCRESM, &TEMP1,
                    &TIME1, &TAUA3X, &TSCHM, &TIME2, &NDNMAX, &MVNMAX, TEMPERATURES,
                    &ERRFLG);
#endif
#ifdef _CRAY
            WRTVER(&NDACT, &NDN, MVTIME, &MVN, &CCNODE, &CCMAX, &CCRESM, &TEMP1,
                   &TIME1, &TAUA3X, &TSCHM, &TIME2, &NDNMAX, &MVNMAX, TEMPERATURES,
                   &ERRFLG);
#endif
            if (ERRFLG != 0)
            {
                if (ERRFLG == 1)
                {
                    NDHARD = HDEF0;
                }
                if ((ERRFLG >= 2) && (ERRFLG <= 3))
                {
                    NDHARD = HDEFWG;
                }
                if (ERRFLG == 4)
                {
                    NDHARD = HDEFTS;
                }
            }
            else
            {
#ifdef _SGI
                dif1dim_(DIFFNAME, TEMPERATURES, &MVN, &NDACT, &CCNODE, &CCMAX, &LKORN,
                         &FMESH, &NSTEPS, &TEMP1,
                         &TIME1, &TIME2, MVTIME, &EXEMOD, &ERRFLG, 120);
                hard_(TEMPERATURES, MVTIME, &NDACT, &MVN, &NDHARD, &TSCHM, &TX, &CCMAX,
                      &CCRESM, &CCRESZ, &DTRESZ, &DTMAX, &EXEMOD, &CCNODE, &FMESH,
                      &TAUA3X, &ERRFLG);
#endif
#ifdef _CRAY
                DIF1DIM(DIFFNAME, TEMPERATURES, &MVN, &NDACT, &CCNODE, &CCMAX, &LKORN,
                        &FMESH, &NSTEPS, &TEMP1,
                        &TIME1, &TIME2, MVTIME, &EXEMOD, &ERRFLG, 120);
                HARD(TEMPERATURES, MVTIME, &NDACT, &MVN, &NDHARD, &TSCHM, &TX, &CCMAX,
                     &CCRESM, &CCRESZ, &DTRESZ, &DTMAX, &EXEMOD, &CCNODE, &FMESH,
                     &TAUA3X, &ERRFLG);
#endif
            }
            ERRFLG = 0;
            fprintf(f, "%d %d\n", NDID[NDACT - 1], NDHARD);
            *dat = NDHARD;
            dat++;
            NDACT++;
        }
        else
        {
            *dat = 100;
            dat++;
        }
    }
    fclose(f);
    delete data;
    delete mesh;
}
