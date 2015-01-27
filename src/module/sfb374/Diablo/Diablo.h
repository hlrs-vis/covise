/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READDIABLO_H
#define _READDIABLO_H
// _SGI oder _CRAY
#define _SGI
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  Diablo calculation module         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                              Christoph Kunz                            **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    int start_FIDAP();

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Parameter names
    char *Data;
    char *Mesh;

    //  Local data
    FILE *geo_fp;
    int n_coord, n_elem, n_groups, n_conn, FIDAP_launch;
    int *el, *vl, *tl;
    float *x_coord;
    float *y_coord;
    float *z_coord;
    char prjname[100], geofile[100], DATAPATH[100];

    //  Shared memory data
    coDoUnstructuredGrid *mesh;
    coDoFloat *data;

public:
    Application(int argc, char *argv[])

    {
        Data = 0L;
        Mesh = 0L;

        Covise::set_module_description("DIABLO Haerteberechnung");
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "grid");
        Covise::add_port(OUTPUT_PORT, "data", "coDoFloat", "data");
        Covise::add_port(PARIN, "Launch_FIDAP", "Boolean", "FIDAP starten");
        Covise::set_port_default("Launch_FIDAP", "FALSE");
        Covise::add_port(PARIN, "DataPath", "Browser", "Diablo Data Path");
        Covise::set_port_default("DataPath", "data/sfb374/diablo/ *");
        Covise::add_port(PARIN, "Projektname", "String", "Projektname");
        Covise::set_port_default("Projektname", "demo.GEO");
        Covise::add_port(PARIN, "Execution-Mode", "Choice", "Execution Mode");
        Covise::set_port_default("Execution-Mode", "5 Check RunDiff RunFast 1DimLin 1DimKug");
        Covise::add_port(PARIN, "TSchmelz", "Scalar", "TSchmelz");
        Covise::set_port_default("TSchmelz", "1720.0");
        Covise::add_port(PARIN, "HDefault", "Scalar", "HDefault");
        Covise::set_port_default("HDefault", "250");
        Covise::add_port(PARIN, "HWeichgl", "Scalar", "HWeichgl");
        Covise::set_port_default("HWeichgl", "250");
        Covise::add_port(PARIN, "HSchmelz", "Scalar", "HSchmelz");
        Covise::set_port_default("HSchmelz", "0");
        Covise::add_port(PARIN, "CGehalt", "Scalar", "CGehalt");
        Covise::set_port_default("CGehalt", "0.39");
        Covise::add_port(PARIN, "cCMax", "Scalar", "cCMax");
        Covise::set_port_default("cCMax", "0.78");
        Covise::add_port(PARIN, "lKorn", "Scalar", "lKorn");
        Covise::set_port_default("lKorn", "5.0");
        Covise::add_port(PARIN, "fMikro", "Scalar", "fMikro");
        Covise::set_port_default("fMikro", "5.0");
        Covise::add_port(PARIN, "fMesh", "Scalar", "fMesh");
        Covise::set_port_default("fMesh", "20");
        Covise::add_port(PARIN, "nSteps", "Scalar", "nSteps");
        Covise::set_port_default("nSteps", "20");
        Covise::add_port(PARIN, "TAbschreck", "Scalar", "TAbschreck");
        Covise::set_port_default("TAbschreck", "300.0");
        Covise::add_port(PARIN, "cCResMart", "Scalar", "cCResMart");
        Covise::set_port_default("cCResMart", "100");
        Covise::add_port(PARIN, "cCResZwi", "Scalar", "cCResZwi");
        Covise::set_port_default("cCResZwi", "50");
        Covise::add_port(PARIN, "dTResZwi", "Scalar", "dTResZwi");
        Covise::set_port_default("dTResZwi", "50");
        Covise::init(argc, argv);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_start_callback(Application::computeCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _READDiablo_H
