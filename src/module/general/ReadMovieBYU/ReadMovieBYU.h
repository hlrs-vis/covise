/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*######################################################################\ 
#                                                                       #
# Description: Read module for MovieBYU  ->.h Bibliothek Datei          #
#                                                                       #
#                             Thilo Krueger                             #
#                 Rechenzentrum Universitaet Stuttgart                  #
#                           Allmandring 30a                             #
#                            70550 Stuttgart                            #
#                 Thilo.Krueger@Rus.Uni-Stuttgart.de                    #
#                                                                       #
\######################################################################*/

#ifndef _READ_MOVIEBYU_H
#define _READ_MOVIEBYU_H

// includes

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

// hier kommen Platzhalter fuer filetype und datatype
#define _FILE_TYPE_BINARY 1
#define _FILE_TYPE_FORTRAN 2
#define _FILE_TYPE_ASCII 3

#define _FILE_SKALAR 4
#define _FILE_VEKTOR 5
#define _FILE_NONE 6
#define _FILE_DISPLACE 7

class Application
{

private:
    // callback-stuff
    static void computeCallback(void *userData, void *callbackData);

    // main
    void compute(const char *port);

    // parameters fuer die Eingabe
    char *gridpath;
    char *datapath;
    // char *partcolors;
    char *colorpath;
    int filetype, datatype; // gridtype rausgenommen
    long timesteps, delta;

    // function declarations
    // Hier kommen die voids fuer die Geometry file

    void read_first_geo(FILE *f, int *np, int *nj, int *npt, int *ncon);
    void read_second_geo(FILE *f, int np, int *&npl);
    void read_coords(FILE *f, int nj, float *&coords);
    void read_iconn(FILE *f, int ncon, int *&iconn);

    // und hier fuer die eventuellen Daten

    void read_skalar(char *datapath, int nj, float *&skalar);
    void read_vektor(char *datapath, int nj, float *&vektor);
    void read_displace(char *datapath, int nj, int time, int skip, float **&displaced);
    void read_color(char *colorpath, int np, char **&colorlist, int *colornumber);

public:
    // Hier werden Die Modul Poeppl definiert und bezeichnet

    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Read file MovieBYU  v1.42");

        // Die hier kommen unten heraus

        Covise::add_port(OUTPUT_PORT, "poly", "UnstructuredGrid", "surface out");
        Covise::add_port(OUTPUT_PORT, "data", "Float|Vec3|UnstructuredGrid", "data out");

        // Die sollen oben herein

        Covise::add_port(PARIN, "gridpath", "Browser", "file gridpath");
        Covise::set_port_default("gridpath", "~/covise *");

        Covise::add_port(PARIN, "datapath", "Browser", "file datapath");
        Covise::set_port_default("datapath", "~/covise *");

        Covise::add_port(PARIN, "colorpath", "Browser", "file colorpath");
        Covise::set_port_default("colorpath", "~/covise *");

        Covise::add_port(PARIN, "filetype", "Choice", "???");
        Covise::set_port_default("filetype", "3 Binary Fortran ASCII");

        Covise::add_port(PARIN, "gridtype", "Choice", "???");
        // no more structured i think
        Covise::set_port_default("gridtype", "1 Unstructured");

        Covise::add_port(PARIN, "datatype", "Choice", "???");
        Covise::set_port_default("datatype", "1 None Scalar Vector Displacement");

        Covise::add_port(PARIN, "timesteps", "IntScalar", "???");
        Covise::set_port_default("timesteps", "1");

        Covise::add_port(PARIN, "delta", "IntScalar", "???");
        Covise::set_port_default("delta", "1");

        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }
};
#endif // _READ_MOVIEBYU_H
