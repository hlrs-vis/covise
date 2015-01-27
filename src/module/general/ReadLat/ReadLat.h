/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:                                                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Dirk Rantzau                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

class Application
{

private:
    static void compute(void *userData, void *callbackData);
    static void quit(void *, void *)
    {
    }

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("read IRIS-Explorer Latice file");
        Covise::add_port(OUTPUT_PORT, "Gitter", "StructuredGrid", "Gitter");
        Covise::add_port(OUTPUT_PORT, "Daten", "Float", "Daten");
        Covise::add_port(PARIN, "Filename", "Browser", "file path");
        Covise::set_port_default("Filename", "data/test.lat *.lat");
        Covise::init(argc, argv);
        Covise::set_start_callback(Application::compute, this);
        Covise::set_quit_callback(Application::quit, this);
    }

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};

#define UNIFORM 0
#define RECTILINEAR 1
#define IRREGULAR 2

#define CHAR 0
#define SHORT 1
#define LONG 2
#define FLOAT 3
#define DOUBLE 4

#define ASCII 1
#define BIN 2

#define PTRLISTSIZ 30

#define LATTICE 101
#define CXDATA 102
#define CXCOORD 103
#define VALUES 200
#define BBOX 204

#define NDIM 0
#define DIMS 1
#define DATA 2
#define COORD 3
#define NDATAVAR 2
#define PRIMTYPE 3
#define D 4

#define COORDTYPE 2
#define C 3
#define SUMCOORD 4
#define PERIMCOORD 5
#define NCOORDVAR 6

#define AT -2
#define BRACKET -3
#define NUMBER -4
#define EQUAL -5
#define SKIP -6
#define READ_LAT_ERROR -7

typedef struct
{
    long Size;
    int Type;
    char *Ptr;
} _PtrList;

typedef struct
{
    int nDim;
    int dims;
    int data;
    int coord;
} _lattice;

typedef struct
{
    int nDim;
    int dims;
    int nDataVar;
    int primType;
    int data;
} _cxData;

typedef struct
{
    int nDim;
    int dims;
    int coordType;
    int bBox;
    int sumCoord;
    int perimCoord;
    int nCoordVar;
    int values;
} _cxCoord;
#endif // _APPLICATION_H
