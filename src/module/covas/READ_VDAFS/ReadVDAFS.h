/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READVDAFS_H
#define _READVDAFS_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for VDAFS data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Reiner Beller                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  09.10.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fstream.h>
#ifndef CO_hp
#include <libc.h>
#endif
#include <ctype.h>
#include "parser.h"

// ================================================================================
// !!! GLOBAL VARIABLES !!!
list<Name> name_list;
// Lists of VDAFS elements
list<Curve> curve_list;
list<Circle> circle_list;
list<Vec3d> point_list;
list<Surf> surf_list;
list<Cons> cons_list;
list<Face> face_list;
list<Top> top_list;
list<Group> group_list;
list<Set> set_list;

// Lists of NURBS data objects
list<NurbsCurve> nurbscurveList; // List of NURBS curves
list<NurbsSurface> nurbssurfaceList; // List of NURBS surfaces
list<TrimCurve> curveDefList; // List of trim curves defining the VDAFS
// elements FACEs in NURBS representation
list<NurbsSurface> surfaceDefList; // List of surfaces defining the VDAFS
// elements FACEs in NURBS representation
// Connection lists
list<int> connectionList; // connections between surfaces and
// trim loops (= closed cons ensemble)
list<int> trimLoopList; // connections between trim loops and their
// trim curve segments

// List of selected points
list<Vec3d> selectedPointList;

// Stream
ifstream *from;
//ofstream ErrFile("error.out", ios::app);
// ================================================================================

// function prototypes

extern void pars(const int fd);
extern void sort_name();
extern Name getname(string);

void setup_POINTS(int &);
coDoPoints *create_points(char *);
void setup_NCUC(int &, int &, int &);
DO_NurbsCurveCol *create_nurbs_curves(char *);
void setup_NSFC(int &, int &, int &, int &);
DO_NurbsSurfaceCol *create_nurbs_surfaces(char *);
void setup_FACEC(int &, int &, int &, int &, int &, int &, int &, int &);
DO_FaceCol *create_nurbs_faces(char *);

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

public:
    Application(int argc, char *argv[])

    {

        Covise::set_module_description("Read VDAFS files.");

        Covise::add_port(PARIN, "dataPath", "Browser", "VDAFS data file");
        Covise::set_port_default("dataPath", "./*.*");
        Covise::add_port(OUTPUT_PORT, "Geometry", "coDoGeometry", "Geometry");

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
#endif // _READVDAFS_H
