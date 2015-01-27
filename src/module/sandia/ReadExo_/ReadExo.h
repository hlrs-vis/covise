/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READEXO_H
#define _READEXO_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Sandia Stl data  	                  **
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
 ** Date:  17.05.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "netcdf.h"
#include "exodusII.h"

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    void parameter(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void parameterCallback(void *userData, void *callbackData);

    //  Parameter names
    char *file_Path;

    //  Local data
    int exoid, j, scal1, scal2, scal3, scal4, vert1;
    int word_size, io_word_size;
    int num_nodes, num_elem, num_elem_blks, num_node_sets, num_side_sets;
    int num_dim, num_conn, num_nodal_variables, num_element_variables, num_global_variables;
    char database_title[MAX_LINE_LENGTH + 1];
    char ex_buf[MAX_LINE_LENGTH + 1];
    float version;
    char **var_names;
    int num_timesteps, timestep;
    float *x_coord, *y_coord, *z_coord;
    int *el, *vl, *tl;
    char tb1[600];

public:
    Application(int argc, char *argv[]);

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _READEXO_H
