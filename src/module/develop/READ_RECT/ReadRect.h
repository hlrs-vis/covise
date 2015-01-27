/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_RECT_H
#define _READ_RECT_H
/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Read module Rect Norsk Hydro data      	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 *\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <api/coModule.h>
using namespace covise;

class Application : public coModule
{

private:
    //  member functions
    virtual int compute();

    coOutputPort *p_outPort1;
    coOutputPort *p_outPort2;
    coOutputPort *p_outPort3;

    coFileBrowserParam *gridfileParam;
    coFileBrowserParam *sdatafileParam;
    coFileBrowserParam *vdatafileParam;
    coIntScalarParam *p_timesteps;
    coIntScalarParam *p_skip;

    int read_grid(FILE *fp, const char *gridName, int x_dim, int y_dim, int z_dim, coDoRectilinearGrid **gridObj);
    int read_scalar(FILE *fp, const char *scalarName, int x_dim, int y_dim, int z_dim, coDoFloat **scalarObj);
    int read_vector(FILE *fp, const char *vectorName, int x_dim, int y_dim, int z_dim, coDoVec3 **vectorObj);

public:
    Application();
    virtual ~Application();
};
#endif
