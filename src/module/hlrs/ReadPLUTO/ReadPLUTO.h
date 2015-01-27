/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READPLUTO_H
#define _READPLUTO_H
/**************************************************************************\ 
**                                                           (C)2011 HLRS **
**                                                                        **
** Description: Read module for PLUTO data         	                   **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                   Steffen Brinkmann, Uwe Woessner                      **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
** Date:  22.11.11  V1.0                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#ifdef BYTESWAP
#define bswap(x) byteSwap(x)
#else
#define bswap(x)
#endif

class ReadPLUTO : public coSimpleModule
{
private:
    coOutputPort *p_mesh;
    coOutputPort *p_rho;
    coOutputPort *p_rholog;
    coOutputPort *p_pressure;
    coOutputPort *p_pressurelog;
    coOutputPort *p_velocity;
    coOutputPort *p_magfield;
    coOutputPort *p_velocity_cart;
    coOutputPort *p_magfield_cart;

    coFileBrowserParam *p_path;
    coChoiceParam *p_precision;
    coChoiceParam *p_file_format;
    coIntScalarParam *p_tbeg;
    coIntScalarParam *p_tend;
    coIntScalarParam *p_skip;
    coBooleanParam *p_axisymm;
    coIntScalarParam *p_n_axisymm;

    virtual int compute(const char *port);
    int openDataFile(int *fd, const char *dataPath);
    int readData(int fd, float *data);
    int readData(int fd, float *rho, float *pr,
                 float *v1, float *v2, float *v3,
                 float *b1, float *b2, float *b3);

    //  Parameter names
    const char *dataPath;
    string dir_path;
    const char *mesh_name;

    const char *rho_name;
    const char *rholog_name;
    const char *pr_name;
    const char *prlog_name;
    const char *vel_name;
    const char *magfield_name;
    const char *vel_cart_name;
    const char *magfield_cart_name;

    //  Local data
    int n_x1;
    int n_x2;
    int n_x3;
    int readDouble;
    int fileFormat;
    int axisymm;
    int n_axisymm;

    float *vec_gridx1;
    float *vec_gridx2;
    float *vec_gridx3;
    float *vec_gridx1_glob;
    float *vec_gridx2_glob;
    float *vec_gridx3_glob;

    float *rho, *pr;
    float *rholog, *prlog;
    float *v1, *v2, *v3;
    float *b1, *b2, *b3;
    float *vx, *vy, *vz;
    float *bx, *by, *bz;

    //  Shared memory data
    coDoStructuredGrid *mesh;

    coDoFloat *DOrho;
    coDoFloat *DOpress;
    coDoFloat *DOrholog;
    coDoFloat *DOpresslog;
    coDoVec3 *DOvel;
    coDoVec3 *DOmagfield;
    coDoVec3 *DOvel_cart;
    coDoVec3 *DOmagfield_cart;

public:
    ReadPLUTO(int argc, char **argv);
    virtual ~ReadPLUTO()
    {
    }
};
#endif // _READPLUTO_H
