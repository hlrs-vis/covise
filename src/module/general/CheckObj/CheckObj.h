/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHECK_OBJ_H
#define _CHECK_OBJ_H

/**************************************************************************\ 
 **                                                      (C)2000 Vircinity **
 **                                                                        **
 ** Description:  COVISE CheckObj application module                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:  Sasha Cioringa                                                **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Date:  28.10.00  V1.0                                                  **
 ** Last:                                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoLines.h>

class CheckObj : public coSimpleModule
{

private:
    // private member functions

    virtual int compute(const char *port);

    //parameters
    coBooleanParam *p_report;

    // ports
    coInputPort *p_inPort;

    int n_el, n_corners, n_coord;
    int all_errors, final_error;
    int *el, *cl, *tl;
    float *x_coord, *y_coord, *z_coord;

    int check_unstructuredgrid(const coDoUnstructuredGrid *);
    int check_polygons(const coDoPolygons *);
    int check_lines(const coDoLines *);
    int check_trianglestrips(const coDoTriangleStrips *);
    int check_all(int *, int *, int, int, int);

public:
    CheckObj(int argc, char *argv[]);
};
#endif // _CHECK_OBJ_H
