/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TUBE_NEW_H
#define _TUBE_NEW_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE Tube application module                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau, Sasha Cioringa                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

//#include <api/coModule.h>
#include <api/coSimpleModule.h>
using namespace covise;

class Tube : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    // parameters
    coFloatParam *p_radius;
    coIntScalarParam *p_parts;
    coChoiceParam *p_option;
    coBooleanParam *p_limitradius;
    coFloatParam *p_maxradius;

    // ports
    coInputPort *p_inLines;
    coInputPort *p_inData;
    coInputPort *p_inDiameter;
    coOutputPort *p_outPoly;
    coOutputPort *p_outNormals;
    coOutputPort *p_outData;

    // private data

    void create_tube();
    void create_stopper();
    void create_arrow();
    void calc_cone_base(int, int, float, float, float *, float *, float *);

    static const int tri;
    static const int open_cylinder;
    static const int closed_cylinder;
    static const int arrows;
    static const float arr_radius_coef; //arrow radius = arr_radius_coef*radius
    static const float arr_angle; //halb angle of the arrow
    static const float arr_hmax; //the arrow can be <= arr_hmax*delta_length
    float cos_array[1024];
    float sin_array[1024];

    int ngon, gennormals, option_param;
    int nlinesWithVertices;
    int npoint, nvert, nlines;
    int nt_point, nt_vert, nt_poly; // Nr of points,vertex,polys for tubes(mantle)
    int na_point, na_vert, na_poly; //Nr of points,vertex,polys for the arrows
    int ns_point, ns_vert, ns_poly; // Nr of points,vert,poly for stoppers
    int nts_vert, nts_poly; // =ntvert+ns_vert and =npoly+ns_poly
    int *vl, *ll;
    int *vt, *lt;

    float radius;
    float maxradius;
    int limitradius;
    float *xl, *yl, *zl;
    float *xt, *yt, *zt;
    float *xn, *yn, *zn;

    float *s_in_changed;
    float *s_out;
    float *s_in;
    int *colors_in_changed;
    int *colors_out;
    int *colors_in;

public:
    Tube(int argc, char *argv[]);
};
#endif
