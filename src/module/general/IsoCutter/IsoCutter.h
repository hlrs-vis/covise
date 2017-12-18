/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _IsoCutter_H
#define _IsoCutter_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2007      ++
// ++ Description:                                                        ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Siegfried Hodri                          ++
// ++                               VISENSO                               ++
// ++                           Nobelstrasse 15                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <api/coSimpleModule.h>
#include <do/coDoData.h>
using namespace covise;

class IsoCutter : public coSimpleModule
{

private:
    //////////  member functions

    virtual int compute(const char *port);
    virtual void preHandleObjects(coInputPort **);

    ////////// ports
    coInputPort *p_inPolygons, *p_inData;
    coOutputPort *p_outPolygons, *p_outData;

    ////////// parameters
    coFloatSliderParam *p_isovalue; // iso values to cut along
    coBooleanParam *p_autominmax;
    coBooleanParam *p_cutdown; // welcher teil soll weggeschnitten werden. <iso oder >iso ?

    coDoPolygons *inPolygons;
    coDoFloat *inData;

    coDoPolygons *outPolygons;
    coDoFloat *outData;

    // direct access for incoming data
    int num_in_poly_list, num_in_corner_list, num_in_coord_list;
    int *in_poly_list, *in_corner_list, *in_coord_list;
    float *in_x_coords, *in_y_coords, *in_z_coords;
    int num_in_data; // size of incoming data
    float *in_data;

    float _min, _max;

    // direct access for outgoing data
    int num_out_poly_list, num_out_corner_list, num_out_coord_list;
    int *out_poly_list, *out_corner_list, *out_coord_list;
    float *out_x_coords, *out_y_coords, *out_z_coords;
    int num_out_data; // size of outgoing data
    float *out_data;

    void calc_cropped_polygons();
    bool add_isocropped_polygon(int poly, unsigned int num_values, float *values);

public:
    IsoCutter(int argc, char *argv[]);
};
#endif
