/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SIMPLIFY_H
#define _SIMPLIFY_H

#include <api/coSimpleModule.h>
using namespace covise;

extern float *x_in, *y_in, *z_in, *s_in, *u_in, *v_in, *w_in, *nu_in, *nv_in, *nw_in;

class SimplifySurface : public coSimpleModule
{

private:
    coInputPort *p_meshIn, *p_dataIn, *p_normalsIn;
    coOutputPort *p_meshOut, *p_dataOut, *p_normalsOut;

    coChoiceParam *param_strategy;
    coFloatParam *param_percent, *param_volume_bound, *param_feature_angle;

    void copyAttributesToOutObj(coInputPort **input_ports, coOutputPort **output_ports, int n);

public:
    SimplifySurface(int argc, char *argv[]);
    virtual int compute();

    coDistributedObject **HandleObjects(coDistributedObject *mesh_object, coDistributedObject *data_object, coDistributedObject *normal_object, const char *Mesh_out_name, const char *Data_out_name, const char *Normal_out_name);
};
#endif // _SIMPLIFY_H
