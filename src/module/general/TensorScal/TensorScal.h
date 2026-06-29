/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TENS_SCAL_NEW_H
#define _TENS_SCAL_NEW_H

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class TensScal : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    // parameters

    coChoiceParam *p_option;
    coChoiceParam *p_voption;

    // ports
    coInputPort *p_inPort;
    coOutputPort *p_outPort;
    coOutputPort *p_voutPort;

    // private functions
    // to be moved to the Tensor class in the library.
    float *S2D_Spur(int nopoints, const float *t_addr);
    float *S3D_Spur(int nopoints, const float *t_addr);
    float *F2D_Spur(int nopoints, const float *t_addr);
    float *F3D_Spur(int nopoints, const float *t_addr);
    float *S2D_Stress(int nopoints, const float *t_addr);
    float *S3D_Stress(int nopoints, const float *t_addr);
    void  S3D_Principal(int nopoints, const float *t_addr, int voption,
			float *xv, float *yv, float *zv);
    float *S3D_Principal_Scalar(int nopoints, const float *t_addr, int option);
public:
    TensScal(int argc, char *argv[]);
};
#endif
