/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUT_GEOMETRY_H
#define _CUT_GEOMETRY_H

#include <stdlib.h>
#include <stdio.h>

#include <api/coModule.h>
using namespace covise;

typedef class Lines_
{
private:
    float *x_, *y_, *z_;
    int *ll_, *cl_;

    int num_ll_, num_cl_, num_points_;

    float *data_;

    enum
    {
        NUMT = 16000,
        PARTS = 30
    };

public:
    Lines_();
    ~Lines_();

    void addLine(int num, float *x, float *y, float *z, float *val);

    coDoLines *getDOLines(const char *objname);
    coDoFloat *getDOData(const char *objname);
} Lines;

////// our class
class Magma_Trace : public coModule
{
private:
    /////////ports
    coInputPort *p_geo_in, *p_data_in;
    coOutputPort *p_geo_out, *p_data_out;

    coIntScalarParam *p_len, *p_skip;
    virtual int compute();
    coDistributedObject *traceLines(const char *name, coDoSet *all_points, const char *data_name, coDoSet *all_data, coDistributedObject **data_out);

public:
    Magma_Trace();
};
#endif
