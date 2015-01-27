/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUT_SOLL_IST_H
#define _CUT_SOLL_IST_H

#include <stdlib.h>
#include <stdio.h>

#include <api/coModule.h>
using namespace covise;

////// our class
class CutSollIst : public coModule
{
private:
    /////////ports
    coInputPort *p_geo_in;
    coOutputPort *p_geo_out, *p_cut1, *p_cut2;

    coFloatParam *distance, *thick;
    coFloatVectorParam *normal;

    virtual int compute();

public:
    CutSollIst();
};
#endif
