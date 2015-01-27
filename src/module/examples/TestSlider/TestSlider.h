/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TESTSLIDER_H
#define _TESTSLIDER_H

#include <api/coModule.h>
using namespace covise;

class TestSlider : public coModule
{
private:
    float cx, cy, cz; // coodinates of the cubes p_center
    float sMin, sMax, sVal; // edge length / 2 of the cube

    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    //  Ports
    coOutputPort *p_polyOut;
    coFloatVectorParam *p_center;
    coFloatSliderParam *p_cusize;

public:
    TestSlider(int argc, char *argv[]);
    virtual ~TestSlider();
};
#endif
