/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / file Lic.h

#ifndef _CARBO_H
#define _CARBO_H

#include "nrutil.h"

#include <api/coModule.h>
using namespace covise;
//#include "Carbo.h"

// LineIntegralConvolution
class Carbo : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();

    void doPolygons(coDoPolygons **polygon,
                    coDoVec3 **vectors,
                    int dimension, float size);
    void carbonDioxide(fvec &ePotential, f2ten &eField, float ucharge,
                       float size, const f2ten &coord);

    //see W.H. Press et al.: Numerical Recipes in C, 2nd edition, pg. 282
    //period > 2*10^8 (or 2*10**8 for "FORTRANers)
    float random2(long *idum);
    inline int ran2int(float number, int steps)
    {
        int value = static_cast<int>(((steps - 1) * number) + 0.5);
        return value;
    };

    //matrix(3,5) - col 4 -> row_scaling, col 5 -> col_pivoting !!
    f2ten gauss3D(f2ten matrix);

    //  member data

    coOutputPort *polygonOutPort;
    //coOutputPort* gridOutPort;
    coOutputPort *vectorOutPort;

    //  parameters

    coIntScalarParam *resolution;
    coFloatParam *domainSize;
    //coFloatParam* distortion;

public:
    Carbo();
    virtual ~Carbo();
};
#endif
