/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CARBO_H
#define _CARBO_H

#include "nrutil.h"

#include "Carbo.h"

class Lic : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();

    void doPolygons(coDoPolygons **polygon,
                    coDoVec3 **vectors,
                    int dimension, float size);

    //  member data

    coOutputPort *polygonOutPort;
    coOutputPort *vectorOutPort;

    //  parameters

    coIntScalarParam *resolution;
    coFloatParam *domainSize;

public:
    Carbo();
    virtual ~Carbo();
};
#endif
