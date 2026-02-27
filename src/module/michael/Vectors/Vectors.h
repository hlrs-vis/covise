/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VECTORS_H
#define _VECTORS_H

#include "nrutil.h"

#include <api/coModule.h>
using namespace covise;

class Vectors : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();

    void doPolygons(coDoPolygons **polygon,
                    coDoVec3 **vectors,
                    int dimension, float size);
    void doVectors(f2ten &eField, const f2ten &coord);

    //  member data

    coOutputPort *polygonOutPort;
    coOutputPort *vectorOutPort;

    //  parameters

    coIntScalarParam *resolution;
    coFloatParam *domainSize;

public:
    Vectors();
    virtual ~Vectors();
};
#endif
