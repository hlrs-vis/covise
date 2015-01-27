/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / file Lic.h

#ifndef _VECTORS_H
#define _VECTORS_H

/***************************************************************************\ 
 **                                                           (C)2001 RUS **
 **                                                                       **
 ** Description:   COVISE LineIntegralConvolution application module      **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 ** Author: M. Muench                                                     **
 **                                                                       **
 ** History:                                                              **
 ** xx. ???? 01         v1                                                **
 ** xxxxxxxx         new covise api                                       **
\***************************************************************************/

#include "nrutil.h"

/***********************************\ 
 *                                 *
 *  place the #include files here  *
 *                                 *
\***********************************/

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
