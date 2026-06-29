/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUBE_H
#define _CUBE_H

#include <api/coModule.h>
using namespace covise;

class Cube : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void postInst();

    //  Ports
    coOutputPort *p_polyOut;
    coFloatVectorParam *p_center;
    coFloatParam *p_cusize;

public:
    Cube();
    virtual ~Cube();
};
#endif
