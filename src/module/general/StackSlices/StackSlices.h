/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef STACK_SLICES_H
#define STACK_SLICES_H


#include <api/coModule.h>
#include <api/coFloatParam.h>

using namespace covise;

class StackSlices : public coModule
{

private:
    virtual int compute(const char *port);

    // parameters
    coChoiceParam *p_direction;
    coFloatParam *p_sliceDistance;

    // ports
    coInputPort *p_gridIn;
    coInputPort *p_dataIn;
    coOutputPort *p_gridOut;
    coOutputPort *p_dataOut;

public:
    StackSlices(int argc, char *argv[]);
};
#endif
