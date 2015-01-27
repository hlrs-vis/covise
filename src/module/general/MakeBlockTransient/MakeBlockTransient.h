/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__MAKETRANSIENT_H)
#define __MAKETRANSIENT_H

// includes
#include <api/coModule.h>
using namespace covise;

class MakeBlockTransient : public coModule
{

private:
    virtual int compute(const char *port);
    int Diagnose();
    coInputPort *p_inport;
    coInputPort *p_accordingTo;
    coOutputPort *p_outport;
    coIntScalarParam *p_blocks;
    coIntScalarParam *p_timesteps;
    coDistributedObject *in_obj_;

public:
    MakeBlockTransient(int argc, char *argv[]);
};
#endif // __MAKETRANSIENT_H
