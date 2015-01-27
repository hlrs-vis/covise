/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SET_OF_REFERENCES_H
#define _SET_OF_REFERENCES_H

// includes
#include <api/coModule.h>
#include <util/coviseCompat.h>

using namespace covise;

class SetOfReferences : public coModule
{

private:
    //ports
    coInputPort *p_inport;
    coOutputPort *p_outport;

    // params
    coIntScalarParam *p_count;
    coBooleanParam *p_timestep;

public:
    virtual int compute(const char *port);
    SetOfReferences(int argc, char *argv[]);
};
#endif
