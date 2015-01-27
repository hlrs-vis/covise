/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INTTOFLOAT_H
#define _INTTOFLOAT_H

#include <api/coSimpleModule.h>

using namespace covise;

class IntToFloat : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    coInputPort *p_input;
    coOutputPort *p_output;

public:
    IntToFloat(int argc, char *argv[]);
};
#endif
