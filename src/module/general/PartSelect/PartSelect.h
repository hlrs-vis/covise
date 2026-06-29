/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PART_SELECT_H
#define _PART_SELECT_H

#include <api/coSimpleModule.h>
using namespace covise;
#define MAX_INPUT_PORTS 4

class PartSelect : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    // parameters

    coStringParam *p_numbers;

    // ports
    coInputPort **p_inPort;
    coOutputPort **p_outPort;

    // private data

public:
    PartSelect(int argc, char *argv[]);
    virtual ~PartSelect();
};
#endif
