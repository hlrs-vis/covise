/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PART_SELECT_H
#define _PART_SELECT_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE PartSelect application module                     **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Sasha Cioringa                                                **
 **                                                                        **
 **                                                                        **
 ** Date:  05.06.01  V1.0                                                  **
\**************************************************************************/

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
