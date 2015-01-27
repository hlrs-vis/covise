/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PRESSURE_LEVELS_H_
#define _PRESSURE_LEVELS_H_

#define numParams 3

#include <api/coSimpleModule.h>
using namespace covise;

class OceanDepth : public coSimpleModule
{
public:
    OceanDepth(int argc, char *argv[]);
    virtual ~OceanDepth();

private:
    virtual int compute(const char *port);
    coInputPort *p_inGrid;
    coInputPort *p_inData[numParams];
    coOutputPort *p_outGrid;
};
#endif
