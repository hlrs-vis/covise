/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRSetTimestepMsg -message to set the timestep of the animation    ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSETTIMESTEPMSG_H
#define COGRSETTIMESTEPMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSetTimestepMsg : public coGRMsg
{
public:
    int getActualTimeStep();
    int getNumTimeSteps();
    coGRSetTimestepMsg(int step, int maxSteps);
    coGRSetTimestepMsg(const char *msg);

private:
    int step_;
    int maxSteps_;
};
}
#endif
