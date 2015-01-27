/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 VISENSO    ++
// ++ coGRObjKinematicsStateMsg - send a kinmeatics state between         ++
// ++                 OpenCOVER and vr-prepare                            ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJKINEMATICSSTATEMSG_H
#define COGROBJKINEMATICSSTATEMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjKinematicsStateMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjKinematicsStateMsg(const char *obj_name, const char *state);

    // reconstruct from received msg
    coGRObjKinematicsStateMsg(const char *msg);
    ~coGRObjKinematicsStateMsg();

    const char *getState();

private:
    char *state_;
};
}
#endif
