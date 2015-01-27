/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 VISENSO    ++
// ++ coGRObjVisMsg - register object message                             ++
// ++                 object name is stored in class coGRObjMsg           ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJREGISTERMSG_H
#define COGROBJREGISTERMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjRegisterMsg : public coGRObjMsg

{
public:
    // construct msg to send
    coGRObjRegisterMsg(const char *obj_name, const char *parent_obj_name, int unregister = 0);

    // reconstruct from received msg
    coGRObjRegisterMsg(const char *msg);
    ~coGRObjRegisterMsg();

    const char *getParentObjName();
    int getUnregister();

private:
    char *parentObjName_;
    int unregister_;
};
}

#endif
