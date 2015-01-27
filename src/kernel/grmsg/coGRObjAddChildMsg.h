/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 VISENSO    ++
// ++ coGRObjVisMsg - mounts or unmounts a SceneObject to another         ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJADDCHILDMSG_H
#define COGROBJADDCHILDMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjAddChildMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjAddChildMsg(const char *obj_name, const char *child_name, int remove = 0);

    // reconstruct from received msg
    coGRObjAddChildMsg(const char *msg);
    ~coGRObjAddChildMsg();

    const char *getChildObjName();
    int getRemove();

private:
    char *childObjName_;
    int remove_;
};
}
#endif
