/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 VISENSO    ++
// ++ coGRObjMsg - handles object depending messages                      ++
// ++              helping class only used by child classes               ++
// ++              Base class for all object messages                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJMSG_H
#define COGROBJMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjMsg : public coGRMsg
{
public:
    const char *getObjName();

protected:
    coGRObjMsg(coGRMsg::Mtype type, const char *obj_name);
    coGRObjMsg(const char *msg);
    virtual ~coGRObjMsg();

private:
    char *obj_name_;
};
}
#endif
