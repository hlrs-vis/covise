/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRcreateViewpointMsg - message to create a viewpoint in COVER     ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRCREATEDEFAULTVIEWPOINTMSG_H
#define COGRCREATEDEFAULTVIEWPOINTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRCreateDefaultViewpointMsg : public coGRMsg
{
public:
    int getViewpointId();
    const char *getName();

    coGRCreateDefaultViewpointMsg(const char *name, int id);
    coGRCreateDefaultViewpointMsg(const char *msg);
    virtual ~coGRCreateDefaultViewpointMsg();

private:
    int id_;
    char *name_;
};
}
#endif
