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

#ifndef COGRCREATEVIEWPOINTMSG_H
#define COGRCREATEVIEWPOINTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRCreateViewpointMsg : public coGRMsg
{
public:
    int getViewpointId();
    const char *getName();
    const char *getView();
    const char *getClipplane();

    coGRCreateViewpointMsg(const char *name, int id, const char *view, const char *clipplane);
    coGRCreateViewpointMsg(const char *msg);
    virtual ~coGRCreateViewpointMsg();

private:
    int id_;

    char *name_;
    char *view_;
    char *plane_;
};
}
#endif
