/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRDeleteViewpointMsg - message to delete a viewpoint in COVER     ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRDELETEIEWPOINTMSG_H
#define COGRDELETEIEWPOINTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRDeleteViewpointMsg : public coGRMsg
{
public:
    int getViewpointId();
    coGRDeleteViewpointMsg(int id);
    coGRDeleteViewpointMsg(const char *msg);

private:
    int id_;
};
}
#endif
