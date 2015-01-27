/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2008 VISENSO    ++
// ++ coGRActivatedViewpointMsg - message to show a viewpoint in COVER    ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRACTIVATEDVIEWPOINTMSG_H
#define COGRACTIVATEDVIEWPOINTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRActivatedViewpointMsg : public coGRMsg
{
public:
    int getViewpointId();
    coGRActivatedViewpointMsg(int id);
    coGRActivatedViewpointMsg(const char *msg);

private:
    int id_;
};
}
#endif
