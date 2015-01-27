/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRShowViewpointMsg - message to show a viewpoint in COVER         ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSHOWVIEWPOINTMSG_H
#define COGRSHOWVIEWPOINTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRShowViewpointMsg : public coGRMsg
{
public:
    int getViewpointId();
    coGRShowViewpointMsg(int id);
    coGRShowViewpointMsg(const char *msg);

private:
    int id_;
};
}
#endif
