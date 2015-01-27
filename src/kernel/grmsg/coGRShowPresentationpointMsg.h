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

#ifndef COGRSHOWPRESENTATIONPOINTMSG_H
#define COGRSHOWPRESENTATIONPOINTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRShowPresentationpointMsg : public coGRMsg
{
public:
    int getPresentationpointId();
    coGRShowPresentationpointMsg(int id);
    coGRShowPresentationpointMsg(const char *msg);

private:
    int id_;
};
}
#endif
