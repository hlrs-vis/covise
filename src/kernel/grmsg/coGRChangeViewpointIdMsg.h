/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRChangeViewpointIdMsg - message to create a viewpoint in COVER   ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRCHANGEVIEWPOINTIDMSG_H
#define COGRCHANGEVIEWPOINTIDMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRChangeViewpointIdMsg : public coGRMsg
{
public:
    int getOldId();
    int getNewId();

    coGRChangeViewpointIdMsg(int oldId, int newId);
    coGRChangeViewpointIdMsg(const char *msg);

private:
    int oldId_;
    int newId_;
};
}
#endif
