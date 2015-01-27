/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRChangeViewpointMsg - message to Change a viewpoint in COVER     ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRCHANGEVIEWPOINTMSG_H
#define COGRCHANGEVIEWPOINTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRChangeViewpointMsg : public coGRMsg
{
public:
    int getViewpointId();
    //const char *getClipplane();

    coGRChangeViewpointMsg(int id /*, const char *clipplane */);
    coGRChangeViewpointMsg(const char *msg);

private:
    int id_;
    //char *plane_;
};
}
#endif
