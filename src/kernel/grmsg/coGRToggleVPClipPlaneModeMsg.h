/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO      ++
// ++ coGRToggleVPClipPlaneModeMsg - message to toggle the flymode of the viepoints ++
// ++                                                                       ++
// ++                                                                       ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRTOGGLEVPCLIPPLANEMODEMSG_H
#define COGRTOGGLEVPCLIPPLANEMODEMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRToggleVPClipPlaneModeMsg : public coGRMsg
{
public:
    int getMode();
    coGRToggleVPClipPlaneModeMsg(int mode);
    coGRToggleVPClipPlaneModeMsg(const char *msg);

private:
    int mode_;
};
}
#endif //COGRTOGGLEVPCLIPPLANEMODEMSG_H
