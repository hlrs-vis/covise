/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO      ++
// ++ coGRToggleFlymodeMsg - message to toggle the flymode of the viepoints ++
// ++                                                                       ++
// ++                                                                       ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRTOGGLEFLYMODEMSG_H
#define COGRTOGGLEFLYMODEMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRToggleFlymodeMsg : public coGRMsg
{
public:
    int getMode();
    coGRToggleFlymodeMsg(int mode);
    coGRToggleFlymodeMsg(const char *msg);

private:
    int mode_;
};
}
#endif //COGRTOGGLEFLYMODEMSG_H
