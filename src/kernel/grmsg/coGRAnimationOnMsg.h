/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO      ++
// ++ coGRAnimationOnMsg - Message to set animation on or off               ++
// ++                                                                       ++
// ++                                                                       ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRANIMATIONONMSG_H
#define COGRANIMATIONONMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRAnimationOnMsg : public coGRMsg
{
public:
    int getMode();
    coGRAnimationOnMsg(int mode);
    coGRAnimationOnMsg(const char *msg);

private:
    int mode_;
};
}
#endif //coGRAnimationOnMsg
