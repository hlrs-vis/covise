/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRTurnTableAnimation - start turntable animation of t seconds     ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRTURNTABLEANIMATIONMSG_H
#define COGRTURNTABLEANIMATIONMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRTurnTableAnimationMsg : public coGRMsg
{
public:
    // construct msg to send
    coGRTurnTableAnimationMsg(float time);

    // reconstruct from received msg
    coGRTurnTableAnimationMsg(const char *msg);

    // specific functions
    float getAnimationTime()
    {
        return time_;
    };

private:
    float time_;
};
}
#endif
