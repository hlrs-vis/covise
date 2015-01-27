/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRSetAnimationSpeedMsg -message to set the speed of animation     ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSETANIMATIONSPEEDMSG_H
#define COGRSETANIMATIONSPEEDMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSetAnimationSpeedMsg : public coGRMsg
{
public:
    float getAnimationSpeed();
    float getAnimationSpeedMin();
    float getAnimationSpeedMax();
    coGRSetAnimationSpeedMsg(float speed, float min, float max);
    coGRSetAnimationSpeedMsg(const char *msg);

private:
    float speed_;
    float speedMin_;
    float speedMax_;
};
}
#endif
