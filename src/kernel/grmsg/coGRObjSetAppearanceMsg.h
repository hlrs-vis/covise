/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2012 VISENSO    ++
// ++ coGRObjSetAppearanceMsg - sets the color of a SceneObject           ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSETAPPEARANCEMSG_H
#define COGROBJSETAPPEARANCEMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSetAppearanceMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjSetAppearanceMsg(const char *obj_name, const char *scopeName, float r, float g, float b, float a);

    // reconstruct from received msg
    coGRObjSetAppearanceMsg(const char *msg);
    ~coGRObjSetAppearanceMsg();

    const char *getScopeName();
    float getR();
    float getG();
    float getB();
    float getA();

private:
    char *scopeName_;
    float r_, g_, b_, a_;
};
}
#endif
