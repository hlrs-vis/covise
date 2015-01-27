/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjVisMsg - stores visibility information for an object         ++
// ++                 visibility type can be: geometry, interactor etc.   ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJMOVEOBJMSG_H
#define COGROBJMOVEOBJMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjMoveObjMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjMoveObjMsg(const char *obj_name, const char *moveName, float x, float y, float z);

    // reconstruct from received msg
    coGRObjMoveObjMsg(const char *msg);

    // specific functions
    const char *getMoveName()
    {
        return moveName_;
    };

    // get position or orientation
    float getX()
    {
        return x_;
    };
    float getY()
    {
        return y_;
    };
    float getZ()
    {
        return z_;
    };

private:
    float x_, y_, z_;
    char *moveName_;
};
}

#endif
