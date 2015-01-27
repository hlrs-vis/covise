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

#ifndef COGROBJMOVEINTERMSG_H
#define COGROBJMOVEINTERMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjMoveInterMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjMoveInterMsg(Mtype type, const char *obj_name, const char *interName, float x, float y, float z);

    // reconstruct from received msg
    coGRObjMoveInterMsg(const char *msg);

    // destructor
    ~coGRObjMoveInterMsg();

    // specific functions
    const char *getInteractorName()
    {
        return interactorName_;
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
    char *interactorName_;
};
}
#endif
