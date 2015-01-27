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

#ifndef COGROBJBOUNDARIESOBJMSG_H
#define COGROBJBOUNDARIESOBJMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjBoundariesObjMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjBoundariesObjMsg(Mtype type, const char *obj_name, const char *boundariesName, float front, float back, float left, float right, float top, float bottom);

    // reconstruct from received msg
    coGRObjBoundariesObjMsg(const char *msg);

    // specific functions
    const char *getBoundariesName()
    {
        return boundariesName_;
    };

    // get boundaries
    float getFront()
    {
        return front_;
    };
    float getBack()
    {
        return back_;
    };
    float getLeft()
    {
        return left_;
    };
    float getRight()
    {
        return right_;
    };
    float getTop()
    {
        return top_;
    };
    float getBottom()
    {
        return bottom_;
    };

private:
    float front_, back_, left_, right_, top_, bottom_;
    const char *boundariesName_;
};
}

#endif
