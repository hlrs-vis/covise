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

#ifndef COGROBJCOLOROBJMSG_H
#define COGROBJCOLOROBJMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjColorObjMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjColorObjMsg(Mtype type, const char *obj_name, int r, int g, int b);

    // reconstruct from received msg
    coGRObjColorObjMsg(const char *msg);

    // get Color
    int getR()
    {
        return r_;
    };
    int getG()
    {
        return g_;
    };
    int getB()
    {
        return b_;
    };

private:
    int r_, g_, b_;
};
}
#endif
