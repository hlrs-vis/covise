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

#ifndef COGROBJSETTRANSPARENCYMSG_H
#define COGROBJSETTRANSPARENCYMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSetTransparencyMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjSetTransparencyMsg(Mtype type, const char *obj_name, float trans);

    // reconstruct from received msg
    coGRObjSetTransparencyMsg(const char *msg);

    // get Color
    float getTransparency()
    {
        return trans_;
    };

private:
    float trans_;
};
}
#endif
