/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 VISENSO    ++
// ++ coGRObjVisMsg - stores visibility information for an object         ++
// ++                 visibility type can be: geometry, interactor etc.   ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJVISMSG_H
#define COGROBJVISMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjVisMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjVisMsg(Mtype type, const char *obj_name, int is_visible);

    // reconstruct from received msg
    coGRObjVisMsg(const char *msg);

    // specific functions
    bool isVisible()
    {
        return is_visible_ != 0;
    };

private:
    int is_visible_;
};
}
#endif
