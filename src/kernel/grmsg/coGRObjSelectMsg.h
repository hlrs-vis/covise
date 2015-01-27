/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 VISENSO    ++
// ++ coGRObjSelectMsg - stores visibility information for an object         ++
// ++                 visibility type can be: geometry, interactor etc.   ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSELECTMSG_H
#define COGROBJSELECTMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSelectMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjSelectMsg(const char *obj_name, int select);

    // reconstruct from received msg
    coGRObjSelectMsg(const char *msg);

    // specific functions
    int isSelected()
    {
        return is_selected_;
    };

private:
    int is_selected_;
};
}
#endif
