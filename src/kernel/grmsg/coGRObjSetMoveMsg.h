/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2008 VISENSO    ++
// ++ coGRObjSetMoveMsg - make a DCS in scene graph moveable or not       ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSETMOVEMSG_H
#define COGROBJSETMOVEMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSetMoveMsg : public coGRObjMsg
{
public:
    bool isMoveable();
    coGRObjSetMoveMsg(Mtype type, const char *obj_name, bool moveable);
    coGRObjSetMoveMsg(const char *msg);

private:
    bool isMoveable_;
};
}
#endif
