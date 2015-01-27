/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2008 VISENSO    ++
// ++ coGRGraphicRessourceMsg - currently used memory in COVER            ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRGRAPHICRESSOURCEMSG_H
#define COGRGraphicRessourceMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRGraphicRessourceMsg : public coGRMsg
{
public:
    float getUsedMBytes();
    coGRGraphicRessourceMsg(float numMBytes);
    coGRGraphicRessourceMsg(const char *msg);

private:
    float numMBytes_;
};
}
#endif
