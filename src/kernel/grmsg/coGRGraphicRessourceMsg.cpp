/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRGraphicRessourceMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRGraphicRessourceMsg::coGRGraphicRessourceMsg(const char *msg)
    : coGRMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%f", &numMBytes_);
}

GRMSGEXPORT coGRGraphicRessourceMsg::coGRGraphicRessourceMsg(float numMBytes)
    : coGRMsg(GRAPHIC_RESSOURCE)
{
    numMBytes_ = numMBytes;
    ostringstream stream;
    stream << numMBytes_;
    addToken(stream.str().c_str());
}

GRMSGEXPORT float coGRGraphicRessourceMsg::getUsedMBytes()
{
    return numMBytes_;
}
