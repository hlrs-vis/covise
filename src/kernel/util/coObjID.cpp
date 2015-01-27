/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coObjID.h"

///   this might change, but remove inter-dependency
#define NODE(x) (((x)&0x7ff00000) >> 20)
#define RECV(x) ((x)&0x000fffff)
using namespace std;

namespace covise
{

#define ID(x) coMsg::recvNo(x) << ':' << coMsg::nodeNo(x)

UTILEXPORT ostream &operator<<(ostream &str, const coObjID &id)
{
    str << "Obj(" << (id.id ? id.id : "invalid") << ")";
    return str;
}
}

using namespace covise;

UTILEXPORT int coObjID::compare(const coObjID &a, const coObjID &b)
{
    if (a.id == b.id)
        return 0;
    if (a.id == NULL)
        return -1;
    if (b.id == NULL)
        return 1;
    return strcmp(a.id, b.id);
}

namespace covise
{

UTILEXPORT ostream &operator<<(ostream &str, const coObjInfo &info)
{
    str << "ObjectInfo: ID=" << info.id
        << ", Block " << info.blockNo << "/" << info.numBlocks
        << ", tStep " << info.timeStep << "/" << info.numTimeSteps
        << " (t=" << info.time << ") reqModID="
        << RECV(info.reqModID) << ":" << NODE(info.reqModID);
    return str;
}
}

int coObjInfo::sequence = 0;
char *coObjInfo::baseName = NULL;
