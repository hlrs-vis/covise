/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSnapshotMsg.h"
#include <cstring>
#include <cstdlib>

using namespace std;
using namespace grmsg;

/*needs a second parameter to be different of second constructor with msg*/
GRMSGEXPORT coGRSnapshotMsg::coGRSnapshotMsg(const char *filename, const char *intention)
    : coGRMsg(SNAPSHOT)
{
    filename_ = strdup(filename);
    addToken(filename_);
    intention_ = strdup(intention);
    addToken(intention_);
    //cerr << "coGRSnapshotMsg::coGRSnapshotMsg   filename " << filename << " intention " << intention << endl;
}

GRMSGEXPORT coGRSnapshotMsg::coGRSnapshotMsg(const char *msg)
    : coGRMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        filename_ = strdup(tok[0].c_str());
        is_valid_ = 1;
    }
    else
    {
        filename_ = strdup("");
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        intention_ = strdup(tok[1].c_str());
        is_valid_ = 1;
    }
    else
    {
        intention_ = strdup("");
        is_valid_ = 0;
    }
    //cerr << "coGRSnapshotMsg::coGRSnapshotMsg   msg " << msg << endl;
}

GRMSGEXPORT coGRSnapshotMsg::~coGRSnapshotMsg()
{
    COGRMSG_SAFEFREE(filename_);
    COGRMSG_SAFEFREE(intention_);
}
