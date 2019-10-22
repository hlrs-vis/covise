/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRKeyWordMsg.h"
#include <cstring>
#include <cstdlib>

using namespace std;
using namespace grmsg;

/*needs a second parameter to be different of second constructor with msg*/
GRMSGEXPORT coGRKeyWordMsg::coGRKeyWordMsg(const char *keyWord, bool kw)
    : coGRMsg(KEYWORD)
{
    (void)kw;
    keyWord_ = strdup(keyWord);
    addToken(keyWord_);
}

GRMSGEXPORT coGRKeyWordMsg::coGRKeyWordMsg(const char *msg)
    : coGRMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        keyWord_ = strdup(tok[0].c_str());
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        keyWord_ = strdup("");
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRKeyWordMsg::~coGRKeyWordMsg()
{
    COGRMSG_SAFEFREE(keyWord_);
} //coGRKeyWordMsg::~coGRKeyWordMsg
