/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRPluginMsg.h"
#include <cstring>
#include <cstdlib>

using namespace std;
using namespace grmsg;

/*needs a second parameter to be different of second constructor with msg*/
GRMSGEXPORT coGRPluginMsg::coGRPluginMsg(const char *plugin, const char *action)
    : coGRMsg(PLUGIN)
{
    plugin_ = strdup(plugin);
    addToken(plugin_);
    action_ = strdup(action);
    addToken(action_);
    //cerr << "coGRPluginMsg::coGRPluginMsg   plugin " << plugin << " action " << action << endl;
}

GRMSGEXPORT coGRPluginMsg::coGRPluginMsg(const char *msg)
    : coGRMsg(msg)
{
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        plugin_ = strdup(tok[0].c_str());
        is_valid_ = 1;
    }
    else
    {
        plugin_ = strdup("");
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        action_ = strdup(tok[1].c_str());
        is_valid_ = 1;
    }
    else
    {
        action_ = strdup("");
        is_valid_ = 0;
    }
    //cerr << "coGRPluginMsg::coGRPluginMsg   msg " << msg << endl;
}

GRMSGEXPORT coGRPluginMsg::~coGRPluginMsg()
{
    COGRMSG_SAFEFREE(plugin_);
    COGRMSG_SAFEFREE(action_);
}
