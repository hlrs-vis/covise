/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2009 VISENSO    ++
// ++ coGRPluginMsg - sends message to make a snapshot                  ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRPLUGINMSG_H
#define COGRPLUGINMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

#include <iostream>

namespace grmsg
{

class GRMSGEXPORT coGRPluginMsg : public coGRMsg
{
public:
    // construct msg to send
    coGRPluginMsg(const char *plugin, const char *action);

    // reconstruct from received msg
    coGRPluginMsg(const char *msg);

    virtual ~coGRPluginMsg();

    // specific functions
    const char *getPlugin() const
    {
        return plugin_;
    };
    const char *getAction() const
    {
        return action_;
    };

private:
    char *plugin_;
    char *action_;
};
}

#endif
