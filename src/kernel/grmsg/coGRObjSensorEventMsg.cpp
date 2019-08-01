/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjSensorEventMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRObjSensorEventMsg::coGRObjSensorEventMsg(const char *msg)
    : coGRObjMsg(msg)
{
    //   sscanf( extractFirstToken().c_str(), "%d", &sensorId_ );
    vector<string> tok = getAllTokens();
    unsigned short boolDummy;

    if (!tok[0].empty())
    {
        sscanf(tok[0].c_str(), "%d", &sensorId_);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        sscanf(tok[1].c_str(), "%hu", &boolDummy);
        isOver_ = boolDummy != 0;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        sscanf(tok[2].c_str(), "%hu", &boolDummy);
        isActive_ = boolDummy != 0;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjSensorEventMsg::coGRObjSensorEventMsg(Mtype type, const char *obj_name, int sensorId, bool isOver, bool isActive)
    : coGRObjMsg(type, obj_name)
{
    sensorId_ = sensorId;
    isOver_ = isOver;
    isActive_ = isActive;

    ostringstream streamSensorId;
    ostringstream streamIsOver;
    ostringstream streamIsActive;

    streamSensorId << sensorId_;
    streamIsOver << isOver_;
    streamIsActive << isActive_;

    addToken(streamSensorId.str().c_str());
    addToken(streamIsOver.str().c_str());
    addToken(streamIsActive.str().c_str());
}

GRMSGEXPORT int coGRObjSensorEventMsg::getSensorId()
{
    return sensorId_;
}

GRMSGEXPORT bool coGRObjSensorEventMsg::isOver()
{
    return isOver_;
}

GRMSGEXPORT bool coGRObjSensorEventMsg::isActive()
{
    return isActive_;
}
