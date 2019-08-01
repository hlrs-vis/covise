/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjSensorMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRObjSensorMsg::coGRObjSensorMsg(const char *msg)
    : coGRObjMsg(msg)
{
    sscanf(extractFirstToken().c_str(), "%d", &sensorId_);
}

GRMSGEXPORT coGRObjSensorMsg::coGRObjSensorMsg(Mtype type, const char *obj_name, int sensorId)
    : coGRObjMsg(type, obj_name)
{
    //fprintf(stderr,"coGRObjSensorMsg id=%d\n", sensorId );
    sensorId_ = sensorId;
    ostringstream stream;
    stream << sensorId_;
    addToken(stream.str().c_str());
}

GRMSGEXPORT int coGRObjSensorMsg::getSensorId()
{
    return sensorId_;
}
