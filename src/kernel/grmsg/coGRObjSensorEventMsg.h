/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2008 VISENSO    ++
// ++ coGRObjSensorEventMsg - send a sensor event of an object (vrml) to GUI ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSENSOREVENTMSG_H
#define COGROBJSENSOREVENTMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSensorEventMsg : public coGRObjMsg
{
public:
    int getSensorId();
    bool isOver();
    bool isActive();

    coGRObjSensorEventMsg(Mtype type, const char *obj_name, int sensorId, bool isOver, bool isActive);
    coGRObjSensorEventMsg(const char *msg);

private:
    int sensorId_;
    bool isOver_;
    bool isActive_;
};
}
#endif
