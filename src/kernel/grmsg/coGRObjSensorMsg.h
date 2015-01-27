/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2008 VISENSO    ++
// ++ coGRObjSensorMsg - send a sensor of an object (vrml) to GUI         ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSENSORMSG_H
#define COGROBJSENSORMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSensorMsg : public coGRObjMsg
{
public:
    int getSensorId();
    coGRObjSensorMsg(Mtype type, const char *obj_name, int sensorId);
    coGRObjSensorMsg(const char *msg);

private:
    int sensorId_;
};
}
#endif
