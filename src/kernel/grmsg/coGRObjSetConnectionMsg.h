/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjVisMsg - stores visibility information for an object         ++
// ++                 visibility type can be: geometry, interactor etc.   ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSETCONNECTIONMSG_H
#define COGROBJSETCONNECTIONMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSetConnectionMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjSetConnectionMsg(coGRMsg::Mtype type, const char *obj_name, const char *connPoint1, const char *connPoint2, int connected, int enabled, const char *obj_name2);
    coGRObjSetConnectionMsg(const char *obj_name, const char *connPoint1, const char *connPoint2, int connected, int enabled, const char *obj_name2);

    // reconstruct from received msg
    coGRObjSetConnectionMsg(const char *msg);

    ~coGRObjSetConnectionMsg();

    const char *getConnPoint1()
    {
        return connPoint1_;
    };
    const char *getConnPoint2()
    {
        return connPoint2_;
    };
    int isConnected()
    {
        return connected_;
    };
    int isEnabled()
    {
        return enabled_;
    }
    const char *getSecondObjName()
    {
        return obj_name2_;
    };

private:
    void initClass(const char *connPoint1, const char *connPoint2, int connected, int enabled, const char *obj_name2);

    char *connPoint1_;
    char *connPoint2_;
    char *obj_name2_;
    int connected_;
    int enabled_;
};
}
#endif
