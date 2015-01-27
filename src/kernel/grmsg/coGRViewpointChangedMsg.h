/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRViewpointChangedMsg - message to send when viewpoint changed    ++
// ++                           in COVER                                  ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRVIEWPOINTCHANGEDMSG_H
#define COGRVIEWPOINTCHANGEDMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRViewpointChangedMsg : public coGRMsg
{
public:
    int getViewpointId();
    const char *getName();
    const char *getView();
    //const char *getClipplane();

    coGRViewpointChangedMsg(int id, const char *name = NULL, const char *view = NULL /*, const char *clipplane=NULL */);
    coGRViewpointChangedMsg(const char *msg);
    virtual ~coGRViewpointChangedMsg();

private:
    int id_;

    char *name_;
    char *view_;
    //char *plane_;
};
}
#endif
