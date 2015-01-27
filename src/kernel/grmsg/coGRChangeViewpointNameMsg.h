/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRChangeViewpointIdMsg - message to create a viewpoint in COVER   ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRCHANGEVIEWPOINTNAMEMSG_H
#define COGRCHANGEVIEWPOINTNAMEMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRChangeViewpointNameMsg : public coGRMsg
{
public:
    int getId();
    char *getName();

    coGRChangeViewpointNameMsg(int id, const char *name);
    coGRChangeViewpointNameMsg(const char *msg);

private:
    int id_;
    char *name_;
};
}
#endif
