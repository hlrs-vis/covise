/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO      ++
// ++ coGRGenericParamChangedMsg - Message sent if a generic param was      ++
// ++                              changed                                  ++
// ++                                                                       ++
// ++                                                                       ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRGENERICPARAMCHANGEDMSG_H
#define COGRGENERICPARAMCHANGEDMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRGenericParamChangedMsg : public coGRMsg
{
public:
    coGRGenericParamChangedMsg(const char *objectName, const char *paramName, const char *value);
    coGRGenericParamChangedMsg(const char *msg);
    const char *getObjectName();
    const char *getParamName();
    const char *getValue();

private:
    char *objectName_;
    char *paramName_;
    char *value_;
};
}
#endif
