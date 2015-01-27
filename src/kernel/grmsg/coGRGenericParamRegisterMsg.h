/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO      ++
// ++ coGRGenericParamRegisterMsg - Message to register a param of a        ++
// ++                               generic object                          ++
// ++                                                                       ++
// ++                                                                       ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRGENERICPARAMREGISTERMSG_H
#define COGRGENERICPARAMREGISTERMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRGenericParamRegisterMsg : public coGRMsg
{
public:
    coGRGenericParamRegisterMsg(const char *objectName, const char *paramName, int paramType, const char *defaultValue);
    coGRGenericParamRegisterMsg(const char *msg);
    const char *getObjectName();
    const char *getParamName();
    int getParamType();
    const char *getDefaultValue();

private:
    char *objectName_;
    char *paramName_;
    int paramType_;
    char *defaultValue_;
};
}
#endif
