/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRSendCurrentDocMsg - send Path and name of current showed Doc    ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSENDCURDOCMSG_H
#define COGRSENDCURDOCMSG_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSendCurrentDocMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRSendCurrentDocMsg(const char *document_name, const char *current_document_, const char *objName);

    // reconstruct from received msg
    coGRSendCurrentDocMsg(const char *msg);

    // specific functions
    const char *getCurrentDocument()
    {
        return current_document_;
    };
    const char *getObjName()
    {
        return objName_;
    };

private:
    const char *current_document_;
    const char *objName_;
};
}
#endif
