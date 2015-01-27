/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjMsg - handles document depending messages                    ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRDOCMSG_H
#define COGRDOCMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRDocMsg : public coGRObjMsg
{
public:
    const char *getDocumentName();

protected:
    coGRDocMsg(coGRMsg::Mtype type, const char *document_name);
    coGRDocMsg(const char *msg);
};
}
#endif
