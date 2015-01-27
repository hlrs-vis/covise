/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRDocMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRDocMsg::coGRDocMsg(coGRMsg::Mtype type, const char *document_name)
    : coGRObjMsg(type, document_name)
{
}

GRMSGEXPORT coGRDocMsg::coGRDocMsg(const char *msg)
    : coGRObjMsg(msg)
{
}

GRMSGEXPORT const char *coGRDocMsg::getDocumentName()
{
    return getObjName();
}
