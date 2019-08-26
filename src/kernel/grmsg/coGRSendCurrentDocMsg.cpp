/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSendCurrentDocMsg.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRSendCurrentDocMsg::coGRSendCurrentDocMsg(const char *document_name, const char *current_document, const char *objName)
    : coGRDocMsg(SEND_CURRENT_DOCUMENT, document_name)
{
    //fprintf( stderr, "coGRSendCurrentDocMsg  document_name  %s current_document %s\n",document_name, current_document);

    current_document_ = current_document;
    addToken(current_document_);

    objName_ = objName;
    addToken(objName_);
}

GRMSGEXPORT coGRSendCurrentDocMsg::coGRSendCurrentDocMsg(const char *msg)
    : coGRDocMsg(msg)
{
    // parse message type
    vector<string> tokens = getAllTokens();

    if (!tokens[0].empty())
    {
        current_document_ = tokens[0].c_str();
        is_valid_ = 1;
    }
    else
    {
        current_document_ = "";
        is_valid_ = 0;
    }

    if (!tokens[1].empty())
    {
        objName_ = tokens[1].c_str();
        is_valid_ = 1;
    }
    else
    {
        objName_ = "";
        is_valid_ = 0;
    }

    //fprintf( stderr, "coGRSendCurrentDocMsg::coGRSendCurrentDocMsg   msg %s\n", msg);
}
