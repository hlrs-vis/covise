/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetDocPageMsg.h"
#include <iostream>
#include <sstream>

using namespace grmsg;

GRMSGEXPORT coGRSetDocPageMsg::coGRSetDocPageMsg(const char *document_name, int page)
    : coGRDocMsg(SET_DOCUMENT_PAGE, document_name)
{
    page_ = page;
    if (page_ >= 0)
    {
        ostringstream stream;
        stream << page_;
        addToken(stream.str().c_str());
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRSetDocPageMsg::coGRSetDocPageMsg(const char *msg)
    : coGRDocMsg(msg)
{
    /// parse message type
    string page_str = extractFirstToken();
    istringstream stream(page_str);
    stream >> page_;
    if (page_ >= 0)
    {
        is_valid_ = 1;
    }
    else
    {
        page_ = -1;
        is_valid_ = 0;
    }
}
