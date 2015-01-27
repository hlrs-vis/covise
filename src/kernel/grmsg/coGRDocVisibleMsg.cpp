/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRDocVisibleMsg.h"
#include <iostream>
#include <sstream>

using namespace grmsg;

GRMSGEXPORT coGRDocVisibleMsg::coGRDocVisibleMsg(const char *document_name, int is_visible)
    : coGRDocMsg(DOC_VISIBLE, document_name)
{
    is_visible_ = is_visible;
    if (is_visible_ == 0 || is_visible_ == 1)
    {
        ostringstream stream;
        stream << is_visible_;
        addToken(stream.str().c_str());
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRDocVisibleMsg::coGRDocVisibleMsg(const char *msg)
    : coGRDocMsg(msg)
{
    /// parse message type
    string is_visible_str = extractFirstToken();
    istringstream stream(is_visible_str);
    stream >> is_visible_;
    if (is_visible_ == 0 || is_visible_ == 1)
    {
        is_valid_ = 1;
    }
    else
    {
        is_visible_ = 0;
        is_valid_ = 0;
    }
}
