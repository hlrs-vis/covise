/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSendDocNumbersMsg.h"
#include <iostream>
#include <sstream>

using namespace grmsg;

GRMSGEXPORT coGRSendDocNumbersMsg::coGRSendDocNumbersMsg(const char *document_name, int minPage, int maxPage)
    : coGRDocMsg(SEND_DOCUMENT_NUMBERS, document_name)
{

    minPage_ = minPage;
    maxPage_ = maxPage;

    if (minPage_ <= maxPage_)
    {
        ostringstream stream1;
        stream1 << minPage_;
        addToken(stream1.str().c_str());
        ostringstream stream2;
        stream2 << maxPage_;
        addToken(stream2.str().c_str());
        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRSendDocNumbersMsg::coGRSendDocNumbersMsg(const char *msg)
    : coGRDocMsg(msg)
{
    // parse message type
    vector<string> tokens = getAllTokens();
    istringstream stream1(tokens.at(0));
    stream1 >> minPage_;
    istringstream stream2(tokens.at(1));
    stream2 >> maxPage_;

    if (minPage_ <= maxPage_)
    {
        is_valid_ = 1;
    }
    else
    {
        minPage_ = maxPage_ = -1;
        is_valid_ = 0;
    }
}
