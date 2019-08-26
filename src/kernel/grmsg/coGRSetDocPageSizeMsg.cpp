/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetDocPageSizeMsg.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRSetDocPageSizeMsg::coGRSetDocPageSizeMsg(const char *document_name, int pageNo, float hsize, float vsize)
    : coGRDocMsg(SET_DOCUMENT_PAGESIZE, document_name)
{

    hSize_ = hsize;
    vSize_ = vsize;
    pageNo_ = pageNo;

    ostringstream stream1;
    stream1 << pageNo;
    addToken(stream1.str().c_str());

    ostringstream stream2;
    stream2 << hsize;
    addToken(stream2.str().c_str());

    ostringstream stream3;
    stream3 << vsize;
    addToken(stream3.str().c_str());

    is_valid_ = 1;
}

GRMSGEXPORT coGRSetDocPageSizeMsg::coGRSetDocPageSizeMsg(const char *msg)
    : coGRDocMsg(msg)
{
    // parse message type
    vector<string> tokens = getAllTokens();

    istringstream stream1(tokens.at(0));
    stream1 >> pageNo_;

    istringstream stream2(tokens.at(1));
    stream2 >> hSize_;

    istringstream stream3(tokens.at(2));
    stream3 >> vSize_;

    is_valid_ = 1;
}
