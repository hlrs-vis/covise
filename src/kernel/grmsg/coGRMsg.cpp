/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRMsg.h"
#include <iostream>
#include <sstream>
#include <cstring>
using namespace std;
using namespace grmsg;

coGRMsg::coGRMsg(Mtype type): is_valid_(type != NO_TYPE), type_(type)
{
    addToken(MsgID.c_str());
    addToken(std::to_string(type).c_str());
}

coGRMsg::coGRMsg(const char *msg)
    : is_valid_(true)
    ,content_(msg)
{
    /// parsed content will always be removed
    /// to enable all children to call extractFirstToken() to
    /// get their token

    /// parse for msg id
    if (extractFirstToken() != MsgID)
    {
        is_valid_ = false;
        return;
    }

    /// parse message type
    type_ = (Mtype)std::stoi(extractFirstToken());
    if (type_ == NO_TYPE)
    {
        is_valid_ = 0;
    }
}


void coGRMsg::addToken(const char *token)
{
    if (token)
        content_ += string(token) + SplitToken;
}

string coGRMsg::getFirstToken()
{
    size_t pos = content_.find(SplitToken);
    return content_.substr(0, pos);
}

string coGRMsg::extractFirstToken()
{
    size_t pos = content_.find(SplitToken);
    string token = content_.substr(0, pos);
    content_ = content_.substr(pos + 1, string::npos);
    return token;
}

vector<string> coGRMsg::getAllTokens()
{
    vector<string> tok;

    istringstream s(content_);
    string temp;

    while (std::getline(s, temp, SplitToken))
    {
        tok.push_back(temp);
    }
    return tok;
}

const char *coGRMsg::c_str() const
{
    return content_.c_str();
}

string coGRMsg::getString() const
{
    return content_;
}

void coGRMsg::print_stdout()
{
    const char *typeStr[] = { "NO_TYPE", "GEO_VISIBLE", "REGISTER", "INTERACTOR_VISIBLE" };
    cout << "coGRMsg::Type = " << typeStr[type_] << endl;
    cout << "coGRMsg::content = " << endl << "----" << endl << content_ << endl << "---- " << endl;
}
