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

GRMSGEXPORT coGRMsg::coGRMsg(Mtype type)
    : is_valid_(1)
    , SplitToken('\n')
    , MsgID(string("GRMSG"))
{
    type_ = type;
    str_ = NULL;
    ostringstream stream;
    stream << type;
    addToken(MsgID.c_str());
    addToken(stream.str().c_str());
}

GRMSGEXPORT coGRMsg::coGRMsg(const char *msg)
    : is_valid_(1)
    , SplitToken('\n')
    , MsgID(string("GRMSG"))
{
    type_ = NO_TYPE;
    str_ = NULL;
    content_ = string(msg);

    /// parsed content will always be removed
    /// to enable all children to call extractFirstToken() to
    /// get their token

    /// parse for msg id
    string id_str = extractFirstToken();
    if (id_str != MsgID)
    {
        is_valid_ = 0;
        return;
    }

    /// parse message type
    string type_str = extractFirstToken();
    istringstream stream(type_str);
    int typei;
    stream >> typei;
    type_ = (Mtype)typei;
}

GRMSGEXPORT void coGRMsg::addToken(const char *token)
{
    if (token == NULL)
    {
        return;
    }
    content_ += string(token) + SplitToken;
}

GRMSGEXPORT string coGRMsg::getFirstToken()
{
    size_t pos = content_.find(SplitToken);
    return content_.substr(0, pos);
}

GRMSGEXPORT string coGRMsg::extractFirstToken()
{
    size_t pos = content_.find(SplitToken);
    string token = content_.substr(0, pos);
    content_ = content_.substr(pos + 1, string::npos);
    return token;
}

GRMSGEXPORT vector<string> coGRMsg::getAllTokens()
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

GRMSGEXPORT const char *coGRMsg::c_str()
{
    return content_.c_str();
    //if (str_ && strcmp(str_, content_.c_str()) == 0)
    //{
    //    return str_;
    //}
    //else
    //{
    //    delete[] str_;
    //    str_ = strcpy(new char[content_.length() + 1], content_.c_str());
    //}



}

GRMSGEXPORT void coGRMsg::print_stdout()
{
    const char *typeStr[] = { "NO_TYPE", "GEO_VISIBLE", "REGISTER", "INTERACTOR_VISIBLE" };
    cout << "coGRMsg::Type = " << typeStr[type_] << endl;
    cout << "coGRMsg::content = " << endl << "----" << endl << content_ << endl << "---- " << endl;
}
