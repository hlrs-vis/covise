/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRGenericParamChangedMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstring>

using namespace grmsg;

GRMSGEXPORT coGRGenericParamChangedMsg::coGRGenericParamChangedMsg(const char *msg)
    : coGRMsg(msg)
{
    string tok;

    tok = extractFirstToken();
    objectName_ = new char[strlen(tok.c_str()) + 1];
    strcpy(objectName_, tok.c_str());

    tok = extractFirstToken();
    paramName_ = new char[strlen(tok.c_str()) + 1];
    strcpy(paramName_, tok.c_str());

    tok = extractFirstToken();
    value_ = new char[strlen(tok.c_str()) + 1];
    strcpy(value_, tok.c_str());
}

GRMSGEXPORT coGRGenericParamChangedMsg::coGRGenericParamChangedMsg(const char *objectName, const char *paramName, const char *value)
    : coGRMsg(GENERIC_PARAM_CHANGED)
{
    objectName_ = new char[strlen(objectName) + 1];
    strcpy(objectName_, objectName);
    addToken(objectName_);

    paramName_ = new char[strlen(paramName) + 1];
    strcpy(paramName_, paramName);
    addToken(paramName_);

    value_ = new char[strlen(value) + 1];
    strcpy(value_, value);
    addToken(value_);
}

GRMSGEXPORT const char *coGRGenericParamChangedMsg::getObjectName()
{
    return objectName_;
}

GRMSGEXPORT const char *coGRGenericParamChangedMsg::getParamName()
{
    return paramName_;
}

GRMSGEXPORT const char *coGRGenericParamChangedMsg::getValue()
{
    return value_;
}
