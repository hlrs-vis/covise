/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRGenericParamRegisterMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstring>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRGenericParamRegisterMsg::coGRGenericParamRegisterMsg(const char *msg)
    : coGRMsg(msg)
{
    string tok;

    tok = extractFirstToken();
    objectName_ = new char[strlen(tok.c_str()) + 1];
    strcpy(objectName_, tok.c_str());

    tok = extractFirstToken();
    paramName_ = new char[strlen(tok.c_str()) + 1];
    strcpy(paramName_, tok.c_str());

    sscanf(extractFirstToken().c_str(), "%d", &paramType_);

    tok = extractFirstToken();
    defaultValue_ = new char[strlen(tok.c_str()) + 1];
    strcpy(defaultValue_, tok.c_str());
}

GRMSGEXPORT coGRGenericParamRegisterMsg::coGRGenericParamRegisterMsg(const char *objectName, const char *paramName, int paramType, const char *defaultValue)
    : coGRMsg(GENERIC_PARAM_REGISTER)
{
    objectName_ = new char[strlen(objectName) + 1];
    strcpy(objectName_, objectName);
    addToken(objectName_);

    paramName_ = new char[strlen(paramName) + 1];
    strcpy(paramName_, paramName);
    addToken(paramName_);

    paramType_ = paramType;
    ostringstream stream;
    stream << paramType_;
    addToken(stream.str().c_str());

    defaultValue_ = new char[strlen(defaultValue) + 1];
    strcpy(defaultValue_, defaultValue);
    addToken(defaultValue_);
}

GRMSGEXPORT const char *coGRGenericParamRegisterMsg::getObjectName()
{
    return objectName_;
}

GRMSGEXPORT const char *coGRGenericParamRegisterMsg::getParamName()
{
    return paramName_;
}

GRMSGEXPORT int coGRGenericParamRegisterMsg::getParamType()
{
    return paramType_;
}

GRMSGEXPORT const char *coGRGenericParamRegisterMsg::getDefaultValue()
{
    return defaultValue_;
}
