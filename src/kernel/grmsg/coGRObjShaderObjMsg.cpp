/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <cstdlib>
#include "coGRObjShaderObjMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRObjShaderObjMsg::coGRObjShaderObjMsg(Mtype type,
                                                     const char *obj_name,
                                                     const char *shader_name,
                                                     const char *paraFloat,
                                                     const char *paraVec2,
                                                     const char *paraVec3,
                                                     const char *paraVec4,
                                                     const char *paraInt,
                                                     const char *paraBool,
                                                     const char *paraMat2,
                                                     const char *paraMat3,
                                                     const char *paraMat4)
    : coGRObjMsg(type, obj_name)
{
    // 	shaderName_ =shader_name;
    // 	paraFloat_ =paraFloat;
    // 	paraVec2_ =paraVec2;
    // 	paraVec3_ =paraVec3;
    // 	paraVec4_ =paraVec4;
    // 	paraInt_ =paraInt;
    // 	paraBool_ =paraBool;
    // 	paraMat2_ =paraMat2;
    // 	paraMat3_ =paraMat3;
    // 	paraMat4_ =paraMat4;

    //Note that     X AND 1 = X for all X in {true, false}
    if (shader_name)
    {
        shaderName_ = strdup(shader_name);
        addToken(shaderName_);
    }
    else
        is_valid_ = 0;

    if (paraFloat)
    {
        paraFloat_ = strdup(paraFloat);
        addToken(paraFloat_);
    }
    else
        is_valid_ = 0;

    if (paraVec2)
    {
        paraVec2_ = strdup(paraVec2);
        addToken(paraVec2_);
    }
    else
        is_valid_ = 0;

    if (paraVec3)
    {
        paraVec3_ = strdup(paraVec3);
        addToken(paraVec3_);
    }
    else
        is_valid_ = 0;

    if (paraVec4)
    {
        paraVec4_ = strdup(paraVec4);
        addToken(paraVec4_);
    }
    else
        is_valid_ = 0;

    if (paraInt)
    {
        paraInt_ = strdup(paraInt);
        addToken(paraInt_);
    }
    else
        is_valid_ = 0;

    if (paraBool)
    {
        paraBool_ = strdup(paraBool);
        addToken(paraBool_);
    }
    else
        is_valid_ = 0;

    if (paraMat2)
    {
        paraMat2_ = strdup(paraMat2);
        addToken(paraMat2_);
    }
    else
        is_valid_ = 0;

    if (paraMat3)
    {
        paraMat3_ = strdup(paraMat3);
        addToken(paraMat3_);
    }
    else
        is_valid_ = 0;

    if (paraMat4)
    {
        paraMat4_ = strdup(paraMat4);
        addToken(paraMat4_);
    }
    else
        is_valid_ = 0;

    is_valid_ = 1;
}

GRMSGEXPORT coGRObjShaderObjMsg::coGRObjShaderObjMsg(const char *msg)
    : coGRObjMsg(msg)
{
    shaderName_ = NULL;
    vector<string> tok = getAllTokens();

    if (!tok[0].empty())
    {
        shaderName_ = strdup(tok[0].c_str());
    }
    else
    {
        shaderName_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        paraFloat_ = strdup(tok[1].c_str());
    }
    else
    {
        paraFloat_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        paraVec2_ = strdup(tok[2].c_str());
    }
    else
    {
        paraVec2_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[3].empty())
    {
        paraVec3_ = strdup(tok[3].c_str());
    }
    else
    {
        paraVec3_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[4].empty())
    {
        paraVec4_ = strdup(tok[4].c_str());
    }
    else
    {
        paraVec4_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[5].empty())
    {
        paraInt_ = strdup(tok[5].c_str());
    }
    else
    {
        paraInt_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[6].empty())
    {
        paraBool_ = strdup(tok[6].c_str());
    }
    else
    {
        paraBool_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[7].empty())
    {
        paraMat2_ = strdup(tok[7].c_str());
    }
    else
    {
        paraMat2_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[8].empty())
    {
        paraMat3_ = strdup(tok[8].c_str());
    }
    else
    {
        paraMat3_ = NULL;
        is_valid_ = 0;
    }

    if (!tok[9].empty())
    {
        paraMat4_ = strdup(tok[9].c_str());
    }
    else
    {
        paraMat4_ = NULL;
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRObjShaderObjMsg::~coGRObjShaderObjMsg()
{
    COGRMSG_SAFEFREE(shaderName_);
    COGRMSG_SAFEFREE(paraFloat_);
    COGRMSG_SAFEFREE(paraVec2_);
    COGRMSG_SAFEFREE(paraVec3_);
    COGRMSG_SAFEFREE(paraVec4_);
    COGRMSG_SAFEFREE(paraInt_);
    COGRMSG_SAFEFREE(paraBool_);
    COGRMSG_SAFEFREE(paraMat2_);
    COGRMSG_SAFEFREE(paraMat2_);
    COGRMSG_SAFEFREE(paraMat3_);
    COGRMSG_SAFEFREE(paraMat4_);
}
