/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjVisMsg - stores visibility information for an object         ++
// ++                 visibility type can be: geometry, interactor etc.   ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSHADEROBJMSG_H
#define COGROBJSHADEROBJMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjShaderObjMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjShaderObjMsg(Mtype type,
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
                        const char *paraMat4);

    // reconstruct from received msg
    coGRObjShaderObjMsg(const char *msg);

    virtual ~coGRObjShaderObjMsg();

    const char *getShaderName()
    {
        return shaderName_;
    };
    const char *getParaFloatName()
    {
        return paraFloat_;
    };
    const char *getParaVec2Name()
    {
        return paraVec2_;
    };
    const char *getParaVec3Name()
    {
        return paraVec3_;
    };
    const char *getParaVec4Name()
    {
        return paraVec4_;
    };
    const char *getParaIntName()
    {
        return paraInt_;
    };
    const char *getParaBoolName()
    {
        return paraBool_;
    };
    const char *getParaMat2Name()
    {
        return paraMat2_;
    };
    const char *getParaMat3Name()
    {
        return paraMat3_;
    };
    const char *getParaMat4Name()
    {
        return paraMat4_;
    };

private:
    char *shaderName_;
    char *paraFloat_;
    char *paraVec2_;
    char *paraVec3_;
    char *paraVec4_;
    char *paraInt_;
    char *paraBool_;
    char *paraMat2_;
    char *paraMat3_;
    char *paraMat4_;
};
}

#endif
