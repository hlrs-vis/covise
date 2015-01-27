/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjTransformAbstractMsg.h"
#include <string>
#include <sstream>
#include <cstdio>

namespace grmsg
{

coGRObjTransformAbstractMsg::coGRObjTransformAbstractMsg(coGRMsg::Mtype type, const char *obj_name,
                                                         float mat00, float mat01, float mat02, float mat03,
                                                         float mat10, float mat11, float mat12, float mat13,
                                                         float mat20, float mat21, float mat22, float mat23,
                                                         float mat30, float mat31, float mat32, float mat33)
    : coGRObjMsg(type, obj_name)
{
    mat_[0][0] = mat00;
    mat_[0][1] = mat01;
    mat_[0][2] = mat02;
    mat_[0][3] = mat03;

    mat_[1][0] = mat10;
    mat_[1][1] = mat11;
    mat_[1][2] = mat12;
    mat_[1][3] = mat13;

    mat_[2][0] = mat20;
    mat_[2][1] = mat21;
    mat_[2][2] = mat22;
    mat_[2][3] = mat23;

    mat_[3][0] = mat30;
    mat_[3][1] = mat31;
    mat_[3][2] = mat32;
    mat_[3][3] = mat33;

    char str[1024];
    sprintf(str, "%f", mat00);
    addToken(str);
    sprintf(str, "%f", mat01);
    addToken(str);
    sprintf(str, "%f", mat02);
    addToken(str);
    sprintf(str, "%f", mat03);
    addToken(str);

    sprintf(str, "%f", mat10);
    addToken(str);
    sprintf(str, "%f", mat11);
    addToken(str);
    sprintf(str, "%f", mat12);
    addToken(str);
    sprintf(str, "%f", mat13);
    addToken(str);

    sprintf(str, "%f", mat20);
    addToken(str);
    sprintf(str, "%f", mat21);
    addToken(str);
    sprintf(str, "%f", mat22);
    addToken(str);
    sprintf(str, "%f", mat23);
    addToken(str);

    sprintf(str, "%f", mat30);
    addToken(str);
    sprintf(str, "%f", mat31);
    addToken(str);
    sprintf(str, "%f", mat32);
    addToken(str);
    sprintf(str, "%f", mat33);
    addToken(str);
}

GRMSGEXPORT coGRObjTransformAbstractMsg::coGRObjTransformAbstractMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (!tok[i * 4 + j].empty())
            {
                string trans = tok[i * 4 + j];
                sscanf(trans.c_str(), "%f", &mat_[i][j]);
            }
            else
                is_valid_ = 0;
        }
    }
}
}
