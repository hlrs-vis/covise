/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjMaterialObjMsg.h"
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRObjMaterialObjMsg::coGRObjMaterialObjMsg(Mtype type,
                                                         const char *obj_name,
                                                         int ambient0,
                                                         int ambient1,
                                                         int ambient2,
                                                         int diffuse0,
                                                         int diffuse1,
                                                         int diffuse2,
                                                         int specular0,
                                                         int specular1,
                                                         int specular2,
                                                         float shininess,
                                                         float transparency)
    : coGRObjMsg(type, obj_name)
{
    _ambient[0] = ambient0;
    _ambient[1] = ambient1;
    _ambient[2] = ambient2;
    _diffuse[0] = diffuse0;
    _diffuse[1] = diffuse1;
    _diffuse[2] = diffuse2;
    _specular[0] = specular0;
    _specular[1] = specular1;
    _specular[2] = specular2;
    _shininess = shininess;
    _transparency = transparency;

    char str[1024];

    sprintf(str, "%d", _ambient[0]);
    addToken(str);
    sprintf(str, "%d", _ambient[1]);
    addToken(str);
    sprintf(str, "%d", _ambient[2]);
    addToken(str);

    sprintf(str, "%d", _diffuse[0]);
    addToken(str);
    sprintf(str, "%d", _diffuse[1]);
    addToken(str);
    sprintf(str, "%d", _diffuse[2]);
    addToken(str);

    sprintf(str, "%d", _specular[0]);
    addToken(str);
    sprintf(str, "%d", _specular[1]);
    addToken(str);
    sprintf(str, "%d", _specular[2]);
    addToken(str);

    sprintf(str, "%f", _shininess);
    addToken(str);

    sprintf(str, "%f", _transparency);
    addToken(str);

    is_valid_ = 1;
}

coGRObjMaterialObjMsg::coGRObjMaterialObjMsg(const char *msg)
    : coGRObjMsg(msg)
{
    vector<string> tok = getAllTokens();

    //ambient color
    if (!tok[0].empty())
    {
        string ambient0 = tok[0];
        sscanf(ambient0.c_str(), "%d", &_ambient[0]);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[1].empty())
    {
        string ambient1 = tok[1];
        sscanf(ambient1.c_str(), "%d", &_ambient[1]);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[2].empty())
    {
        string ambient2 = tok[2];
        sscanf(ambient2.c_str(), "%d", &_ambient[2]);
        is_valid_ = is_valid_ & 1;
    }
    else
    {
        is_valid_ = 0;
    }

    //diffuse color
    if (!tok[3].empty())
    {
        string diffuse0 = tok[3];
        sscanf(diffuse0.c_str(), "%d", &_diffuse[0]);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[4].empty())
    {
        string diffuse1 = tok[4];
        sscanf(diffuse1.c_str(), "%d", &_diffuse[1]);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[5].empty())
    {
        string diffuse2 = tok[5];
        sscanf(diffuse2.c_str(), "%d", &_diffuse[2]);
    }
    else
    {
        is_valid_ = 0;
    }
    //specular color
    if (!tok[6].empty())
    {
        string specular0 = tok[6];
        sscanf(specular0.c_str(), "%d", &_specular[0]);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[7].empty())
    {
        string specular1 = tok[7];
        sscanf(specular1.c_str(), "%d", &_specular[1]);
    }
    else
    {
        is_valid_ = 0;
    }

    if (!tok[8].empty())
    {
        string specular2 = tok[8];
        sscanf(specular2.c_str(), "%d", &_specular[2]);
    }
    else
    {
        is_valid_ = 0;
    }

    //shininess
    if (!tok[9].empty())
    {
        string shininess = tok[9];
        sscanf(shininess.c_str(), "%f", &_shininess);
    }
    else
    {
        is_valid_ = 0;
    }

    //transparency
    if (!tok[10].empty())
    {
        string transparency = tok[10];
        sscanf(transparency.c_str(), "%f", &_transparency);
    }
    else
    {
        is_valid_ = 0;
    }
}
