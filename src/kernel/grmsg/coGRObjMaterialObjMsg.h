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

#ifndef COGROBJMATERIALOBJMSG_H
#define COGROBJMATERIALOBJMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjMaterialObjMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjMaterialObjMsg(Mtype type, const char *obj_name, int ambient0, int ambient1, int ambient2, int diffuse0, int diffuse1, int diffuse2, int specular0, int specular1, int specular2, float shininess, float transparency);
    // reconstruct from received msg
    coGRObjMaterialObjMsg(const char *msg);

    // get Colors
    int *getAmbient()
    {
        return _ambient;
    };
    int *getDiffuse()
    {
        return _diffuse;
    };
    int *getSpecular()
    {
        return _specular;
    };
    float getShininess()
    {
        return _shininess;
    };
    float getTransparency()
    {
        return _transparency;
    };

private:
    int _ambient[3];
    int _diffuse[3];
    int _specular[3];
    float _shininess;
    float _transparency;
};
}

#endif
