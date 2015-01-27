/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __coMaterial_H

#define __coMaterial_H

//#include "util/coFileUtil.h"
#include "util/coLinkList.h"

class coMaterial
{
public:
    char *name;
    float ambientColor[3];
    float diffuseColor[3];
    float specularColor[3];
    float emissiveColor[3];
    float shininess;
    float transparency;
    coMaterial(const char *n, float *ambient, float *diffuse, float *specular, float *emissive, float shininess, float transparency);
    coMaterial(const char *n, const char *filename);
    ~coMaterial();
};

class coMaterialList : public coLinkList<coMaterial *>
{
public:
    coMaterialList(const char *);
    void add(const char *);
    coMaterial *get(const char *str);
};

#endif
