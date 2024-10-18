/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMaterial.h

#ifndef _VRMLNODEMATERIAL_
#define _VRMLNODEMATERIAL_

#include "VrmlNodeTemplate.h"
#include "VrmlSFColor.h"
#include "VrmlSFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeMaterial : public VrmlNodeTemplate
{

public:
    // Define the fields of Material nodes
    static void initFields(VrmlNodeMaterial *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeMaterial(VrmlScene *);

    virtual VrmlNodeMaterial *toMaterial() const;

    virtual void render(Viewer *);

    float ambientIntensity()
    {
        return d_ambientIntensity.get();
    }
    float *diffuseColor()
    {
        return d_diffuseColor.get();
    }
    float *emissiveColor()
    {
        return d_emissiveColor.get();
    }
    float shininess()
    {
        return d_shininess.get();
    }
    float *specularColor()
    {
        return d_specularColor.get();
    }
    float transparency()
    {
        return d_transparency.get();
    }

private:
    VrmlSFFloat d_ambientIntensity;
    VrmlSFColor d_diffuseColor;
    VrmlSFColor d_emissiveColor;
    VrmlSFFloat d_shininess;
    VrmlSFColor d_specularColor;
    VrmlSFFloat d_transparency;
};
}
#endif //_VRMLNODEMATERIAL_
