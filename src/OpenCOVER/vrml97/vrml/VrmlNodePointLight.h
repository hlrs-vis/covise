/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePointLight.h

#ifndef _VRMLNODEPOINTLIGHT_
#define _VRMLNODEPOINTLIGHT_

#include "VrmlNodeLight.h"
#include "VrmlSFFloat.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodePointLight : public VrmlNodeLight
{

public:
    // Define the fields of pointLight nodes
    static void initFields(VrmlNodePointLight *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodePointLight(VrmlScene *);
    virtual ~VrmlNodePointLight();

    // Bindable/scoped nodes must notify the scene of their existence.
    virtual void addToScene(VrmlScene *s, const char *relUrl);


    virtual void render(Viewer *);
    //LarryD Mar 04/99
    virtual const VrmlSFVec3f &getAttenuation() const;
    //LarryD Mar 04/99
    virtual const VrmlSFVec3f &getLocation() const;
    virtual float getRadius() //LarryD Mar 04/99
    {
        return d_radius.get();
    }

protected:
    VrmlSFVec3f d_attenuation;
    VrmlSFVec3f d_location;
    VrmlSFFloat d_radius;
};
}
#endif //_VRMLNODEPOINTLIGHT_
