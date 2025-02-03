/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeLight.h

#ifndef _VRMLNODELIGHT_
#define _VRMLNODELIGHT_

#include "VrmlNode.h"
#include "VrmlSFBool.h"
#include "VrmlSFColor.h"
#include "VrmlSFFloat.h"

#include "Viewer.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeLight : public VrmlNodeChild
{

public:
    static void initFields(VrmlNodeLight *node, VrmlNodeType *t);
    
    VrmlNodeLight(VrmlScene *, const std::string &name);

    virtual void render(Viewer *);

    virtual float getAmbientIntensity() //LarryD Mar 04/99
    {
        return d_ambientIntensity.get();
    }
    virtual float getIntensity() //LarryD Mar 04/99
    {
        return d_intensity.get();
    }
    virtual bool getOn() //LarryD Mar 04/99
    {
        return d_on.get();
    }
    virtual float *getColor() //LarryD Mar 04/99
    {
        return d_color.get();
    }
    Viewer::Object getViewerObject() {return d_viewerObject;};

protected:
    VrmlSFFloat d_ambientIntensity;
    VrmlSFColor d_color;
    VrmlSFFloat d_intensity;
    VrmlSFBool d_on;
    Viewer::Object d_viewerObject; // move to VrmlNode.h ? ...
};
}
#endif //_VRMLNODELIGHT_
