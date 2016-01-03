/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeMatrixLight.h

#ifndef _VRMLNODEMatrixLight_
#define _VRMLNODEMatrixLight_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;
using namespace vrml;

class VRML97COVEREXPORT VrmlNodeMatrixLight : public VrmlNodeChild
{

public:
    // Define the fields of MatrixLight nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeMatrixLight(VrmlScene *scene = 0);
    VrmlNodeMatrixLight(const VrmlNodeMatrixLight &n);
    virtual ~VrmlNodeMatrixLight();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeMatrixLight *toMatrixLight() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }
    static void update();

private:
    // Fields
    VrmlSFInt d_numMatrixLight;
    VrmlSFFloat d_fraction_changed;

    VrmlSFBool d_enabled;
    VrmlSFBool d_loop;
    osg::ref_ptr<coMatrixLightEffect> precipitationEffect;
};
#endif //_VRMLNODEMatrixLight_
