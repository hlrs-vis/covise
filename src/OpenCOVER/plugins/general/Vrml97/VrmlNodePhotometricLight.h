/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePhotometricLight.h

#ifndef _VRMLNODEPhotometricLight_
#define _VRMLNODEPhotometricLight_

#include <util/coTypes.h>
#include "coIES.h"

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

class VRML97COVEREXPORT VrmlNodePhotometricLight : public VrmlNodeChild
{

public:
    // Define the fields of PhotometricLight nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodePhotometricLight(VrmlScene *scene = 0);
    VrmlNodePhotometricLight(const VrmlNodePhotometricLight &n);
    virtual ~VrmlNodePhotometricLight();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodePhotometricLight *toPhotometricLight() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual void render(Viewer *);
    
    static void updateAll();
    void update();
    
    static std::list<VrmlNodePhotometricLight *> allPhotometricLights;

private:
    // Fields
    VrmlSFInt d_lightNumber;
    VrmlSFString d_IESFile;

    coIES *iesFile;
    static osg::ref_ptr<osg::Uniform> photometricLightMatrix;
    Viewer::Object d_viewerObject;
    osg::ref_ptr<osg::MatrixTransform> lightNodeInSceneGraph;
    static const int MAX_LIGHTS = 4;


};
#endif //_VRMLNODEPhotometricLight_
