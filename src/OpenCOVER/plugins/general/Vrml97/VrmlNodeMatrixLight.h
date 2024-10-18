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

class VRML97COVEREXPORT VrmlNodeMatrixLight : public VrmlNodeChild
{

public:
    // Define the fields of MatrixLight nodes
    static void initFields(VrmlNodeMatrixLight *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeMatrixLight(VrmlScene *scene = 0);
    VrmlNodeMatrixLight(const VrmlNodeMatrixLight &n);
    virtual ~VrmlNodeMatrixLight();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNodeMatrixLight *toMatrixLight() const;

    virtual void render(Viewer *);
    
    static void updateAll();
    void update();
    
    static std::list<VrmlNodeMatrixLight *> allMatrixLights;

private:
    // Fields
    VrmlSFInt d_lightNumber;
    VrmlSFInt d_numRows;
    VrmlSFInt d_numColumns;
    VrmlSFString d_IESFile;

    coIES *iesFile;
    static osg::ref_ptr<osg::Uniform> matrixLightMatrix; // used to be photometricLightMatrix
    Viewer::Object d_viewerObject;
    osg::ref_ptr<osg::MatrixTransform> lightNodeInSceneGraph;
    static const int MAX_LIGHTS = 4;


};
#endif //_VRMLNODEMatrixLight_
