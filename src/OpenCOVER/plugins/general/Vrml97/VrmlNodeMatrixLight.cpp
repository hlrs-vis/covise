/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeMatrixLight.cpp
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <util/byteswap.h>

#include "VrmlNodeMatrixLight.h"
#include "ViewerOsg.h"
#include <osg/Quat>

// static initializations
std::list<VrmlNodeMatrixLight *> VrmlNodeMatrixLight::allMatrixLights;
osg::ref_ptr<osg::Uniform> VrmlNodeMatrixLight::matrixLightMatrix;

// MatrixLight factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeMatrixLight(scene);
}

void VrmlNodeMatrixLight::updateAll()
{
    for(std::list<VrmlNodeMatrixLight *>::iterator it = allMatrixLights.begin();it != allMatrixLights.end(); it++)
    {
        (*it)->update();
    }
}
void VrmlNodeMatrixLight::update()
{
    osg::MatrixList worldMatrices = lightNodeInSceneGraph->getWorldMatrices();
    osg::Matrixf firstMat = worldMatrices[0];
       // matrixLightMatrix->setElement(d_lightNumber.get(), firstMat); 
    osg::Matrixf invFirstMat;
    if(invFirstMat.invert_4x4(firstMat))
    {
        matrixLightMatrix->setElement(d_lightNumber.get(), invFirstMat); 
    }
}

// Define the built in VrmlNodeType:: "MatrixLight" fields

VrmlNodeType *VrmlNodeMatrixLight::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("MatrixLight", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("lightNumber", VrmlField::SFINT32);
    t->addExposedField("IESFile", VrmlField::SFSTRING);
    static osg::Matrixf lightMatrices[MAX_LIGHTS];
    matrixLightMatrix =new osg::Uniform(osg::Uniform::FLOAT_MAT4, "matrixLightMatrix", MAX_LIGHTS);
    osg::StateSet *state = cover->getObjectsRoot()->getOrCreateStateSet();
    state->addUniform(matrixLightMatrix);

    return t;
}

VrmlNodeType *VrmlNodeMatrixLight::nodeType() const
{
    return defineType(0);
}

VrmlNodeMatrixLight::VrmlNodeMatrixLight(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_lightNumber(0)
    , d_viewerObject(0)
    , d_IESFile("")
{
    setModified();
    lightNodeInSceneGraph = new osg::MatrixTransform();
    allMatrixLights.push_back(this);
}

void VrmlNodeMatrixLight::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeMatrixLight::VrmlNodeMatrixLight(const VrmlNodeMatrixLight &n)
    : VrmlNodeChild(n.d_scene)
    , d_lightNumber(n.d_lightNumber)
    , d_viewerObject(n.d_viewerObject)
    , d_IESFile(n.d_IESFile)
    , lightNodeInSceneGraph(n.lightNodeInSceneGraph)
{
    allMatrixLights.push_back(this);
    setModified();
}

VrmlNodeMatrixLight::~VrmlNodeMatrixLight()
{
    allMatrixLights.remove(this);
}

VrmlNode *VrmlNodeMatrixLight::cloneMe() const
{
    return new VrmlNodeMatrixLight(*this);
}

VrmlNodeMatrixLight *VrmlNodeMatrixLight::toMatrixLight() const
{
    return (VrmlNodeMatrixLight *)this;
}

void VrmlNodeMatrixLight::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_viewerObject && isModified())
    {
        viewer->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }
    if (d_viewerObject)
    {
        viewer->insertReference(d_viewerObject);
    }
    d_viewerObject = viewer->beginObject(name(), 0, this);

    ((osgViewerObject *)d_viewerObject)->pNode = lightNodeInSceneGraph;
    ((ViewerOsg *)viewer)->addToScene((osgViewerObject *)d_viewerObject);

    viewer->endObject();

    clearModified();
}

ostream &VrmlNodeMatrixLight::printFields(ostream &os, int indent)
{
    if (!d_lightNumber.get())
        PRINT_FIELD(lightNumber);
    if (!d_IESFile.get())
        PRINT_FIELD(IESFile);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeMatrixLight::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(lightNumber, SFInt)
    else if
        TRY_FIELD(IESFile, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "lightNumber") == 0)
    {
    }
    if (strcmp(fieldName, "IESFile") == 0)
    {
        iesFile = new coIES(d_IESFile.get());
        osg::ref_ptr<osg::Texture2D> lightTexture = new osg::Texture2D();
        lightTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
        lightTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
        lightTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP);
        lightTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP);
        lightTexture->setImage(iesFile->getTexture());
        osg::StateSet *state = cover->getObjectsRoot()->getOrCreateStateSet();

        state->setTextureAttributeAndModes(5+d_lightNumber.get(), lightTexture, osg::StateAttribute::ON);
    }
}

const VrmlField *VrmlNodeMatrixLight::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "lightNumber") == 0)
        return &d_lightNumber;
    if (strcmp(fieldName, "IESFile") == 0)
        return &d_IESFile;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
