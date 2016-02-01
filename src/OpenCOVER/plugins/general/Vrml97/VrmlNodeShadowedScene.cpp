/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeShadowedScene.cpp
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

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNodeLight.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>
#include <cover/coVRLighting.h>
#include <cover/coVRShadowManager.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <math.h>

#include "VrmlNodeShadowedScene.h"
#include "ViewerOsg.h"
#include <osg/MatrixTransform>
#include <osg/Quat>

// ShadowedScene factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeShadowedScene(scene);
}

// Define the built in VrmlNodeType:: "ShadowedScene" fields

VrmlNodeType *VrmlNodeShadowedScene::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("ShadowedScene", creator);
    }

    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("technique", VrmlField::SFSTRING);
    t->addExposedField("shadowLight", VrmlField::SFNODE);
    t->addExposedField("jitteringScale", VrmlField::SFFLOAT);
    t->addExposedField("softnessWidth", VrmlField::SFFLOAT);
    t->addExposedField("textureSize",VrmlField::SFVEC2F);

    return t;
}

VrmlNodeType *VrmlNodeShadowedScene::nodeType() const
{
    return defineType(0);
}

VrmlNodeShadowedScene::VrmlNodeShadowedScene(VrmlScene *scene)
    : VrmlNodeGroup(scene)
    , d_technique("ShadowMap")
    , d_jitterScale(32)
    , d_softnessWidth(0.005)
    , d_textureSize(1024,1024)
{
    d_shadowObject = 0;
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeShadowedScene::VrmlNodeShadowedScene(const VrmlNodeShadowedScene &n)
    : VrmlNodeGroup(n.d_scene)
    , d_technique(n.d_technique)
    , d_shadowLight(n.d_shadowLight)
    , d_jitterScale(n.d_jitterScale)
    , d_softnessWidth(n.d_softnessWidth)
    , d_textureSize(n.d_textureSize)
{
    d_shadowObject = 0;
}

VrmlNodeShadowedScene::~VrmlNodeShadowedScene()
{
}

VrmlNode *VrmlNodeShadowedScene::cloneMe() const
{
    return new VrmlNodeShadowedScene(*this);
}

void VrmlNodeShadowedScene::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_shadowObject && isModified())
    {
        viewer->removeObject(d_shadowObject);
        d_shadowObject = 0;
    }
    checkAndRemoveNodes(viewer);
    if (d_shadowObject)
    {
        viewer->insertReference(d_shadowObject);
    }
    else if (d_children.size() > 0)
    {
        d_shadowObject = viewer->beginObject(name(), 0, this);
    }
    if(isModified())
    {
        viewer->setShadow(d_technique.get());
        coVRShadowManager::instance()->setSoftnessWidth(d_softnessWidth.get());
        coVRShadowManager::instance()->setJitteringScale(d_jitterScale.get());
        osg::Vec2s ts;
        ts[0] = d_textureSize.get()[0];
        ts[1] = d_textureSize.get()[1];
        coVRShadowManager::instance()->setTextureSize(ts);
        // set shadow Light
        VrmlNodeLight *ln = dynamic_cast<VrmlNodeLight *>(d_shadowLight.get());
        if(ln!=NULL)
        {

            osgViewerObject *obj = (osgViewerObject *)ln->getViewerObject();
            if(obj->pNode!=NULL)
            {
                osg::LightSource * ls = dynamic_cast<osg::LightSource *>(obj->pNode.get());
                coVRLighting::instance()->setShadowLight(ls);
            }
        }
    }
    if (d_children.size() > 0)
    {

        // Render children
        VrmlNodeGroup::render(viewer);
        viewer->endObject();
    }
    clearModified();
}

ostream &VrmlNodeShadowedScene::printFields(ostream &os, int indent)
{
    if (!d_technique.get())
        PRINT_FIELD(technique);
    if(!d_shadowLight.get())
        PRINT_FIELD(shadowLight);
    if(!d_jitterScale.get())
        PRINT_FIELD(jitterScale);
    if(!d_softnessWidth.get())
        PRINT_FIELD(softnessWidth);
    if(!d_textureSize.get())
        PRINT_FIELD(textureSize);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeShadowedScene::setField(const char *fieldName,
                                     const VrmlField &fieldValue)
{
    if
        TRY_FIELD(technique, SFString)
    else if
        TRY_FIELD(shadowLight,SFNode)
    else if
        TRY_FIELD(softnessWidth,SFFloat)
    else if
        TRY_FIELD(jitterScale,SFFloat)
    else if
        TRY_FIELD(textureSize,SFVec2f)
    else
        VrmlNodeGroup::setField(fieldName, fieldValue);
    setModified();
}

const VrmlField *VrmlNodeShadowedScene::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "technique") == 0)
        return &d_technique;
    else if(strcmp(fieldName,"shadowLight") == 0)
        return &d_shadowLight;
    else if(strcmp(fieldName,"softnessWidth") == 0)
        return &d_softnessWidth;
    else if(strcmp(fieldName,"jitterScale") == 0)
        return &d_jitterScale;
    else if(strcmp(fieldName,"textureSize") == 0)
        return &d_textureSize;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
