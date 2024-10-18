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

void VrmlNodeShadowedScene::initFields(VrmlNodeShadowedScene *node, VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t,
                    exposedField("technique", node->d_technique),
                    exposedField("shadowLight", node->d_shadowLight),
                    exposedField("jitteringScale", node->d_jitterScale),
                    exposedField("softnessWidth", node->d_softnessWidth),
                    exposedField("textureSize", node->d_textureSize));

}

const char *VrmlNodeShadowedScene::name()
{
    return "ShadowedScene";
}


VrmlNodeShadowedScene::VrmlNodeShadowedScene(VrmlScene *scene)
    : VrmlNodeGroup(scene, name())
    , d_technique("ShadowMap")
    , d_jitterScale(32)
    , d_softnessWidth(0.005)
    , d_textureSize(1024,1024)
{
    d_shadowObject = 0;
}

VrmlNodeShadowedScene::VrmlNodeShadowedScene(const VrmlNodeShadowedScene &n)
    : VrmlNodeGroup(n)
    , d_technique(n.d_technique)
    , d_shadowLight(n.d_shadowLight)
    , d_jitterScale(n.d_jitterScale)
    , d_softnessWidth(n.d_softnessWidth)
    , d_textureSize(n.d_textureSize)
{
    d_shadowObject = 0;
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
