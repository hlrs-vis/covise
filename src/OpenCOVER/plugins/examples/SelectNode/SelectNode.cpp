/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: SelectNode OpenCOVER Plugin selects vertices of a mesh      **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include "SelectNode.h"
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coIntersection.h>
#include <osg/Drawable>
#include <osg/Geometry>
#include <osg/Material>

SelectNode::SelectNode()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

// this is called if the plugin is removed at runtime
SelectNode::~SelectNode()
{
}

bool SelectNode::destroy()
{
    while (transformSphere->getNumParents() != 0)
        cover->getObjectsRoot()->removeChild(transformSphere.get());
    return true;
}

bool
SelectNode::init()
{

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 5.0);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    mySphereDrawable->setColor(osg::Vec4(1.0f, 0.2f, 0.0f, 1.0f));
    geometryNode = new osg::Geode();
    ((osg::Geode *)geometryNode.get())->addDrawable(mySphereDrawable);
    geometryNode->setStateSet(createRedStateSet());
    transformSphere = new osg::MatrixTransform();
    transformSphere->addChild(geometryNode.get());
    transformSphere->setNodeMask(transformSphere->getNodeMask() & ~Isect::Intersection);
    transformSphere->setName("Avatar SelectionSphere");
    return true;
}

void SelectNode::preFrame()
{
    osg::Vec3 pos(0.0, 0.0, 0.0);
    if (cover->getIntersectedNode())
    {
        //pos = cover->getIntersectionHitPointWorld()*cover->getInvBaseMat();
        osg::Vec3 pointerPos = coIntersection::instance()->hitInformation.getLocalIntersectPoint();

        osg::Drawable *drawable = coIntersection::instance()->hitInformation.getDrawable();
        osg::Geometry *geometry = drawable->asGeometry();
        if (geometry)
        {
            osg::Vec3Array *vertices = dynamic_cast<osg::Vec3Array *>(geometry->getVertexArray());
            if (vertices)
            {
                double minSquareDist = DBL_MAX;
                osgUtil::Hit::VecIndexList vl = coIntersection::instance()->hitInformation.getVecIndexList();
                for (osgUtil::Hit::VecIndexList::iterator it = vl.begin(); it != vl.end(); it++)
                {
                    if (((*vertices)[(*it)] - pointerPos).length2() < minSquareDist)
                    {
                        minSquareDist = ((*vertices)[(*it)] - pointerPos).length2();
                        pos = (*vertices)[(*it)];
                    }
                }
            }
        }

        //int getPrimitiveIndex() const { return _primitiveIndex; }

        if (transformSphere->getNumParents() == 0)
            cover->getObjectsRoot()->addChild(transformSphere.get());
    }
    else
    {
        while (transformSphere->getNumParents() != 0)
            cover->getObjectsRoot()->removeChild(transformSphere.get());
    }
    osg::Vec3 worldPos = pos * cover->getBaseMat();
    float size = cover->getInteractorScale(worldPos) / 1000.0;

    osg::Matrix mat;
    mat.makeScale(size, size, size);
    mat.setTrans(pos);
    transformSphere->setMatrix(mat);
}

osg::StateSet *
SelectNode::createRedStateSet()
{
    osg::Material *material = new osg::Material();

    material->setColorMode(osg::Material::OFF);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.2f, 0.1f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);

    osg::StateSet *stateSet = new osg::StateSet();
    stateSet->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    return stateSet;
}

COVERPLUGIN(SelectNode)
