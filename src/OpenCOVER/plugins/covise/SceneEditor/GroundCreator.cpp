/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GroundCreator.h"

#include <QDir>
#include <iostream>

#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/CullFace>

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

GroundCreator::GroundCreator()
{
}

GroundCreator::~GroundCreator()
{
}

SceneObject *GroundCreator::createFromXML(QDomElement *root)
{
    Ground *wf = new Ground();
    if (!buildFromXML(wf, root))
    {
        delete wf;
        return NULL;
    }
    return wf;
}

bool GroundCreator::buildFromXML(SceneObject *so, QDomElement *root)
{
    if (!buildGeometryFromXML((Ground *)so, root))
    {
        return false;
    }
    return SceneObjectCreator::buildFromXML(so, root);
}

bool GroundCreator::buildGeometryFromXML(Ground *ground, QDomElement *root)
{
    float width(10000.0f);
    float length(10000.0f);
    QDomElement geoElem = root->firstChildElement("geometry");
    if (!geoElem.isNull())
    {
        QDomElement w = geoElem.firstChildElement("width");
        if (!w.isNull())
        {
            width = w.attribute("value").toFloat();
        }
        QDomElement l = geoElem.firstChildElement("length");
        if (!l.isNull())
        {
            length = l.attribute("value").toFloat();
        }
    }
    width = width / 2;
    length = length / 2;

    osg::Vec4Array *colorArray = new osg::Vec4Array(1);
    (*colorArray)[0].set(1, 1, 1, 1);
    osg::Vec3Array *coordArray = new osg::Vec3Array(4);
    (*coordArray)[0].set(-width, -length, 0.0f);
    (*coordArray)[1].set(width, -length, 0.0f);
    (*coordArray)[2].set(width, length, 0.0f);
    (*coordArray)[3].set(-width, length, 0.0f);
    osg::Vec3Array *normalArray = new osg::Vec3Array(1);
    (*normalArray)[0].set(0, 0, 1);

    osg::Geometry *geometry = new osg::Geometry();
    geometry->setColorArray(colorArray);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometry->setVertexArray(coordArray);
    geometry->setNormalArray(normalArray);
    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
    geometry->setUseDisplayList(true);

    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.7f, 0.7f, 0.7f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);

    // stateset for culling back
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.0);
    osg::CullFace *cullFace = new osg::CullFace();
    cullFace->setMode(osg::CullFace::BACK);
    osg::StateSet *stateSet = geometry->getOrCreateStateSet();
    //stateSet->setRenderingHint(osg::StateSet::OPAQUE_BIN); // be carefull because of shadowed scene
    //stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF); // be carefull because of shadowed scene
    stateSet->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);
    stateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);

    geometry->setStateSet(stateSet);

    osg::Geode *geode = new osg::Geode();
    geode->addDrawable(geometry);
    ground->setGeometryNode(geode);
    geode->setNodeMask((opencover::Isect::Visible & (~opencover::Isect::Intersection) & (~opencover::Isect::Pick)));
    //geode->setNodeMask((opencover::Isect::Visible & (~opencover::Isect::Intersection) & (~opencover::Isect::Pick)) | Shadow::CastsShadowTraversalMask | Shadow::ReceivesShadowTraversalMask);
    //geode->setNodeMask(Shadow::CastsShadowTraversalMask | Shadow::ReceivesShadowTraversalMask);

    return true;
}
