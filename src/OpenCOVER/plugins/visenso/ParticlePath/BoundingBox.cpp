/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BoundingBox.h"
#include "Const.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>

BoundingBox::BoundingBox(osg::ref_ptr<osg::Group> parent)
    : parentNode(parent)
{
    boxGeode = new osg::Geode();
    float min = -0.5;
    float max = 0.5;

    osg::Vec3 bpoints[8];
    bpoints[0].set(min, min, min);
    bpoints[1].set(max, min, min);
    bpoints[2].set(max, max, min);
    bpoints[3].set(min, max, min);
    bpoints[4].set(min, min, max);
    bpoints[5].set(max, min, max);
    bpoints[6].set(max, max, max);
    bpoints[7].set(min, max, max);

    osg::Geometry *lineGeometry[12];
    osg::Vec3Array *vArray[12];
    osg::DrawArrays *drawable[12];

    for (int i = 0; i < 12; i++)
    {
        lineGeometry[i] = new osg::Geometry();
        vArray[i] = new osg::Vec3Array();
        lineGeometry[i]->setVertexArray(vArray[i]);
        drawable[i] = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2);
        lineGeometry[i]->addPrimitiveSet(drawable[i]);
        boxGeode->addDrawable(lineGeometry[i]);
    }

    // lines
    vArray[0]->push_back(bpoints[0]);
    vArray[0]->push_back(bpoints[1]);
    vArray[1]->push_back(bpoints[1]);
    vArray[1]->push_back(bpoints[2]);
    vArray[2]->push_back(bpoints[2]);
    vArray[2]->push_back(bpoints[3]);
    vArray[3]->push_back(bpoints[3]);
    vArray[3]->push_back(bpoints[0]);
    vArray[4]->push_back(bpoints[4]);
    vArray[4]->push_back(bpoints[5]);
    vArray[5]->push_back(bpoints[5]);
    vArray[5]->push_back(bpoints[6]);
    vArray[6]->push_back(bpoints[6]);
    vArray[6]->push_back(bpoints[7]);
    vArray[7]->push_back(bpoints[7]);
    vArray[7]->push_back(bpoints[4]);
    vArray[8]->push_back(bpoints[0]);
    vArray[8]->push_back(bpoints[4]);
    vArray[9]->push_back(bpoints[3]);
    vArray[9]->push_back(bpoints[7]);
    vArray[10]->push_back(bpoints[2]);
    vArray[10]->push_back(bpoints[6]);
    vArray[11]->push_back(bpoints[1]);
    vArray[11]->push_back(bpoints[5]);

    osg::Material *material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    osg::StateSet *stateSet = boxGeode->getOrCreateStateSet();
    stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(material);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    boxGeode->setStateSet(stateSet);
    boxGeode->setNodeMask(boxGeode->getNodeMask() & (~opencover::Isect::Intersection) & (~opencover::Isect::Pick));

    parentNode->addChild(boxGeode);
}

BoundingBox::~BoundingBox()
{
    if (parentNode->containsNode(boxGeode))
        parentNode->removeChild(boxGeode);
}
