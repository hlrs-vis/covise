/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Arrow.h"
#include "Const.h"

Arrow::Arrow(osg::ref_ptr<osg::Group> parent, osg::Vec4 color)
    : parentNode(parent)
    , isInScene(false)
    , shouldBeVisible(true)
    , isInvalid(true)
{
    geode = new osg::Geode();

    material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0] * 0.3f, color[1] * 0.3f, color[2] * 0.3f, color[3]));
    osg::StateSet *stateSet;

    cylinderG = new osg::Cylinder(EPSILON_CENTER, 0.5f * ARROW_SIZE, 0.1f);
    cylinderD = new osg::ShapeDrawable(cylinderG.get());
    cylinderD->setUseDisplayList(false);
    stateSet = cylinderD->getOrCreateStateSet();
    stateSet->setAttributeAndModes(material.get(), osg::StateAttribute::PROTECTED);
    cylinderD->setStateSet(stateSet);
    geode->addDrawable(cylinderD.get());

    coneG = new osg::Cone(EPSILON_CENTER, 1.5f * ARROW_SIZE, 2.5f * ARROW_SIZE);
    coneD = new osg::ShapeDrawable(coneG.get());
    coneD->setUseDisplayList(false);
    stateSet = coneD->getOrCreateStateSet();
    stateSet->setAttributeAndModes(material.get(), osg::StateAttribute::PROTECTED);
    coneD->setStateSet(stateSet);
    geode->addDrawable(coneD.get());
}

Arrow::~Arrow()
{
    setVisible(false);
}

void Arrow::update(osg::Vec3 position, osg::Vec3 vector)
{
    float length = vector.length();
    if (length < EPSILON)
    {
        isInvalid = true;
        if (isInScene)
        {
            isInScene = false;
            parentNode->removeChild(geode.get());
        }
    }

    osg::Vec3 lineCenter = position + vector * 0.5f;
    osg::Vec3 lineEnd = position + vector;
    osg::Matrix m;
    m.makeRotate(osg::Vec3(0.0f, 0.0f, length), vector);

    cylinderG->setCenter(lineCenter);
    cylinderG->setHeight(length);
    cylinderG->setRotation(m.getRotate());
    cylinderD->dirtyBound();
    coneG->setCenter(lineEnd);
    coneG->setRotation(m.getRotate());
    coneD->dirtyBound();

    isInvalid = false;
    if (shouldBeVisible && !isInScene)
    {
        isInScene = true;
        parentNode->addChild(geode.get());
    }
}

void Arrow::setVisible(bool visible)
{
    shouldBeVisible = visible;
    if (isInScene)
    {
        if (!visible || isInvalid)
            parentNode->removeChild(geode.get());
    }
    else
    {
        if (visible && !isInvalid)
            parentNode->addChild(geode.get());
    }
}
