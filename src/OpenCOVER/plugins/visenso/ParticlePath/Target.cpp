/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Target.h"
#include "Const.h"

Target::Target(osg::ref_ptr<osg::Group> parent)
    : GenericGuiObject("ParticlePath.Target")
    , parentNode(parent)
{
    transform = new osg::MatrixTransform();
    geode = new osg::Geode();
    transform->addChild(geode);

    material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, TARGET_COLOR);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(TARGET_COLOR[0] * 0.3f, TARGET_COLOR[1] * 0.3f, TARGET_COLOR[2] * 0.3f, TARGET_COLOR[3]));

    geometry = new osg::Sphere(osg::Vec3(0.0f, 0.0f, 0.0f), TARGET_SIZE);
    drawable = new osg::ShapeDrawable(geometry.get());
    osg::StateSet *stateSet = drawable->getOrCreateStateSet();
    stateSet->setAttributeAndModes(material.get(), osg::StateAttribute::PROTECTED);
    drawable->setStateSet(stateSet);
    geode->addDrawable(drawable.get());

    // add gui params
    p_visible = addGuiParamBool("Visible", false);
    p_position = addGuiParamVec3("Position", osg::Vec3(0.0f, 0.0f, 0.0f));

    setPosition(osg::Vec3(0.0, 0.0, 0.0));
}

Target::~Target()
{
    setVisible(false);
}

void Target::guiParamChanged(GuiParam *guiParam)
{
    if (guiParam == p_visible)
    {
        setVisible(p_visible->getValue());
    }
    else if (guiParam == p_position)
    {
        setPosition(p_position->getValue());
    }
}

void Target::setVisible(bool visible)
{
    if (parentNode->containsNode(transform.get()))
    {
        if (!visible)
            parentNode->removeChild(transform.get());
    }
    else
    {
        if (visible)
            parentNode->addChild(transform.get());
    }
}

void Target::setPosition(osg::Vec3 pos)
{
    osg::Matrix m;
    m.makeTranslate(TRACE_CENTER + pos);
    transform->setMatrix(m);
}
