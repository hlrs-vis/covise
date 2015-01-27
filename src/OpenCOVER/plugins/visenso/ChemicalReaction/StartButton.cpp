/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "StartButton.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include "cover/coTranslator.h"

using namespace opencover;
using namespace covise;

StartButton::StartButton()
    : coVRIntersectionInteractor(1.0f, coInteraction::ButtonA, "StartButton", "StartButton", coInteraction::Medium)
    , was_clicked(false)
{
    moveTransform->setMatrix(osg::Matrix::translate(-11.0f, 0.0f, -5.0f));

    createGeometry();
    this->enableIntersection();
    this->hide();
}

StartButton::~StartButton()
{
}

void
StartButton::createGeometry()
{
    geometryNode = new osg::Geode(); // every coVRIntersectionInteractor has to create this geode!

    // Box

    boxGeometry = new osg::Box(osg::Vec3(0.0f, 0.0f, 0.0f), 2.2f, 0.25f, 1.5f);
    boxDrawable = new osg::ShapeDrawable(boxGeometry.get());
    boxGeode = new osg::Geode();
    boxGeode->addDrawable(boxDrawable.get());

    osg::Material *boxMaterial = new osg::Material();
    boxMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.5f, 0.2f, 0.2f, 1.0f));
    boxMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
    boxMaterial->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    boxMaterial->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
    boxGeode->getOrCreateStateSet()->setAttributeAndModes(boxMaterial);

    // Cylinder

    cylinderGeometry = new osg::Cylinder(osg::Vec3(0.0f, 0.0f, 0.0f), 1.0f, 0.25f);
    osg::Matrix m;
    m.makeRotate(osg::Vec3(0.0f, 0.0f, 1.0f), osg::Vec3(0.0f, 1.0f, 0.0f));
    cylinderGeometry->setRotation(m.getRotate());
    cylinderDrawable = new osg::ShapeDrawable(cylinderGeometry.get());
    cylinderGeode = new osg::Geode();
    cylinderGeode->addDrawable(cylinderDrawable.get());

    osg::Material *cylinderMaterial = new osg::Material();
    cylinderMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.5f, 0.2f, 1.0f));
    cylinderMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
    cylinderMaterial->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    cylinderMaterial->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
    cylinderGeode->getOrCreateStateSet()->setAttributeAndModes(cylinderMaterial);

    // Text

    textDrawable = new osgText::Text();
    textDrawable->setCharacterSize(0.65f);
    textDrawable->setAlignment(osgText::Text::CENTER_CENTER);
    textDrawable->setAxisAlignment(osgText::Text::XZ_PLANE);
    textDrawable->setPosition(osg::Vec3(0.0f, -0.15f, 0.0f));
    textDrawable->setFont(coVRFileManager::instance()->getFontFile(NULL));

    textGeode = new osg::Geode();
    textGeode->addDrawable(textDrawable.get());

    osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
    osg::Material *textMaterial = new osg::Material;
    textMaterial->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setEmission(osg::Material::FRONT_AND_BACK, textColor);
    textGeode->getOrCreateStateSet()->setAttributeAndModes(textMaterial);

    scaleTransform->addChild(textGeode.get());
}

void StartButton::startInteraction()
{
    was_clicked = true;
    coVRIntersectionInteractor::startInteraction();
}

bool StartButton::wasClicked()
{
    if (was_clicked)
    {
        was_clicked = false;
        return true;
    }
    return false;
}

void StartButton::setButtonState(ButtonState state)
{
    if (state == BUTTON_STATE_START)
    {
        osgText::String starts(coTranslator::coTranslate("Start"), osgText::String::ENCODING_UTF8);
        textDrawable->setText(starts);
        if (scaleTransform->containsNode(boxGeode.get()))
        {
            scaleTransform->removeChild(boxGeode.get());
        }
        if (!scaleTransform->containsNode(cylinderGeode.get()))
        {
            scaleTransform->addChild(cylinderGeode.get());
        }
    }
    else
    {
        osgText::String resets(coTranslator::coTranslate("Reset"), osgText::String::ENCODING_UTF8);
        textDrawable->setText(resets);
        if (scaleTransform->containsNode(cylinderGeode.get()))
        {
            scaleTransform->removeChild(cylinderGeode.get());
        }
        if (!scaleTransform->containsNode(boxGeode.get()))
        {
            scaleTransform->addChild(boxGeode.get());
        }
    }
}

void StartButton::setVisible(bool visible)
{
    if (visible)
        this->show();
    else
        this->hide();
}
