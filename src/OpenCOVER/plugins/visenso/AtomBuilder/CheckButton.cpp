/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CheckButton.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
using namespace opencover;
using namespace covise;
CheckButton::CheckButton(osg::Vec3 pos, float size)
    : coVRIntersectionInteractor(size, coInteraction::ButtonA, "CheckButton", "CheckButton", coInteraction::Medium)
    , was_clicked(false)
{
    osg::Matrix m;
    m.makeTranslate(pos);
    moveTransform->setMatrix(m);

    createGeometry();
    this->enableIntersection();
    this->hide();
}

CheckButton::~CheckButton()
{
}

void
CheckButton::createGeometry()
{

    geometryNodeCheck_ = coVRFileManager::instance()->loadIcon("atombaukasten/button_check");
    geometryNodeOk_ = coVRFileManager::instance()->loadIcon("atombaukasten/button_ok");
    geometryNodeNotOk_ = coVRFileManager::instance()->loadIcon("atombaukasten/button_notok");

    setButtonState(BUTTON_STATE_CHECK);

    /* Cylinder
   geometryNode = new osg::Geode(); // every coVRIntersectionInteractor has to create this geode!

   cylinderGeometry = new osg::Cylinder(osg::Vec3(0.0f, 0.0f, 0.0f), 1.0f, 0.25f);
   osg::Matrix m;
   m.makeRotate(osg::Vec3(0.0f,0.0f,1.0f), osg::Vec3(0.0f,1.0f,0.0f));
   cylinderGeometry->setRotation(m.getRotate());
   cylinderDrawable = new osg::ShapeDrawable(cylinderGeometry.get());
   cylinderGeode = new osg::Geode();
   cylinderGeode->addDrawable(cylinderDrawable.get());

   osg::Material* cylinderMaterial = new osg::Material();
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
   textDrawable->setText("CheckCheckCheck");

   textGeode = new osg::Geode();
   textGeode->addDrawable(textDrawable.get());

   osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
   osg::Material* textMaterial = new osg::Material;
   textMaterial->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
   textMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
   textMaterial->setEmission(osg::Material::FRONT_AND_BACK, textColor);
   textGeode->getOrCreateStateSet()->setAttributeAndModes(textMaterial);

   scaleTransform->addChild(textGeode.get());
   scaleTransform->addChild(cylinderGeode.get());
   */
}

void CheckButton::startInteraction()
{
    was_clicked = true;
    coVRIntersectionInteractor::startInteraction();
}

bool CheckButton::wasClicked()
{
    if (was_clicked)
    {
        was_clicked = false;
        return true;
    }
    return false;
}

void CheckButton::setText(string t)
{
    textDrawable->setText(t);
}

void CheckButton::setVisible(bool visible)
{
    if (visible)
        this->show();
    else
        this->hide();
}

void CheckButton::setButtonState(ButtonState state)
{
    if (state == BUTTON_STATE_CHECK)
    {

        if (scaleTransform->containsNode(geometryNodeOk_.get()))
        {
            scaleTransform->removeChild(geometryNodeOk_.get());
        }
        if (scaleTransform->containsNode(geometryNodeNotOk_.get()))
        {
            scaleTransform->removeChild(geometryNodeNotOk_.get());
        }
        if (!scaleTransform->containsNode(geometryNodeCheck_.get()))
        {
            scaleTransform->addChild(geometryNodeCheck_);
        }
    }
    else if (state == BUTTON_STATE_OK)
    {
        if (scaleTransform->containsNode(geometryNodeNotOk_.get()))
        {
            scaleTransform->removeChild(geometryNodeNotOk_.get());
        }
        if (scaleTransform->containsNode(geometryNodeCheck_))
        {
            scaleTransform->removeChild(geometryNodeCheck_);
        }
        if (!scaleTransform->containsNode(geometryNodeOk_.get()))
        {
            scaleTransform->addChild(geometryNodeOk_.get());
        }
    }
    else if (state == BUTTON_STATE_NOTOK)
    {
        if (scaleTransform->containsNode(geometryNodeOk_.get()))
        {
            scaleTransform->removeChild(geometryNodeOk_.get());
        }
        if (scaleTransform->containsNode(geometryNodeCheck_.get()))
        {
            scaleTransform->removeChild(geometryNodeCheck_.get());
        }
        if (!scaleTransform->containsNode(geometryNodeNotOk_))
        {
            scaleTransform->addChild(geometryNodeNotOk_.get());
        }
    }
    else
        fprintf(stderr, "CheckButton::setButtonState: ERROR: unknow state\n");
}
