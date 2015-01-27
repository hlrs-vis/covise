/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Atom.h"
#include "Elements.h"

#include "StartMolecule.h"
#include "EndMolecule.h"

#include <cover/coVRFileManager.h>

#include "cover/coTranslator.h"

#include <math.h>

Atom::Atom(AtomConfig _startConfig, StartMolecule *_startMolecule)
    : startConfig(_startConfig)
    , endConfig(AtomConfig())
    , startMolecule(_startMolecule)
    , endMolecule(NULL)
{
    transform = new osg::MatrixTransform();

    // Sphere

    osg::Vec4 sphereColor = ELEMENT_COLORS[startConfig.element];
    osg::Material *sphereMaterial = new osg::Material();
    sphereMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, sphereColor);
    sphereMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(sphereColor[0] * 0.7f, sphereColor[1] * 0.7f, sphereColor[2] * 0.7f, 1.0f));
    sphereMaterial->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    sphereMaterial->setShininess(osg::Material::FRONT_AND_BACK, 25.0f);

    sphereGeometry = new osg::Sphere(osg::Vec3(0.0f, 0.0f, 0.0f), 1.0f);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(1.0f);
    sphereDrawable = new osg::ShapeDrawable(sphereGeometry.get(), hint);
    //sphereDrawable->setColor(color);
    sphereGeode = new osg::Geode();
    sphereGeode->addDrawable(sphereDrawable.get());
    sphereGeode->getOrCreateStateSet()->setAttributeAndModes(sphereMaterial);
    transform->addChild(sphereGeode.get());

    // Text

    std::string symbol(coTranslator::coTranslate(ELEMENT_SYMBOLS[startConfig.element]));
    textDrawable = new osgText::Text3D();
    textDrawable->setDrawMode(osgText::Text3D::TEXT);
    if (symbol.length() == 1)
        textDrawable->setCharacterSize(0.85f);
    else
        textDrawable->setCharacterSize(0.6f);
    textDrawable->setCharacterDepth(0.5f);
    textDrawable->setAlignment(osgText::Text3D::CENTER_CENTER);
    textDrawable->setAxisAlignment(osgText::Text3D::XZ_PLANE);
    textDrawable->setPosition(osg::Vec3(0.0f, -0.5f, 0.0f));
    textDrawable->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textDrawable->setText(symbol);

    textGeode = new osg::Geode();
    textGeode->addDrawable(textDrawable.get());

    osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
    if (sphereColor[0] + sphereColor[1] + sphereColor[2] > 1.9f)
        textColor = osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    osg::Material *textMaterial = new osg::Material;
    textMaterial->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setEmission(osg::Material::FRONT_AND_BACK, textColor);
    textGeode->getOrCreateStateSet()->setAttributeAndModes(textMaterial);

    transform->addChild(textGeode.get());

    // Set position
    reset();
}

Atom::~Atom()
{
}

void Atom::reset()
{
    osg::Matrix m = osg::Matrix::translate(startConfig.position);
    float scale = GET_ELEMENT_RADIUS(startConfig.element, startConfig.charge);
    m.preMultScale(osg::Vec3(scale, scale, scale));
    transform->setMatrix(m);
}

void Atom::animate(float animationTime)
{
    osg::Vec3 from = startConfig.position;
    osg::Vec3 to = endMolecule->getPosition() - startMolecule->getPosition() + endConfig.position;

    float fromRadius = GET_ELEMENT_RADIUS(startConfig.element, startConfig.charge);
    float toRadius = GET_ELEMENT_RADIUS(endConfig.element, endConfig.charge);

    // way percentage
    float percent;
    if (animationTime < 0.5f)
        percent = pow(animationTime * 2.0f, 2.0f) / 2.0f;
    else
        percent = 1.0f - pow((1.0f - animationTime) * 2.0f, 2.0f) / 2.0f;

    // add a offset in y-direction to get a curve
    float tmp = 2.0f * animationTime;
    if (animationTime > 0.5f)
        tmp = 2.0f - 2.0f * animationTime;
    float yOffset;
    if (tmp < 0.5f)
        yOffset = pow(tmp * 2.0f, 2.0f) / 2.0f;
    else
        yOffset = 1.0f - pow((1.0f - tmp) * 2.0f, 2.0f) / 2.0f;
    if (from[0] < to[0])
        yOffset *= -1.0f;
    yOffset *= (from - to).length();

    // set translation
    osg::Vec3 trans;
    trans = from + (to - from) * percent + osg::Vec3(0.0f, 0.3f, 0.0f) * yOffset;
    osg::Matrix m = osg::Matrix::translate(trans);

    // set scaling
    float scale = (1.0f - percent) * fromRadius + percent * toRadius;
    m.preMultScale(osg::Vec3(scale, scale, scale));

    transform->setMatrix(m);
}
