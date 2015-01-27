/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Molecule.h"

#include "Elements.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include "cover/coTranslator.h"

using namespace opencover;
using namespace covise;

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

Molecule::Molecule(Design *_design)
    : design(_design)
{
    textDrawable = new osgText::Text();
    textGeode = new osg::Geode();
    textGeode->addDrawable(textDrawable.get());
}

Molecule::~Molecule()
{
    if (cover->getObjectsRoot()->containsNode(textGeode.get()))
        cover->getObjectsRoot()->removeChild(textGeode.get());
}

void Molecule::resetAtoms()
{
    for (std::vector<Atom *>::iterator it = atoms.begin(); it < atoms.end(); ++it)
    {
        (*it)->reset();
    }
}

void Molecule::animateAtoms(float animationTime)
{
    for (std::vector<Atom *>::iterator it = atoms.begin(); it < atoms.end(); ++it)
    {
        (*it)->animate(animationTime);
    }
}

void Molecule::showName()
{
    // Text
    textDrawable->setCharacterSize(0.5f);
    textDrawable->setAlignment(osgText::Text::CENTER_TOP);
    textDrawable->setAxisAlignment(osgText::Text::XZ_PLANE);
    textDrawable->setFont(coVRFileManager::instance()->getFontFile(NULL));
    osgText::String os(coTranslator::coTranslate(design->name), osgText::String::ENCODING_UTF8);
    textDrawable->setText(os);

    // Position
    float xMax(-999.0f), xMin(999.0f), zMin(999.0f);
    for (std::vector<AtomConfig>::iterator it = design->config.begin(); it < design->config.end(); ++it)
    {
        float radius = GET_ELEMENT_RADIUS((*it).element, (*it).charge);
        xMax = MAX(xMax, (*it).position[0] + radius);
        xMin = MIN(xMin, (*it).position[0] - radius);
        zMin = MIN(zMin, (*it).position[2] - radius);
    }
    textDrawable->setPosition(getPosition() + osg::Vec3((xMax + xMin) / 2.0f, -0.3f, zMin - 0.3f));

    // Color
    osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
    osg::Material *textMaterial = new osg::Material;
    textMaterial->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setEmission(osg::Material::FRONT_AND_BACK, textColor);
    textGeode->getOrCreateStateSet()->setAttributeAndModes(textMaterial);

    // Add
    cover->getObjectsRoot()->addChild(textGeode.get());
}
