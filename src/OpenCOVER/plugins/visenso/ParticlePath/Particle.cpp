/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Particle.h"
#include "Const.h"

Particle::Particle(osg::ref_ptr<osg::Group> parent)
    : parentNode(parent)
{
    transform = new osg::MatrixTransform();
    geode = new osg::Geode();
    transform->addChild(geode);

    geometry = new osg::Sphere(osg::Vec3(0.0, 0.0, 0.0), PARTICLE_SIZE);
    drawable = new osg::ShapeDrawable(geometry.get());
    geode->addDrawable(drawable.get());

    setVisible(true);
}

Particle::~Particle()
{
    setVisible(false);
}

void Particle::update(osg::Vec3 position)
{
    osg::Matrix m;
    m.makeTranslate(position);
    transform->setMatrix(m);
}

void Particle::setVisible(bool visible)
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

void Particle::setColor(osg::Vec4 color)
{
    material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0] * 0.3f, color[1] * 0.3f, color[2] * 0.3f, color[3]));

    osg::StateSet *stateSet = drawable->getOrCreateStateSet();
    stateSet->setAttributeAndModes(material.get());
    drawable->setStateSet(stateSet);
}
