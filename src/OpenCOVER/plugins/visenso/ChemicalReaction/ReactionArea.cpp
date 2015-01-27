/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReactionArea.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
using namespace opencover;
using namespace covise;

bool isInReactionArea(osg::Vec3 p)
{
    if (p[0] > AREA_X_MAX)
        return false;
    if (p[0] < AREA_X_MIN)
        return false;
    if (p[2] > AREA_Z_MAX)
        return false;
    if (p[2] < AREA_Z_MIN)
        return false;
    return true;
};

ReactionArea::ReactionArea()
{
    osg::Material *material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.7f, 0.7f, 0.7f, 1.0f));
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.3f, 0.3f, 0.3f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 25.0f);

    geode = new osg::Geode();
    geode->getOrCreateStateSet()->setAttributeAndModes(material);

    osg::Box *box;
    osg::Drawable *drawable;

    box = new osg::Box(osg::Vec3(0.0f, 0.0f, AREA_Z_MAX), AREA_X_MAX - AREA_X_MIN + AREA_BORDER_WIDTH, AREA_BORDER_WIDTH, AREA_BORDER_WIDTH);
    drawable = new osg::ShapeDrawable(box);
    geode->addDrawable(drawable);

    box = new osg::Box(osg::Vec3(0.0f, 0.0f, AREA_Z_MIN), AREA_X_MAX - AREA_X_MIN + AREA_BORDER_WIDTH, AREA_BORDER_WIDTH, AREA_BORDER_WIDTH);
    drawable = new osg::ShapeDrawable(box);
    geode->addDrawable(drawable);

    box = new osg::Box(osg::Vec3(AREA_X_MAX, 0.0f, 0.0f), AREA_BORDER_WIDTH, AREA_BORDER_WIDTH, AREA_Z_MAX - AREA_Z_MIN + AREA_BORDER_WIDTH);
    drawable = new osg::ShapeDrawable(box);
    geode->addDrawable(drawable);

    box = new osg::Box(osg::Vec3(AREA_X_MIN, 0.0f, 0.0f), AREA_BORDER_WIDTH, AREA_BORDER_WIDTH, AREA_Z_MAX - AREA_Z_MIN + AREA_BORDER_WIDTH);
    drawable = new osg::ShapeDrawable(box);
    geode->addDrawable(drawable);
}

ReactionArea::~ReactionArea()
{
    setVisible(false);
}

void ReactionArea::setVisible(bool visible)
{
    if (cover->getObjectsRoot()->containsNode(geode.get()))
    {
        if (!visible)
            cover->getObjectsRoot()->removeChild(geode.get());
    }
    else
    {
        if (visible)
            cover->getObjectsRoot()->addChild(geode.get());
    }
}
