/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Light.h"

#include <osgShadow/ShadowedScene>

using namespace covise;
using namespace opencover;

Light::Light()
{
    _name = "";
    _type = SceneObjectTypes::LIGHT;
}

Light::~Light()
{
    //TODO check if other lights are there
    // remove light from shadowed scene
    osgShadow::ShadowedScene *shadowedScene = dynamic_cast<osgShadow::ShadowedScene *>(opencover::cover->getObjectsRoot()->getParent(0));
    if (shadowedScene)
        shadowedScene->setShadowTechnique(NULL);
}

int Light::setNode(osg::Node *n)
{
    _node = n;

    return 1;
}

osg::Node *Light::getNode()
{
    return _node.get();
}
