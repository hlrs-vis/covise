/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Ground.h"

#include <cover/coVRPluginSupport.h>

Ground::Ground()
{
    _name = "";
    _type = SceneObjectTypes::GROUND;
}

Ground::~Ground()
{
}

int Ground::setNode(osg::Node *n)
{
    _node = n;

    return 1;
}

osg::Node *Ground::getNode()
{
    return _node.get();
}
