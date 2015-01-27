/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Avatar.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
using namespace covise;
using namespace opencover;

#define MIN(x, y) (x < y ? x : y)

Avatar::Avatar()
{
    avatarNode = coVRFileManager::instance()->loadIcon("firefighter");
    transform = new osg::MatrixTransform();
    transform->addChild(avatarNode.get());
    addChild(transform.get());
}

Avatar::~Avatar()
{
    removeChild(transform.get());
}

void Avatar::update(osg::Vec3 position, osg::Vec3 orientation)
{
    orientation[2] = 0.0f;
    orientation.normalize();

    osg::Matrix m;
    m = osg::Matrix::rotate(osg::Vec3(0.0f, -1.0f, 0.0f), orientation);
    m = m * osg::Matrix::translate(position);
    transform->setMatrix(m);
}
