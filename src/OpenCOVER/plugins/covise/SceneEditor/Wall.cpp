/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Wall.h"

Wall::Wall(Room *r, std::string name, float width, float height, Wall::Alignment align, osg::Vec3 p)
    : Barrier(r, name, width, height, (Barrier::Alignment)align, p)
{
}

Wall::~Wall()
{
}
