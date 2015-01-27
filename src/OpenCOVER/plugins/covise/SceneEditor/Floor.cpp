/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Floor.h"

Floor::Floor(Room *r, std::string name, float width, float height, osg::Vec3 pos)
    : Barrier(r, name, width, height, Barrier::BOTTOM, pos)
{
    _transformNode->setName("Floor");
}

Floor::~Floor()
{
}
