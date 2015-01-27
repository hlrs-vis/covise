/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Ceiling.h"

Ceiling::Ceiling(Room *r, std::string name, float width, float height, osg::Vec3 pos)
    : Barrier(r, name, width, height, Barrier::TOP, pos)
{
}

Ceiling::~Ceiling()
{
}
