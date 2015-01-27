/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WALL_H
#define WALL_H

#include "Barrier.h"

class Wall : public Barrier
{
public:
    enum Alignment
    {
        FRONT = Barrier::FRONT,
        BACK = Barrier::BACK,
        LEFT = Barrier::LEFT,
        RIGHT = Barrier::RIGHT
    };

    Wall(Room *r, std::string name, float width, float height, Wall::Alignment align, osg::Vec3 pos = osg::Vec3(0.0, 0.0, 0.0));
    virtual ~Wall();

protected:
};

#endif
