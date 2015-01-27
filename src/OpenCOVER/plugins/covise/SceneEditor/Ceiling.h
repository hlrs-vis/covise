/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CEILING_H
#define CEILING_H

#include "Barrier.h"

class Ceiling : public Barrier
{
public:
    Ceiling(Room *r, std::string name, float widht, float height, osg::Vec3 pos = osg::Vec3(0.0, 0.0, 0.0));
    virtual ~Ceiling();

protected:
};

#endif
