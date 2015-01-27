/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coIntersecCheckboxMenuItem.h"

coIntersecCheckboxMenuItem::coIntersecCheckboxMenuItem(const char *name,
                                                       bool donotaskme,
                                                       coCheckboxGroup *group)
    : coCheckboxMenuItem(name, donotaskme, group)
    , _isIntersected(false)
{
}

coIntersecCheckboxMenuItem::~coIntersecCheckboxMenuItem()
{
}

int
    //coIntersecCheckboxMenuItem::hit(osg::Vec3& vec, osgUtil::Hit* p_hit)
    coIntersecCheckboxMenuItem::hit(vruiHit *hit)
{
    int ret = coCheckboxMenuItem::hit(hit);
    _isIntersected = true;
    return ret;
}

void
coIntersecCheckboxMenuItem::miss()
{
    coCheckboxMenuItem::miss();
    _isIntersected = false;
}

bool
coIntersecCheckboxMenuItem::isIntersected() const
{
    return _isIntersected;
}
