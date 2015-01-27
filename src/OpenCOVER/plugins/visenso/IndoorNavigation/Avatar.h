/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AVATAR_H
#define _AVATAR_H

#include <osg/MatrixTransform>

class Avatar : public osg::Group
{
public:
    Avatar();
    ~Avatar();

    void update(osg::Vec3 position, osg::Vec3 orientation);

private:
    osg::ref_ptr<osg::Node> avatarNode;
    osg::ref_ptr<osg::MatrixTransform> transform;
};

#endif
