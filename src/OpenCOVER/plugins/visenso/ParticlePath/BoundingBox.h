/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BOUNDING_BOX_H
#define _BOUNDING_BOX_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Material>

#include <iostream>
#include <vector>

class BoundingBox
{
public:
    BoundingBox(osg::ref_ptr<osg::Group> parent);
    ~BoundingBox();

private:
    osg::ref_ptr<osg::Group> parentNode;

    osg::ref_ptr<osg::Geode> boxGeode;
};

#endif
