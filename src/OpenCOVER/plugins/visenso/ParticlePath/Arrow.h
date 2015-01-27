/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ARROW_H
#define _ARROW_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Material>

#include <iostream>
#include <vector>

class Arrow
{
public:
    Arrow(osg::ref_ptr<osg::Group> parent, osg::Vec4 color);
    ~Arrow();

    void update(osg::Vec3 position, osg::Vec3 vector);
    void setVisible(bool visible);

private:
    osg::ref_ptr<osg::Group> parentNode;

    bool isInScene;
    bool shouldBeVisible;
    bool isInvalid;

    osg::ref_ptr<osg::Material> material;

    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::ShapeDrawable> cylinderD;
    osg::ref_ptr<osg::Cylinder> cylinderG;
    osg::ref_ptr<osg::ShapeDrawable> coneD;
    osg::ref_ptr<osg::Cone> coneG;
};

#endif
