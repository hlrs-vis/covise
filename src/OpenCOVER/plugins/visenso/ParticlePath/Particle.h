/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PARTICLE_H
#define _PARTICLE_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Material>

#include <iostream>
#include <vector>

class Particle
{
public:
    Particle(osg::ref_ptr<osg::Group> parent);
    ~Particle();

    void update(osg::Vec3 position);
    void setVisible(bool visible);
    void setColor(osg::Vec4 color);

private:
    osg::ref_ptr<osg::Group> parentNode;

    osg::ref_ptr<osg::Material> material;

    osg::ref_ptr<osg::MatrixTransform> transform;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::ShapeDrawable> drawable;
    osg::ref_ptr<osg::Sphere> geometry;
};

#endif
