/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PATH_H
#define _PATH_H

#include "Arrow.h"
#include "Particle.h"
#include "Tracer.h"

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/Geode>

#include <iostream>
#include <vector>

class Path
{
public:
    Path(osg::ref_ptr<osg::Group> parent);
    ~Path();

    void preFrame();
    void calculateNewPath();
    void setInactive();

    Tracer *tracer;

private:
    osg::ref_ptr<osg::Group> parentNode;

    float animationTime;

    osg::ref_ptr<osg::Geode> pathGeode;
    osg::ref_ptr<osg::Geometry> pathGeometry;

    Particle *particle;
    Arrow *velocityArrow;
    Arrow *combinedForceArrow;
    Arrow *electricForceArrow;
    Arrow *magneticForceArrow;
};

#endif
