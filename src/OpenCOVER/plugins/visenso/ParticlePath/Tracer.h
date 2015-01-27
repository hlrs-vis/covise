/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACER_H
#define _TRACER_H

#include <osg/Vec3d>
#include <osg/Matrix>

#include <iostream>
#include <vector>

typedef struct
{
    osg::Vec3d position;
    osg::Vec3d velocity;
    osg::Vec3d electricForce;
    osg::Vec3d magneticForce;
    osg::Vec3d combinedForce;
} TracerStep;

typedef struct
{
    double mass; // in kg
    double charge; // in C
    double velocity; // in m/s
    double angle; // in radian
    double electricField; // in V/m
    double magneticField; // in F
} Config;

class Tracer
{
public:
    Tracer();
    ~Tracer();

    void trace();

    Config config;
    std::vector<TracerStep> result;

private:
    void init();
    void step();
    void calculateForces(TracerStep &step);
};

#endif
