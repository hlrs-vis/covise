#ifndef SPLINE_H
#define SPLINE_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <math.h>
#include <iostream>
#include <Position.h>
#include "Entity.h"

class Spline
{

public:
    Spline();
    ~Spline();

    // Coordinates
    double x0;
    double y0;

    double x1;
    double y1;


    //polynomial
    double a0;
    double b0;
    double c0; // is zero
    double d0;

    int n; // number of vertices in spline
    double step;
    std::vector<osg::Vec3> splineTraj;

    osg::Vec3 getSplinePos(int i);

    void poly3Spline(Position*);

};

#endif // SPLINE_H
