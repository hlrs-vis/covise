#ifndef SPLINE_H
#define SPLINE_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <math.h>
#include <iostream>
#include <Position.h>


class Spline
{

public:
    Spline(Position *initPos);
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

    double step;

    std::vector<double> xx;
    std::vector<double> yy;

};

#endif // SPLINE_H
