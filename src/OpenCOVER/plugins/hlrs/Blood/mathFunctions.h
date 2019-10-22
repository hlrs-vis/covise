#ifndef MATHFUNCTIONS_H
#define MATHFUNCTIONS_H

#include <iostream>
#include <cmath>
#include "globalConstants.h"

//double abs(double x) {
//    if(x >= 0) {
//        return x;
//    } else {
//        return x * (-1);
//    }
//}

double volumeOfSphere(double radius) {
    return 4/3 * PI * pow(radius, 2.0);
}

double crossSectionalArea(double radius) {
        return PI * radius * radius;
}

double instVelocity(double initPosition, double finalPosition, double deltaTime) {
    //negative velocity means particle is moving downwards
    double deltaPos = finalPosition - initPosition;
    return deltaPos/deltaTime;
}

double netVelocity(osg::Vec3 velocity) {
    double x2 = pow(velocity.x(), 2.0); 
    double y2 = pow(velocity.y(), 2.0);
    double z2 = pow(velocity.z(), 2.0);
    return sqrt(x2 + y2 + z2);
}

#endif /* MATHFUNCTIONS_H */

