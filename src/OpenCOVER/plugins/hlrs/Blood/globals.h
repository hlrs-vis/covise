#ifndef GLOBALS_H
#define GLOBALS_H

#include <iostream>
#include <cmath>
using namespace std;

const double PI = 3.1415926;
const double RHO_BLOOD = 1060.0; //units: kg/m^3
const double GRAVITY = 9.81;

const double COEFF_STATIC_FRICTION = 0.65; //coefficient of static friction for steel on steel: 0.5-0.8 (average to 0.65)
const double COEFF_KINETIC_FRICTION = 0.16;

const double KIN_VISC_BLOOD = 0.04;   //kinematic viscosity of whole blood (blood + plasma)
const double KIN_VISC_PLASMA = 0.015; //kinematic viscosity of plasma
const double DYN_VISC_BLOOD = 0.0035; //units: N*S/m^2, Pa*s

//from gen.cpp
const int REYNOLDS_THRESHOLD = 2230;
const int REYNOLDS_LIMIT = 170000;
const double CD_TURB = 0.15;

//enum for the different drag models to be used
enum cdModel {
    CD_STOKES,
    CD_MOLERUS,
    CD_MUSCHELK,
    CD_NONE
};

inline double volumeOfSphere(double radius) {
	return 4/3 * PI * pow(radius, 2.0);
}

inline double crossSectionalArea(double radius) {
    return PI * radius * radius;
}

inline int signOf(int number) {
    if(number == 0) {
        return 1;
    } else {
        return number / abs(number);
    }
}

inline ostream& operator<<(ostream& os, const osg::Vec3 toPrint) {
    os << toPrint.x() << " " << toPrint.y() << " " << toPrint.z();
    return os;
}

inline double pythagoras(osg::Vec3 aVector) {
	double x2 = aVector.x() * aVector.x(); 
	double y2 = aVector.y() * aVector.y();
	double z2 = aVector.z() * aVector.z();
	
	return sqrt(x2 + y2 + z2);
}

inline double signedLength(osg::Vec3 anotherVector) {
    if(anotherVector.x() < 0 || anotherVector.y() < 0 || anotherVector.z() < 0) {
        return -1 * anotherVector.length();
    } else {
        return anotherVector.length();
    }
}

inline osg::Vec3 normalize(osg::Vec3 thirdVector) {
	return thirdVector/thirdVector.length();
}

inline osg::Vec3 absoluteValue(osg::Vec3 fourthVector) {
	return osg::Vec3(abs(fourthVector.x()), abs(fourthVector.y()), abs(fourthVector.z()));
}

#endif
