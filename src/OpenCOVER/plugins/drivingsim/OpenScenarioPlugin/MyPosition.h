#ifndef MYPOSITION_H
#define MYPOSITION_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <OpenScenario/schema/oscPosition.h>


class MyPosition : public OpenScenario::oscPosition
{

public:
    MyPosition();
    ~MyPosition();

    // Road Coordinates
    std::string roadId;
    int laneId;
    double offset;
    double s;

    // Relative Coordinates (Road/Object)
    double dx;
    double dy;
    double dz;
    std::string relObject;

    //absolute Coordinates
    double x;
    double y;
    double z;

    std::string getPositionElement();



};

#endif // MYPOSITION_H
