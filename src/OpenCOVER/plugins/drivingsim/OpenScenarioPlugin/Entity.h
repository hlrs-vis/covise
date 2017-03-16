#ifndef ENTITY_H
#define ENTITY_H

using namespace std;
#include<iostream>
#include<string>
#include <vector>
#include "AgentVehicle.h"

class Entity {

 private:
    float speed;
	string name;
	//bool currentlyActiveInAct;//decides if new entity position comes from Maneuver or from initialization

 public:
	Vehicle* entityGeometry;
	osg::Vec3 entityPosition;
	osg::Vec3 directionVector;

    Entity(string entityName);
    void move();
    osg::Vec3 &getPosition();
    void setPosition(osg::Vec3 &newPosition);   
	string getName();
	void setSpeed(float speed_temp);
	float getSpeed();
	void setDirection(osg::Vec3 &newDirection);

};

#endif // ENTITY_H
