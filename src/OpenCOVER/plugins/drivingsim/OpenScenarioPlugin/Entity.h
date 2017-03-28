#ifndef ENTITY_H
#define ENTITY_H

using namespace std;
#include<iostream>
#include<string>
#include <vector>
#include "AgentVehicle.h"

class Entity {

 public:
	string name;
	string catalogReferenceName;
	string filepath;
	float speed;
	string roadId;
	int laneId;
	float inits;
	AgentVehicle *entityGeometry;
	osg::Vec3 entityPosition;
	osg::Vec3 directionVector;

    Entity(string entityName, string catalogReferenceName);
	~Entity();
	void setInitEntityPosition(osg::Vec3 init);
	void setInitEntityPosition(Road *r);
    void moveLongitudinal();
	string &getName();
	void setSpeed(float speed_temp);
	float &getSpeed();
    osg::Vec3 &getPosition();
    void setPosition(osg::Vec3 &newPosition);   
	void setDirection(osg::Vec3 &newDirection);

};

#endif // ENTITY_H
