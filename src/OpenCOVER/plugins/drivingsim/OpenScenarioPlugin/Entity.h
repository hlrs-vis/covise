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
	bool currentlyActiveInAct;//decides if new entity position comes from Maneuver or from initialization

 public:
	Vehicle* car;
	vector<float> entityPosition;
    Entity(string entityName, Vehicle* car);
    void move();
    vector<float> getPosition();
    void setPosition(vector<float> newPosition);   
	string getName();
	void setSpeed(float speed_temp);
	float getSpeed();

};

#endif // ENTITY_H
