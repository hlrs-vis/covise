#ifndef ACT_H
#define ACT_H

#include "Entity.h"
#include "Maneuver.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>

class Act {

 private:
	string name;
	int numberOfExecutions;
	int executionCounter;

 public:
	Act(string actName, int numberOfExecutions, list<Maneuver*> maneuverList_temp, list<Entity*> activeEntityList_temp);
	~Act();
	bool actCondition;
	list<Entity*> activeEntityList;
	list<Maneuver*> maneuverList;
	int getNumberOfExecutions();
	int getExecutionCounter();
	void setExecutionCounter();
	string getName();
	//list<Maneuver*> getManeuverList();
	bool getActCondition();
	void setActCondition();

};

#endif // ACT_H