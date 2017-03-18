#ifndef ACT_H
#define ACT_H

#include "Entity.h"
#include "Maneuver.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>
#include <DrivingSim/OpenScenario/schema/oscAct.h>

class Act : public OpenScenario::oscAct
{

 private:
	string name;
	int numberOfExecutions;
	int executionCounter;

 public:
	Act();
	~Act();
	virtual void finishedParsing();
	bool actCondition;
	list<Entity*> activeEntityList;
	list<Maneuver*> maneuverList;
	void initialize(int numberOfExecutions, list<Maneuver*> &maneuverList_temp, list<Entity*> &activeEntityList_temp);
	int getNumberOfExecutions();
	int getExecutionCounter();
	void setExecutionCounter();
	string getName();
	//list<Maneuver*> getManeuverList();
	bool getActCondition();
	void setActCondition();

};

#endif // ACT_H