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

 public:
	string name;
	list<Entity*> activeEntityList;
	list<Maneuver*> maneuverList;
	int numberOfExecutions;

	//conditions
	bool actCondition;
	bool actFinished;
	string startConditionType;
	float startTime;
	string endConditionType;
	float endTime;

	Act();
	~Act();
	virtual void finishedParsing();
	void initialize(int noe, list<Maneuver*> &maneuverList_temp, list<Entity*> &activeEntityList_temp);
	string getName();
	int getNumberOfExecutions();

};

#endif // ACT_H