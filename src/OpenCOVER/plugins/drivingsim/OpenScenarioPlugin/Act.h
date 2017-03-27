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
	int numberOfExecutions;//from sequence
	//int executionCounter;

 public:
	Act();
	~Act();
	virtual void finishedParsing();
	float startTime;
	float endTime;
	string startConditionType;
	string endConditionType;
	bool actFinished;
	bool actCondition;
	list<Entity*> activeEntityList;
	list<Maneuver*> maneuverList;
	void initialize(int noe, list<Maneuver*> &maneuverList_temp, list<Entity*> &activeEntityList_temp);
	int getNumberOfExecutions();
	int getExecutionCounter();
	void setExecutionCounter();
	string getName();
	//list<Maneuver*> getManeuverList();
	bool getActCondition();
	void setActCondition();
	void simulationTimeConditionControl(float simulationTime);
	Maneuver* getManeuverByName(string maneuverName);

};

#endif // ACT_H