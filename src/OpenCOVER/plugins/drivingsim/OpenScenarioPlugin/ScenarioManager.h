#ifndef SCENARIOMANAGER_H
#define SCENARIOMANAGER_H

#include "Act.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>


class ScenarioManager {

public:
	list<Act*> actList;
	list<Entity*> entityList;
	float simulationTime;

	//conditions
	bool scenarioCondition;
	string endConditionType;
    float endTime;

	ScenarioManager();
	~ScenarioManager();
	Entity* getEntityByName(string entityName);
    bool conditionControl();
    bool conditionControl(Act* act);
    bool conditionControl(Maneuver* maneuver);

};

#endif // SCENARIOMANAGER_H
