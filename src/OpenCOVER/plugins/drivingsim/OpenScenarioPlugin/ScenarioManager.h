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
    bool anyActTrue;
	bool scenarioCondition;
	string endConditionType;
    float endTime;

	ScenarioManager();
	~ScenarioManager();
	Entity* getEntityByName(string entityName);
    bool conditionControl();
    bool conditionControl(Act* act);
    bool conditionControl(Event* event, Maneuver *maneuver);
    void conditionManager();

};

#endif // SCENARIOMANAGER_H
