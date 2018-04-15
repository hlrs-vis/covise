#ifndef SCENARIOMANAGER_H
#define SCENARIOMANAGER_H

#include "Act.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>

class Condition;
class Maneuver;
class ScenarioManager {

public:
    std::list<Act*> actList;
	list<Entity*> entityList;
	float simulationTime;

	//conditions
    bool anyActTrue;
	bool scenarioCondition;
	string endConditionType;
    float endTime;

	ScenarioManager();
	void restart();
	~ScenarioManager();
	Entity* getEntityByName(string entityName);
    Maneuver* getManeuverByName(string maneuverName);
	void initializeEntities();
    bool conditionControl(Condition* condition);
    bool conditionControl(Act* act);
    bool conditionControl(Event* event, Maneuver *maneuver);
    void conditionManager();

    void addCondition(Condition* condition);
    std::list<Condition*> endConditionList;

};

#endif // SCENARIOMANAGER_H
