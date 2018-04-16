#ifndef SCENARIOMANAGER_H
#define SCENARIOMANAGER_H

#include "Act.h"
#include "../../../DrivingSim/OpenScenario/schema/oscEntity.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>

class Condition;
class Maneuver;
class Event;
using namespace OpenScenario;

class ScenarioManager {

public:
    std::list<Act*> actList;
	list<Entity*> entityList;
	float simulationTime;

	//conditions
	bool scenarioCondition;

	ScenarioManager();
	void restart();
	~ScenarioManager();
    Entity* getEntityByName(std::string entityName);
    Maneuver* getManeuverByName(std::string maneuverName);
    Event* getEventByName(std::string eventName);
	void initializeEntities();
    bool conditionControl(Condition* condition);

    void conditionManager();
    void actConditionManager();
    void eventConditionManager(Act *currentAct);

    void addCondition(Condition* condition);
    void initializeCondition(Condition* condition);
    std::list<Condition*> endConditionList;

    void resetReferencePositionStatus();

};

#endif // SCENARIOMANAGER_H
