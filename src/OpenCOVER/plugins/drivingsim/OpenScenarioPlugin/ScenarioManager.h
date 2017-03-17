#ifndef SCENARIOMANAGER_H
#define SCENARIOMANAGER_H

#include "Act.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>


class ScenarioManager {

public:
	ScenarioManager();
	~ScenarioManager();
	int numberOfActs;
	list<Act*> actList;
	list<Entity*> entityList;
	Entity* getEntityByName(string entityName);

};

#endif // SCENARIOMANAGER_H
