#ifndef SCENARIOMANAGER_H
#define SCENARIOMANAGER_H

#include "Act.h"

using namespace std;
#include<iostream>
#include<string>
#include <list>


class ScenarioManager {

 private:
	int numberOfActs;
	int numberOfEntities;

 public:
	list<Act*> actList;
	list<Entity*> entityList;
    ScenarioManager();
	~ScenarioManager();
	void setNumberOfActs(int numberOfActs_temp);
	void setNumberOfEntities(int numberOfEntities_temp);
	//list<Act*> getActList();
	//list<Entity*> getEntityList();
	Entity* getEntityByName(string entityName);

};

#endif // SCENARIOMANAGER_H
