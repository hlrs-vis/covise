#include "ScenarioManager.h"
#include <vector>

ScenarioManager::ScenarioManager(){}
void ScenarioManager::setNumberOfActs(int numberOfActs_temp){
numberOfActs = numberOfActs_temp;}
void ScenarioManager::setNumberOfEntities(int numberOfEntities_temp){
numberOfEntities = numberOfEntities_temp;}
/*list<Entity*> ScenarioManager::getEntityList(){
return entityList;}
list<Act*> ScenarioManager::getActList(){
return actList;}*/
Entity* ScenarioManager::getEntityByName(string entityName){
	for(list<Entity*>::iterator entity_iter = entityList.begin(); entity_iter != entityList.end(); entity_iter++){
		if((*entity_iter)->getName()!=entityName){continue;}
	    else{
		return (*entity_iter);}}}



