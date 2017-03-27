#include "ScenarioManager.h"
#include <vector>

ScenarioManager::ScenarioManager():
	simulationTime(0),
	scenarioCondition(true)
{
}

Entity* ScenarioManager::getEntityByName(string entityName)
{
	for (list<Entity*>::iterator entity_iter = entityList.begin(); entity_iter != entityList.end(); entity_iter++)
	{
		if ((*entity_iter)->getName() != entityName)
		{
			continue;
			return 0;
		}
		else
		{
			return (*entity_iter);
		}
	}
return 0;
}

void ScenarioManager::conditionControl()
{
	if (endTime<simulationTime)
		{
			scenarioCondition = false;
		}
}

void ScenarioManager::conditionControl(Act* act)
{
	if (act->startConditionType=="time")
		{
			act->simulationTimeConditionControl(simulationTime);
		}
}

void ScenarioManager::conditionControl(Maneuver* maneuver)
{
	if (maneuver->startConditionType=="time")
	{
		maneuver->simulationTimeConditionControl(simulationTime);
	}
	if (maneuver->startConditionType=="distance")
	{
		maneuver->distanceToEntityConditionControl(getEntityByName(maneuver->activeCar), getEntityByName(maneuver->passiveCar));
	}
	if (maneuver->startConditionType=="termination")
	{
		for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
		{
			for(list<Maneuver*>::iterator maneuver_iter2 = (*act_iter)->maneuverList.begin(); maneuver_iter2 != (*act_iter)->maneuverList.end(); maneuver_iter2++)
			{
				maneuver->maneuverTerminitionControl((*maneuver_iter2));
			}
		}
	}
}




