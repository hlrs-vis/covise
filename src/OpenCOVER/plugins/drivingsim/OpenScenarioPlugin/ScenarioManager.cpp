#include "ScenarioManager.h"
#include <vector>
#include "ReferencePosition.h"

ScenarioManager::ScenarioManager():
	simulationTime(0),
    scenarioCondition(true),
    anyActTrue(false)
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

void ScenarioManager::conditionManager(){
    if(conditionControl()){
        for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
        {
            if(conditionControl((*act_iter)))
            {
                anyActTrue = true;
                for(list<Maneuver*>::iterator maneuver_iter = (*act_iter)->maneuverList.begin(); maneuver_iter != (*act_iter)->maneuverList.end(); maneuver_iter++)
                {
                    conditionControl(*maneuver_iter);
                }
            }
        }
    }
}

bool ScenarioManager::conditionControl()
{
	if (endTime<simulationTime)
		{
			scenarioCondition = false;
            return false;
		}
    return true;
}

bool ScenarioManager::conditionControl(Act* act)
{
	if (act->startConditionType=="time")
	{
		if(act->startTime<simulationTime && act->actFinished==false)
		{
            act->actCondition = true;
            return act->actCondition;
		}
		else
		{
			act->actCondition = false;
            return act->actCondition;
		}

		if (endTime != 0)
		{
			if(act->startTime<simulationTime && endTime>simulationTime && act->actFinished==false)
			{
				act->actCondition = true;
                return act->actCondition;
			}
			else
			{
				act->actCondition = false;
                return act->actCondition;
			}
		}
	}
    return false;
}

bool ScenarioManager::conditionControl(Maneuver* maneuver)
{
    if ((maneuver->finishedEntityActions) == (maneuver->activeEntites*maneuver->actionVector.size()))
    {
        for(list<Entity*>::iterator entity_iter = entityList.begin(); entity_iter != entityList.end(); entity_iter++)
        {
            (*entity_iter)->resetActionAttributes();
        }
        maneuver->maneuverFinished = true;
        maneuver->maneuverCondition = false;
        return maneuver->maneuverCondition;
    }
	if (maneuver->startConditionType=="time")
	{
		if(maneuver->startTime<simulationTime && maneuver->maneuverFinished != true)
		{
			maneuver->maneuverCondition = true;
            return maneuver->maneuverCondition;
		}
		else
		{
			maneuver->maneuverCondition = false;
            return maneuver->maneuverCondition;
		}
	}
	if (maneuver->startConditionType=="distance")
	{
		auto activeCar = getEntityByName(maneuver->activeCarName);
		auto passiveCar = getEntityByName(maneuver->passiveCarName);
        if (activeCar->refPos->s-passiveCar->refPos->s >= maneuver->relativeDistance && maneuver->maneuverFinished == false)
		{
			maneuver->maneuverCondition = true;
            return maneuver->maneuverCondition;

		}

	}
	if (maneuver->startConditionType=="termination")
	{
		for(list<Act*>::iterator act_iter = actList.begin(); act_iter != actList.end(); act_iter++)
		{
			for(list<Maneuver*>::iterator terminatedManeuver = (*act_iter)->maneuverList.begin(); terminatedManeuver != (*act_iter)->maneuverList.end(); terminatedManeuver++)
			{
				if ((*terminatedManeuver)->maneuverFinished == true && maneuver->maneuverFinished == false && (*terminatedManeuver)->getName() == maneuver->startAfterManeuver)
				{
					maneuver->maneuverCondition = true;
                    return maneuver->maneuverCondition;

				}
			}
		}
	}
    return false;
}




