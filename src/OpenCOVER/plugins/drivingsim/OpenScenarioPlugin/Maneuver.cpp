#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>
#include "OpenScenarioPlugin.h"
#include "Act.h"

using namespace std;

Maneuver::Maneuver():
	maneuverCondition(false),
	maneuverFinished(false),
    trajectoryCatalogReference(""),
	startAfterManeuver(""),
	startConditionType("termination"),
	targetSpeed(0)
{
}
Maneuver::~Maneuver()
{
}
void Maneuver::initialize(std::list<Entity*> &activeEntityList_temp)
{
    activeEntityList = activeEntityList_temp;
}


void Maneuver::finishedParsing()
{
	name = oscManeuver::name.getValue();
}

void Maneuver::checkConditions()
{
    if (startConditionType == "time")
    {
        if (startTime<OpenScenarioPlugin::instance()->scenarioManager->simulationTime && maneuverFinished != true)
        {
            maneuverCondition = true;
        }
        else
        {
            maneuverCondition = false;
        }
    }
    if (startConditionType == "distance")
    {
        auto activeCar = OpenScenarioPlugin::instance()->scenarioManager->getEntityByName(activeCarName);
        auto passiveCar = OpenScenarioPlugin::instance()->scenarioManager->getEntityByName(passiveCarName);
        if (activeCar->entityPosition[0] - passiveCar->entityPosition[0] >= relativeDistance && maneuverFinished == false)
        {
            maneuverCondition = true;
        }

    }
    if (startConditionType == "termination")
    {
        for (list<Act*>::iterator act_iter = OpenScenarioPlugin::instance()->scenarioManager->actList.begin(); act_iter != OpenScenarioPlugin::instance()->scenarioManager->actList.end(); act_iter++)
        {
            for (list<Maneuver*>::iterator terminatedManeuver = (*act_iter)->maneuverList.begin(); terminatedManeuver != (*act_iter)->maneuverList.end(); terminatedManeuver++)
            {
                if ((*terminatedManeuver)->maneuverFinished == true && maneuverFinished == false && (*terminatedManeuver)->getName() == startAfterManeuver)
                {
                    maneuverCondition = true;
                }
            }
        }
    }
}

string &Maneuver::getName()
{
	return name;
}

void Maneuver::changeSpeedOfEntity(Entity *aktivCar, float dt)
{
	float negativeAcceleration = 50;
	float dv = negativeAcceleration*dt;
	if(aktivCar->getSpeed()>targetSpeed)
	{
		aktivCar->setSpeed(aktivCar->getSpeed()-dv);
	}
	else
	{
	aktivCar->setSpeed(targetSpeed);
	}
}
