#ifndef ACTION_H
#define ACTION_H

#include <string>
#include <list>
#include <OpenScenario/schema/oscAction.h>
#include "StoryElement.h"


class Entity;
class Trajectory;

class Action: public OpenScenario::oscAction, public StoryElement
{
public:
    Action();

	virtual void stop();
	virtual void start();

    std::string trajectoryCatalogReference;
    std::string routeCatalogReference;

    Trajectory* actionTrajectory;
    std::list<Entity*> actionEntityList;

    void setTrajectory(Trajectory* init_Traj);


};

#endif // ACTION_H
