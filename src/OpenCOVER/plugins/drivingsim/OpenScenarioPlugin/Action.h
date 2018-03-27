#ifndef ACTION_H
#define ACTION_H

#include <string>
#include <list>
#include <OpenScenario/schema/oscAction.h>


class Entity;
class Trajectory;

class Action: public OpenScenario::oscAction
{
public:
    Action();

    std::string trajectoryCatalogReference;
    std::string routeCatalogReference;

    bool actionCompleted;
    Trajectory* actionTrajectory;
    std::list<Entity*> actionEntityList;

    void setTrajectory(Trajectory* init_Traj);


};

#endif // ACTION_H
