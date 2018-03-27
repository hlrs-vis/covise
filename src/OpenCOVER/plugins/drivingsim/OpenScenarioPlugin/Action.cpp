#include "Action.h"

Action::Action():
    trajectoryCatalogReference(""),
    routeCatalogReference(""),
    actionCompleted(false),
    actionTrajectory(NULL),
    actionEntityList(NULL)
{

}

void Action::setTrajectory(Trajectory* init_Traj)
{
    actionTrajectory = init_Traj;
}
