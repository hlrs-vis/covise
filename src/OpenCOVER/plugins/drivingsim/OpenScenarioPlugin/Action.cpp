#include "Action.h"
#include "Trajectory.h"

Action::Action():
    trajectoryCatalogReference(""),
    routeCatalogReference(""),
    actionTrajectory(NULL),
    actionEntityList(NULL)
{

}

void Action::stop()
{
	StoryElement::stop();
}

void Action::start()
{
	StoryElement::start();
}

void Action::setTrajectory(Trajectory* init_Traj)
{
    actionTrajectory = init_Traj;
}
