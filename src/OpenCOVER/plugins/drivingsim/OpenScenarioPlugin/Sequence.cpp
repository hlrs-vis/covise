#include "Sequence.h"
#include "StoryElement.h"

Sequence::Sequence():
    StoryElement(),
    activeManeuver(NULL),
    executions(0)
{

}

void Sequence::initialize(std::list<Entity*> actorList_temp, std::list<::Maneuver*> maneuverList_temp)
{
    actorList = actorList_temp;
    maneuverList = maneuverList_temp;
}
