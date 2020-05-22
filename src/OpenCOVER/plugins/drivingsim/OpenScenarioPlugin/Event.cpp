#include "Event.h"
#include "Condition.h"
#include "StoryElement.h"
#include "Action.h"
#include "Sequence.h"
#include "Entity.h"
#include "LaneChange.h"

Event::Event():
    StoryElement(),
    finishedEntityActions(0)
{

}

void Event::start(Sequence *currentSequence)
{
    activeEntites = currentSequence->actorList.size();
    for (auto entity_iter = currentSequence->actorList.begin(); entity_iter != currentSequence->actorList.end(); entity_iter++)
    {
        Entity* currentEntity = (*entity_iter);
        for (auto action_iter = Action.begin(); action_iter != Action.end(); action_iter++)
        {
            //::Action* currentAction = *action_iter);
			::Action* currentAction = dynamic_cast<::Action*>(*action_iter);
            if (currentAction->Private.exists())
            {
                if (currentAction->Private->Routing.exists())
                {
                    if (currentAction->Private->Routing.exists())
                    {
                        if (currentAction->Private->Routing->FollowTrajectory.exists())
                        {
                            Trajectory* currentTrajectory = currentAction->actionTrajectory;

                            currentEntity->startFollowTrajectory(currentTrajectory);

                        }
						
                    }
                }
				if (currentAction->Private->Lateral.exists())
				{
					if (currentAction->Private->Lateral->LaneChange.exists())
					{
						if (currentAction->Private->Lateral->LaneChange->Dynamics.exists())
						{
							LaneChange* lc = dynamic_cast<LaneChange*>(currentAction->Private->Lateral->LaneChange.getObject());
							currentEntity->startDoLaneChange(lc);
						}
					}
				}
            }
        }
    }
    StoryElement::start();
}
void Event::stop()
{
	finishedEntityActions=0;
	activeEntites=0;
	StoryElement::stop();
}


void Event::addCondition(Condition *condition)
{
    startConditionList.push_back(condition);

}
