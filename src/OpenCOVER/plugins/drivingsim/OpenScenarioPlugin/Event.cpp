#include "Event.h"

Event::Event():
    finishedEntityActions(0),
    eventFinished(false)
{

}

void Event::initialize(int numEntites)
{
    activeEntites = numEntites;
}
