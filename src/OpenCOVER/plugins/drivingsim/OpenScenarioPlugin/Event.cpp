#include "Event.h"

Event::Event():
finishedEntityActions(0)
{

}

void Event::initialize(int numEntites)
{
    activeEntites = numEntites;
}
