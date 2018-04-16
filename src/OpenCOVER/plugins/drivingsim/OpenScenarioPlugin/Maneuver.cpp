#include "Maneuver.h"
#include <cover/coVRPluginSupport.h>
#include <iterator>
#include <math.h>
#include "OpenScenarioPlugin.h"
#include "Act.h"
#include "StoryElement.h"

using namespace std;

Maneuver::Maneuver():
    StoryElement(),
    activeEvent(NULL)
{
}
Maneuver::~Maneuver()
{
}
void Maneuver::initialize(::Event* event_temp)
{
    eventList.push_back(event_temp);
}


void Maneuver::finishedParsing()
{
    maneuverName = oscManeuver::name.getValue();
}

string &Maneuver::getName()
{
    return maneuverName;
}
