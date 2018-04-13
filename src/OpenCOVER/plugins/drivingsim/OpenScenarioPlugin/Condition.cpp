#include "Condition.h"
#include <OpenScenario/schema/oscByEntity.h>

using namespace std;

Condition::Condition():
    oscCondition(),
    isTrue(false),
    activeCar(NULL),
    passiveCar(NULL),
    checkedAct(NULL),
    checkedEvent(NULL),
    checkedManeuver(NULL),
    checkedSequence(NULL)
{}
Condition::~Condition(){}

void Condition::initalize(oscByEntity *condition)
{


}
