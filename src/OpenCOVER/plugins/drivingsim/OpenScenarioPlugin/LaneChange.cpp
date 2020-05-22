#include "LaneChange.h"
#include "Entity.h"
#include "OpenScenario/schema/oscSpeedDynamics.h"
#include "ReferencePosition.h"

using namespace OpenScenario;

LaneChange::LaneChange()
{
}


void LaneChange::getAbsolutePositionLc(ReferencePosition* relativePos, ReferencePosition* position)
{
	if (Dynamics->distance.exists())
	{
		double distance = Dynamics->distance;
		
		
	}
}