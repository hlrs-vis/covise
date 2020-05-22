#ifndef LANE_CHANGE_H
#define LANE_CHANGE_H

#include<string>
#include<OpenScenario/schema/oscLaneChange.h>
#include "OpenScenario/schema/oscSpeedDynamics.h"

//#include<VehicleUtil/RoadSystem/Lane.h>
class ReferencePosition;

class LaneChange : public OpenScenario::oscLaneChange
{
public:
	LaneChange();
	//Lane *getWidth;

	void getAbsolutePositionLc(ReferencePosition* relativePos, ReferencePosition* position);
};



#endif