#include "Source.h"
#include "OpenScenarioPlugin.h"
#include <DrivingSim/OpenScenario/schema/oscTrafficDefinition.h>
#include <DrivingSim/OpenScenario/schema/oscVehicleDistribution.h>
#include <DrivingSim/OpenScenario/schema/oscDriverDistribution.h>

using namespace OpenScenario;
using namespace opencover;

Source::Source() :
	oscSource()
{
}

Source::~Source()
{
}

void Source::finishedParsing()
{
	Road* myRoad=NULL;
	RoadSystem *system = RoadSystem::Instance();

	double s, t;
	// find road
	if (oscRoad* position = Position->Road.getObject())
	{
		myRoad = system->getRoad(position->roadId);
		s = position->s;
		t = position->t;
	}
	else if (oscWorld* position = Position->World.getObject())
	{
		Vector3D v(position->x, position->y, position->z);
		double uInit = 0.0;
		Vector2D st = system->searchPosition(v,myRoad, uInit);
		s = st.u;
		t = st.v;
	}
	else if (oscLane* position = Position->Lane.getObject())
	{
		myRoad = system->getRoad(position->roadId);
		s = position->s;
		int lane = position->laneId;
		double width=0, widthToLane=0;
		myRoad->getLaneWidthAndOffset(s, lane, width, widthToLane);
		t = widthToLane + width / 2.0; // center of lane
	}
	else
	{
		fprintf(stderr, "Relative positions are not supported for Sources\n");
	}
	oscTrafficDefinition* traffic = TrafficDefinition.getObject();
	if (traffic !=NULL && myRoad != NULL)
	{
		oscVehicleDistribution* vd = traffic->VehicleDistribution.getObject();
		oscDriverDistribution* dd = traffic->DriverDistribution.getObject();
		if (vd != NULL && dd != NULL)
		{
				
		}
	}
	else
	{
		fprintf(stderr, "no Traffic definition \n");
	}
	OpenScenarioPlugin::instance()->addSource(this);
}
