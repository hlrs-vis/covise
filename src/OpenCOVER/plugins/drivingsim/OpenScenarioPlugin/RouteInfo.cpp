#include "RouteInfo.h"

using namespace std;

RouteInfo::RouteInfo():
oscRoute()
{
}
RouteInfo::~RouteInfo(){}

void RouteInfo::finishedParsing()
{
}

void RouteInfo::addEntity(Entity *e)
{
	EntityInfo *ei = new EntityInfo(e);
	entityInfos.push_back(ei);
}
void RouteInfo::removeEntity(Entity *e)
{
	for (auto it = entityInfos.begin(); it != entityInfos.end(); it++)
	{
		EntityInfo *ei = (*it);
		if (ei->getEntity() == e)
		{
			entityInfos.erase(it);
			delete ei;
			break;
		}
	}
}
void RouteInfo::update()
{
	for (auto it = entityInfos.begin(); it != entityInfos.end(); it++)
	{
		(*it)->update();
	}
}

EntityInfo::EntityInfo(Entity *e)
: entity(e)
, st(0., 0.)
{
	currentWaypoint = NULL;
	currentRoad = NULL;
	currentLane = NULL;
	osg::Vec3 pos = e->getPosition();
	Vector3D p(pos.x(), pos.y(), pos.z());
	double longPosition;
	st = RoadSystem::Instance()->searchPosition(p, currentRoad, longPos);
	if (currentRoad != NULL)
	{
	    int laneNumber = currentRoad->searchLane(st[0], st[1]);
		if (laneNumber != NULL)
		{
            RoadTransition rt;
			entity->entityGeometry->addRouteTransition(rt);
		}

	}
}
EntityInfo::~EntityInfo()
{

}
void EntityInfo::update()
{
}
