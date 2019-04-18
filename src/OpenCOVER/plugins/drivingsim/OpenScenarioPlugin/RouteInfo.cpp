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
{
	entity = e;
	currentWaypoint = NULL;
	currentRoad = NULL;
	currentLane = NULL;
	osg::Vec3 pos = e->getPosition();
	Vector3D p(pos.x(), pos.y(), pos.z());
	Vector2D st = RoadSystem::Instance()->searchPosition(p, currentRoad, longPos);
	s = st[0];
	t = st[1];
	if (currentRoad != NULL)
	{
	    int laneNumber = currentRoad->searchLane(s,t);
		if (laneNumber != 0)
		{
			//RoadTransition rt(;
			//entity->entityGeometry->addRouteTransition();
		}

	}
}
EntityInfo::~EntityInfo()
{

}
void EntityInfo::update()
{
}
