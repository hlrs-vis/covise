#ifndef ROUTE_INFO_H
#define ROUTE_INFO_H

#include<string>
#include<vector>
#include <OpenScenario/schema/oscPosition.h>
#include <OpenScenario/schema/oscRoute.h>
#include <osg/Vec3>
#include "Entity.h"
#include <VehicleUtil/RoadSystem/RoadSystem.h>


class EntityInfo
{
public:
	EntityInfo(Entity *e);
	~EntityInfo();
	void update();
	Entity *getEntity() { return entity; };
private:
	Entity *entity;
	OpenScenario::oscPosition *currentWaypoint;
	Road *currentRoad;
	Lane *currentLane;
	Vector2D st;
	double longPos;
};

class RouteInfo : public OpenScenario::oscRoute
{

private:
	

public:
    std::list<EntityInfo *> entityInfos;
	RouteInfo();
	~RouteInfo();
	virtual void finishedParsing();
	void addEntity(Entity *);
	void removeEntity(Entity *);
	void update();
};

#endif // ROUTE_INFO_H
