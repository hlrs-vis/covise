#include "Entity.h"


Entity::Entity(string entityName):
	name(entityName),
	initActionType(NULL)
{
	directionVector.set(1, 0, 0);
	entityGeometry = new AgentVehicle(name, new CarGeometry(name, "C:\\src\\covise\\test\\volvo\\volvo_blue_nrm.3ds", true));
}

void Entity::moveLongitudinal()
{
	directionVector.set(1, 0, 0);
	float step_distance = 0.1*speed*opencover::cover->frameDuration();
	entityPosition[0] = entityPosition[0] + step_distance;
}

osg::Vec3 &Entity::getPosition()
{
	return entityPosition;
}

void Entity::setPosition(osg::Vec3 &newPosition)
{
	entityPosition = newPosition;
	entityGeometry->setPosition(newPosition, directionVector);
}

string &Entity::getName()
{
	return name;
}

void Entity::setSpeed(float speed_temp)
{
	speed = speed_temp;
}

float &Entity::getSpeed()
{
	return speed;
}

void Entity::setDirection(osg::Vec3 &dir)
{
	directionVector = dir;
}