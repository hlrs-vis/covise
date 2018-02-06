#include "Entity.h"

using namespace std;

Entity::Entity(string entityName, string catalogReferenceName):
	name(entityName),
    catalogReferenceName(catalogReferenceName),
    totalDistance(0),
    visitedVertices(0),
    absVertPosIsSet(false),
    finishedCurrentTraj(false)
{
	directionVector.set(1, 0, 0);
}

void Entity::setInitEntityPosition(osg::Vec3 initPos)
{
	entityGeometry = new AgentVehicle(name, new CarGeometry(name, filepath, true));
	entityGeometry->setPosition(initPos, directionVector);
}

void Entity::setInitEntityPosition(Road *r)
{
    entityGeometry = new AgentVehicle(name, new CarGeometry(name, filepath, true),0,r,inits,laneId,speed,1);
    // Road r; s inits;
	auto vtrans = entityGeometry->getVehicleTransform();
	osg::Vec3 pos(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());
	entityPosition = pos;
	entityGeometry->setPosition(pos, directionVector);
}

void Entity::moveLongitudinal()
{
	float step_distance = speed*opencover::cover->frameDuration();
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
    directionVector.normalize();

}

void Entity::setTrajectoryDirection(osg::Vec3 init_targetPosition)
{
    // entity is heading to targetPosition
    targetPosition = init_targetPosition;
    totaldirectionVector = targetPosition - entityPosition;
    totaldirectionVectorLength = totaldirectionVector.length();

    directionVector = totaldirectionVector;
    directionVector.normalize();
}


void Entity::getTrajSpeed(float deltat)
{

    // calculate length of targetvector
    speed = totaldirectionVectorLength/deltat;

}

void Entity::followTrajectory(int verticesCounter,std::list<Entity*> &finishedEntityList)
{

    //calculate step distance
    //float step_distance = speed*opencover::cover->frameDuration();
    float step_distance = speed*1/60;

    if(totalDistance == 0)
    {
        totalDistance = totaldirectionVectorLength;
    }
    //calculate remaining distance
    totalDistance = totalDistance-step_distance;
    //calculate new position
    newPosition = entityPosition+(directionVector*step_distance);
    if (totalDistance <= 0)
    {
        visitedVertices++;
        totalDistance = 0;
        if (visitedVertices == verticesCounter)
        {
            /* entity maneuver finished: bool
             * is set to true in here
             * has to be checked in conditionManager before entity is added to active ManeuverEntities
             */
            //maneuverCondition = false;
            //maneuverFinished = true;
            //activeEntityList.remove(this);
            finishedCurrentTraj = true;
            finishedEntityList.push_back(this);
        }
    }

    entityPosition = newPosition;
    entityGeometry->setPosition(newPosition, directionVector);
}

void Entity::setAbsVertPos(){
    if(!absVertPosIsSet){
        absVertPos = entityPosition;
        absVertPosIsSet = true;
    }
}
