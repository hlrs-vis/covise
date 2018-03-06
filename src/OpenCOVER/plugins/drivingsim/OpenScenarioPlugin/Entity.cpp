#include "Entity.h"

using namespace std;

Entity::Entity(string entityName, string catalogReferenceName):
	name(entityName),
    catalogReferenceName(catalogReferenceName),
    totalDistance(0),
    visitedVertices(0),
    refPosIsSet(false),
    finishedCurrentTraj(false),
    activeShape("Polyline"),
    visitedSplineVertices(0),
    refPos(NULL)
{
	directionVector.set(1, 0, 0);
}


void Entity::setInitEntityPosition(osg::Matrix m)
{
    entityGeometry = new AgentVehicle(name, new CarGeometry(name, filepath, true));
    entityGeometry->setTransform(m);
    osg::Vec3 pos(m(3,0),m(3,1),m(3,2));
    entityPosition = pos;
}

void Entity::setInitEntityPosition(osg::Vec3 initPos)
{
	entityGeometry = new AgentVehicle(name, new CarGeometry(name, filepath, true));
	entityGeometry->setPosition(initPos, directionVector);
}

void Entity::setInitEntityPosition(osg::Vec3 initPos, osg::Vec3 initDirVec)
{
    entityGeometry = new AgentVehicle(name, new CarGeometry(name, filepath, true));
    directionVector.set(initDirVec[0], initDirVec[1], initDirVec[2]);
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
    // entityPosition[0] = entityPosition[0] + step_distance;
    entityPosition = entityPosition+(directionVector*step_distance);

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
    float step_distance = speed*opencover::cover->frameDuration();
    //float step_distance = speed*1/60;

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
            finishedCurrentTraj = true;
            finishedEntityList.push_back(this);
        }
    }

    entityPosition = newPosition;
    entityGeometry->setPosition(newPosition, directionVector);
}

void Entity::setRefPos()
{
    if(!refPosIsSet)
    {
        referencePosition = entityPosition;
        refPosIsSet = true;
    }
}

void Entity::setRefPos(osg::Vec3 newReferencePosition)
{
    referencePosition = newReferencePosition;
}

void Entity::setActiveShape(std::string shapestring)
{
    activeShape = shapestring;
}

void Entity::setSplinePos(osg::Vec3 initPos)
{
    splinePos = initPos;
}

void Entity::followSpline()
{
    totalDistance = 1.0;
    targetPosition = splinePos;
    directionVector = targetPosition-entityPosition;
    totaldirectionVectorLength = directionVector.length();
    directionVector.normalize();

    float step_distance = speed*opencover::cover->frameDuration();

    if(splineDistance == 0)
    {
        splineDistance = totaldirectionVectorLength;
    }
    //calculate remaining distance
    splineDistance = splineDistance-step_distance;
    //calculate new position
    newPosition = entityPosition+(directionVector*step_distance);
    if (splineDistance <= 0)
    {
        visitedSplineVertices++;
        splineDistance = 0;
        if (visitedSplineVertices == 10)
        {
            totalDistance = 0;
            activeShape = "Polyline";
            visitedVertices++;

        }
    }

    entityPosition = newPosition;
    entityGeometry->setPosition(newPosition, directionVector);

}
