#include "Entity.h"
#include "ReferencePosition.h"
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
    refPos(NULL),
    newRefPos(NULL)
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

void Entity::setInitEntityPosition(ReferencePosition* init_refPos)
{
    entityGeometry = new AgentVehicle(name, new CarGeometry(name, filepath, true),0,init_refPos->road,init_refPos->s,init_refPos->laneId,speed,1);
    // Road r; s inits;
    auto vtrans = entityGeometry->getVehicleTransform();
    osg::Vec3 pos(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());
    entityPosition = pos;
    entityGeometry->setPosition(pos, directionVector);
}


void Entity::moveLongitudinal()
{
    if(refPos->road != NULL)
    {
        float step_distance = speed*opencover::cover->frameDuration();
        double ds = 1.0;
        double dt = 0.0;

        refPos->move(ds,dt,step_distance);

        Transform vehicleTransform = refPos->road->getRoadTransform(refPos->s, refPos->t);
        entityGeometry->setTransform(vehicleTransform,refPos->hdg);
        cout << name << " is driving on Road: " << refPos->roadId << endl;
    }
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

void Entity::setTrajectoryDirection()
{
    // entity is heading to targetPosition
    targetPosition = refPos->getPosition();
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

void Entity::followTrajectory(int verticesCounter,std::list<Entity*> *activeEntityList)
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
        cout << name << " arrived at vertice " << visitedVertices << endl;
        visitedVertices++;
        totalDistance = 0;
        if (visitedVertices == verticesCounter)
        {
            activeEntityList->remove(this);
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

void Entity::setTrajectoryDirectionOnRoad()
{
    targetPosition = refPos->getPosition();
    totaldirectionVector = targetPosition - newRefPos->getPosition();
    totaldirectionVectorLength = totaldirectionVector.length();

    directionVector = totaldirectionVector;
    directionVector.normalize();
}

void Entity::followTrajectoryOnRoad(int verticesCounter,std::list<Entity*> *activeEntityList)
{

    float step_distance = opencover::cover->frameDuration()*speed;

    if(totalDistance == 0)
    {
        totalDistance = totaldirectionVectorLength;
    }
    //calculate remaining distance
    totalDistance = totalDistance-step_distance;

    if(totalDistance <= 0)
    {
        cout << "Arrived at " << visitedVertices << endl;
        visitedVertices++;
        totalDistance = 0;
        if(visitedVertices == verticesCounter)
        {
            activeEntityList->remove(this);
        }
    }


    if(refPos->road != NULL)
    {
        double ds = refPos->s - newRefPos->s;
        double dt = refPos->t - newRefPos->t;

        newRefPos->move(ds,dt,step_distance);
        Transform vehicleTransform = newRefPos->road->getRoadTransform(newRefPos->s, newRefPos->t);
        entityGeometry->setTransform(vehicleTransform,newRefPos->hdg);
    }
    else
    {
        newRefPos->move(directionVector,step_distance);
        osg::Vec3 pos = newRefPos->getPosition();
        entityGeometry->setPosition(pos, directionVector);
    }


}
