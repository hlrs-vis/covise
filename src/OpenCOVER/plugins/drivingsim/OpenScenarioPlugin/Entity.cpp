#include "Entity.h"
#include "ReferencePosition.h"
#include "Action.h"
#include "Event.h"
#include <OpenScenario/schema/oscVehicle.h>
#include <OpenScenario/schema/oscEntity.h>
#include <OpenScenario/OpenScenarioBase.h>
#include "OpenScenarioPlugin.h"
using namespace std;
using namespace OpenScenario;

Entity::Entity(oscObject *obj):
	object(obj),
	name(obj->name),
    totalDistance(0),
    visitedVertices(0),
    refPos(NULL),
    newRefPos(NULL),
    dt(0.0)
{
	directionVector.set(1, 0, 0);

	std::string catalogReferenceName= object->CatalogReference->entryName;
	vehicle = ((oscVehicle*)(OpenScenarioPlugin::instance()->osdb->getCatalogObjectByCatalogReference("VehicleCatalog", catalogReferenceName)));

	std::string geometryFileName;
	for (oscFileArrayMember::iterator it = vehicle->Properties->File.begin(); it != vehicle->Properties->File.end(); it++)
	{
		oscFile* file = ((oscFile*)(*it));
		geometryFileName = file->filepath.getValue();
		break;
	}

	agentVehicle = new AgentVehicle(name, new CarGeometry(name, geometryFileName, true));
}

Entity::~Entity()
{
}

void Entity::setInitEntityPosition(ReferencePosition* init_refPos)
{
    //entityGeometry = new AgentVehicle(name, new CarGeometry(name, filepath, true),0,init_refPos->road,init_refPos->s,init_refPos->laneId,speed,1);

    /*if(init_refPos->road != NULL)
    {
        auto vtrans = agentVehicle->getVehicleTransform();
        osg::Vec3 pos(vtrans.v().x(), vtrans.v().y(), vtrans.v().z());
        entityPosition = pos;
		agentVehicle->setTransform(vtrans,init_refPos->hdg);

    }
    else
    {*/
        entityPosition = init_refPos->xyz;

        directionVector[0] = cos(init_refPos->hdg);
        directionVector[1] = sin(init_refPos->hdg);


		agentVehicle->setPosition(entityPosition, directionVector);

   // }

}


void Entity::moveLongitudinal()
{
    if(refPos->road != NULL && speed > 0)
    {
        float step_distance = speed*opencover::cover->frameDuration();
        double ds = 1.0;
        double dt = 0.0;

        refPos->move(ds,dt,step_distance);

        Transform vehicleTransform = refPos->road->getRoadTransform(refPos->s, refPos->t);
		agentVehicle->setTransform(vehicleTransform,refPos->hdg);
        //cout << name << " is driving on Road: " << refPos->roadId << endl;
    }
    else
    {
		agentVehicle->setPosition(refPos->xyz, directionVector);
    }

}

osg::Vec3 Entity::getPosition()
{
    return refPos->getPosition();
}

void Entity::setPosition(osg::Vec3 &newPosition)
{
	entityPosition = newPosition;
	agentVehicle->setPosition(newPosition, directionVector);
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

void Entity::setTrajSpeed(float deltat)
{

    // calculate length of targetvector

    speed = totaldirectionVectorLength/deltat;


}

void Entity::setTrajectoryDirection()
{
    targetPosition = newRefPos->getPosition();
    totaldirectionVector = targetPosition - refPos->getPosition();
    totaldirectionVectorLength = totaldirectionVector.length();

    directionVector = totaldirectionVector;
    directionVector.normalize();

}

void Entity::followTrajectory(Event* event, int verticesCounter)
{

    float step_distance = opencover::cover->frameDuration()*speed;

    if(totalDistance == 0)
    {
        totalDistance = totaldirectionVectorLength;
    }
    //calculate remaining distance
    totalDistance = totalDistance-step_distance;

    directionVector = newRefPos->getPosition() - refPos->getPosition();
    directionVector.normalize();
    refPos->move(directionVector,step_distance);
    osg::Vec3 pos = refPos->getPosition();

	agentVehicle->setPosition(pos, directionVector);

    if(totalDistance <= 0)
    {
        cout << "Arrived at " << visitedVertices << endl;
        visitedVertices++;
        totalDistance = 0;
        if(visitedVertices == verticesCounter)
        {
            if(totaldirectionVectorLength<0.01)
            {
                speed = 0;
            }
            visitedVertices = 0;
            event->finishedEntityActions = event->finishedEntityActions+1;

            refPos->update();
        }
    }
}

void Entity::longitudinalSpeedAction(Event* event, double init_targetSpeed, int shape)
{
    float targetSpeed = (float) init_targetSpeed;

    //linear
    if(shape == 0)
    {
        if (dt == 0)
        {
            old_speed = speed;

            if (targetSpeed>old_speed)
            {
                acceleration = 50;
            }
            else
            {
                acceleration = -50;
            }
        }
    }
    // step
    else
    {
        old_speed = targetSpeed;
        acceleration = 1000;
    }

    float frametime = opencover::cover->frameDuration();
    dt += frametime;

    cout << getName() << " is breaking! New speed: " << speed << endl;
    float t_end = (targetSpeed-old_speed)/acceleration;
    if(dt>=t_end)
    {
        speed = targetSpeed;
        dt = 0.0;
        event->finishedEntityActions = event->finishedEntityActions+1;

    }
    else
    {
        speed = acceleration*dt+old_speed;
    }


}

void Entity::resetActionAttributes()
{
    totalDistance = 0;
    visitedVertices = 0;

    dt = 0.0;
}
