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
    delete refPos;
    delete newRefPos;
}

void Entity::setInitEntityPosition(ReferencePosition* init_refPos)
{
    dt = 0.0;

    refPos = init_refPos;
    newRefPos = new ReferencePosition(refPos);
    lastRefPos = new ReferencePosition(refPos);
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
        float step_distance = speed*OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
        double ds;
        double hdg;
        if(refPos->laneId>0)
        {
            ds = -1.0;
            hdg = refPos->hdg + 3.14159;
        }
        else
        {
            ds = 1.0;
            hdg = refPos->hdg;
        }

        refPos->move(ds,0.0,step_distance);

        Transform vehicleTransform = refPos->road->getRoadTransform(refPos->s, refPos->t);
        agentVehicle->setTransform(vehicleTransform,hdg);
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
    osg::Vec3 segmentVector= newRefPos->getPosition() - refPos->getPosition();
    segmentLength = segmentVector.length();

    directionVector = segmentVector/ segmentLength;

}
void Entity::startFollowTrajectory(Trajectory *t)
{
    trajectory = t;
    currentVertex = 0;
    distanceTraveledFromLastVertex = 0;
    if(t->Vertex.size()>1)
    {
        Position* currentPos;
        currentPos = ((Position*)(trajectory->Vertex[currentVertex]->Position.getObject()));
        currentPos->getAbsolutePosition(refPos, newRefPos); // update newRefPos (relative to Entity position)
        *lastRefPos = *newRefPos;
        *refPos = lastRefPos;
        currentVertex++;
        currentPos = ((Position*)(trajectory->Vertex[currentVertex]->Position.getObject()));
        currentPos->getAbsolutePosition(lastRefPos, newRefPos); // update newRefPos (relative to Entity position)
        if (t->domain.getValue() == 0) // domain == time
        {
            // calculate speed from trajectory vertices
            speed = segmentLength / trajectory->getReference(currentVertex);
        }
    }
}
 
void Entity::followTrajectory(Event* event)
{

    osg::Vec3 segmentVector = newRefPos->getPosition() - lastRefPos->getPosition();
    segmentLength = segmentVector.length();
    if (trajectory->domain.getValue() == 0) // domain == time
    {
        // calculate speed from trajectory vertices
        speed = segmentLength / trajectory->getReference(currentVertex);
    }
    float stepDistance = speed * OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
    while ((stepDistance + distanceTraveledFromLastVertex) > segmentLength)
    {
        currentVertex++;

        if (currentVertex == trajectory->Vertex.size())
        {
            break;
        }
        if (trajectory->domain.getValue() == 0) // domain == time
        {
            // calculate speed from trajectory vertices
            speed = segmentLength / trajectory->getReference(currentVertex);
        }
        directionVector = newRefPos->getPosition() - lastRefPos->getPosition();
        directionVector.normalize();
        float moveWithinSegment = segmentLength- distanceTraveledFromLastVertex;

        distanceTraveledFromLastVertex += moveWithinSegment;
        refPos->move(directionVector, moveWithinSegment);
        distanceTraveledFromLastVertex = 0;

        Position* currentPos;
        currentPos = ((Position*)(trajectory->Vertex[currentVertex]->Position.getObject()));
        *lastRefPos = *newRefPos;

        currentPos->getAbsolutePosition(lastRefPos,newRefPos); // update newRefPos (relative to last vertex) 

        stepDistance -= moveWithinSegment;
    }

    if (currentVertex == trajectory->Vertex.size())
    {
        stepDistance = 0;
        event->finishedEntityActions++;
        refPos->update();
    }
    directionVector = newRefPos->getPosition() - lastRefPos->getPosition();
    directionVector.normalize();
    if (stepDistance > 0)
    {
        refPos->move(directionVector, stepDistance);
    }
    distanceTraveledFromLastVertex += stepDistance;
    osg::Vec3 pos = refPos->getPosition();

    agentVehicle->setPosition(pos, directionVector);
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

    dt += OpenScenarioPlugin::instance()->scenarioManager->simulationStep;

    cout << getName() << " is breaking! New speed: " << speed << endl;
    float t_end = (targetSpeed-old_speed)/acceleration;
    if(dt>=t_end)
    {
        speed = targetSpeed;
        dt = 0.0;
        event->finishedEntityActions++;

    }
    else
    {
        speed = acceleration*dt+old_speed;
    }


}

