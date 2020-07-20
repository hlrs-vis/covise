#include "Entity.h"
#include "ReferencePosition.h"
#include "Action.h"
#include "Event.h"
#include "LaneChange.h"
#include "Position.h"
#include <OpenScenario/schema/oscVehicle.h>
#include <OpenScenario/schema/oscEntity.h>
#include <OpenScenario/OpenScenarioBase.h>
#include "OpenScenario/schema/oscSpeedDynamics.h"
#include "OpenScenarioPlugin.h"
using namespace std;
using namespace OpenScenario;

Entity::Entity(oscObject* obj) :
	object(obj),
	name(obj->name),
	refPos(nullptr),
	newRefPos(nullptr),
	dt(0.0),
	agentVehicle(nullptr)
{
	directionVector.set(1, 0, 0);
	lateralVector.set(0, 1, 0);

	std::string catalogReferenceName= object->CatalogReference->entryName;
	vehicle = ((oscVehicle*)(OpenScenarioPlugin::instance()->osdb->getCatalogObjectByCatalogReference("VehicleCatalog", catalogReferenceName)));

	if (vehicle)
	{
		std::string geometryFileName;
		for (oscFileArrayMember::iterator it = vehicle->Properties->File.begin(); it != vehicle->Properties->File.end(); it++)
		{
			oscFile* file = ((oscFile*)(*it));
			geometryFileName = file->filepath.getValue();
			break;
		}

		agentVehicle = new AgentVehicle(name, new CarGeometry(name, geometryFileName, true));
	}
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
		

		if (agentVehicle)
		{
			agentVehicle->setPosition(entityPosition, directionVector);
		}

   // }

}


void Entity::moveLongitudinal()
{
    if(refPos->road != NULL && speed > 0 && agentVehicle)
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
		
		vehicleUtil::Transform vehicleTransform = refPos->road->getRoadTransform(refPos->s, refPos->t);
        agentVehicle->setTransform(vehicleTransform,hdg);
        //cout << name << " is driving on Road: " << refPos->roadId << endl;
    }
    else
    {
		if(agentVehicle)
		{
			agentVehicle->setPosition(refPos->xyz, directionVector);
		}
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
		//refPos->update();//hinzugefügt
    }
    distanceTraveledFromLastVertex += stepDistance;
    osg::Vec3 pos = refPos->getPosition();

    agentVehicle->setPosition(pos, directionVector);
	//cout << "actual position: s=" << refPos->s << "t=" << refPos->t << endl;
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

void Entity::finishedEntityActionCounter(Event* event)
{
	event->finishedEntityActions++;
}

void Entity::doLaneChange(LaneChange* lc, Event* event)
{
	if (lc->Target.exists())
	{
		//step lanechange
		if (lc->Dynamics->shape == oscSpeedDynamics::step)
		{
			if (lc->Target->Relative.exists())
			{
				refPos->t += tDistance;
			}

			if (lc->Target->Absolute.exists())
			{
				refPos->t += tDistance;
			}
			event->finishedEntityActions++;
		}
		//linear lanechange
		if (lc->Dynamics->shape == oscSpeedDynamics::linear)
		{

			if (lc->Target->Relative.exists() || lc->Target->Absolute.exists())
			{
				if (lc->Dynamics->distance.exists())			
				{
					osg::Vec3 segmentVector = newRefPos->getPosition() - lastRefPos->getPosition();
					segmentLength = segmentVector.length();
					float stepDistance = speed * OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
					directionVector = newRefPos->getPosition() - lastRefPos->getPosition();
					directionVector.normalize();

					//if true: lanechange finished
					if (distanceTraveledFromLastStep + stepDistance > segmentLength)
					{
						refPos->update();
						event->finishedEntityActions++;
						stepDistance = 0;
					}
					//if true: lanechange finished
					else if (distanceTraveledFromLastStep + stepDistance == segmentLength){
						refPos->move(directionVector, stepDistance);
						refPos->update();
						event->finishedEntityActions++;
					}
					else {
						//move Entity
						if (stepDistance > 0)
						{
							refPos->move(directionVector, stepDistance);
							refPos->update();
							distanceTraveledFromLastStep += stepDistance; //count the traveled distance
						}
					}
					osg::Vec3 pos = refPos->getPosition();
					agentVehicle->setPosition(pos, directionVector);				
				}
				if (lc->Dynamics->time.exists())
				{
					if (lc->Dynamics->time == 0)
					{
						refPos->t += tDistance;
						event->finishedEntityActions++;
					}
					else
					{
						double simulationStep = OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
						double segmentLength = speed * timeLc;
						double stepDistance = speed * simulationStep;
						double deltaS = sqrt(segmentLength * segmentLength - tDistance * tDistance);

						//get newRefPos
						getDestinationPositionLc(refPos, newRefPos, tDistance, deltaS);
						directionVector = newRefPos->getPosition() - refPos->getPosition();
						directionVector.normalize();
						double stepDistanceT = stepDistance * directionVector[1];


						//determine hdg
						double phi = atan(directionVector[1] / directionVector[0]);
						cout << "phi= " << phi << endl;

						//check progress 
						if (timeLc > simulationStep)
						{
							if (stepDistance != 0)
							{
								refPos->move(directionVector, stepDistance);
								refPos->update();
							}
							refPos->hdg += phi;
							timeLc -= simulationStep;
							//calculate the distance 
							if (tDistance > 0)
							{
								tDistance -= fabs(stepDistanceT);
							}
							else
							{
								tDistance += fabs(stepDistanceT);
							}
							distanceTraveledFromLastStep += stepDistance;
						}

						else
						{
							//lanechange is finishing
							if (fabs(speed) > 0)
							{
								refPos->hdg = 0;
								event->finishedEntityActions++;
								timeLc = 0;
							}

						}
						osg::Vec3 pos = refPos->getPosition();
						agentVehicle->setPosition(pos, directionVector);
					}
				}
			}
		}
		if (lc->Dynamics->shape == oscSpeedDynamics::sinusoidal)
		{
			if (lc->Target->Relative.exists() || lc->Target->Absolute.exists())
			{
				if (lc->Dynamics->distance.exists())
				{
					double simulationStep = OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
					//traveled distance in s + actual stepDistance;
					dsLane += speed * simulationStep;

					double nextT = sinusPosition(dsLane);
					float stepDistanceT = nextT - refPos->t;
					double travelDistanceT = newRefPos->t - refPos->t;
					double travelDistanceS = speed * simulationStep;

					directionVector[0] = travelDistanceS;
					directionVector[1] = stepDistanceT;
					float stepWidth = directionVector.length();

					//check progress of lanechange
					if (fabs(dsLane) < fabs(distance))
					{
						if (fabs(stepDistanceT) > 0)
						{
							refPos->move(speed * simulationStep, stepDistanceT, stepWidth);
						}
					}
					else
					{
						if (fabs(speed) > 0)
						{
								cout << "lanechange vorbei!!" << endl;
								refPos->hdg = 0;
								refPos->update();
								event->finishedEntityActions++;
						}
					}
					osg::Vec3 pos = refPos->getPosition();
					agentVehicle->setPosition(pos, directionVector);
				}

				if (lc->Dynamics->time.exists())
				{
					if (lc->Dynamics->time == 0)
					{
						refPos->t += tDistance;
						event->finishedEntityActions++;
					}
					else
					{
						double simulationStep = OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
						double deltaS = speed * timeLc;

						cout << "deltaS" << deltaS << endl;
						getDestinationPositionLc(refPos, newRefPos, tDistance, deltaS);
						dsLane += speed * simulationStep;

						//sinus parameter
						a = (lastRefPos->t - newRefPos->t) / 2;
						b = 3.14159 / (newRefPos->s - lastRefPos->s);
						d = (lastRefPos->t + newRefPos->t) / 2;
						//get new t-position
						double nextT = sinusPosition(dsLane);
						float stepDistanceT = nextT - refPos->t;
						double travelDistanceT = newRefPos->t - refPos->t;
						double travelDistanceS = speed * simulationStep;
						
						directionVector[0] = travelDistanceS;
						directionVector[1] = stepDistanceT;
						float stepWidth = directionVector.length();
						directionVector.normalize();
						
						//check progress and move Entity
						if (timeLc > simulationStep)
						{
							if (fabs(stepDistanceT) > 0)
							{
								refPos->move(travelDistanceS, stepDistanceT, stepWidth);
								timeLc -= simulationStep;
								if (tDistance > 0)
								{
									tDistance -= fabs(stepDistanceT);
								}
								else
								{
									tDistance += fabs(stepDistanceT);
								}
							}
							
						}
						else if (timeLc == simulationStep) {

							if (fabs(stepDistanceT) > 0)
							{
								refPos->move(travelDistanceS, stepDistanceT, stepWidth);
								event->finishedEntityActions++;
							}
						}
						else
						{
							if (fabs(speed) > 0)
							{
								refPos->hdg = 0;
								cout << "lanechange vorbei!" << endl;
								event->finishedEntityActions++;
							}
						}
						osg::Vec3 pos = refPos->getPosition();
						agentVehicle->setPosition(pos, directionVector);
					}
				}
			}
		}
		if (lc->Dynamics->shape == oscSpeedDynamics::cubic)
		{
			if (lc->Dynamics->distance.exists())
			{
				if (lc->Target->Relative.exists() || lc->Target->Absolute.exists())
				{
					//get new t and s position
					double simulationStep = OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
					dsLane += speed * simulationStep;
					double nextT = cubicPosition(dsLane);
					float stepDistanceT = nextT - refPos->t;
					double travelDistanceT = newRefPos->t - refPos->t;
					double travelDistanceS = speed * simulationStep;

					directionVector[0] = travelDistanceS;
					directionVector[1] = stepDistanceT;
					float stepDistance = directionVector.length();
					directionVector.normalize();

					if (fabs(dsLane) < fabs(distance))
					{
						if (fabs(stepDistanceT) > 0)
						{
							refPos->move(travelDistanceS,stepDistanceT, stepDistance);
							
						}
					}
					else if (fabs(dsLane) == fabs(distance)) {
						if (fabs(stepDistanceT) > 0)
						{
							refPos->move(travelDistanceS, stepDistanceT, stepDistance);
							event->finishedEntityActions++;
						}
					}
					else
					{
						if (fabs(speed) > 0)
						{
							cout << "lanechange vorbei!!" << endl;
							refPos->hdg = 0;
							event->finishedEntityActions++;
						}
					}
					osg::Vec3 pos = refPos->getPosition();
					agentVehicle->setPosition(pos, directionVector);
				}
			}
			if (lc->Dynamics->time.exists())
			{
				if (lc->Dynamics->time == 0)
				{
					refPos->t += tDistance;
					event->finishedEntityActions++;
				}
				else
				{
					double simulationStep = OpenScenarioPlugin::instance()->scenarioManager->simulationStep;
					double deltaS = speed * timeLc;
					getDestinationPositionLc(refPos, newRefPos, tDistance, deltaS);
					dsLane += speed * simulationStep;


					double nextT = cubicPosition(dsLane);
					float stepDistanceT = nextT - refPos->t;
					double travelDistanceT = newRefPos->t - refPos->t;
					double travelDistanceS = speed * simulationStep;

					//phi for hdg
					directionVector[0] = speed * simulationStep;
					directionVector[1] = stepDistanceT;
					float stepWidth = directionVector.length();
					directionVector.normalize();
					

					if (timeLc > simulationStep)
					{
						if (fabs(stepDistanceT) > 0)
						{
							refPos->move(travelDistanceS,stepDistanceT, stepWidth); 
							refPos->update();

							timeLc -= simulationStep;
							if (tDistance > 0)
							{
								tDistance -= fabs(stepDistanceT);
							}
							else
							{
								tDistance += fabs(stepDistanceT);
							}
						}
					}
					else if (timeLc == simulationStep) {
						if (fabs(stepDistanceT) > 0)
						{
							refPos->move(travelDistanceS, stepDistanceT, stepWidth);
							refPos->update();
							event->finishedEntityActions++;
						}
					}
					else
					{
						if (fabs(speed) > 0)
						{
							cout << "lanechange vorbei!" << endl;
							refPos->hdg = 0;
							refPos->update();
							event->finishedEntityActions++;
						}

					}
					osg::Vec3 pos = refPos->getPosition();
					agentVehicle->setPosition(pos, directionVector);
				}
			}
		}	
	}
}

double Entity::getRelativeLcDistance(int value, double targetOff)
{
	double distance = 0;
	double targetOffset = targetOff;
	double width, Laneoffset;
	double widthStartLane, widthDestinationLane; 
	double startOffset = refPos->offset; 
	//get width of the start and destination lanes
	int destinationLaneId = refPos->laneId + value;
	refPos->road->getLaneWidthAndOffset(refPos->s, refPos->laneId, widthStartLane, Laneoffset);
	if (widthStartLane < 0)
	{
		widthStartLane = widthStartLane * -1;
	}
	refPos->road->getLaneWidthAndOffset(refPos->s, destinationLaneId, widthDestinationLane, Laneoffset);
	if (widthDestinationLane < 0)
	{
		widthDestinationLane = widthDestinationLane * -1;
	}
	if (value > 1)
	{
		for (int i = 1; i < value; i++)
		{
			int actualLane = refPos->laneId + i;
			refPos->road->getLaneWidthAndOffset(refPos->s, actualLane, width, Laneoffset);
			if (width < 0)
			{
				width = width * -1;
			}
			distance += width;
		}
	}
	if (value < -1)
	{
		for (int i = -1; i > value; i--)
		{
			int actualLane = refPos->laneId + i;
			refPos->road->getLaneWidthAndOffset(refPos->s, actualLane, width, Laneoffset);
			if (width < 0)
			{
				width = width * -1;
			}
			distance += width;
		}
	}
	if (value > 0) {
		startOffset = startOffset * -1;
	}
	else {
		targetOffset = targetOffset * -1;
	}
	distance += 0.5 * widthStartLane + 0.5 * widthDestinationLane + startOffset +targetOffset;
	return distance;
}

double Entity::getAbsoluteLcDistance(int value, double targetOff)
{
	double distance = 0;
	double width, offset;
	double startOffset = refPos->offset;
	int actualLane = refPos->laneId;
	double targetOffset = targetOff;
	double widthDestinationLane,widthStartLane;
	int destinationLaneId = value;
	refPos->road->getLaneWidthAndOffset(refPos->s, actualLane, widthStartLane, offset);
	if (widthStartLane < 0)
	{
		widthStartLane = widthStartLane * -1;
	}

	refPos->road->getLaneWidthAndOffset(refPos->s, destinationLaneId, widthDestinationLane, offset);
	if (widthDestinationLane < 0)
	{
		widthDestinationLane = widthDestinationLane * -1;
	}
	if (actualLane < destinationLaneId) {
		startOffset = startOffset * -1;
	}
	else {
		targetOffset = targetOffset * -1;
	}
	

	if ((actualLane-1) > value)
	{
		while (actualLane-1 != value)
			{
			refPos->road->getLaneWidthAndOffset(refPos->s, actualLane, width, offset);
			if (width < 0)
			{
				width = width * -1;
			}
			distance += width;
			actualLane--;
			}
		}
	if ((actualLane + 1) < value)
	{
		while ((actualLane+1) != value)
		{
			refPos->road->getLaneWidthAndOffset(refPos->s, actualLane+1, width, offset);
			if (width < 0)
			{
				width = width * -1;
			}
			distance += width;
			actualLane++;
		}
	}
	if (actualLane)
	distance +=0.5* widthDestinationLane+0.5*widthStartLane + startOffset +targetOffset;
	return distance;
	}

void Entity::getDestinationPositionLc(ReferencePosition* relativePos, ReferencePosition* position,double width,double distance)
{
	double ds = distance;
	double dt = width;
	position->update(relativePos->roadId, relativePos->s + ds, relativePos->t + dt);
}
void Entity::startDoLaneChange(LaneChange* lc)
{
	distanceTraveledFromLastStep = 0;
	tDistance = 0;
	dsLane = 0;
	dtLane = 0;
	int value = 0;
	double targetOffset = 0;
	
	if (lc->targetLaneOffset.exists()) {
		targetOffset = lc->targetLaneOffset.getValue();
	}
	if (lc->Dynamics->time.exists()) {
		timeLc = lc->Dynamics->time;
	}
	*lastRefPos = *refPos;
	//calculate distance in t-direction relative to an object
	if (lc->Target->Relative.exists())
	{
		string object = lc->Target->Relative->object.getValue();	
		if (object != "$owner"&& object != name) {
			int relObjectLaneId = OpenScenarioPlugin::instance()->scenarioManager->getEntityByName(object)->refPos->laneId;
			int destinationLaneId = relObjectLaneId + lc->Target->Relative->value;
			value = destinationLaneId - refPos->laneId;
		}
		else {
			value = lc->Target->Relative->value;
		}	
		if (value < 0)
		{
			tDistance -= getRelativeLcDistance(value, targetOffset);
		}
		else
		{
			tDistance += getRelativeLcDistance(value, targetOffset);
		}
	}
	if (lc->Target->Absolute.exists())
	{
		int value = lc->Target->Absolute->value;
		if (value < refPos->laneId)
		{
			tDistance -= getAbsoluteLcDistance(value, targetOffset);
		}
		else 
		{
			tDistance += getAbsoluteLcDistance(value, targetOffset);
		}
	}
	//get newRefPos
	if (lc->Dynamics->distance.exists())
	{
		if (lc->Dynamics->shape == oscSpeedDynamics::linear|| lc->Dynamics->shape == oscSpeedDynamics::sinusoidal|| lc->Dynamics->shape == oscSpeedDynamics::cubic)
		{
			distance = lc->Dynamics->distance;

			getDestinationPositionLc(refPos, newRefPos, tDistance, distance);
			*lastRefPos = *refPos;
		}

	}
	//calculate sinus coefficients
	if (lc->Dynamics->shape == oscSpeedDynamics::sinusoidal)
	{
		if (lc->Dynamics->distance.exists())
		{
			a = (refPos->t - newRefPos->t)/2;
			b = 3.14159 / distance;
			d = (refPos->t + newRefPos->t) / 2;
		}
		if (lc->Dynamics->time.exists())
		{
			*lastRefPos =* refPos;
		}
	}
	if (lc->Dynamics->shape == oscSpeedDynamics::cubic)
	{
		if (lc->Dynamics->distance.exists()|| lc->Dynamics->time.exists())
		{
			*lastRefPos = *refPos;
		}
	}
}
double Entity::sinusPosition(double x)
{
	double y = a * cos(b * x) + d;
	return y;
}

double Entity::cubicPosition(double x)
{
	double f0 = lastRefPos->t;
	double df0 = lastRefPos->hdg;
	double f1 = newRefPos->t;
	double length = newRefPos->s - lastRefPos->s;
	double angle = newRefPos->hdg - df0;
	double df1 = tan(angle * 2.0 * 3.14159 / 360);

	//calculate coefficients
	double d = (df1 + df0 - 2.0 * f1 / length + 2.0 * f0 / length) / (length * length);
	double c = (f1 - d * length * length * length - df0 * length - f0) / (length * length);

	//cubic function
	double y = f0 + df0 * x + c * x * x + d * x * x * x;
	return y;
}
