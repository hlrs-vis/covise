#include "TestDynamics.h"
#include <iostream>
#include <cmath>

#include "SteeringWheel.h"
#include "RoadSystem/Types.h"

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>
#include <math.h>


TestDynamics::TestDynamics()
	
{
	state.deltaX = 0.0;
	state.deltaY = 0.0;
	state.deltaPsi = 0.0;
	state.vX = 0.0;
	state.vY = 0.0;
	state.vPsi = 0.0;
	state.aX = 0.0;
	state.aY = 0.0;
	state.aPsi = 0.0;
	state.X = 0.0;
	state.Y = 0.0;
	state.Z = 0.0;
	state.psi = 0.0;
	state.fxfl = 0.0;
	state.fxfr = 0.0;
	state.fxrr = 0.0;
	state.fxrl = 0.0;
	state.fyfl = 0.0;
	state.fyfr = 0.0;
	state.fyrr = 0.0;
	state.fyrl = 0.0;
	state.fx = 0.0;
	state.fy = 0.0;
	state.fdrag = 0.0;
	state.frollfl = 0.0;
	state.frollfr = 0.0;
	state.frollrr = 0.0;
	state.frollrl = 0.0;
	state.fbrakefl = 0.0;
	state.fbrakefr = 0.0;
	state.fbrakerr = 0.0;
	state.fbrakerl = 0.0;
	state.fdrivefl = 0.0;
	state.fdrivefr = 0.0;
	state.fdriverr = 0.0;
	state.fdriverl = 0.0;
	state.fengine = 0.0;
	state.fbrake = 0.0;
	state.mz = 0.0;
	state.betaf = 0.0;
	state.betar = 0.0;
	state.delta = 0.0;
		
	accelerator = 0.0;
	steering = 0.0;
	brake = 0.0;
	
	leftRoad = true;
	
	
}

std::pair<Road *, double> TestDynamics::getStartPositionOnRoad()
{
	cout << "--------------------" << endl;
	cout << "--------------------" << endl;
	cout << "getStartPositionOnRoad reached" << endl;
	
	
	RoadSystem *system = RoadSystem::Instance();
	
	if(system->getNumRoads() > 0)
	{
		xodrLoaded = true;
		cout << "road system found" << endl;
	}
	cout << "system number of roads: " << system->getNumRoads() << endl;
	cout << "--------------------" << endl;
	cout << "--------------------" << endl;
	for (int roadIt = 0; roadIt < system->getNumRoads(); ++roadIt)
	{
        
		Road *road = system->getRoad(roadIt);
		cout << "road number " << roadIt << " found." << endl;
		if (road->getLength() >= 2.0 * targetS)
        {
            LaneSection *section = road->getLaneSection(targetS);
            if (section)
            {
                for (int laneIt = -1; laneIt >= -section->getNumLanesRight(); --laneIt)
                {
                    Lane *lane = section->getLane(laneIt);
                    if (lane->getLaneType() == Lane::DRIVING)
                    {
                        double t = 0.0;
                        for (int laneWidthIt = -1; laneWidthIt > laneIt; --laneWidthIt)
                        {
                            t -= section->getLane(laneWidthIt)->getWidth(targetS);
                        }
                        t -= 0.5 * lane->getWidth(targetS);
						currentRoad[0] = road;
                        return std::make_pair(road, t);
                    }
                }
            }
        }
    }
	return std::make_pair((Road *)NULL, 0.0);
	
}

void TestDynamics::initState() 
{
	if (startPos.first)
	{
		cout << "--------------------" << endl;
		cout << "--------------------" << endl;
		cout << "startPos Loop reached" << endl;
		RoadPoint rP = startPos.first->getRoadPoint(targetS, startPos.second);
		leftRoad = false;
		currentLongPos[0] = targetS;
        currentLongPos[1] = -1.0;
        currentLongPos[2] = -1.0;
        currentLongPos[3] = -1.0;
		state.X = rP.x();
		state.Y = rP.y();
		state.Z = rP.z();
		cout << "Road point coordinates are:" << endl << "X: " << rP.x() << endl << "Y: " << rP.y() << endl << "Z: " << rP.z() << endl;
		cout << "--------------------" << endl;
		cout << "--------------------" << endl;
		
		/*osg::Matrix relTrans;
		relTrans.makeTranslate(state.Y, state.Z, state.X);
		
		chassisTrans = relTrans * chassisTrans;*/
		
		globalPos.makeTranslate(rP.x(), rP.z(), -rP.y());
		/*state.X = 0.0;
		state.Y = 0.0;
		state.Z = 0.0;*/
		
	}
	else 
	{
		/*std::get<0>(y)[2] = -0.2; //Initial position
        std::get<2>(y)[0] = 1.0; //Initial orientation (Important: magnitude be one!)*/
        cout << "--------------------" << endl;
		cout << "--------------------" << endl;
		cout << "else statement reached" << endl;
		cout << "--------------------" << endl;
		cout << "--------------------" << endl;
		currentRoad[0] = NULL;
        currentRoad[1] = NULL;
        currentRoad[2] = NULL;
        currentRoad[3] = NULL;
        currentLongPos[0] = -1.0;
        currentLongPos[1] = -1.0;
        currentLongPos[2] = -1.0;
        currentLongPos[3] = -1.0;
        leftRoad = true;
	}
	
}

MovementState TestDynamics::deltaFunction(double inputArray[], double dT)
{
	state.delta = - steeringRatio * steering; //positive moves right
	
	state.vX = inputArray[0];
	state.vY = inputArray[1];
	state.vPsi = inputArray[2];
	
	/*state.betar = - atan((state.vY + a1 * state.vPsi) / (state.vX+0.00000000000000001));
	state.betaf = - atan((state.vY - a2 * state.vPsi)/ (state.vX+0.00000000000000001)) + state.delta;*/
	if (state.vX == 0)
	{
		if (state.vY == 0)
		{	
			state.betar = 0;
		} else 
		{
			state.betar = M_PI / 2;
		}
	} else 
	{
		state.betar = - atan((state.vY + a2 * state.vPsi) / (state.vX));
	}
	
	//calculate betaf:
	if (state.vX == 0)
	{
		if (state.vY == 0)
		{	
			state.betaf = 0;
		} else 
		{
			state.betaf = M_PI / 2 + state.delta;
		}
	} else 
	{
		state.betaf = - atan((state.vY - a1 * state.vPsi)/ (state.vX)) + state.delta;
	}
	
	//calculate forces:
	//calculate fdrag:
	state.fdrag = - state.vX * state.vX * cAero;
	
	//calculate fengine:
	state.fengine = enginePower * accelerator; 
	
	//calculate fdrive:
	state.fdrivefl = 0.5 * (1 - powerDist) * state.fengine;
	state.fdrivefr = 0.5 * (1 - powerDist) * state.fengine;
	state.fdriverr = 0.5 * powerDist * state.fengine;
	state.fdriverl = 0.5 * powerDist * state.fengine;
	
	//calculate fbrake:
	state.fbrake = brakePower * brake;
	
	//calculate brake forces and roll resistance on wheels:
	state.fbrakefl = - 0.5 * (1 - brakeDist) * state.fbrake * tanh(state.vX);
	state.fbrakefr = - 0.5 * (1 - brakeDist) * state.fbrake * tanh(state.vX);
	state.fbrakerr = - 0.5 * brakeDist * state.fbrake * tanh(state.vX);
	state.fbrakerl = - 0.5 * brakeDist * state.fbrake * tanh(state.vX);
	state.frollfl = - mass * g * mu * state.vX * copysign(1.0, state.vX);
	state.frollfr = - mass * g * mu * state.vX * copysign(1.0, state.vX);
	state.frollrr = - mass * g * mu * state.vX * copysign(1.0, state.vX);
	state.frollrl = - mass * g * mu * state.vX * copysign(1.0, state.vX);
	
	//calculate fy:
	state.fyfl = mass * g * Df * sin(Cf * atan((Bf * state.betaf - Ef * (Bf * state.betaf - atan(Bf * state.betaf)))));
	state.fyfr = mass * g * Df * sin(Cf * atan((Bf * state.betaf - Ef * (Bf * state.betaf - atan(Bf * state.betaf)))));
	state.fyrr = mass * g * Dr * sin(Cr * atan((Br * state.betar - Er * (Br * state.betar - atan(Br * state.betar)))));
	state.fyrl = mass * g * Dr * sin(Cr * atan((Br * state.betar - Er * (Br * state.betar - atan(Br * state.betar)))));
	
	//cout << "std::abs(state.vY) " << std::abs(state.vY) << endl;
	state.fyfl += + /*tanh(state.vPsi) **/ lateralMu * state.vPsi - tanh(state.vY) * lateralMu * sqrt(std::abs(state.vY));
	state.fyfr += + /*tanh(state.vPsi) **/ lateralMu * state.vPsi - tanh(state.vY) * lateralMu * sqrt(std::abs(state.vY));
	state.fyrr += - /*tanh(state.vPsi) **/ lateralMu * state.vPsi - tanh(state.vY) * lateralMu * sqrt(std::abs(state.vY));
	state.fyrl += - /*tanh(state.vPsi) **/ lateralMu * state.vPsi - tanh(state.vY) * lateralMu * sqrt(std::abs(state.vY));
	
	
	//calculate driving forces:
	state.fxfl = state.fbrakefl + state.fdrivefl;
	state.fxfr = state.fbrakefr + state.fdrivefr;
	state.fxrr = state.fbrakerr + state.fdriverr;
	state.fxrl = state.fbrakerl + state.fdriverl;
	
	//friction circle
	double tempAnglefl = atan(state.fyfl / (state.fxfl + 0.00000000000000000000000000001));
	double tempAnglefr = atan(state.fyfr / (state.fxfr + 0.00000000000000000000000000001));
	double tempAnglerr = atan(state.fyrr / (state.fxrr + 0.00000000000000000000000000001));
	double tempAnglerl = atan(state.fyrl / (state.fxrl + 0.00000000000000000000000000001));
	if (frictionCircleLimit < sqrt(state.fyfl * state.fyfl + state.fxfl * state.fxfl))
	{
		double tempAnglefl = atan(state.fyfl / (state.fxfl + 0.00000000000000000000000000001));
		state.fyfl = std::abs(sin(tempAnglefl) * frictionCircleLimit) * copysign(1.0, state.fyfl);
		state.fxfl = std::abs(cos(tempAnglefl) * frictionCircleLimit) * copysign(1.0, state.fxfl);
	}
	if (frictionCircleLimit < sqrt(state.fyfr * state.fyfr + state.fxfr * state.fxfr))
	{
		double tempAnglefr = atan(state.fyfr / (state.fxfr + 0.00000000000000000000000000001));
		state.fyfr = std::abs(sin(tempAnglefr) * frictionCircleLimit) * copysign(1.0, state.fyfr);
		state.fxfr = std::abs(cos(tempAnglefr) * frictionCircleLimit) * copysign(1.0, state.fxfr);
	}
	if (frictionCircleLimit < sqrt(state.fyrr * state.fyrr + state.fxrr * state.fxrr))
	{
		double tempAnglerr = atan(state.fyrr / (state.fxrr + 0.00000000000000000000000000001));
		state.fyrr = std::abs(sin(tempAnglerr) * frictionCircleLimit) * copysign(1.0, state.fyrr);
		state.fxrr = std::abs(cos(tempAnglerr) * frictionCircleLimit) * copysign(1.0, state.fxrr);
	}
	if (frictionCircleLimit < sqrt(state.fyrl * state.fyrl + state.fxrl * state.fxrl))
	{
		double tempAnglerl = atan(state.fyrl / (state.fxrl + 0.00000000000000000000000000001));
		state.fyrl = std::abs(sin(tempAnglerl) * frictionCircleLimit) * copysign(1.0, state.fyrl);
		state.fxrl = std::abs(cos(tempAnglerl) * frictionCircleLimit) * copysign(1.0, state.fxrl);
	}
	
	//calculate overall forces	
	state.fx = state.fxfl + state.fxfr + state.fxrr + state.fxrl + state.fdrag + state.frollfl + state.frollfr + state.frollrr + state.frollrl;
	state.fy = state.fyfl + state.fyfr + state.fyrr + state.fyrl;
	state.mz = - state.fyfl * a1 - state.fyfr * a1 + state.fyrr * a2 + state.fyrl * a2;
		
	//calculate accelerations:
	state.aX = state.fx / mass - state.vPsi * state.vY;
	state.aY = state.fy / mass + state.vPsi * state.vX;
	state.aPsi = state.mz / inertia;
	
	//calculate updated velocities and positions:
	state.vX = state.aX * dT;
	state.vY = state.aY * dT;
	state.vPsi = state.aPsi * dT;
	
	MovementState out;
	
	out.vX = state.vX;
	out.vY = state.vY;
	out.vPsi = state.vPsi;
	
	return out;
	
}


void TestDynamics::timeStep(double dT)
{
	accelerator = InputDevice::instance()->getAccelerationPedal(); //negative moves forward
	steering = InputDevice::instance()->getSteeringWheelAngle();
	brake = InputDevice::instance()->getBrakePedal();
	
	for (int i = 0; i < integrationSteps; i++)
	{		
		double vXOld = state.vX;
		double vYOld = state.vY;
		//state.vX= cos(state.deltaPsi) * vXOld - sin(state.deltaPsi) * vYOld;
		//state.vY = sin(state.deltaPsi) * vXOld + cos(state.deltaPsi) * vYOld;
		
		double initialVX = state.vX;
		double initialVY = state.vY;
		double initialVPsi = state.vPsi;
		
		double inputOne [] = {state.vX, state.vY, state.vPsi};
		MovementState fOne = deltaFunction(inputOne, dT);
		
		double inputTwo [] = {initialVX + fOne.vX / 2.0, initialVY + fOne.vY / 2.0, initialVPsi + fOne.vPsi / 2.0};
		MovementState fTwo = deltaFunction(inputTwo, dT);
		
		double inputThree [] = {initialVX + fTwo.vX / 2.0, initialVY + fTwo.vY / 2.0, initialVPsi + fTwo.vPsi / 2.0};
		MovementState fThree = deltaFunction(inputThree, dT);
		
		double inputFour [] = {initialVX + fThree.vX, initialVY + fThree.vY, initialVPsi + fThree.vPsi};
		MovementState fFour = deltaFunction(inputFour, dT);
		
		state.vX = initialVX + 1.0 / 6.0 * (fOne.vX + 2 * fTwo.vX + 2 * fThree.vX + fFour.vX);
		state.vY = initialVY + 1.0 / 6.0 * (fOne.vY + 2 * fTwo.vY + 2 * fThree.vY + fFour.vY);
		state.vPsi = initialVPsi + 1.0 / 6.0 * (fOne.vPsi + 2 * fTwo.vPsi + 2 * fThree.vPsi + fFour.vPsi);
	}
	
	//limit values:
	if (state.vX > 100) 
	{
		state.vX = 100;
	}
	if (state.vY > 100)
	{
		state.vY = 100;
	}
	if (state.vPsi > 100)
	{
		state.vPsi = 100;
	}
	if (state.vX < -100) 
	{
		state.vX = -100;
	}
	if (state.vY < -100)
	{
		state.vY = -100;
	}
	if (state.vPsi < -100)
	{
		state.vPsi = -100;
	}
	if (std::abs(state.vX) < vXLimit) 
	{
		state.vX = 0;
	}
	if (std::abs(state.vY) < vYLimit)
	{
		state.vY = 0;
	}
	if (std::abs(state.vPsi) < vPsiLimit)
	{
		state.vPsi = 0;
	}
	
	
	state.deltaX = state.vX * dT * integrationSteps;
	state.deltaY = state.vY * dT * integrationSteps;
	state.deltaPsi = state.vPsi * dT * integrationSteps;
	
	/*//calculate pitch:
	tStateIn.momentumPitch = (tStateIn.forceX * cogHeight - springRate * tStateIn.theta - damping *  tStateIn.pitchRate);
	tStateIn.pitchRateDot = tStateIn.momentumPitch / mass;
	tStateIn.pitchRate = tStateIn.pitchRate + tStateIn.pitchRateDot * dT;
	tStateIn.deltaTheta = tStateIn.pitchRate * dT;
	tStateIn.theta += tStateIn.deltaTheta;*/
	
	/*cout << "fxfl " << state.fxfl << endl;
	cout << "fyfl " << state.fyfl << endl;
	cout << "-------------------" << endl;
	cout << "fxrl " << state.fxrl << endl;
	cout << "fyrl " << state.fyrl << endl;
	cout << "-------------------" << endl;
	cout << "vX " << state.vX << endl;
	cout << "vY " << state.vY << endl;
	cout << "vPsi " << state.vPsi << endl;
	cout << "-------------------" << endl;
	cout << "delta " << state.delta << endl;
	cout << "betaf " << state.betaf << endl;
	cout << "betar " << state.betar << endl;*/
}

void TestDynamics::setVehicleTransformation(const osg::Matrix &m)
{
	chassisTrans = m;
}

/*double TestDynamics::getRoadHeight(VrmlNodeVehicle *vehicle)
{
	
	
	osg::Vec2d vehiclePos = vehicle->getPos();
	cout << "vehicle pos X: " << vehiclePos.x() << endl;
	cout << "vehicle pos Y: " << vehiclePos.y() << endl;
	Vector2D vIn = Vector2D(vehiclePos.x(), vehiclePos.y());
	RoadSystem system = RoadSystem::Instance();
	cout << "number of roads in system: " << system->getNumRoads() << endl;
	for(int roadIt = 0; roadIt < system->getNumRoads(); roadIt++)
	{
		Road *road = system->getRoad(roadIt);
		
		if(road->isOnRoad(vIn))
		{
			RoadPoint point = road->getRoadPoint(vehiclePos.x(), vehiclePos.y());
			cout << "road point height: " << point.z() << endl;
		}
	}
		
	
	
	
}*/

void TestDynamics::move(VrmlNodeVehicle *vehicle)
{
	state.psi += state.deltaPsi;
	
	
	if (xodrLoaded == false)
	{
		startPos = getStartPositionOnRoad();
		if (xodrLoaded == true)
		{
			initState();
		}	
	}
	
	double h = cover->frameDuration() / integrationSteps;
	timeStep(h);
	
	osg::Matrix testMatrix1;
	osg::Matrix testMatrix2;
	osg::Matrix testMatrix2AndAHalf;
	/*testMatrix1.makeTranslate(0.0, 0.0, 1.0);
	testMatrix2.makeRotate(M_PI / 2, 0.0, 1.0, 0.0);
	testMatrix2AndAHalf.makeRotate(-M_PI / 2, 0.0, 1.0, 0.0);*/
	testMatrix1.makeTranslate(state.deltaY, 0.0, -state.deltaX);
	testMatrix2.makeRotate(state.psi, 0.0, 1.0, 0.0);
	testMatrix2AndAHalf.makeRotate(-state.psi, 0.0, 1.0, 0.0);
	/*osg::Matrix testMatrix3;
	testMatrix3 = testMatrix3 * testMatrix2AndAHalf * testMatrix1 * testMatrix2;*/
	globalPos = globalPos * testMatrix2AndAHalf * testMatrix1 * testMatrix2;
	
	
	if (xodrLoaded == true)
	{
		if (leftRoad == false)
		{
			Vector2D vecIn(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
			double oldX = state.X;
			double oldY = state.Y;
			double oldZ = state.Z;
			RoadSystem *rSystem = RoadSystem::Instance();
			if(currentRoad[0])
			{
				/*double inX = 10.00;
				double inY = 10.00;
				Vector2D inVec(inX, inY);
				RoadPoint point = currentRoad[0]->getRoadPoint(inX, inY);
				double inX3d = 35.0;
				double inY3d = -60.0;
				double inZ3d = 0.0;
				double longPos = 0.0;*/
				osg::Vec3d tempVec = globalPos.getTrans();
				Vector3D searchInVec(tempVec.x(), -tempVec.z(), tempVec.y());
				
				Vector2D searchOutVec = RoadSystem::Instance()->searchPositionFollowingRoad(searchInVec, currentRoad[0], currentLongPos[0]);
				currentLongPos[0] = searchOutVec.x();
				//Vector3D normalVector = currentRoad[0]->getNormalVector(currentLongPos[0]);
				
				/*if (printCounter < printMax)
				{
					cout << "*~*~*~*~*~*" << endl;
					//cout << "normal vector: " << endl << "x: " << normalVector.x() << endl << "y: " << normalVector.y() << endl << "z: " << normalVector.z() << endl;
					
					//double temp = sqrt(tangentVector.x() * tangentVector.x() + tangentVector.y() * tangentVector.y());
					//cout << "lengt in xy-plane: " << temp << endl;
					cout << "currentLongPos[0]: " << currentLongPos[0] << endl;
					cout << "tempvec: " << tempVec.x() <<  ";" << tempVec.y() <<  ";" << tempVec.z() << endl;
					cout << "searchOutVec: " << searchOutVec.x() <<  ";" << searchOutVec.y() << endl;
				}	*/
				RoadPoint point = currentRoad[0]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
				
				if (!isnan(point.x()))
				{
					globalPos.makeTranslate(point.x(), point.z(), -point.y());
					/*if (printCounter < printMax)
					{
						cout << "road length " << currentRoad[0]->getLength() << endl;
						cout << "is point on road? " << currentRoad[0]->isOnRoad(inVec)<< endl;
						cout << "point.x() " << point.x() << endl;
						cout << "point.y() " << point.y() << endl;
						cout << "point.z() " << point.z() << endl;
						cout << "*~*~*~*~*~*" << endl;
					}*/	
					/*printedOnce = true;*/
				} else 
				{
					leftRoad = true;
				}
				/*cout << "---------------------" << endl;
				cout << "current road exists" << endl;
				cout << "state.deltaX" << state.deltaX << endl;
				cout << "state.deltaY" << state.deltaY << endl;
				cout << "---------------------" << endl;
				Vector3D vecPos(state.X + state.deltaX, state.Y + state.deltaY, state.Z);
				
				if (rSystem)
				{
					cout << "---------------------" << endl;
					cout << "vector change reached" << endl;
					cout << "---------------------" << endl;
					vecIn = RoadSystem::Instance()->searchPositionFollowingRoad(vecPos, currentRoad[0], currentLongPos[0]);
					RoadPoint point = currentRoad[0]->getRoadPoint(state.X, state.Y);
					if (!isnan(point.x()))
					{
						state.X = point.x();
						state.Y = point.y();
						state.Z = point.z();
						cout << "test road point x value " << point.x() << endl;
						cout << "state.X: " << state.X << endl;
						cout << "test road point y value " << point.y() << endl;
						cout << "state.Y: " << state.Y << endl;
					}
				}
				if (!vecIn.isNaV())
				{
					RoadPoint point = currentRoad[0]->getRoadPoint(vecIn.x(), vecIn.y());
					state.X = point.y();
					state.Y = -point.x();
					state.Z = point.z();
				}*/
				
			}
			
			/*cout << "translate input Y: " << oldY - state.Y << endl;
			cout << "translate input X: " << oldX - state.X << endl;*/
			
			Vector3D tangentVector = currentRoad[0]->getTangentVector(currentLongPos[0]);
			//cout << "tangent vector: " << endl << "x: " << tangentVector.x() << endl << "y: " << tangentVector.y() << endl << "z: " << tangentVector.z() << endl;
			
			
			osg::Matrix idTrans;
			idTrans.makeIdentity();
			osg::Matrix relRotYaw;
			relRotYaw.makeRotate(state.deltaPsi, 0, 1, 0);
			
			osg::Matrix relRotRoll;
			osg::Matrix relRotPitch;
			if (!isnan(tangentVector.x()) && !isnan(tangentVector.y()) && !isnan(tangentVector.z()))
			{
				relRotRoll.makeRotate(atan((tangentVector.x() * tangentVector.z()) / (tangentVector.x() *tangentVector.x() + tangentVector.y() * tangentVector.y())), 0, 0, 1);
				relRotPitch.makeRotate(-atan((-tangentVector.y() * tangentVector.z()) / (tangentVector.x() * tangentVector.x() + tangentVector.y() * tangentVector.y())), 1, 0, 0);
			}
			
			rotationPos = rotationPos * relRotYaw;// * relRotRoll * relRotPitch;
			
			/*osg::Vec3d swapVec = globalPos.getTrans();
			osg::Matrix  swappedCoords;
			swappedCoords.makeTranslate(swapVec.x(), swapVec.z(), -swapVec.y());*/
			
			chassisTrans = rotationPos * relRotRoll * relRotPitch * globalPos;
			
			
			bodyTrans = chassisTrans  * rotationPos;
			
			/*if (printCounter < printMax) 
			{
				osg::Vec3d testVec = chassisTrans.getTrans();
				cout << "chassis trans: " << endl << "y: " << testVec.x() << endl << "z: " << testVec.y() << endl << "x: " << testVec.z() << endl;
			}*/
			
			vehicle->setVRMLVehicle(chassisTrans);
			vehicle->setVRMLVehicleBody(bodyTrans);
			
			osg::Matrix relRotTireYaw;
			relRotTireYaw.makeRotate(steering, 0, 1, 0);
			
			vehicle->setVRMLVehicleFrontWheels(relRotTireYaw, relRotTireYaw);
			vehicle->setVRMLVehicleRearWheels(idTrans, idTrans);
			
		}
	}	
	if (leftRoad == true)
	{
		osg::Matrix relTrans;
		relTrans.makeTranslate(state.deltaY, 0, -state.deltaX);
		osg::Matrix relRotYaw;
		relRotYaw.makeRotate(state.deltaPsi, 0, 1, 0);
		osg::Matrix relRotRoll;
		relRotRoll.makeRotate(0, 0, 0, 1);
		osg::Matrix relRotPitch;
		relRotPitch.makeRotate(0, 1, 0, 0);
			
		chassisTrans = relTrans * relRotRoll * relRotYaw * chassisTrans;
		bodyTrans = bodyTrans * relRotPitch * relRotRoll;
		
		osg::Matrix idTrans;
		idTrans.makeIdentity();

		vehicle->setVRMLVehicle(chassisTrans);
		vehicle->setVRMLVehicleBody(bodyTrans);
		
		osg::Matrix relRotTireYaw;
		relRotTireYaw.makeRotate(steering, 0, 1, 0);
		
		vehicle->setVRMLVehicleFrontWheels(relRotTireYaw, relRotTireYaw);
		vehicle->setVRMLVehicleRearWheels(idTrans, idTrans);
		
		currentRoad[0] = NULL;
		osg::Vec3d tempVec = globalPos.getTrans();
		Vector3D searchInVec(tempVec.x(), -tempVec.z(), tempVec.y());
		Vector2D v_c = RoadSystem::Instance()->searchPosition(searchInVec, currentRoad[0], currentLongPos[0]);
		std::cout << "search road from scratch" << std::endl;
		if (!v_c.isNaV())
		{
			RoadPoint point = currentRoad[0]->getRoadPoint(v_c.u(), v_c.v());
			globalPos.makeTranslate(point.x(), point.z(), -point.y());
			state.psi = 0;
			rotationPos.makeRotate(0, 0, 1, 0);
			leftRoad = false;
		}
	}
	
	
	
	osg::Vec3d testVec = globalPos.getTrans();
		
	if (printCounter < printMax) 
	{
		cout << "global pos: " << endl << "x: " << testVec.x() << endl << "y: " << testVec.y() << endl << "z: " << testVec.z() << endl;
		cout << "state.fy: " << state.fy<< endl;
		cout << "state.vY: " << state.vY<< endl;
		cout << "state.fx: " << state.fx<< endl;
		cout << "state.vX: " << state.vX<< endl;
		cout << "state.deltaPsi: " << state.deltaPsi << endl;
		cout << "psi " << state.psi << endl;
		cout << "state.deltaX: " << state.deltaX << endl;
		cout << "state.deltaY: " << state.deltaY << endl;
		cout << "leftRoad " << leftRoad << endl;
		cout << "__/|___/|___/|___/|___/|" << endl;
		
		//printCounter +=1;
	}
	
}

void TestDynamics::resetState()
{
	state.deltaX = 0;
	state.deltaY = 0;
	state.deltaPsi = 0;
	state.vX = 0;
	state.vY = 0;
	state.vPsi = 0;
	state.aX = 0;
	state.aY = 0;
	state.aPsi = 0;
	state.X = 0;
	state.Y = 0;
	state.Z = 0;
	state.psi = -M_PI /2;
	state.fxfl = 0;
	state.fxfr = 0;
	state.fxrr = 0;
	state.fxrl = 0;
	state.fyfl = 0;
	state.fyfr = 0;
	state.fyrr = 0;
	state.fyrl = 0;
	state.fx = 0;
	state.fy = 0;
	state.fdrag = 0;
	state.frollfl = 0;
	state.frollfr = 0;
	state.frollrr = 0;
	state.frollrl = 0;
	state.fbrakefl = 0;
	state.fbrakefr = 0;
	state.fbrakerr = 0;
	state.fbrakerl = 0;
	state.fdrivefl = 0;
	state.fdrivefr = 0;
	state.fdriverr = 0;
	state.fdriverl = 0;
	state.fengine = 0;
	state.fbrake = 0;
	state.mz = 0;
	state.betaf = 0;
	state.betar = 0;
	state.delta = 0;
	
	accelerator=0;
	steering=0;
	brake=0;

	//chassisTrans.makeTranslate(0, 0, 0);
	osg::Matrix rotate;
	rotate.makeRotate(0, 0, 1, 0);
	rotationPos.makeRotate(-M_PI /2, 0, 1, 0);
	//chassisTrans.makeIdentity();
	//chassisTrans.makeRotate(M_PI/2, 0, 1, 0);
	// = chassisTrans * rotate;
	//bodyTrans.makeIdentity();
	
	
	xodrLoaded = false;
}
