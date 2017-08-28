#include "TestDynamics.h"
#include <iostream>
#include <cmath>

#include "SteeringWheel.h"
#include <VehicleUtil/RoadSystem/Types.h>

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>
#include <util/unixcompat.h>
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

	targetS = 3.0;
	
	mass = 1900;
	cAero = 0.00002;
	mu = 0.005;
	lateralMu = 500;
	enginePower = 20000;
	brakePower = 8000;
	g = 9.81;
	inertia = 5000;
	//F = Fz · D · sin(C · arctan(B·slip - E · (B·slip - arctan(B·slip))))
	Bf = 7.5; //10=tarmac; 4=ice
	Cf = 1.9; //~2
	Df = 0.8; //1=tarmace; 0.1=ice
	Ef = 0.99; //0.97=tarmac; 1=ice
	Br = 6; //10=tarmac; 4=ice
	Cr = 1.9; //~2
	Dr = 0.6; //1=tarmace; 0.1=ice
	Er = 0.97; //0.97=tarmac; 1=ice
	a1 = 1.6;
	a2 = 1.65;
	vXLimit = 0.001;
	vYLimit = 0.01;
	vPsiLimit = 0.001;
	
	powerDist = 0.8; //1=RWD
	brakeDist = 0.5;
	steeringRatio = 0.15;
	frictionCircleLimit = 6000;
	integrationSteps = 5.0;
	xodrLoaded = false;
	printedOnce = false;
	printCounter = 1;
	printMax = 100;
	
	Car2OddlotRotation.makeRotate(M_PI/2,0,0,1);
	Oddlot2CarRotation.makeRotate(-M_PI/2,0,0,1);
	Oddlot2OpencoverRotation.makeRotate(-M_PI/2,1,0,0);
	Opencover2OddlotRotation.makeRotate(M_PI/2,1,0,0);
	Car2OpencoverRotation = Car2OddlotRotation * Oddlot2OpencoverRotation;
	Opencover2CarRotation = Opencover2OddlotRotation * Oddlot2CarRotation;
	
	currentRoad[0] = NULL;
	
	tireDist = 0.0;
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
	state.frollfl = - mass * g * mu * state.vX * std::copysign(1.0, state.vX);
	state.frollfr = - mass * g * mu * state.vX * std::copysign(1.0, state.vX);
	state.frollrr = - mass * g * mu * state.vX * std::copysign(1.0, state.vX);
	state.frollrl = - mass * g * mu * state.vX * std::copysign(1.0, state.vX);
	
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
		state.fyfl = std::abs(sin(tempAnglefl) * frictionCircleLimit) * std::copysign(1.0, state.fyfl);
		state.fxfl = std::abs(cos(tempAnglefl) * frictionCircleLimit) * std::copysign(1.0, state.fxfl);
	}
	if (frictionCircleLimit < sqrt(state.fyfr * state.fyfr + state.fxfr * state.fxfr))
	{
		double tempAnglefr = atan(state.fyfr / (state.fxfr + 0.00000000000000000000000000001));
		state.fyfr = std::abs(sin(tempAnglefr) * frictionCircleLimit) * std::copysign(1.0, state.fyfr);
		state.fxfr = std::abs(cos(tempAnglefr) * frictionCircleLimit) * std::copysign(1.0, state.fxfr);
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
	
	//test
	
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
	
	//test
	
	if (false/*xodrLoaded == true*/)
	{
		
		if (!leftRoad)
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
	//test
	/*double vXGlobal = 0.01;
	double vYGlobal = 0.005;
	double vZGlobal = 0.001;
	double vYawGlobal = 0;
	double vRollGlobal = 0;
	double vPitchGlobal = 0;
	
	double forwardSpeed = 0.10;*/
	
	osg::Matrix roadPoint;
	roadPoint.makeTranslate(0,0,0);
	
	if (true/*eftRoad == true*/)
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
		/*
		osg::Matrix rotMatrixInv = rotMatrix;
		rotMatrixInv = rotMatrixInv.inverse(rotMatrixInv);
		
		osg::Matrix CCSpeeds = globalSpeedMatrix * rotMatrixInv * Opencover2CarRotation;
		
		std::cout << "CCSpeeds:" << std::endl;
		std::cout << CCSpeeds(0,0) << "," << CCSpeeds(0,1) << "," << CCSpeeds(0,2) << "," << CCSpeeds(0,3) << std::endl;
		std::cout << CCSpeeds(1,0) << "," << CCSpeeds(1,1) << "," << CCSpeeds(1,2) << "," << CCSpeeds(1,3) << std::endl;
		std::cout << CCSpeeds(2,0) << "," << CCSpeeds(2,1) << "," << CCSpeeds(2,2) << "," << CCSpeeds(2,3) << std::endl;
		std::cout << CCSpeeds(3,0) << "," << CCSpeeds(3,1) << "," << CCSpeeds(3,2) << "," << CCSpeeds(3,3) << std::endl;
		std::cout << "---"<< std::endl;*/
		
		//test
		/*
		
		osg::Matrix xMatrix;
		double x1 = rotMatrix(0,0);
		double x2 = rotMatrix(0,1);
		double x3 = rotMatrix(0,2);
		if(accelerator > 0)
		{
			std::cout << "accelerator on" << std::endl;
			xMatrix.makeRotate(0.001,x1,x2,x3);
		}
		
		osg::Matrix yMatrix;
		double y1 = rotMatrix(1,0);
		double y2 = rotMatrix(1,1);
		double y3 = rotMatrix(1,2);
		if(brake > 0)
		{
			std::cout << "brake on" << std::endl;
			yMatrix.makeRotate(0.001,y1,y2,y3);
		}
		
		osg::Matrix zMatrix;
		double z1 = rotMatrix(2,0);
		double z2 = rotMatrix(2,1);
		double z3 = rotMatrix(2,2);
		if(steering > 0)
		{
			std::cout << "steering on" << std::endl;
			//zMatrix.makeRotate(0.001,z1,z2,z3);
			yMatrix.makeRotate(-0.001,y1,y2,y3);
		}
		
		//forwardMove.makeTranslate(-forwardSpeed * z1, -forwardSpeed * z2, -forwardSpeed * z3);
		globalSpeedMatrix.makeTranslate(-vYGlobal, vZGlobal, -vXGlobal);
		
		std::cout << xMatrix(0,0) << "," << xMatrix(0,1) << "," << xMatrix(0,2) << "," << xMatrix(0,3) << std::endl;
		std::cout << xMatrix(1,0) << "," << xMatrix(1,1) << "," << xMatrix(1,2) << "," << xMatrix(1,3) << std::endl;
		std::cout << xMatrix(2,0) << "," << xMatrix(2,1) << "," << xMatrix(2,2) << "," << xMatrix(2,3) << std::endl;
		std::cout << xMatrix(3,0) << "," << xMatrix(3,1) << "," << xMatrix(3,2) << "," << xMatrix(3,3) << std::endl;
		std::cout << "---"<< std::endl;
		
		osg::Matrix cogToChassis;
		double chassisOffsetZ = 2;
		cogToChassis.makeTranslate(chassisOffsetZ * z1, chassisOffsetZ * z2, chassisOffsetZ * z3);
		//cogToChassis.makeTranslate(0,0,2);
		rotMatrix = rotMatrix * xMatrix * yMatrix * zMatrix;
		std::cout << rotMatrix(0,0) << "," << rotMatrix(0,1) << "," << rotMatrix(0,2) << "," << rotMatrix(0,3) << std::endl;
		std::cout << rotMatrix(1,0) << "," << rotMatrix(1,1) << "," << rotMatrix(1,2) << "," << rotMatrix(1,3) << std::endl;
		std::cout << rotMatrix(2,0) << "," << rotMatrix(2,1) << "," << rotMatrix(2,2) << "," << rotMatrix(2,3) << std::endl;
		std::cout << rotMatrix(3,0) << "," << rotMatrix(3,1) << "," << rotMatrix(3,2) << "," << rotMatrix(3,3) << std::endl;
		std::cout << "---"<< std::endl;
		cogPos = cogPos * globalSpeedMatrix;
		chassisTrans = rotMatrix * cogPos * cogToChassis;
		
		std::cout << cogPos(0,0) << "," << cogPos(0,1) << "," << cogPos(0,2) << "," << cogPos(0,3) << std::endl;
		std::cout << cogPos(1,0) << "," << cogPos(1,1) << "," << cogPos(1,2) << "," << cogPos(1,3) << std::endl;
		std::cout << cogPos(2,0) << "," << cogPos(2,1) << "," << cogPos(2,2) << "," << cogPos(2,3) << std::endl;
		std::cout << cogPos(3,0) << "," << cogPos(3,1) << "," << cogPos(3,2) << "," << cogPos(3,3) << std::endl;
		std::cout << "---"<< std::endl;
		std::cout << chassisTrans(0,0) << "," << chassisTrans(0,1) << "," << chassisTrans(0,2) << "," << chassisTrans(0,3) << std::endl;
		std::cout << chassisTrans(1,0) << "," << chassisTrans(1,1) << "," << chassisTrans(1,2) << "," << chassisTrans(1,3) << std::endl;
		std::cout << chassisTrans(2,0) << "," << chassisTrans(2,1) << "," << chassisTrans(2,2) << "," << chassisTrans(2,3) << std::endl;
		std::cout << chassisTrans(3,0) << "," << chassisTrans(3,1) << "," << chassisTrans(3,2) << "," << chassisTrans(3,3) << std::endl;
		std::cout << "---"<< std::endl;
		
		osg::Matrix frontLeft;
		frontLeft.makeTranslate(1.5,1,0.3);
		
		osg::Matrix frontLeftPos = frontLeft * Car2OpencoverRotation * rotMatrix * cogPos;
		
		std::cout << frontLeftPos(0,0) << "," << frontLeftPos(0,1) << "," << frontLeftPos(0,2) << "," << frontLeftPos(0,3) << std::endl;
		std::cout << frontLeftPos(1,0) << "," << frontLeftPos(1,1) << "," << frontLeftPos(1,2) << "," << frontLeftPos(1,3) << std::endl;
		std::cout << frontLeftPos(2,0) << "," << frontLeftPos(2,1) << "," << frontLeftPos(2,2) << "," << frontLeftPos(2,3) << std::endl;
		std::cout << frontLeftPos(3,0) << "," << frontLeftPos(3,1) << "," << frontLeftPos(3,2) << "," << frontLeftPos(3,3) << std::endl;
		std::cout << "---"<< std::endl;
		
		/*osg::Matrix frontLeftPosInOddlot = frontLeftPos * Opencover2OddlotRotation;
		
		std::cout << frontLeftPosInOddlot(0,0) << "," << frontLeftPosInOddlot(0,1) << "," << frontLeftPosInOddlot(0,2) << "," << frontLeftPosInOddlot(0,3) << std::endl;
		std::cout << frontLeftPosInOddlot(1,0) << "," << frontLeftPosInOddlot(1,1) << "," << frontLeftPosInOddlot(1,2) << "," << frontLeftPosInOddlot(1,3) << std::endl;
		std::cout << frontLeftPosInOddlot(2,0) << "," << frontLeftPosInOddlot(2,1) << "," << frontLeftPosInOddlot(2,2) << "," << frontLeftPosInOddlot(2,3) << std::endl;
		std::cout << frontLeftPosInOddlot(3,0) << "," << frontLeftPosInOddlot(3,1) << "," << frontLeftPosInOddlot(3,2) << "," << frontLeftPosInOddlot(3,3) << std::endl;
		std::cout << "---"<< std::endl;
		
		osg::Matrix frontLeftSusp;
		frontLeftSusp.makeTranslate(1.5,1,+0.3-0.4);
		
		osg::Matrix frontLeftSuspPos = frontLeftSusp * Car2OpencoverRotation * rotMatrix * cogPos;
		
		std::cout << frontLeftSuspPos(0,0) << "," << frontLeftSuspPos(0,1) << "," << frontLeftSuspPos(0,2) << "," << frontLeftSuspPos(0,3) << std::endl;
		std::cout << frontLeftSuspPos(1,0) << "," << frontLeftSuspPos(1,1) << "," << frontLeftSuspPos(1,2) << "," << frontLeftSuspPos(1,3) << std::endl;
		std::cout << frontLeftSuspPos(2,0) << "," << frontLeftSuspPos(2,1) << "," << frontLeftSuspPos(2,2) << "," << frontLeftSuspPos(2,3) << std::endl;
		std::cout << frontLeftSuspPos(3,0) << "," << frontLeftSuspPos(3,1) << "," << frontLeftSuspPos(3,2) << "," << frontLeftSuspPos(3,3) << std::endl;
		std::cout << "---"<< std::endl;
		
		double dist = sqrt((frontLeftPos(3,0)-frontLeftSuspPos(3,0))*(frontLeftPos(3,0)-frontLeftSuspPos(3,0))+(frontLeftPos(3,1)-frontLeftSuspPos(3,1))*(frontLeftPos(3,1)-frontLeftSuspPos(3,1))+(frontLeftPos(3,2)-frontLeftSuspPos(3,2))*(frontLeftPos(3,2)-frontLeftSuspPos(3,2)));
		std::cout << "dist: " << dist << std::endl;
		
		tireContactPoint.makeTranslate(1.5,1,+0.3-0.4-0.1);
		tireContactPoint = tireContactPoint * Car2OpencoverRotation * rotMatrix * cogPos;
		
		osg::Matrix tempMatrix = tireContactPoint * Opencover2OddlotRotation;
		osg::Vec3d tempVec = tempMatrix.getTrans();
		Vector3D searchInVec(tempVec.x(), tempVec.y(), tempVec.z());
		std::cout << "1" << std::endl;
		//std::cout << "fl1 in" << std:: endl << "pointx" << tempVec.x() << std::endl << "pointy" << tempVec.y() << std::endl << "pointz" << tempVec.z() << std::endl;
		std:;cout << "currentRoad[0]: " << currentRoad[0]->getLength() << " currentLongPos: " << currentLongPos[0] << std::endl;
		Vector2D searchOutVec = RoadSystem::Instance()->searchPositionFollowingRoad(searchInVec, currentRoad[0], currentLongPos[0]);
		//std::cout << "fl2 out" << std::endl << "pointx" << searchOutVec.x() << std::endl << "pointy" << searchOutVec.y() << std::endl;
		//currentLongPosFL1 = searchOutVec.x();
		std::cout << "2" << std::endl;
		RoadPoint point = currentRoad[0]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
		std::cout << "3" << std::endl;
		if (!isnan(point.x()))
		{
			//carState.roadHeightFL1 = point.z();
			std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
			tireDist = sqrt((tempMatrix(3,0)-point.x())*(tempMatrix(3,0)-point.x())+(tempMatrix(3,1)-point.y())*(tempMatrix(3,1)-point.y())+(tempMatrix(3,2)-point.z())*(tempMatrix(3,2)-point.z()));
		} else 
		{
			std::cout << "tire fl1 left road!" << std::endl;
			leftRoad = true;
		}
		std::cout << "tireDist: " << tireDist << std::endl;*/
		/*
		double motorOffsetXFL = -1.2;
		double motorOffsetYFL = 0;
		double motorOffsetZFL = -0.5;
		double motorOffsetXFR = 0.3;
		double motorOffsetYFR = 0;
		double motorOffsetZFR = -0.5;
		double motorOffsetXR = -0.45;
		double motorOffsetYR = 0;
		double motorOffsetZR = 0.75;
		
		double l = sqrt(rotMatrix(2,2) * rotMatrix(2,2) + rotMatrix(2,0) * rotMatrix(2,0));
		//double yawAngle = atan(rotMatrix(2,2) / rotMatrix(2,0)) - M_PI / 2;
		double yawAngle;
		if(rotMatrix(2,0) > 0)
		{
			yawAngle = asin(rotMatrix(2,2) / l) - M_PI / 2;
		}
		else 
		{
			yawAngle = -(asin(rotMatrix(2,2) / l) - M_PI / 2);
		}
		osg::Matrix carAngle;
		carAngle.makeRotate(yawAngle,0,1,0);
		carAngle = rotMatrix * carAngle;
		
		double mpLZ = (-motorOffsetXFL * carAngle(1,0) - motorOffsetZFL * carAngle(1,2)) / carAngle(1,1);
		double mpRZ = (-motorOffsetXFR * carAngle(1,0) - motorOffsetZFR * carAngle(1,2)) / carAngle(1,1);
		double mpBZ = (-motorOffsetXR * carAngle(1,0) - motorOffsetZR * carAngle(1,2)) / carAngle(1,1);
		std::cout << "yawAngle " << yawAngle << "mpLZ " << mpLZ << " mpRZ " << mpRZ << " mpBZ " << mpBZ << std::endl;
		std::cout << carAngle(0,0) << "," << carAngle(0,1) << "," << carAngle(0,2) << "," << carAngle(0,3) << std::endl;
		std::cout << carAngle(1,0) << "," << carAngle(1,1) << "," << carAngle(1,2) << "," << carAngle(1,3) << std::endl;
		std::cout << carAngle(2,0) << "," << carAngle(2,1) << "," << carAngle(2,2) << "," << carAngle(2,3) << std::endl;
		std::cout << carAngle(3,0) << "," << carAngle(3,1) << "," << carAngle(3,2) << "," << carAngle(3,3) << std::endl;*/
		/*double fGravValue = -4000;
		osg::Matrix fGrav;
		osg::Matrix inverseRotMatrix = rotMatrix;
		inverseRotMatrix = inverseRotMatrix.inverse(inverseRotMatrix);
		fGrav.makeTranslate(0,fGravValue,0);
		fGrav = fGrav * inverseRotMatrix * Opencover2CarRotation;
		std::cout << fGrav(0,0) << "," << fGrav(0,1) << "," << fGrav(0,2) << "," << fGrav(0,3) << std::endl;
		std::cout << fGrav(1,0) << "," << fGrav(1,1) << "," << fGrav(1,2) << "," << fGrav(1,3) << std::endl;
		std::cout << fGrav(2,0) << "," << fGrav(2,1) << "," << fGrav(2,2) << "," << fGrav(2,3) << std::endl;
		std::cout << fGrav(3,0) << "," << fGrav(3,1) << "," << fGrav(3,2) << "," << fGrav(3,3) << std::endl;*/
		
		
		/*std::cout << "=(^-.-^)="<< std::endl;*/
		vehicle->setVRMLVehicle(chassisTrans);
		vehicle->setVRMLVehicleBody(bodyTrans);
		
		osg::Matrix relRotTireYaw;
		relRotTireYaw.makeRotate(steering, 0, 1, 0);
		
		vehicle->setVRMLVehicleFrontWheels(relRotTireYaw, relRotTireYaw);
		vehicle->setVRMLVehicleRearWheels(idTrans, idTrans);
		
		//currentRoad[0] = NULL;
		osg::Vec3d tempVec = chassisTrans.getTrans();
		Vector3D searchInVec(tempVec.x(), -tempVec.z(), tempVec.y());
		//Vector3D searchInVec(8, 13, 0);
		//currentLongPos[0] = 50;
		//Vector2D v_c = RoadSystem::Instance()->searchPositionFollowingRoad(searchInVec, currentRoad[0], currentLongPos[0]);
		if(currentRoad[0])
		{
			Vector2D v_c = currentRoad[0]->searchPositionNoBorder(searchInVec, currentLongPos[0]);
			//std::cout << "search road from scratch" << std::endl;
			if (!v_c.isNaV())
			{
				currentLongPos[0] = v_c.x();
				RoadPoint point = currentRoad[0]->getRoadPoint(v_c.u(), v_c.v());
				roadPoint.makeTranslate(point.x(), point.z(), -point.y());
				/*globalPos.makeTranslate(point.x(), point.z(), -point.y());
				state.psi = 0;
				rotationPos.makeRotate(0, 0, 1, 0);*/
				leftRoad = false;
			}
			else
			{
				leftRoad = true;
			}
		}
		else 
		{
			leftRoad = true;
		}
		
	}
	
	
	
	
	osg::Vec3d testVec = globalPos.getTrans();
	printCounter++;
	if (printCounter > printMax) 
	{
		cout << "chassis trans: " << endl << "x: " << chassisTrans.getTrans().x() << endl << "y: " << chassisTrans.getTrans().y() << endl << "z: " << chassisTrans.getTrans().z() << endl;
		cout << "globalPos: " << endl << "x: " << globalPos.getTrans().x() << endl << "y: " << globalPos.getTrans().y() << endl << "z: " << globalPos.getTrans().z() << endl;
		cout << "road point: " << endl << "x: " << roadPoint.getTrans().x() << endl << "y: " << roadPoint.getTrans().y() << endl << "z: " << roadPoint.getTrans().z() << endl;
		cout << "current long pos: " << currentLongPos[0] << endl;
		/*cout << "state.fy: " << state.fy<< endl;
		cout << "state.vY: " << state.vY<< endl;
		cout << "state.fx: " << state.fx<< endl;
		cout << "state.vX: " << state.vX<< endl;
		cout << "state.deltaPsi: " << state.deltaPsi << endl;
		cout << "psi " << state.psi << endl;
		cout << "state.deltaX: " << state.deltaX << endl;
		cout << "state.deltaY: " << state.deltaY << endl;*/
		cout << "road: " << currentRoad[0] << endl;
		cout << "leftRoad " << leftRoad << endl;
		cout << "__/|___/|___/|___/|___/|" << endl;
		
		printCounter = 0;
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
