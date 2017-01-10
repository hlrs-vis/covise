#ifndef __TestDynamics_H
#define __TestDynamics_H

#include "Vehicle.h"
#include "RemoteVehicle.h"
#include "VehicleDynamics.h"

#include "RoadSystem/RoadSystem.h"
#include "RoadSystem/Types.h"
#include "TestDynamicsUtil.h"

#include <iostream>
#include <math.h>

class TestState
{
public: 
	double deltaX;
	double deltaY;
	double deltaPsi;
	double vX;
	double vY;
	double vPsi;
	double aX;
	double aY;
	double aPsi;
	double X;
	double Y;
	double Z;
	double psi;
	double fxfl;
	double fxfr;
	double fxrr;
	double fxrl;
	double fyfl;
	double fyfr;
	double fyrr;
	double fyrl;
	double fx;
	double fy;
	double fdrag;
	double frollfl;
	double frollfr;
	double frollrr;
	double frollrl;
	double fbrakefl;
	double fbrakefr;
	double fbrakerr;
	double fbrakerl;
	double fdrivefl;
	double fdrivefr;
	double fdriverr;
	double fdriverl;
	double fengine;
	double fbrake;
	double mz;
	double betaf;
	double betar;
	double delta;
};
class MovementState
{
	
	public:
		double vX;
		double vY;
		double vPsi;
		/*double getVX(){return vX;};
		double getVY(){return vY;};
		double getVPsi(){return vPsi;};
		void setVX(double vX){ vX=vX;};
		void setVY(double vY){ vY=vY;};
		void setVPsi(double vPsi){ vPsi=vPsi;};*/
};

class TestDynamics : public VehicleDynamics
{
public:
	TestDynamics();
	
	void initState();
	double getVelocity(){ return abs(state.vX);};
	void timeStep(double dT);
	MovementState deltaFunction(double inputArray[], double dT);
	
	void move(VrmlNodeVehicle *vehicle);
	/*double getRoadHeight(VrmlNodeVehicle *vehicle);*/
	
	void resetState();

	const osg::Matrix &getVehicleTransformation(){ return chassisTrans;};
	virtual void setVehicleTransformation(const osg::Matrix &);
	osg::Matrix chassisTrans;
	osg::Matrix bodyTrans;
	
	std::pair<Road *, double> getStartPositionOnRoad();
	

protected:
	Road *currentRoad[4];
	double currentLongPos[4];
	std::pair<Road *, double> startPos;
	bool leftRoad;
	
	double targetS = 3.0;
	
private:
	TestState state;
	TestState stateOut;
	double accelerator;
	double steering;
	double brake;
	double mass = 1900;
	double cAero = 0.00002;
	double mu = 0.005;
	double lateralMu = 500;
	double enginePower = 20000;
	double brakePower = 8000;
	double g = 9.81;
	double inertia = 5000;
	//F = Fz · D · sin(C · arctan(B·slip - E · (B·slip - arctan(B·slip))))
	double Bf = 7.5; //10=tarmac; 4=ice
	double Cf = 1.9; //~2
	double Df = 0.8; //1=tarmace; 0.1=ice
	double Ef = 0.99; //0.97=tarmac; 1=ice
	double Br = 6; //10=tarmac; 4=ice
	double Cr = 1.9; //~2
	double Dr = 0.6; //1=tarmace; 0.1=ice
	double Er = 0.97; //0.97=tarmac; 1=ice
	double a1 = 1.6;
	double a2 = 1.65;
	double vXLimit = 0.001;
	double vYLimit = 0.01;
	double vPsiLimit = 0.001;
	/*double damping = 60;
	double springRate = 60;
	double cogHeight = 0.1;*/
	double powerDist = 0.8; //1=RWD
	double brakeDist = 0.5;
	double steeringRatio = 0.15;
	double frictionCircleLimit = 6000;
	double integrationSteps = 5.0;
	bool xodrLoaded = false;
	bool printedOnce = false;
	int printCounter = 0;
	int printMax = 1;
	osg::Matrix globalPos;
	osg::Matrix rotationPos;
};

#endif
