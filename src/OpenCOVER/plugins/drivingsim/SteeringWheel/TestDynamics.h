#ifndef __TestDynamics_H
#define __TestDynamics_H

#include "Vehicle.h"
#include "RemoteVehicle.h"
#include "VehicleDynamics.h"

#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include <VehicleUtil/RoadSystem/Types.h>
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
	
	std::vector<Road*> roadList[4];
	std::string currentRoadName;
	int currentRoadId;
	double currentHeight;
	double roadHeightIncrement;
	double roadHeightIncrementInit;
	double roadHeightIncrementDelta;
	double roadHeightIncrementMax;
	double roadHeightDelta;
	
	bool singleRoadSwitch;
	
	double targetS;
	
private:
	TestState state;
	TestState stateOut;
	double accelerator;
	double steering;
	double brake;
	double mass;
	double cAero;
	double mu;
	double lateralMu;
	double enginePower;
	double brakePower;
	double g;
	double inertia;
	//F = Fz · D · sin(C · arctan(B·slip - E · (B·slip - arctan(B·slip))))
	double Bf; //10=tarmac; 4=ice
	double Cf; //~2
	double Df; //1=tarmace; 0.1=ice
	double Ef; //0.97=tarmac; 1=ice
	double Br; //10=tarmac; 4=ice
	double Cr; //~2
	double Dr; //1=tarmace; 0.1=ice
	double Er; //0.97=tarmac; 1=ice
	double a1;
	double a2;
	double vXLimit;
	double vYLimit;
	double vPsiLimit;
	/*double damping = 60;
	double springRate = 60;
	double cogHeight = 0.1;*/
	double powerDist; //1=RWD
	double brakeDist;
	double steeringRatio;
	double frictionCircleLimit;
	double integrationSteps;
	bool xodrLoaded;
	bool printedOnce;
	int printCounter;
	int printMax;
	
	osg::Matrix Car2OddlotRotation;
	osg::Matrix Oddlot2CarRotation;
	osg::Matrix Oddlot2OpencoverRotation;
	osg::Matrix Opencover2OddlotRotation;
	osg::Matrix Car2OpencoverRotation;
	osg::Matrix Opencover2CarRotation;
	
	osg::Matrix globalSpeedMatrix;
	osg::Matrix globalPos;
	osg::Matrix rotationPos;
	osg::Matrix cogPos;
	osg::Matrix rotMatrix;
	osg::Matrix tireContactPoint;
	double tireDist;
};

#endif
