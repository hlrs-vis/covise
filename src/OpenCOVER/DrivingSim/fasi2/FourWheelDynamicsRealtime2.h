/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FourWheelDynamicsRealtime2_h
#define __FourWheelDynamicsRealtime2_h

#include "FWDCarState.h"
#include "FWDState.h"
#include "FWDIntegrator.h"
#include "../VehicleUtil/gealg/CarDynamics.h"
#include "XenomaiTask.h"
#include "ValidateMotionPlatform.h"
#include "CanOpenController.h"
#include "XenomaiSteeringWheel.h"
#include "RoadPointFinder.h"
#include <list>

#include "RoadSystem/RoadSystem.h"
#include "KLSM.h"

class FourWheelDynamicsRealtime2 : public XenomaiTask
{
public:
    FourWheelDynamicsRealtime2();
    ~FourWheelDynamicsRealtime2();
	
	FWDCarState carState;
	
	osg::Matrix Car2OddlotRotation;
	osg::Matrix Oddlot2CarRotation;
	osg::Matrix Oddlot2OpencoverRotation;
	osg::Matrix Opencover2OddlotRotation;
	osg::Matrix Car2OpencoverRotation;
	osg::Matrix Opencover2CarRotation;
	
    void move();
    void initState();
    void resetState();

    double getVelocity()
    {
		return abs(speedState.vX);
    }

    virtual double getEngineTorque()
    {
        return carState.Tcomb;
    }
    virtual double getTyreSlipFL()
    {
        if(std::abs(speedState.slipFL) < carState.slipSoundLimit)
		{
			return std::abs(speedState.slipFL) / carState.slipSoundLimit;
		} else
		{
			return 1.0;
		}
    }
    virtual double getTyreSlipFR()
    {
        if(std::abs(speedState.slipFR) < carState.slipSoundLimit)
		{
			return std::abs(speedState.slipFR) / carState.slipSoundLimit;
		} else
		{
			return 1.0;
		}
    }
    virtual double getTyreSlipRL()
    {
        if(std::abs(speedState.slipRL) < carState.slipSoundLimit)
		{
			return std::abs(speedState.slipRL) / carState.slipSoundLimit;
		} else
		{
			return 1.0;
		}
    }
    virtual double getTyreSlipRR()
    {
        if(std::abs(speedState.slipRR) < carState.slipSoundLimit)
		{
			return std::abs(speedState.slipRR) / carState.slipSoundLimit;
		} else 
		{
			return 1.0;
		}
    }
    
    virtual void setTyreSlipFL(double inSlip)
	{
		speedState.slipFL = inSlip;
	}
	virtual void setTyreSlipFR(double inSlip)
	{
		speedState.slipFR = inSlip;
	}
	virtual void setTyreSlipRR(double inSlip)
	{
		speedState.slipRR = inSlip;
	}
	virtual void setTyreSlipRL(double inSlip)
	{
		speedState.slipRL = inSlip;
	}
	
    virtual double getAcceleration()
    {
        return 0.0;
    }
    virtual double getEngineSpeed()
    {
        return rpms;
    }

    virtual void setEngineSpeed(double inRPM)
    {
        rpms = inRPM;
    }
    
    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }
    
    virtual double getShiftParameter()
    {
        return carState.tireRadR * 2 * M_PI / (carState.gearRatio * carState.finalDrive);
    }
    void setVehicleTransformation(const osg::Matrix &m);

    void platformToGround();
    void platformMiddleLift();
    void platformReturnToAction();
    void centerWheel();
    void setSportDamper(bool);

    double getSteerWheelAngle()
    {
        return steerWheelAngle;
    }

    std::pair<Road *, Vector2D> getStartPositionOnRoad();
    virtual const osg::Matrix &getVehicleTransformation()
    {
        return chassisTrans;
    };
	
	double getRoadHeight(osg::Matrix inMatrix, Road currentRoad, double currentLongPos);

protected:
    void run();

    void determineGroundPlane();
    bool newIntersections;

    unsigned long overruns;
    static const RTIME period = 1000000;
    bool runTask;
    bool taskFinished;
    bool pause;

    ValidateMotionPlatform *motPlat;
    bool returningToAction;
    bool movingToGround;
    bool doCenter;

    CanOpenController *steerCon;
    XenomaiSteeringWheel *steerWheel;
	RoadPointFinder *roadPointFinder;
	
    double steerPosition;
    int32_t steerSpeed;
    double steerWheelAngle;
    double rpms;

    osg::Matrix chassisTrans;

    osg::Quat wheelQuatFL;
    osg::Quat wheelQuatFR;
    osg::Quat wheelQuatRL;
    osg::Quat wheelQuatRR;

    cardyn::StateVectorType y;
    cardyn::ExpressionVectorType dy;
    

    //gealg::mv<1, 0x0f>::type i_proj;

    std::vector<gealg::mv<3, 0x040201>::type> r_i;
    std::vector<gealg::mv<3, 0x040201>::type> n_i;
    std::vector<gealg::mv<3, 0x040201>::type> r_n;
    std::vector<gealg::mv<3, 0x040201>::type> t_n;
    std::vector<gealg::mv<3, 0x040201>::type> r_o;
    std::vector<gealg::mv<3, 0x040201>::type> t_o;
    std::vector<gealg::mv<3, 0x040201>::type> i_w;
    cardyn::StateVectorType y_frame;
    gealg::mv<6, 0x060504030201LL>::type getRoadSystemContactPoint(const gealg::mv<3, 0x040201>::type &, Road *&, double &);

    Road *currentRoad;
	Road *currentRoadFL1;
	Road *currentRoadFL2;
	Road *currentRoadFL3;
	
	Road *currentRoadFR1;
	Road *currentRoadFR2;
	Road *currentRoadFR3;
	
	Road *currentRoadRR1;
	Road *currentRoadRR2;
	Road *currentRoadRR3;
	
	Road *currentRoadRL1;
	Road *currentRoadRL2;
	Road *currentRoadRL3;
	
	Road *currentRoadArray[12];
	
	double currentLongPos;
    double currentLongPosFL1;
	double currentLongPosFL2;
	double currentLongPosFL3;
	
	double currentLongPosFR1;
	double currentLongPosFR2;
	double currentLongPosFR3;
	
    double currentLongPosRR1;
	double currentLongPosRR2;
	double currentLongPosRR3;
	
	double currentLongPosRL1;
	double currentLongPosRL2;
	double currentLongPosRL3;
	
	double  currentLongPosArray[12];

    std::pair<Road *, Vector2D> startPos;
    bool leftRoad;
    double k_wf_Slider;
    double d_wf_Slider;
    double k_wr_Slider;
    double d_wr_Slider;
    double clutchPedal;
	
	//timing
	double lastTicks;
	double timerCounter = 0;
	double time = 0;
	
	//added code
	FWDState speedState;
	FWDIntegrator integrator;
	double  contactPatch = 0.1;
	
	
	bool xodrLoaded = false;
	bool printedOnce = false;
	int printCounter = 0;
	int printMax = 1;
};
#endif
