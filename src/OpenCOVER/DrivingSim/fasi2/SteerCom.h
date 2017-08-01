#ifndef __SteerCom_h
#define __SteerCom_h

#include "XenomaiTask.h"
#include "CanOpenController.h"
#include "XenomaiSteeringWheel.h"

class SteerCom : public XenomaiTask
{
public:
	SteerCom();
	~SteerCom();
	
	void setCurrent(double inCurrent);
	double getSteeringWheelAngle();
	void centerCall();
	
protected:
	void run();
	unsigned long overruns;
	static const RTIME period = 1000000;
	bool runTask;
	bool taskFinished;
	
	CanOpenController *steerCon;
    XenomaiSteeringWheel *steerWheel;
	
	double current;
	double steerWheelAngle;
	
};

#endif