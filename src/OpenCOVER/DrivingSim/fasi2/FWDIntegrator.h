#ifndef __FWDIntegrator_h
#define __FWDintegrator_h

#include "FWDCarState.h"
#include "FWDState.h"
#include "FWDAccelerationState.h"
#include "FWDPositionState.h"

class FWDIntegrator
{
public:
	FWDIntegrator();
	FWDState integrate(FWDState inSpeedState, FWDState inPosState, FWDCarState carState, double dT);
};

#endif