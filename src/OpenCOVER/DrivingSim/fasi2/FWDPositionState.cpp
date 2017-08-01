#include "FWDPositionState.h"
#include <cmath> 

FWDPositionState::FWDPositionState()
{
	dX = 0;
	dY = 0;
	dZ = 0;
	dYaw = 0;
	dRoll = 0;
	dPitch = 0;
	
	localZPosSuspFL = 0;
	localZPosSuspFR = 0;
	localZPosSuspRR = 0;
	localZPosSuspRL = 0;
	
	phiFL1 = 0;
	phiFL2 = 0;
	phiFL3 = 0;
	phiFR1 = 0;
	phiFR2 = 0;
	phiFR3 = 0;
	phiRR1 = 0;
	phiRR2 = 0;
	phiRR3 = 0;
	phiRL1 = 0;
	phiRL2 = 0;
	phiRL3 = 0;
}