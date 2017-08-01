#ifndef __FWDPositionState_h
#define __FWDPositionState_h

class FWDPositionState
{
public:
	FWDPositionState();
	
	double dX;
	double dY;
	double dZ;
	double dYaw;
	double dRoll;
	double dPitch;
	
	double localZPosSuspFL;
	double localZPosSuspFR;
	double localZPosSuspRR;
	double localZPosSuspRL;
	
	double phiFL1;
	double phiFL2;
	double phiFL3;
	double phiFR1;
	double phiFR2;
	double phiFR3;
	double phiRR1;
	double phiRR2;
	double phiRR3;
	double phiRL1;
	double phiRL2;
	double phiRL3;
};
#endif
