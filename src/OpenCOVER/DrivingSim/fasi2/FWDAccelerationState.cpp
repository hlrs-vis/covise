#include "FWDAccelerationState.h"
#include <cmath> 

FWDAccelerationState::FWDAccelerationState()
{
	aX = 0;
	aY = 0;
	aZ = 0;
	aYaw = 0;
	aRoll = 0;
	aPitch = 0;
	aSuspZFL = 0;
	aSuspZFR = 0;
	aSuspZRR = 0;
	aSuspZRL = 0;
	aOmegaYFL = 0;
	aOmegaYFR = 0;
	aOmegaYRR = 0;
	aOmegaYRL = 0;
	aOmegaZFL = 0;
	aOmegaZFR = 0;
	phiDotFL1 = 0;
	phiDotFL2 = 0;
	phiDotFL3 = 0;
	phiDotFR1 = 0;
	phiDotFR2 = 0;
	phiDotFR3 = 0;
	phiDotRR1 = 0;
	phiDotRR2 = 0;
	phiDotRR3 = 0;
	phiDotRL1 = 0;
	phiDotRL2 = 0;
	phiDotRL3 = 0;
	aEngineRPM = 0;
	TcolumnCombined = 0;
	Tclutch = 0;
	TclutchMax = 0;
	slipFL = 0;
	slipFR = 0;
	slipRR = 0;
	slipRL = 0;
	FweightedFL = 0;
	FweightedFR = 0;
	FweightedRR = 0;
	FweightedRL = 0;
	FtireFL = 0;
	FtireFR = 0;
	FtireRR = 0;
	FtireRL = 0;
	FxFL = 0;
	FxFR = 0;
	FxRR = 0;
	FxRL = 0;
	FyFL = 0;
	FyFR = 0;
	FyRR = 0;
	FyRL = 0;
	genericOut1 = 0;
	genericOut2 = 0;
	genericOut3 = 0;
	genericOut4 = 0;
	genericOut5 = 0;
	genericOut6 = 0;
	genericOut7 = 0;
	genericOut8 = 0;
}