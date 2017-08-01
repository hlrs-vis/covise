#include "FWDState.h"
#include <cmath> 

FWDState::FWDState()
{
	vX = 0;
	vY = 0;
	vZ = 0;
	vYaw = 0;
	vRoll = 0;
	vPitch = 0;
	vSuspZFL = 0;
	vSuspZFR = 0;
	vSuspZRR = 0;
	vSuspZRL = 0;
	OmegaYFL = 0;
	OmegaYFR = 0;
	OmegaYRR = 0;
	OmegaYRL = 0;
	OmegaZFL = 0;
	OmegaZFR = 0;
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
	engineRPM = 0;
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

void FWDState::limitSpeeds()
	{
		double speedLimit = 50;
		double vxSpeedLimit = 100;
		double rotLimit = 10;
		if(std::abs(vX)>vxSpeedLimit)
		{
			if(vX>vxSpeedLimit)
			{
				vX=vxSpeedLimit;
			}else
			{
				vX=-vxSpeedLimit;
			}	
		}
		if(std::abs(vY)>speedLimit)
		{
			if(vY>speedLimit)
			{
				vY=speedLimit;
			}else
			{
				vY=-speedLimit;
			}	
		}
		if(std::abs(vZ)>speedLimit)
		{
			if(vZ>speedLimit)
			{
				vZ=speedLimit;
			}else
			{
				vZ=-speedLimit;
			}	
		}
		if(std::abs(vYaw)>rotLimit)
		{
			if(vYaw>rotLimit)
			{
				vYaw=rotLimit;
			}else
			{
				vYaw=-rotLimit;
			}	
		}
		if(std::abs(vRoll)>rotLimit)
		{
			if(vRoll>rotLimit)
			{
				vRoll=rotLimit;
			}else
			{
				vRoll=-rotLimit;
			}	
		}
		if(std::abs(vPitch)>rotLimit)
		{
			if(vPitch>rotLimit)
			{
				vPitch=rotLimit;
			}else
			{
				vPitch=-rotLimit;
			}	
		}
		if(std::abs(vSuspZFL)>speedLimit)
		{
			if(vSuspZFL>speedLimit)
			{
				vSuspZFL=speedLimit;
			}else
			{
				vSuspZFL=-speedLimit;
			}	
		}
		if(std::abs(vSuspZFR)>speedLimit)
		{
			if(vSuspZFR>speedLimit)
			{
				vSuspZFR=speedLimit;
			}else
			{
				vSuspZFR=-speedLimit;
			}	
		}
		if(std::abs(vSuspZRR)>speedLimit)
		{
			if(vSuspZRR>speedLimit)
			{
				vSuspZRR=speedLimit;
			}else
			{
				vSuspZRR=-speedLimit;
			}	
		}
		if(std::abs(vSuspZRL)>speedLimit)
		{
			if(vSuspZRL>speedLimit)
			{
				vSuspZRL=speedLimit;
			}else
			{
				vSuspZRL=-speedLimit;
			}	
		}
		/*if(std::abs(OmegaYFL)>speedLimit)
		{
			if(OmegaYFL>speedLimit)
			{
				OmegaYFL=speedLimit;
			}else
			{
				OmegaYFL=-speedLimit;
			}	
		}
		if(std::abs(OmegaYFR)>speedLimit)
		{
			if(OmegaYFR>speedLimit)
			{
				OmegaYFR=speedLimit;
			}else
			{
				OmegaYFR=-speedLimit;
			}	
		}
		if(std::abs(OmegaYRR)>speedLimit)
		{
			if(OmegaYRR>speedLimit)
			{
				OmegaYRR=speedLimit;
			}else
			{
				OmegaYRR=-speedLimit;
			}	
		}
		if(std::abs(OmegaYRL)>speedLimit)
		{
			if(OmegaYRL>speedLimit)
			{
				OmegaYRL=speedLimit;
			}else
			{
				OmegaYRL=-speedLimit;
			}	
		}*/
		if(std::abs(OmegaZFL)>speedLimit)
		{
			if(OmegaZFL>speedLimit)
			{
				OmegaZFL=speedLimit;
			}else
			{
				OmegaZFL=-speedLimit;
			}	
		}
		if(std::abs(OmegaZFR)>speedLimit)
		{
			if(OmegaZFR>speedLimit)
			{
				OmegaZFR=speedLimit;
			}else
			{
				OmegaZFR=-speedLimit;
			}	
		}
	}
void FWDState::threshold() 
	{
		double thresholdValue = 0.00001;
		if (std::abs(vX) < thresholdValue) {vX = 0;}
		if (std::abs(vY) < thresholdValue){vY = 0;}
		if (std::abs(vZ) < thresholdValue){vZ = 0;}
		if (std::abs(vYaw) < thresholdValue) {vYaw = 0;}
		if (std::abs(vRoll) < thresholdValue) {vRoll = 0;}
		if (std::abs(vPitch) < thresholdValue) {vPitch = 0;}
		if (std::abs(vSuspZFL) < thresholdValue) {vSuspZFL = 0;}
		if (std::abs(vSuspZFR) < thresholdValue) {vSuspZFR = 0;}
		if (std::abs(vSuspZRR) < thresholdValue) {vSuspZRR = 0;}
		if (std::abs(vSuspZRL) < thresholdValue) {vSuspZRL = 0;}
		if (std::abs(OmegaYFL) < thresholdValue) {OmegaYFL = 0;}
		if (std::abs(OmegaYFR) < thresholdValue) {OmegaYFR = 0;}
		if (std::abs(OmegaYRR) < thresholdValue) {OmegaYRR = 0;}
		if (std::abs(OmegaYRL) < thresholdValue) {OmegaYRL = 0;}
		if (std::abs(OmegaZFL) < thresholdValue) {OmegaZFL = 0;}
		if (std::abs(OmegaZFR) < thresholdValue) {OmegaZFR = 0;}
	}