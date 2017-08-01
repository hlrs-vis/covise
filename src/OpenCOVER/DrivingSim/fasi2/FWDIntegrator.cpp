#include "FWDIntegrator.h"

#include "FWDCarState.h"

#include <iostream>

FWDIntegrator::FWDIntegrator()
{
}

FWDState FWDIntegrator::integrate(FWDState inSpeedState, FWDState inPosState, FWDCarState carState, double dT)
{
	double initialVX = inSpeedState.vX;
	double initialVY = inSpeedState.vY;
	double initialVZ = inSpeedState.vZ;
	double initialVYaw = inSpeedState.vYaw;
	double initialVRoll = inSpeedState.vRoll;
	double initialVPitch = inSpeedState.vPitch;
	double initialVSuspZFL = inSpeedState.vSuspZFL;
	double initialVSuspZFR = inSpeedState.vSuspZFR;
	double initialVSuspZRR = inSpeedState.vSuspZRR;
	double initialVSuspZRL = inSpeedState.vSuspZRL;
	double initialOmegaYFL = inSpeedState.OmegaYFL;
	double initialOmegaYFR = inSpeedState.OmegaYFR;
	double initialOmegaYRR = inSpeedState.OmegaYRR;
	double initialOmegaYRL = inSpeedState.OmegaYRL;
	double initialOmegaZFL = inSpeedState.OmegaZFL;
	double initialOmegaZFR = inSpeedState.OmegaZFR;
	double initialOmegaZRR;
	double initialOmegaZRL;
	double initialRpm = inSpeedState.engineRPM;
	double initialPhiDotFL1 = inSpeedState.phiDotFL1;
	double initialPhiDotFL2 = inSpeedState.phiDotFL2;
	double initialPhiDotFL3 = inSpeedState.phiDotFL3;
	double initialPhiDotFR1 = inSpeedState.phiDotFR1;
	double initialPhiDotFR2 = inSpeedState.phiDotFR2;
	double initialPhiDotFR3 = inSpeedState.phiDotFR3;
	double initialPhiDotRR1 = inSpeedState.phiDotRR1;
	double initialPhiDotRR2 = inSpeedState.phiDotRR2;
	double initialPhiDotRR3 = inSpeedState.phiDotRR3;
	double initialPhiDotRL1 = inSpeedState.phiDotRL1;
	double initialPhiDotRL2 = inSpeedState.phiDotRL2;
	double initialPhiDotRL3 = inSpeedState.phiDotRL3;
	
	double FAxleLoadPitchR;
	double FAxleLoadPitchF;
	double FAxleLoadRollL;
	double FAxleLoadRollR;
	double FAxleLoadFL;
	double FAxleLoadFR;
	double FAxleLoadRR;
	double FAxleLoadRL;
	
	//wheel loads due to angle
	FAxleLoadPitchR=cos(carState.localPitch)*(carState.lFront+carState.cogH*tan(-carState.localPitch))/(carState.lFront+carState.lRear); //higher means more load on this side
	FAxleLoadPitchF=cos(carState.localPitch)*(carState.lRear-carState.cogH*tan(-carState.localPitch))/(carState.lFront+carState.lRear); //higher means more load on this side
	FAxleLoadRollL=cos(carState.localRoll)*((carState.sRearH+carState.sFrontH)/2+carState.cogH*tan(-carState.localRoll))/((carState.sRearH+carState.sFrontH)); //higher means more load on this side
	FAxleLoadRollR=cos(carState.localRoll)*((carState.sRearH+carState.sFrontH)/2-carState.cogH*tan(-carState.localRoll))/((carState.sRearH+carState.sFrontH)); //higher means more load on this side
	FAxleLoadFL=(2*carState.mTotal*carState.aGrav/2)*FAxleLoadPitchF*FAxleLoadRollL;
	FAxleLoadFR=(2*carState.mTotal*carState.aGrav/2)*FAxleLoadPitchF*FAxleLoadRollR;
	FAxleLoadRR=(2*carState.mTotal*carState.aGrav/2)*FAxleLoadPitchR*FAxleLoadRollR;
	FAxleLoadRL=(2*carState.mTotal*carState.aGrav/2)*FAxleLoadPitchR*FAxleLoadRollL;
	
	//suspension forces
	//-try with Fts = a1 * z + a2 * z * z;
	//==>Fts = c1 * (offset + roadHeight - suspHeight) + c2 * (offset + roadHeight * suspHeight)^2
	/* old version:
	double FssFL = (carState.jointOffsetFL + carState.globalPosSuspFL.getTrans().z() - carState.globalPosJointFL.getTrans().z())* carState.csFL + carState.FsFL;
	double FsdFL = (initialVSuspZFL - initialVZ + sin(initialVPitch) * carState.lFront - sin(initialVRoll) * carState.sFrontH) * carState.dsFL;
	double FtsFL = (carState.tireRadF + carState.roadHeightFL2 - carState.globalPosSuspFL.getTrans().z()) * carState.ctFL 
					+ (carState.tireRadF + carState.roadHeightFL2 - carState.globalPosSuspFL.getTrans().z()) 
					* (carState.tireRadF + carState.roadHeightFL2 - carState.globalPosSuspFL.getTrans().z()) * carState.ctFL + carState.FtFL;
	double FtdFL = -initialVSuspZFL * carState.dtFL;
	
	double FssFR = (carState.jointOffsetFR + carState.globalPosSuspFR.getTrans().z() - carState.globalPosJointFR.getTrans().z())* carState.csFR + carState.FsFR;
	double FsdFR = (initialVSuspZFR - initialVZ + sin(initialVPitch) * carState.lFront + sin(initialVRoll) * carState.sFrontH) * carState.dsFR;
	double FtsFR = (carState.tireRadF + carState.roadHeightFR2 - carState.globalPosSuspFR.getTrans().z()) * carState.ctFR 
					+ (carState.tireRadF + carState.roadHeightFR2 - carState.globalPosSuspFR.getTrans().z())
					* (carState.tireRadF + carState.roadHeightFR2 - carState.globalPosSuspFR.getTrans().z()) * carState.ctFR + carState.FtFR;
	double FtdFR = -initialVSuspZFR * carState.dtFR;
	
	double FssRR = (carState.jointOffsetRR + carState.globalPosSuspRR.getTrans().z() - carState.globalPosJointRR.getTrans().z())* carState.csRR + carState.FsRR;
	double FsdRR = (initialVSuspZRR - initialVZ - sin(initialVPitch) * carState.lRear + sin(initialVRoll) * carState.sRearH ) * carState.dsRR;
	double FtsRR = (carState.tireRadR + carState.roadHeightRR2 - carState.globalPosSuspRR.getTrans().z()) * carState.ctRR 
					+ (carState.tireRadR + carState.roadHeightRR2 - carState.globalPosSuspRR.getTrans().z())
					* (carState.tireRadR + carState.roadHeightRR2 - carState.globalPosSuspRR.getTrans().z()) * carState.ctRR + carState.FtRR;
	double FtdRR = -initialVSuspZRR * carState.dtRR;
	
	double FssRL = (carState.jointOffsetRL + carState.globalPosSuspRL.getTrans().z() - carState.globalPosJointRL.getTrans().z())* carState.csRL + carState.FsRL;
	double FsdRL = (initialVSuspZRL - initialVZ - sin(initialVPitch) * carState.lRear - sin(initialVRoll) * carState.sRearH ) * carState.dsRL;
	double FtsRL = (carState.tireRadR + carState.roadHeightRL2 - carState.globalPosSuspRL.getTrans().z()) * carState.ctRL 
					+ (carState.tireRadR + carState.roadHeightRL2 - carState.globalPosSuspRL.getTrans().z()) 
					* (carState.tireRadR + carState.roadHeightRL2 - carState.globalPosSuspRL.getTrans().z()) * carState.ctRL + carState.FtRL;
	double FtdRL = -initialVSuspZRL * carState.dtRL;
	*/
	
	double FssFL = (inPosState.vSuspZFL)* carState.csFL + carState.FsFL;
	double FsdFL = (initialVSuspZFL/* - initialVZ + sin(initialVPitch) * carState.lFront - sin(initialVRoll) * carState.sFrontH*/) * carState.dsFL;
	double FtsFL = -(carState.localZPosTireFL) * carState.ctFL/* + (carState.localZPosTireFL) * (carState.localZPosTireFL) * carState.ctFL + carState.FtFL*/;
	double FtdFL = -carState.tireDefSpeedFL * carState.dtFL;
	
	double FssFR = (inPosState.vSuspZFR)* carState.csFR + carState.FsFR;
	double FsdFR = (initialVSuspZFR/* - initialVZ + sin(initialVPitch) * carState.lFront + sin(initialVRoll) * carState.sFrontH*/) * carState.dsFR;
	double FtsFR = -(carState.localZPosTireFR) * carState.ctFR/* + (carState.localZPosTireFR) * (carState.localZPosTireFR) * carState.ctFR + carState.FtFR*/;
	double FtdFR = -carState.tireDefSpeedFR * carState.dtFR;
	
	double FssRR = (inPosState.vSuspZRR)* carState.csRR + carState.FsRR;
	double FsdRR = (initialVSuspZRR/* - initialVZ - sin(initialVPitch) * carState.lRear + sin(initialVRoll) * carState.sRearH*/) * carState.dsRR;
	double FtsRR = -(carState.localZPosTireRR) * carState.ctRR/* + (carState.localZPosTireRR) * (carState.localZPosTireRR) * carState.ctRR + carState.FtRR*/;
	double FtdRR = -carState.tireDefSpeedRR * carState.dtRR;
	
	double FssRL = (inPosState.vSuspZRL)* carState.csRL + carState.FsRL;
	double FsdRL = (initialVSuspZRL/* - initialVZ - sin(initialVPitch) * carState.lRear - sin(initialVRoll) * carState.sRearH*/) * carState.dsRL;
	double FtsRL = -(carState.localZPosTireRL) * carState.ctRL/* + (carState.localZPosTireRL) * (carState.localZPosTireRL) * carState.ctRL + carState.FtRL*/;
	double FtdRL = -carState.tireDefSpeedRL * carState.dtRL;
	
	if(FtsFL < 0)
	{
		FtsFL = 0;
	}
	/*if(FtdFL < 0)
	{
		FtdFL = 0;
	}*/
	
	if(FtsFR < 0)
	{
		FtsFR = 0;
	}
	/*if(FtdFR < 0)
	{
		FtdFR = 0;
	}*/
	
	if(FtsRR < 0)
	{
		FtsRR = 0;
	}
	/*if(FtdRR < 0)
	{
		FtdRR = 0;
	}*/
	
	if(FtsRL < 0)
	{
		FtsRL = 0;
	}
	/*if(FtdRL < 0)
	{
		FtdRL = 0;
	}*/
	
	//combined weighted forces
	double FweightedFL = FAxleLoadFL * (FssFL + FsdFL + FssFR + FsdFR) * (FssFL + FsdFL + FssRL + FsdRL) / ((FssRR + FsdRR + FssRL + FsdRL) * (FssFR + FsdFR + FssRR + FsdRR));
	if(isnan(FweightedFL))
	{
		FweightedFL = 0;
	}
	double FweightedFR = FAxleLoadFR * (FssFL + FsdFL + FssFR + FsdFR) * (FssFR + FsdFR + FssRR + FsdRR) / ((FssRR + FsdRR + FssRL + FsdRL) * (FssFL + FsdFL + FssRL + FsdRL));
	if(isnan(FweightedFR))
	{
		FweightedFR = 0;
	}
	double FweightedRR = FAxleLoadRR * (FssRR + FsdRR + FssRL + FsdRL) * (FssFR + FsdFR + FssRR + FsdRR) / ((FssFL + FsdFL + FssFR + FsdFR) * (FssFL + FsdFL + FssRL + FsdRL));
	if(isnan(FweightedRR))
	{
		FweightedRR = 0;
	}
	double FweightedRL = FAxleLoadRL * (FssRR + FsdRR + FssRL + FsdRL) * (FssFL + FsdFL + FssRL + FsdRL) / ((FssFL + FsdFL + FssFR + FsdFR) * (FssFR + FsdFR + FssRR + FsdRR));
	if(isnan(FweightedRL))
	{
		FweightedRL = 0;
	}
	
	//forces in Z on tire
	double FtireFL = FtsFL + FtdFL;
	double FtireFR = FtsFR + FtdFR;
	double FtireRR = FtsRR + FtdRR;
	double FtireRL = FtsRL + FtdRL;
	
	//TMEasy
	//contact patch length; use Fz from suspension equations, either Fweighted or Ftire
	double LFL = sqrt(4 * carState.tireRadF * std::abs(FtireFL) / carState.cR);
	double LFR = sqrt(4 * carState.tireRadF * std::abs(FtireFR) / carState.cR);
	double LRR = sqrt(4 * carState.tireRadR * std::abs(FtireRR) / carState.cR);
	double LRL = sqrt(4 * carState.tireRadR * std::abs(FtireRL) / carState.cR);
	
	
	//bore radius
	//double RPFL = 0.25 * (LFL + carState.B);
	//double RPFR = 0.25 * (LFR + carState.B);
	//double RPRR = 0.25 * (LRR + carState.B);
	//double RPRL = 0.25 * (LRL + carState.B);
	double RBFL = 0.25 * (LFL + carState.B) * 2 / 3;
	double RBFR = 0.25 * (LFR + carState.B) * 2 / 3;
	double RBRR = 0.25 * (LRR + carState.B) * 2 / 3;
	double RBRL = 0.25 * (LRL + carState.B) * 2 / 3;
	
	//torque around y, rolling resistance
	double TyFL = -carState.tireRadF * carState.fRoll * carState.d * initialOmegaYFL * FtireFL;
	double TyFR = -carState.tireRadF * carState.fRoll * carState.d * initialOmegaYFR * FtireFR;
	double TyRR = -carState.tireRadR * carState.fRoll * carState.d * initialOmegaYRR * FtireRR;
	double TyRL = -carState.tireRadR * carState.fRoll * carState.d * initialOmegaYRL * FtireRL;
	
	//coefficient for dynamic tire radius
	double lambdaFL = carState.lambdaN + (carState.lambda2N - carState.lambdaN) * (FtireFL/carState.FzN);
	double lambdaFR = carState.lambdaN + (carState.lambda2N - carState.lambdaN) * (FtireFR/carState.FzN);
	double lambdaRR = carState.lambdaN + (carState.lambda2N - carState.lambdaN) * (FtireRR/carState.FzN);
	double lambdaRL = carState.lambdaN + (carState.lambda2N - carState.lambdaN) * (FtireRL/carState.FzN);
	
	//vertical tire stiffness
	double czFL = carState.czN + (carState.cz2N - carState.czN) * (FtireFL/carState.FzN);
	double czFR = carState.czN + (carState.cz2N - carState.czN) * (FtireFR/carState.FzN);
	double czRR = carState.czN + (carState.cz2N - carState.czN) * (FtireRR/carState.FzN);
	double czRL = carState.czN + (carState.cz2N - carState.czN) * (FtireRL/carState.FzN);
	
	//dynamic tire radius
	double rDynFL = lambdaFL * carState.tireRadF + (1 - lambdaFL) * (carState.tireRadF - FtireFL/czFL);
	double rDynFR = lambdaFR * carState.tireRadF + (1 - lambdaFR) * (carState.tireRadF - FtireFR/czFR);
	double rDynRR = lambdaRR * carState.tireRadR + (1 - lambdaRR) * (carState.tireRadR - FtireRR/czRR);
	double rDynRL = lambdaRL * carState.tireRadR + (1 - lambdaRL) * (carState.tireRadR - FtireRL/czRL);
	
	//bore slip
	double sBFL = -RBFL * initialOmegaZFL / (rDynFL * std::abs(initialOmegaYFL) + carState.vN);
	double sBFR = -RBFR * initialOmegaZFR / (rDynFR * std::abs(initialOmegaYFR) + carState.vN);
	double sBRR = -RBRR * initialOmegaZRR / (rDynRR * std::abs(initialOmegaYRR) + carState.vN);
	double sBRL = -RBRL * initialOmegaZRL / (rDynRL * std::abs(initialOmegaYRL) + carState.vN);
	
	//parameters for slip curves
	//Fx_=Fz/FzN*(2*Fx_(FzN)-0.5*Fx_(2FzN)-(Fx_(FzN)-0.5*Fx_(2FzN))*Fz/FzN
	double FxMFL = FtireFL / carState.FzN * (2 * carState.FxMN - 0.5 * carState.FxM2N - (carState.FxMN - 0.5 * carState.FxM2N) * FtireFL / carState.FzN);
	double FxMFR = FtireFR / carState.FzN * (2 * carState.FxMN - 0.5 * carState.FxM2N - (carState.FxMN - 0.5 * carState.FxM2N) * FtireFR / carState.FzN);
	double FxMRR = FtireRR / carState.FzN * (2 * carState.FxMN - 0.5 * carState.FxM2N - (carState.FxMN - 0.5 * carState.FxM2N) * FtireRR / carState.FzN);
	double FxMRL = FtireRL / carState.FzN * (2 * carState.FxMN - 0.5 * carState.FxM2N - (carState.FxMN - 0.5 * carState.FxM2N) * FtireRL / carState.FzN);
	double dFx0FL = FtireFL / carState.FzN * (2 * carState.dFx0N - 0.5 * carState.dFx02N - (carState.dFx0N - 0.5 * carState.dFx02N) * FtireFL / carState.FzN);
	double dFx0FR = FtireFR / carState.FzN * (2 * carState.dFx0N - 0.5 * carState.dFx02N - (carState.dFx0N - 0.5 * carState.dFx02N) * FtireFR / carState.FzN);
	double dFx0RR = FtireRR / carState.FzN * (2 * carState.dFx0N - 0.5 * carState.dFx02N - (carState.dFx0N - 0.5 * carState.dFx02N) * FtireRR / carState.FzN);
	double dFx0RL = FtireRL / carState.FzN * (2 * carState.dFx0N - 0.5 * carState.dFx02N - (carState.dFx0N - 0.5 * carState.dFx02N) * FtireRL / carState.FzN);
	double FxGFL = FtireFL / carState.FzN * (2 * carState.FxGN - 0.5 * carState.FxG2N - (carState.FxGN - 0.5 * carState.FxG2N) * FtireFL / carState.FzN);
	double FxGFR = FtireFR / carState.FzN * (2 * carState.FxGN - 0.5 * carState.FxG2N - (carState.FxGN - 0.5 * carState.FxG2N) * FtireFR / carState.FzN);
	double FxGRR = FtireRR / carState.FzN * (2 * carState.FxGN - 0.5 * carState.FxG2N - (carState.FxGN - 0.5 * carState.FxG2N) * FtireRR / carState.FzN);
	double FxGRL = FtireRL / carState.FzN * (2 * carState.FxGN - 0.5 * carState.FxG2N - (carState.FxGN - 0.5 * carState.FxG2N) * FtireRL / carState.FzN);
	double FyMFL = FtireFL / carState.FzN * (2 * carState.FyMN - 0.5 * carState.FyM2N - (carState.FyMN - 0.5 * carState.FyM2N) * FtireFL / carState.FzN);
	double FyMFR = FtireFR / carState.FzN * (2 * carState.FyMN - 0.5 * carState.FyM2N - (carState.FyMN - 0.5 * carState.FyM2N) * FtireFR / carState.FzN);
	double FyMRR = FtireRR / carState.FzN * (2 * carState.FyMN - 0.5 * carState.FyM2N - (carState.FyMN - 0.5 * carState.FyM2N) * FtireRR / carState.FzN);
	double FyMRL = FtireRL / carState.FzN * (2 * carState.FyMN - 0.5 * carState.FyM2N - (carState.FyMN - 0.5 * carState.FyM2N) * FtireRL / carState.FzN);
	double dFy0FL = FtireFL / carState.FzN * (2 * carState.dFy0N - 0.5 * carState.dFy02N - (carState.dFy0N - 0.5 * carState.dFy02N) * FtireFL / carState.FzN);
	double dFy0FR = FtireFR / carState.FzN * (2 * carState.dFy0N - 0.5 * carState.dFy02N - (carState.dFy0N - 0.5 * carState.dFy02N) * FtireFR / carState.FzN);
	double dFy0RR = FtireRR / carState.FzN * (2 * carState.dFy0N - 0.5 * carState.dFy02N - (carState.dFy0N - 0.5 * carState.dFy02N) * FtireRR / carState.FzN);
	double dFy0RL = FtireRL / carState.FzN * (2 * carState.dFy0N - 0.5 * carState.dFy02N - (carState.dFy0N - 0.5 * carState.dFy02N) * FtireRL / carState.FzN);
	double FyGFL = FtireFL / carState.FzN * (2 * carState.FyGN - 0.5 * carState.FyG2N - (carState.FyGN - 0.5 * carState.FyG2N) * FtireFL / carState.FzN);
	double FyGFR = FtireFR / carState.FzN * (2 * carState.FyGN - 0.5 * carState.FyG2N - (carState.FyGN - 0.5 * carState.FyG2N) * FtireFR / carState.FzN);
	double FyGRR = FtireRR / carState.FzN * (2 * carState.FyGN - 0.5 * carState.FyG2N - (carState.FyGN - 0.5 * carState.FyG2N) * FtireRR / carState.FzN);
	double FyGRL = FtireRL / carState.FzN * (2 * carState.FyGN - 0.5 * carState.FyG2N - (carState.FyGN - 0.5 * carState.FyG2N) * FtireRL / carState.FzN);
	//sx_=sx_(FzN)+(sx_(2FzN)-sx_(FzN))*(Fz/FzN-1)
	double sxMFL = carState.sxMN + (carState.sxM2N - carState.sxMN) * (FtireFL / carState.FzN - 1);
	double sxMFR = carState.sxMN + (carState.sxM2N - carState.sxMN) * (FtireFR / carState.FzN - 1);
	double sxMRR = carState.sxMN + (carState.sxM2N - carState.sxMN) * (FtireRR / carState.FzN - 1);
	double sxMRL = carState.sxMN + (carState.sxM2N - carState.sxMN) * (FtireRL / carState.FzN - 1);
	//
	double sxGFL = carState.sxGN + (carState.sxG2N - carState.sxGN) * (FtireFL / carState.FzN - 1);
	double sxGFR = carState.sxGN + (carState.sxG2N - carState.sxGN) * (FtireFR / carState.FzN - 1);
	double sxGRR = carState.sxGN + (carState.sxG2N - carState.sxGN) * (FtireRR / carState.FzN - 1);
	double sxGRL = carState.sxGN + (carState.sxG2N - carState.sxGN) * (FtireRL / carState.FzN - 1);
	//
	double syMFL = carState.syMN + (carState.syM2N - carState.syMN) * (FtireFL / carState.FzN - 1);
	double syMFR = carState.syMN + (carState.syM2N - carState.syMN) * (FtireFR / carState.FzN - 1);
	double syMRR = carState.syMN + (carState.syM2N - carState.syMN) * (FtireRR / carState.FzN - 1);
	double syMRL = carState.syMN + (carState.syM2N - carState.syMN) * (FtireRL / carState.FzN - 1);
	//
	double syGFL = carState.syGN + (carState.syG2N - carState.syGN) * (FtireFL / carState.FzN - 1);
	double syGFR = carState.syGN + (carState.syG2N - carState.syGN) * (FtireFR / carState.FzN - 1);
	double syGRR = carState.syGN + (carState.syG2N - carState.syGN) * (FtireRR / carState.FzN - 1);
	double syGRL = carState.syGN + (carState.syG2N - carState.syGN) * (FtireRL / carState.FzN - 1);
	//combined slip parameters
	double sxRoofFL = sxMFL / (sxMFL + syMFL) * (FxMFL / dFx0FL) / (FxMFL / dFx0FL + FyMFL / dFy0FL);
	double sxRoofFR = sxMFR / (sxMFR + syMFR) * (FxMFR / dFx0FR) / (FxMFR / dFx0FR + FyMFR / dFy0FR);
	double sxRoofRR = sxMRR / (sxMRR + syMRR) * (FxMRR / dFx0RR) / (FxMRR / dFx0RR + FyMRR / dFy0RR);
	double sxRoofRL = sxMRL / (sxMRL + syMRL) * (FxMRL / dFx0RL) / (FxMRL / dFx0RL + FyMRL / dFy0RL);
	double syRoofFL = syMFL / (sxMFL + syMFL) * (FyMFL / dFy0FL) / (FxMFL / dFx0FL + FyMFL / dFy0FL);
	double syRoofFR = syMFR / (sxMFR + syMFR) * (FyMFR / dFy0FR) / (FxMFR / dFx0FR + FyMFR / dFy0FR);
	double syRoofRR = syMRR / (sxMRR + syMRR) * (FyMRR / dFy0RR) / (FxMRR / dFx0RR + FyMRR / dFy0RR);
	double syRoofRL = syMRL / (sxMRL + syMRL) * (FyMRL / dFy0RL) / (FxMRL / dFx0RL + FyMRL / dFy0RL);
	
	//speeds in wheel direction
	double cosWAngleFL = cos(-carState.wheelAngleZFL);
	double sinWAngleFL = sin(-carState.wheelAngleZFL);
	double cosWAngleFR = cos(-carState.wheelAngleZFR);
	double sinWAngleFR = sin(-carState.wheelAngleZFR);
	double cosWAngleRR = cos(-carState.wheelAngleZRR);
	double sinWAngleRR = sin(-carState.wheelAngleZRR);
	double cosWAngleRL = cos(-carState.wheelAngleZRL);
	double sinWAngleRL = sin(-carState.wheelAngleZRL);
	double vTangF = initialVYaw * carState.rwF;
	double vTangR = initialVYaw * carState.rwR;
	
	double vXWFL = cosWAngleFL * (initialVX - vTangF * carState.sinawF) - sinWAngleFL * initialVY;
	double vYWFL = sinWAngleFL * initialVX + cosWAngleFL * (initialVY + vTangF * carState.cosawF);
	double vXWFR = cosWAngleFR * (initialVX + vTangF * carState.sinawF) - sinWAngleFR * initialVY;
	double vYWFR = sinWAngleFR * initialVX + cosWAngleFR * (initialVY + vTangF * carState.cosawF);
	double vXWRR = cosWAngleRR * (initialVX + vTangR * carState.sinawR) - sinWAngleRR * initialVY;
	double vYWRR = sinWAngleRR * initialVX + cosWAngleRR * (initialVY - vTangR * carState.cosawR);
	double vXWRL = cosWAngleRL * (initialVX - vTangR * carState.sinawR) - sinWAngleRL * initialVY;
	double vYWRL = sinWAngleRL * initialVX + cosWAngleRL * (initialVY - vTangR * carState.cosawR);
	
	//slip values 
	double syFL = -vYWFL / (rDynFL * std::abs(initialOmegaYFL) + carState.vN);
	double syFR = -vYWFR / (rDynFR * std::abs(initialOmegaYFR) + carState.vN);
	double syRR = -vYWRR / (rDynRR * std::abs(initialOmegaYRR) + carState.vN);
	double syRL = -vYWRL / (rDynRL * std::abs(initialOmegaYRL) + carState.vN);
	
	//normalized slip TODO: how is slip in y dependant on wheel rotation speed???
	double sxNFL = -(vXWFL - rDynFL * initialOmegaYFL) / (rDynFL * std::abs(initialOmegaYFL) * sxRoofFL + carState.vN); //add angle to vx
	double sxNFR = -(vXWFR - rDynFR * initialOmegaYFR) / (rDynFR * std::abs(initialOmegaYFR) * sxRoofFR + carState.vN); //add angle to vx
	double sxNRR = -(vXWRR - rDynRR * initialOmegaYRR) / (rDynRR * std::abs(initialOmegaYRR) * sxRoofRR + carState.vN); //add angle to vx
	double sxNRL = -(vXWRL - rDynRL * initialOmegaYRL) / (rDynRL * std::abs(initialOmegaYRL) * sxRoofRL + carState.vN); //add angle to vx
	double syNFL = -vYWFL / (rDynFL * std::abs(initialOmegaYFL) * syRoofFL + carState.vN); //add angle to vy
	double syNFR = -vYWFR / (rDynFR * std::abs(initialOmegaYFR) * syRoofFR + carState.vN); //add angle to vy
	double syNRR = -vYWRR / (rDynRR * std::abs(initialOmegaYRR) * syRoofRR + carState.vN); //add angle to vy
	double syNRL = -vYWRL / (rDynRL * std::abs(initialOmegaYRL) * syRoofRL + carState.vN); //add angle to vy
	
	//combined slip
	double sFL = sqrt(sxNFL * sxNFL + syNFL * syNFL);
	double sFR = sqrt(sxNFR * sxNFR + syNFR * syNFR);
	double sRR = sqrt(sxNRR * sxNRR + syNRR * syNRR);
	double sRL = sqrt(sxNRL * sxNRL + syNRL * syNRL);
	
	//slipAngle
	double cosPhiFL = sxNFL / (sFL + carState.vN);
	double cosPhiFR = sxNFR / (sFR + carState.vN);
	double cosPhiRR = sxNRR / (sRR + carState.vN);
	double cosPhiRL = sxNRL / (sRL + carState.vN);
	double sinPhiFL = syNFL / (sFL + carState.vN);
	double sinPhiFR = syNFR / (sFR + carState.vN);
	double sinPhiRR = syNRR / (sRR + carState.vN);
	double sinPhiRL = syNRL / (sRL + carState.vN);
	
	//parameters for combined slip
	double dF0FL = sqrt(dFx0FL * sxRoofFL * cosPhiFL * dFx0FL * sxRoofFL * cosPhiFL + dFy0FL * syRoofFL * sinPhiFL * dFy0FL * syRoofFL * sinPhiFL);
	double dF0FR = sqrt(dFx0FR * sxRoofFR * cosPhiFR * dFx0FR * sxRoofFR * cosPhiFR + dFy0FR * syRoofFR * sinPhiFR * dFy0FR * syRoofFR * sinPhiFR);
	double dF0RR = sqrt(dFx0RR * sxRoofRR * cosPhiRR * dFx0RR * sxRoofRR * cosPhiRR + dFy0RR * syRoofRR * sinPhiRR * dFy0RR * syRoofRR * sinPhiRR);
	double dF0RL = sqrt(dFx0RL * sxRoofRL * cosPhiRL * dFx0RL * sxRoofRL * cosPhiRL + dFy0RL * syRoofRL * sinPhiRL * dFy0RL * syRoofRL * sinPhiRL);
	double FMFL = sqrt(FxMFL * cosPhiFL * FxMFL * cosPhiFL + FyMFL * sinPhiFL * FyMFL * sinPhiFL);
	double FMFR = sqrt(FxMFR * cosPhiFR * FxMFR * cosPhiFR + FyMFR * sinPhiFR * FyMFR * sinPhiFR);
	double FMRR = sqrt(FxMRR * cosPhiRR * FxMRR * cosPhiRR + FyMRR * sinPhiRR * FyMRR * sinPhiRR);
	double FMRL = sqrt(FxMRL * cosPhiRL * FxMRL * cosPhiRL + FyMRL * sinPhiRL * FyMRL * sinPhiRL);
	double FGFL = sqrt(FxGFL * cosPhiFL * FxGFL * cosPhiFL + FyGFL * sinPhiFL * FyGFL * sinPhiFL);
	double FGFR = sqrt(FxGFR * cosPhiFR * FxGFR * cosPhiFR + FyGFR * sinPhiFR * FyGFR * sinPhiFR);
	double FGRR = sqrt(FxGRR * cosPhiRR * FxGRR * cosPhiRR + FyGRR * sinPhiRR * FyGRR * sinPhiRR);
	double FGRL = sqrt(FxGRL * cosPhiRL * FxGRL * cosPhiRL + FyGRL * sinPhiRL * FyGRL * sinPhiRL);
	double sMFL = sqrt((sxMFL * cosPhiFL / sxRoofFL) * (sxMFL * cosPhiFL / sxRoofFL) + (syMFL * sinPhiFL / syRoofFL) * (syMFL * sinPhiFL / syRoofFL));
	double sMFR = sqrt((sxMFR * cosPhiFR / sxRoofFR) * (sxMFR * cosPhiFR / sxRoofFR) + (syMFR * sinPhiFR / syRoofFR) * (syMFR * sinPhiFR / syRoofFR));
	double sMRR = sqrt((sxMRR * cosPhiRR / sxRoofRR) * (sxMRR * cosPhiRR / sxRoofRR) + (syMRR * sinPhiRR / syRoofRR) * (syMRR * sinPhiRR / syRoofRR));
	double sMRL = sqrt((sxMRL * cosPhiRL / sxRoofRL) * (sxMRL * cosPhiRL / sxRoofRL) + (syMRL * sinPhiRL / syRoofRL) * (syMRL * sinPhiRL / syRoofRL));
	double sGFL = sqrt((sxGFL * cosPhiFL / sxRoofFL) * (sxGFL * cosPhiFL / sxRoofFL) + (syGFL * sinPhiFL / syRoofFL) * (syGFL * sinPhiFL / syRoofFL));
	double sGFR = sqrt((sxGFR * cosPhiFR / sxRoofFR) * (sxGFR * cosPhiFR / sxRoofFR) + (syGFR * sinPhiFR / syRoofFR) * (syGFR * sinPhiFR / syRoofFR));
	double sGRR = sqrt((sxGRR * cosPhiRR / sxRoofRR) * (sxGRR * cosPhiRR / sxRoofRR) + (syGRR * sinPhiRR / syRoofRR) * (syGRR * sinPhiRR / syRoofRR));
	double sGRL = sqrt((sxGRL * cosPhiRL / sxRoofRL) * (sxGRL * cosPhiRL / sxRoofRL) + (syGRL * sinPhiRL / syRoofRL) * (syGRL * sinPhiRL / syRoofRL));
	
	//bore torque; TODO advanced bore torque model for parking (in lecture notes)
	/*double TBFL = RBFL * dF0FL * sBFL * 9 / 8;
	double TBFR = RBFR * dF0FR * sBFR * 9 / 8;
	double TBRR = RBRR * dF0RR * sBRR * 9 / 8;
	double TBRL = RBRL * dF0RL * sBRL * 9 / 8;*/
	double RBFL1 = 0.9 * RBFL;
	double RBFL2 = RBFL;
	double RBFL3 = 1.1 * RBFL;
	double RBFR1 = 0.9 * RBFR;
	double RBFR2 = RBFR;
	double RBFR3 = 1.1 * RBFR;
	double RBRR1 = 0.9 * RBRR;
	double RBRR2 = RBRR;
	double RBRR3 = 1.1 * RBRR;
	double RBRL1 = 0.9 * RBRL;
	double RBRL2 = RBRL;
	double RBRL3 = 1.1 * RBRL;
	
	double boreXGFL = FtireFL / carState.FzN * (2 * carState.boreXGN - 0.5 * carState.boreXG2N - (carState.boreXGN - 0.5 * carState.boreXG2N) * FtireFL / carState.FzN);
	double boreXGFR = FtireFR / carState.FzN * (2 * carState.boreXGN - 0.5 * carState.boreXG2N - (carState.boreXGN - 0.5 * carState.boreXG2N) * FtireFR / carState.FzN);
	double boreXGRR = FtireRR / carState.FzN * (2 * carState.boreXGN - 0.5 * carState.boreXG2N - (carState.boreXGN - 0.5 * carState.boreXG2N) * FtireRR / carState.FzN);
	double boreXGRL = FtireRL / carState.FzN * (2 * carState.boreXGN - 0.5 * carState.boreXG2N - (carState.boreXGN - 0.5 * carState.boreXG2N) * FtireRL / carState.FzN);
	double boreYGFL = FtireFL / carState.FzN * (2 * carState.boreYGN - 0.5 * carState.boreYG2N - (carState.boreYGN - 0.5 * carState.boreYG2N) * FtireFL / carState.FzN);
	double boreYGFR = FtireFR / carState.FzN * (2 * carState.boreYGN - 0.5 * carState.boreYG2N - (carState.boreYGN - 0.5 * carState.boreYG2N) * FtireFR / carState.FzN);
	double boreYGRR = FtireRR / carState.FzN * (2 * carState.boreYGN - 0.5 * carState.boreYG2N - (carState.boreYGN - 0.5 * carState.boreYG2N) * FtireRR / carState.FzN);
	double boreYGRL = FtireRL / carState.FzN * (2 * carState.boreYGN - 0.5 * carState.boreYG2N - (carState.boreYGN - 0.5 * carState.boreYG2N) * FtireRL / carState.FzN);
	double boreGFL = (boreXGFL + boreYGFL) / 2.0;
	double boreGFR = (boreXGFR + boreYGFR) / 2.0;
	double boreGRR = (boreXGRR + boreYGRR) / 2.0;
	double boreGRL = (boreXGRL + boreYGRL) / 2.0;
	
	double TBmaxFL1;
	if (boreGFL == 0)
	{
		TBmaxFL1 = RBFL1 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxFL1 = RBFL1 * boreGFL;
	}
	double TBmaxFL2;
	if (boreGFL == 0)
	{
		TBmaxFL2 = RBFL2 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxFL2 = RBFL2 * boreGFL;
	}
	double TBmaxFL3;
	if (boreGFL == 0)
	{
		TBmaxFL3 = RBFL3 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxFL3 = RBFL3 * boreGFL;
	}
	
	double TBmaxFR1;
	if (boreGFR == 0)
	{
		TBmaxFR1 = RBFR1 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxFR1 = RBFR1 * boreGFR;
	}
	double TBmaxFR2;
	if (boreGFR == 0)
	{
		TBmaxFR2 = RBFR2 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxFR2 = RBFR2 * boreGFR;
	}
	double TBmaxFR3;
	if (boreGFR == 0)
	{
		TBmaxFR3 = RBFR3 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxFR3 = RBFR3 * boreGFR;
	}
	
	double TBmaxRR1;
	if (boreGRR == 0)
	{
		TBmaxRR1 = RBRR1 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxRR1 = RBRR1 * boreGRR;
	}
	double TBmaxRR2;
	if (boreGRR == 0)
	{
		TBmaxRR2 = RBRR2 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxRR2 = RBRR2 * boreGRR;
	}
	double TBmaxRR3;
	if (boreGRR == 0)
	{
		TBmaxRR3 = RBRR3 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxRR3 = RBRR3 * boreGRR;
	}
	
	double TBmaxRL1;
	if (boreGRL == 0)
	{
		TBmaxRL1 = RBRL1 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxRL1 = RBRL1 * boreGRL;
	}
	double TBmaxRL2;
	if (boreGRL == 0)
	{
		TBmaxRL2 = RBRL2 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxRL2 = RBRL2 * boreGRL;
	}
	double TBmaxRL3;
	if (boreGRL == 0)
	{
		TBmaxRL3 = RBRL3 * (carState.boreXGN + carState.boreYGN) / 2;
	} else
	{
		TBmaxRL3 = RBRL3 * boreGRL;
	}
	
	double cPhiFL1 = carState.cBore * RBFL1 * RBFL1;
	double cPhiFL2 = carState.cBore * RBFL2 * RBFL2;
	double cPhiFL3 = carState.cBore * RBFL3 * RBFL3;
	double cPhiFR1 = carState.cBore * RBFR1 * RBFR1;
	double cPhiFR2 = carState.cBore * RBFR2 * RBFR2;
	double cPhiFR3 = carState.cBore * RBFR3 * RBFR3;
	double cPhiRR1 = carState.cBore * RBRR1 * RBRR1;
	double cPhiRR2 = carState.cBore * RBRR2 * RBRR2;
	double cPhiRR3 = carState.cBore * RBRR3 * RBRR3;
	double cPhiRL1 = carState.cBore * RBRL1 * RBRL1;
	double cPhiRL2 = carState.cBore * RBRL2 * RBRL2;
	double cPhiRL3 = carState.cBore * RBRL3 * RBRL3;
	double dPhiFL1 = carState.dBore * RBFL1 * RBFL1;
	double dPhiFL2 = carState.dBore * RBFL2 * RBFL2;
	double dPhiFL3 = carState.dBore * RBFL3 * RBFL3;
	double dPhiFR1 = carState.dBore * RBFR1 * RBFR1;
	double dPhiFR2 = carState.dBore * RBFR2 * RBFR2;
	double dPhiFR3 = carState.dBore * RBFR3 * RBFR3;
	double dPhiRR1 = carState.dBore * RBRR1 * RBRR1;
	double dPhiRR2 = carState.dBore * RBRR2 * RBRR2;
	double dPhiRR3 = carState.dBore * RBRR3 * RBRR3;
	double dPhiRL1 = carState.dBore * RBRL1 * RBRL1;
	double dPhiRL2 = carState.dBore * RBRL2 * RBRL2;
	double dPhiRL3 = carState.dBore * RBRL3 * RBRL3;
	
	double TBstFL1 = cPhiFL1 * inPosState.phiDotFL1;
	double TBstFL2 = cPhiFL2 * inPosState.phiDotFL2;
	double TBstFL3 = cPhiFL3 * inPosState.phiDotFL3;
	double TBstFR1 = cPhiFR1 * inPosState.phiDotFR1;
	double TBstFR2 = cPhiFR2 * inPosState.phiDotFR2;
	double TBstFR3 = cPhiFR3 * inPosState.phiDotFR3;
	double TBstRR1 = cPhiRR1 * inPosState.phiDotRR1;
	double TBstRR2 = cPhiRR2 * inPosState.phiDotRR2;
	double TBstRR3 = cPhiRR3 * inPosState.phiDotRR3;
	double TBstRL1 = cPhiRL1 * inPosState.phiDotRL1;
	double TBstRL2 = cPhiRL2 * inPosState.phiDotRL2;
	double TBstRL3 = cPhiRL3 * inPosState.phiDotRL3;
	
	if(std::abs(TBstFL1) > TBmaxFL1)
	{
		TBstFL1 = TBstFL1 * TBmaxFL1 / std::abs(TBstFL1);
	}
	if(std::abs(TBstFL2) > TBmaxFL2)
	{
		TBstFL2 = TBstFL2 * TBmaxFL2 / std::abs(TBstFL2);
	}
	if(std::abs(TBstFL3) > TBmaxFL3)
	{
		TBstFL3 = TBstFL3 * TBmaxFL3 / std::abs(TBstFL3);
	}
	
	if(std::abs(TBstFR1) > TBmaxFR1)
	{
		TBstFR1 = TBstFR1 * TBmaxFR1 / std::abs(TBstFR1);
	}
	if(std::abs(TBstFR2) > TBmaxFR2)
	{
		TBstFR2 = TBstFR2 * TBmaxFR2 / std::abs(TBstFR2);
	}
	if(std::abs(TBstFR3) > TBmaxFR3)
	{
		TBstFR3 = TBstFR3 * TBmaxFR3 / std::abs(TBstFR3);
	}
	
	if(std::abs(TBstRR1) > TBmaxRR1)
	{
		TBstRR1 = TBstRR1 * TBmaxRR1 / std::abs(TBstRR1);
	}
	if(std::abs(TBstRR2) > TBmaxRR2)
	{
		TBstRR2 = TBstRR2 * TBmaxRR2 / std::abs(TBstRR2);
	}
	if(std::abs(TBstRR3) > TBmaxRR3)
	{
		TBstRR3 = TBstRR3 * TBmaxRR3 / std::abs(TBstRR3);
	}
	
	if(std::abs(TBstRL1) > TBmaxRL1)
	{
		TBstRL1 = TBstRL1 * TBmaxRL1 / std::abs(TBstRL1);
	}
	if(std::abs(TBstRL2) > TBmaxRL2)
	{
		TBstRL2 = TBstRL2 * TBmaxRL2 / std::abs(TBstRL2);
	}
	if(std::abs(TBstRL3) > TBmaxRL3)
	{
		TBstRL3 = TBstRL3 * TBmaxRL3 / std::abs(TBstRL3);
	}
	
	//TODO add omegaz for entire car to bore motion
	double phiADotFL1;
	if(dF0FL == 0)
	{
		phiADotFL1 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBFL1 * RBFL1 * initialOmegaZFL + rDynFL * std::abs(initialOmegaYFL) * TBstFL1) / (((carState.dFx0N * carState.dFy0N) / 2) * RBFL1 * RBFL1 + rDynFL * std::abs(initialOmegaYFL) * dPhiFL1);
	} else
	{
		phiADotFL1 = -(dF0FL * RBFL1 * RBFL1 * initialOmegaZFL + rDynFL * std::abs(initialOmegaYFL) * TBstFL1) / (dF0FL * RBFL1 * RBFL1 + rDynFL * std::abs(initialOmegaYFL) * dPhiFL1);
	}
	double phiADotFL2;
	if(dF0FL == 0)
	{
		phiADotFL2 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBFL2 * RBFL2 * initialOmegaZFL + rDynFL * std::abs(initialOmegaYFL) * TBstFL2) / (((carState.dFx0N * carState.dFy0N) / 2) * RBFL2 * RBFL2 + rDynFL * std::abs(initialOmegaYFL) * dPhiFL2);
	} else
	{
		phiADotFL2 = -(dF0FL * RBFL2 * RBFL2 * initialOmegaZFL + rDynFL * std::abs(initialOmegaYFL) * TBstFL2) / (dF0FL * RBFL2 * RBFL2 + rDynFL * std::abs(initialOmegaYFL) * dPhiFL2);
	}
	double phiADotFL3;
	if(dF0FL == 0)
	{
		phiADotFL3 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBFL3 * RBFL3 * initialOmegaZFL + rDynFL * std::abs(initialOmegaYFL) * TBstFL3) / (((carState.dFx0N * carState.dFy0N) / 2) * RBFL3 * RBFL3 + rDynFL * std::abs(initialOmegaYFL) * dPhiFL3);
	} else
	{
		phiADotFL3 = -(dF0FL * RBFL3 * RBFL3 * initialOmegaZFL + rDynFL * std::abs(initialOmegaYFL) * TBstFL3) / (dF0FL * RBFL3 * RBFL3 + rDynFL * std::abs(initialOmegaYFL) * dPhiFL3);
	}
	
	double phiADotFR1;
	if(dF0FR == 0)
	{
		phiADotFR1 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBFR1 * RBFR1 * initialOmegaZFR + rDynFR * std::abs(initialOmegaYFR) * TBstFR1) / (((carState.dFx0N * carState.dFy0N) / 2) * RBFR1 * RBFR1 + rDynFR * std::abs(initialOmegaYFR) * dPhiFR1);
	} else
	{
		phiADotFR1 = -(dF0FR * RBFR1 * RBFR1 * initialOmegaZFR + rDynFR * std::abs(initialOmegaYFR) * TBstFR1) / (dF0FR * RBFR1 * RBFR1 + rDynFR * std::abs(initialOmegaYFR) * dPhiFR1);
	}
	double phiADotFR2;
	if(dF0FR == 0)
	{
		phiADotFR2 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBFR2 * RBFR2 * initialOmegaZFR + rDynFR * std::abs(initialOmegaYFR) * TBstFR2) / (((carState.dFx0N * carState.dFy0N) / 2) * RBFR2 * RBFR2 + rDynFR * std::abs(initialOmegaYFR) * dPhiFR2);
	} else
	{
		phiADotFR2 = -(dF0FR * RBFR2 * RBFR2 * initialOmegaZFR + rDynFR * std::abs(initialOmegaYFR) * TBstFR2) / (dF0FR * RBFR2 * RBFR2 + rDynFR * std::abs(initialOmegaYFR) * dPhiFR2);
	}
	double phiADotFR3;
	if(dF0FR == 0)
	{
		phiADotFR3 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBFR3 * RBFR3 * initialOmegaZFR + rDynFR * std::abs(initialOmegaYFR) * TBstFR3) / (((carState.dFx0N * carState.dFy0N) / 2) * RBFR3 * RBFR3 + rDynFR * std::abs(initialOmegaYFR) * dPhiFR2);
	} else
	{
		phiADotFR3 = -(dF0FR * RBFR3 * RBFR3 * initialOmegaZFR + rDynFR * std::abs(initialOmegaYFR) * TBstFR3) / (dF0FR * RBFR3 * RBFR3 + rDynFR * std::abs(initialOmegaYFR) * dPhiFR3);
	}
	
	double phiADotRR1;
	if(dF0RR == 0)
	{
		phiADotRR1 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBRR1 * RBRR1 * initialOmegaZRR + rDynRR * std::abs(initialOmegaYRR) * TBstRR1) / (((carState.dFx0N * carState.dFy0N) / 2) * RBRR1 * RBRR1 + rDynRR * std::abs(initialOmegaYRR) * dPhiRR1);
	} else
	{
		phiADotRR1 = -(dF0RR * RBRR1 * RBRR1 * initialOmegaZRR + rDynRR * std::abs(initialOmegaYRR) * TBstRR1) / (dF0RR * RBRR1 * RBRR1 + rDynRR * std::abs(initialOmegaYRR) * dPhiRR1);
	}
	double phiADotRR2;
	if(dF0RR == 0)
	{
		phiADotRR2 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBRR2 * RBRR2 * initialOmegaZRR + rDynRR * std::abs(initialOmegaYRR) * TBstRR2) / (((carState.dFx0N * carState.dFy0N) / 2) * RBRR2 * RBRR2 + rDynRR * std::abs(initialOmegaYRR) * dPhiRR2);
	} else
	{
		phiADotRR2 = -(dF0RR * RBRR2 * RBRR2 * initialOmegaZRR + rDynRR * std::abs(initialOmegaYRR) * TBstRR2) / (dF0RR * RBRR2 * RBRR2 + rDynRR * std::abs(initialOmegaYRR) * dPhiRR2);
	}
	double phiADotRR3;
	if(dF0RR == 0)
	{
		phiADotRR3 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBRR3 * RBRR3 * initialOmegaZRR + rDynRR * std::abs(initialOmegaYRR) * TBstRR3) / (((carState.dFx0N * carState.dFy0N) / 2) * RBRR3 * RBRR3 + rDynRR * std::abs(initialOmegaYRR) * dPhiRR3);
	} else
	{
		phiADotRR3 = -(dF0RR * RBRR3 * RBRR3 * initialOmegaZRR + rDynRR * std::abs(initialOmegaYRR) * TBstRR3) / (dF0RR * RBRR3 * RBRR3 + rDynRR * std::abs(initialOmegaYRR) * dPhiRR3);
	}
	
	double phiADotRL1;
	if(dF0RL == 0)
	{
		phiADotRL1 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBRL1 * RBRL1 * initialOmegaZRL + rDynRL * std::abs(initialOmegaYRL) * TBstRL1) / (((carState.dFx0N * carState.dFy0N) / 2) * RBRL1 * RBRL1 + rDynRL * std::abs(initialOmegaYRL) * dPhiRL1);
	} else
	{
		phiADotRL1 = -(dF0RL * RBRL1 * RBRL1 * initialOmegaZRL + rDynRL * std::abs(initialOmegaYRL) * TBstRL1) / (dF0RL * RBRL1 * RBRL1 + rDynRL * std::abs(initialOmegaYRL) * dPhiRL1);
	}
	double phiADotRL2;
	if(dF0RL == 0)
	{
		phiADotRL2 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBRL2 * RBRL2 * initialOmegaZRL + rDynRL * std::abs(initialOmegaYRL) * TBstRL2) / (((carState.dFx0N * carState.dFy0N) / 2) * RBRL2 * RBRL2 + rDynRL * std::abs(initialOmegaYRL) * dPhiRL2);
	} else
	{
		phiADotRL2 = -(dF0RL * RBRL2 * RBRL2 * initialOmegaZRL + rDynRL * std::abs(initialOmegaYRL) * TBstRL2) / (dF0RL * RBRL2 * RBRL2 + rDynRL * std::abs(initialOmegaYRL) * dPhiRL2);
	}
	double phiADotRL3;
	if(dF0RL == 0)
	{
		phiADotRL3 = -(((carState.dFx0N * carState.dFy0N) / 2) * RBRL3 * RBRL3 * initialOmegaZRL + rDynRL * std::abs(initialOmegaYRL) * TBstRL3) / (((carState.dFx0N * carState.dFy0N) / 2) * RBRL3 * RBRL3 + rDynRL * std::abs(initialOmegaYRL) * dPhiRL3);
	} else
	{
		phiADotRL3 = -(dF0RL * RBRL3 * RBRL3 * initialOmegaZRL + rDynRL * std::abs(initialOmegaYRL) * TBstRL3) / (dF0RL * RBRL3 * RBRL3 + rDynRL * std::abs(initialOmegaYRL) * dPhiRL3);
	}
	
	
	double TBDFL1 = TBstFL1 + dPhiFL1 * phiADotFL1;
	double TBDFL2 = TBstFL2 + dPhiFL2 * phiADotFL2;
	double TBDFL3 = TBstFL3 + dPhiFL3 * phiADotFL3;
	double TBDFR1 = TBstFR1 + dPhiFR1 * phiADotFR1;
	double TBDFR2 = TBstFR2 + dPhiFR2 * phiADotFR2;
	double TBDFR3 = TBstFR3 + dPhiFR3 * phiADotFR3;
	double TBDRR1 = TBstRR1 + dPhiRR1 * phiADotRR1;
	double TBDRR2 = TBstRR2 + dPhiRR2 * phiADotRR2;
	double TBDRR3 = TBstRR3 + dPhiRR3 * phiADotRR3;
	double TBDRL1 = TBstRL1 + dPhiRL1 * phiADotRL1;
	double TBDRL2 = TBstRL2 + dPhiRL2 * phiADotRL2;
	double TBDRL3 = TBstRL3 + dPhiRL3 * phiADotRL3;
	
	double phiDotFL1 = 0;
	if (std::abs(TBDFL1) < TBmaxFL1)
	{
		phiDotFL1 = phiADotFL1;
	}
	double phiDotFL2 = 0;
	if (std::abs(TBDFL2) < TBmaxFL2)
	{
		phiDotFL2 = phiADotFL2;
	}
	double phiDotFL3 = 0;
	if (std::abs(TBDFL3) < TBmaxFL3)
	{
		phiDotFL3 = phiADotFL3;
	}
	
	double phiDotFR1 = 0;
	if (std::abs(TBDFR1) < TBmaxFR1)
	{
		phiDotFR1 = phiADotFR1;
	}
	double phiDotFR2 = 0;
	if (std::abs(TBDFR2) < TBmaxFR2)
	{
		phiDotFR2 = phiADotFR2;
	}
	double phiDotFR3 = 0;
	if (std::abs(TBDFR3) < TBmaxFR3)
	{
		phiDotFR3 = phiADotFR3;
	}
	
	double phiDotRR1 = 0;
	if (std::abs(TBDRR1) < TBmaxRR1)
	{
		phiDotRR1 = phiADotRR1;
	}
	double phiDotRR2 = 0;
	if (std::abs(TBDRR2) < TBmaxRR2)
	{
		phiDotRR2 = phiADotRR2;
	}
	double phiDotRR3 = 0;
	if (std::abs(TBDRR3) < TBmaxRR3)
	{
		phiDotRR3 = phiADotRR3;
	}
	
	double phiDotRL1 = 0;
	if (std::abs(TBDRL1) < TBmaxRL1)
	{
		phiDotRL1 = phiADotRL1;
	}
	double phiDotRL2 = 0;
	if (std::abs(TBDRL2) < TBmaxRL2)
	{
		phiDotRL2 = phiADotRL2;
	}
	double phiDotRL3 = 0;
	if (std::abs(TBDRL3) < TBmaxRL3)
	{
		phiDotRL3 = phiADotRL3;
	}
	
		
	//combined forces
	double FFL;
	if(sFL <= sMFL)
	{
		FFL = dF0FL * sFL / (1 + (sFL / (sMFL + carState.vN) + dF0FL * sMFL / (FMFL + carState.vN) - 2));
	} else if (sFL <= sGFL)
	{
		double sigma = (sFL - sMFL) / (sGFL - sMFL);
		FFL = FMFL - (FMFL - FGFL) * sigma * sigma * (3 - 2 * sigma);
	} else 
	{
		FFL = FGFL;
	}
	double FFR;
	if(sFR <= sMFR)
	{
		FFR = dF0FR * sFR / (1 + (sFR / (sMFR + carState.vN) + dF0FR * sMFR / (FMFR + carState.vN) - 2));
	} else if (sFR <= sGFR)
	{
		double sigma = (sFR - sMFR) / (sGFR - sMFR);
		FFR = FMFR - (FMFR - FGFR) * sigma * sigma * (3 - 2 * sigma);
	} else 
	{
		FFR = FGFR;
	}
	double FRR;
	if(sRR <= sMRR)
	{
		FRR = dF0RR * sRR / (1 + (sRR / (sMRR + carState.vN) + dF0RR * sMRR / (FMRR + carState.vN) - 2));
	} else if (sRR <= sGRR)
	{
		double sigma = (sRR - sMRR) / (sGRR - sMRR);
		FRR = FMRR - (FMRR - FGRR) * sigma * sigma * (3 - 2 * sigma);
	} else 
	{
		FRR = FGRR;
	}
	double FRL;
	if(sRL <= sMRL)
	{
		FRL = dF0RL * sRL / (1 + (sRL / (sMRL + carState.vN) + dF0RL * sMRL / (FMRL + carState.vN) - 2));
	} else if (sRL <= sGRL)
	{
		double sigma = (sRL - sMRL) / (sGRL - sMRL);
		FRL = FMRL - (FMRL - FGRL) * sigma * sigma * (3 - 2 * sigma);
	} else 
	{
		FRL = FGRL;
	}
	
	//directional forces
	double FxFL = FFL * cosPhiFL;
	double FxFR = FFR * cosPhiFR;
	double FxRR = FRR * cosPhiRR;
	double FxRL = FRL * cosPhiRL;
	double FyFL = FFL * sinPhiFL;
	double FyFR = FFR * sinPhiFR;
	double FyRR = FRR * sinPhiRR;
	double FyRL = FRL * sinPhiRL;
	
	//n
	double nFL;
	if (std::abs(syFL) <= carState.sy0)
	{
		double we = carState.sy0 / carState.syS;
		double se = std::abs(syFL) / carState.sy0;
		nFL = LFL * ((1 - we) * (1 - se) + we * (1 * (3 - 2 * se) * se * se));
	} else if (std::abs(syFL) <= carState.syS)
	{
		double we = carState.sy0 / carState.syS;
		nFL = LFL * -(1 - we) * (std::abs(syFL) - carState.sy0) * ((carState.syS - std::abs(syFL))/(carState.syS - carState.sy0)) * ((carState.syS - std::abs(syFL))/(carState.syS - carState.sy0));
	} else 
	{
		nFL = 0;
	}
	double nFR;
	if (std::abs(syFR) <= carState.sy0)
	{
		double we = carState.sy0 / carState.syS;
		double se = std::abs(syFR) / carState.sy0;
		nFR = LFR * ((1 - we) * (1 - se) + we * (1 * (3 - 2 * se) * se * se));
	} else if (std::abs(syFR) <= carState.syS)
	{
		double we = carState.sy0 / carState.syS;
		nFR = LFR * -(1 - we) * (std::abs(syFR) - carState.sy0) * ((carState.syS - std::abs(syFR))/(carState.syS - carState.sy0)) * ((carState.syS - std::abs(syFR))/(carState.syS - carState.sy0));
	} else 
	{
		nFR = 0;
	}
	double nRR;
	if (std::abs(syRR) <= carState.sy0)
	{
		double we = carState.sy0 / carState.syS;
		double se = std::abs(syRR) / carState.sy0;
		nRR = LRR * ((1 - we) * (1 - se) + we * (1 * (3 - 2 * se) * se * se));
	} else if (std::abs(syRR) <= carState.syS)
	{
		double we = carState.sy0 / carState.syS;
		nRR = LRR * -(1 - we) * (std::abs(syRR) - carState.sy0) * ((carState.syS - std::abs(syRR))/(carState.syS - carState.sy0)) * ((carState.syS - std::abs(syRR))/(carState.syS - carState.sy0));
	} else 
	{
		nRR = 0;
	}
	double nRL;
	if (std::abs(syRL) <= carState.sy0)
	{
		double we = carState.sy0 / carState.syS;
		double se = std::abs(syRL) / carState.sy0;
		nRL = LRL * ((1 - we) * (1 - se) + we * (1 * (3 - 2 * se) * se * se));
	} else if (std::abs(syRL) <= carState.syS)
	{
		double we = carState.sy0 / carState.syS;
		nRL = LRL * -(1 - we) * (std::abs(syRL) - carState.sy0) * ((carState.syS - std::abs(syRL))/(carState.syS - carState.sy0)) * ((carState.syS - std::abs(syRL))/(carState.syS - carState.sy0));
	} else 
	{
		nRL = 0;
	}
	
	//self aligning torque
	double TsFL = 0;
	double TsFR = 0;
	double TsRR = 0;
	double TsRL = 0;
	if(vXWFL >= 0)
	{
		TsFL = nFL * FyFL;
	} else
	{
		TsFL = -nFL * FyFL;
	}
	if(vXWFR >= 0)
	{
		TsFR = nFR * FyFR;
	} else
	{
		TsFR = -nFR * FyFR;
	}
	if(vXWRR >= 0)
	{
		TsRR = nRR * FyRR;
	} else
	{
		TsRR = -nRR * FyRR;
	}
	if(vXWRL >= 0)
	{
		TsRL = nRL * FyRL;
	} else
	{
		TsRL = -nRL * FyRL;
	}
	
	
	//forces from gravity
	double FgravXFL = sin(carState.roadAngleFL) * carState.mTotal / 4.0;//FweightedFL;
	double FgravXFR = sin(carState.roadAngleFR) * carState.mTotal / 4.0;//FweightedFR;
	double FgravXRR = sin(carState.roadAngleRR) * carState.mTotal / 4.0;//FweightedRR;
	double FgravXRL = sin(carState.roadAngleRL) * carState.mTotal / 4.0;//FweightedRL;
	
	//steering
	/*double vWheelDiff = initialOmegaZFL - initialOmegaZFR;
	double vWheelCombined = (initialOmegaZFL + initialOmegaZFR) / 2;
	
	double TcolumnS = carState.cRack * (carState.posSteeringWheel * carState.steeringRatio - carState.posWheelCombined);
	double TcolumnD = carState.dRack * (carState.vSteeringWheel - vWheelCombined);
	double TrackS = carState.cInt * carState.deltaWheel;
	double TrackD = carState.dInt * vWheelDiff;*/
	
	
	//temp stuff
	double Facc = 0;//carState.acceleratorAngle * 3000;
	double Fsteering = 0;//carState.posSteeringWheel * 30 * tanh(initialVX);
	double Fbrake = 0;//0.1 * carState.brakeForce * tanh(initialVX);
	double FresX = 0;//100 * tanh(initialVX);
	double FresY = 0;//10 * tanh(initialVY);
	double FresRotZ = 0;//10 * tanh(initialVYaw);
	
	//tire forces in car coordinates
	double FxFLCC = cosWAngleFL * FxFL + sinWAngleFL * FyFL;
	double FyFLCC = -sinWAngleFL * FxFL + cosWAngleFL * FyFL;
	double FxFRCC = cosWAngleFR * FxFR + sinWAngleFR * FyFR;
	double FyFRCC = -sinWAngleFR * FxFR + cosWAngleFR * FyFR;
	double FxRRCC = cosWAngleRR * FxRR + sinWAngleRR * FyRR;
	double FyRRCC = -sinWAngleRR * FxRR + cosWAngleRR * FyRR;
	double FxRLCC = cosWAngleRL * FxRL + sinWAngleRL * FyRL;
	double FyRLCC = -sinWAngleRL * FxRL + cosWAngleRL * FyRL;
	
	/* old drive train
	double TdriveFL = 0;
	double TdriveFR = 0;
	double TdriveRR	= 0;
	double TdriveRL = 0;
	if (carState.gear > 0)
	{
		TdriveRR = 10 * engineTorque / 2;
		TdriveRL = 10 * engineTorque / 2;
	} 
	else if (carState.gear < 0)
	{
		TdriveRR = -10 * engineTorque / 2;
		TdriveRL = -10 * engineTorque / 2;
	}*/

	//brake
	double TbrakeMaxFL = carState.brakeForce / 4;
	double TbrakeMaxFR = carState.brakeForce / 4;
	double TbrakeMaxRR = carState.brakeForce / 4;
	double TbrakeMaxRL = carState.brakeForce / 4;
	
	//double TbrakeFL = -0.1 * carState.brakeForce * tanh(initialOmegaYFL) / 4;
	//double TbrakeFR = -0.1 * carState.brakeForce * tanh(initialOmegaYFL) / 4;
	//double TbrakeRR = -0.1 * carState.brakeForce * tanh(initialOmegaYFL) / 4;
	//double TbrakeRL = -0.1 * carState.brakeForce * tanh(initialOmegaYFL) / 4;
	double TbrakeFL = std::abs(initialOmegaYFL) * carState.bRate + carState.Tstat;
	double TbrakeFR = std::abs(initialOmegaYFR) * carState.bRate + carState.Tstat;
	double TbrakeRR = std::abs(initialOmegaYRR) * carState.bRate + carState.Tstat;
	double TbrakeRL = std::abs(initialOmegaYRL) * carState.bRate + carState.Tstat;
	
	if (TbrakeFL > TbrakeMaxFL)
	{
		TbrakeFL = TbrakeMaxFL;
	}
	if (TbrakeFR > TbrakeMaxFR)
	{
		TbrakeFR = TbrakeMaxFR;
	}
	if (TbrakeRR > TbrakeMaxRR)
	{
		TbrakeRR = TbrakeMaxRR;
	}
	if (TbrakeRL > TbrakeMaxRL)
	{
		TbrakeRL = TbrakeMaxRL;
	}
	
	if (initialOmegaYFL<0)
	{
		TbrakeFL = TbrakeFL;// * std::pow(tanh(abs(initialOmegaYFL)),0.01);
	} else
	{
		TbrakeFL = -TbrakeFL;// * std::pow(tanh(initialOmegaYFL),0.01);
	}
	if (initialOmegaYFR<0)
	{
		TbrakeFR = TbrakeFR;// * std::pow(tanh(abs(initialOmegaYFR)),0.01);
	} else
	{
		TbrakeFR = -TbrakeFR;// * std::pow(tanh(initialOmegaYFR),0.01);
	}
	if (initialOmegaYRR<0)
	{
		TbrakeRR = TbrakeRR;// * std::pow(tanh(abs(initialOmegaYRR)),0.01);
	} else
	{
		TbrakeRR = -TbrakeRR;// * std::pow(tanh(initialOmegaYRR),0.01);
	}
	if (initialOmegaYRL<0)
	{
		TbrakeRL = TbrakeRL;// * std::pow(tanh(abs(initialOmegaYRL)),0.01);
	} else
	{
		TbrakeRL = -TbrakeRL;//* std::pow(tanh(initialOmegaYRL),0.01);
	}
	
	/*TbrakeFL = -TbrakeFL * tanh(initialOmegaYFL);
	TbrakeFR = -TbrakeFR * tanh(initialOmegaYFR);
	TbrakeRR = -TbrakeRR * tanh(initialOmegaYRR);
	TbrakeRL = -TbrakeRL * tanh(initialOmegaYRL);*/
	
	//drive train
	double averageWheelSpeed = (initialOmegaYRL + initialOmegaYRR) / 2;
	double TlossEng = -carState.lossCoefEngine * initialRpm;
	double TlossDrive = -carState.lossCoefDrive * averageWheelSpeed;
	double TengineIn = TlossEng + carState.Tcomb;
	double TdriveIn = TbrakeRL + TbrakeRR - rDynRR * FxRR + TyRR - rDynRL * FxRL + TyRL + TlossDrive;
	double Tclutch = 0;
	double TclutchMAX = 0;
	double Mengine = 0;
	double MWheelYRR = 0;
	double MWheelYRL = 0;
	double omegaEngineDot = 0;
	double aWheelYRR = 0;
	double aWheelYRL = 0;
	if (carState.clutchSwitch == 0)
	{
		TclutchMAX = carState.clutchState * carState.frictionCoefDynClutch;
		if (initialRpm > averageWheelSpeed)
		{
			Tclutch = TclutchMAX;
		}
		else 
		{
			Tclutch = -TclutchMAX;
		}
		Mengine = TengineIn - Tclutch - initialRpm * carState.bEngine;
		MWheelYRR = (Tclutch * carState.finalDrive * carState.gearRatio - averageWheelSpeed * carState.bDrive) / 2 - rDynRR * FxRR + TyRR + TbrakeRR;
		MWheelYRL = (Tclutch * carState.finalDrive * carState.gearRatio - averageWheelSpeed * carState.bDrive) / 2 - rDynRL * FxRL + TyRL + TbrakeRL;
		omegaEngineDot = Mengine / carState.inertiaEngine;
		aWheelYRR = MWheelYRR / (carState.inertiaWheelY + 0.5 * carState.inertiaDrive);
		aWheelYRL = MWheelYRL / (carState.inertiaWheelY + 0.5 * carState.inertiaDrive);
	}
	else 
	{
		TclutchMAX = carState.clutchState * carState.frictionCoefStatClutch;
		Tclutch = (carState.inertiaDrive * TengineIn + carState.inertiaEngine * TdriveIn / (carState.finalDrive * carState.gearRatio) - (carState.inertiaDrive * carState.bEngine - carState.inertiaEngine * carState.bDrive) * initialRpm) / (carState.inertiaEngine + carState.inertiaDrive);
		Mengine = (TengineIn + TdriveIn / (carState.finalDrive * carState.gearRatio)) / (carState.bEngine + carState.bDrive);
		MWheelYRR = (TengineIn / 2 + (TbrakeRR - rDynRR * FxRR + TyRR) / (carState.finalDrive * carState.gearRatio)) / (carState.bEngine + carState.bDrive);
		MWheelYRL = (TengineIn / 2 + (TbrakeRL - rDynRL * FxRL + TyRL) / (carState.finalDrive * carState.gearRatio)) / (carState.bEngine + carState.bDrive);
		
		omegaEngineDot = Mengine / (carState.inertiaEngine + carState.inertiaDrive);
		aWheelYRR = MWheelYRR / ((carState.inertiaEngine + 0.5 * carState.inertiaDrive + carState.inertiaWheelY) * carState.finalDrive * carState.gearRatio);
		aWheelYRL = MWheelYRL / ((carState.inertiaEngine + 0.5 * carState.inertiaDrive + carState.inertiaWheelY) * carState.finalDrive * carState.gearRatio);
	}
	
	/*if(carState.gear == 0)
	{
		std::cout << "omegaEngineDot: " << omegaEngineDot << " aWheelYRR: " << aWheelYRR << " aWheelYRL: " << aWheelYRL << std::endl;
	}*/
	
	double FxRoadComb = FxFLCC + FxFRCC + FxRRCC + FxRLCC + FgravXFL + FgravXFR + FgravXRR + FgravXRL;
	double FyRoadComb = FyFLCC + FyFRCC + FyRRCC + FyRLCC + 0;
	
	//aeeroodynamic drag
	double Fdrag = 0;
	if(initialVX > 0)
	{
		Fdrag = -initialVX * initialVX * carState.cDrag;
	} else
	{
		Fdrag = initialVX * initialVX * carState.cDrag;
	}
	
	//Forces on body
	double ForceX = Fdrag + FxRoadComb/*carState.aGrav * carState.mCar * sin(carState.localPitch)*/;
	double ForceY = FyRoadComb/*+ carState.aGrav * carState.mCar * sin(carState.localRoll)*/;
	double ForceZ = FssFL + FsdFL + FssFR + FsdFR + FssRR + FsdRR + FssRL + FsdRR - carState.aGrav * carState.mCar / (cos(carState.localRoll) * cos(carState.localPitch));
	double MXtires = FyFLCC * carState.cogH + FyFRCC * carState.cogH + FyRRCC * carState.cogH + FyRLCC * carState.cogH;
	double MX = (-FssRR - FsdRR + FssRL + FsdRL) * carState.sRearH +(FssFL + FsdFL - FssFR - FsdFR) * carState.sFrontH + MXtires;
	double MYtires = -(FxFLCC + FxFRCC + FxRRCC + FxRLCC) * carState.cogH;
	double MY = (FssRR + FsdRR + FssRL + FsdRL) * carState.lRear - (FssFL + FsdFL + FssFR + FsdFR) * carState.lFront + MYtires;
	double MZtires = FyFLCC * carState.lFront + FyFRCC * carState.lFront - FyRRCC * carState.lRear - FyRLCC * carState.lRear - FxFLCC * carState.sFrontH + FxFRCC * carState.sFrontH + FxRRCC * carState.sRearH - FxRLCC * carState.sRearH;
	double MZ = Fsteering * carState.lFront - FresRotZ + MZtires;
	
	//Forces on suspension
	double FsuspZFL = -FssFL - FsdFL + FtsFL + FtdFL - carState.aGrav * carState.mSusFL;
	double FsuspZFR = -FssFR - FsdFR + FtsFR + FtdFR - carState.aGrav * carState.mSusFR;
	double FsuspZRR = -FssRR - FsdRR + FtsRR + FtdRR - carState.aGrav * carState.mSusRR;
	double FsuspZRL = -FssRL - FsdRL + FtsRL + FtdRL - carState.aGrav * carState.mSusRL;
	
	//Forces on wheels
	//double MWheelZFL = /*TsFL - TBFL*/ + TrackS - TrackD + TcolumnS + TcolumnD; //TODO removed bore torque
	//double MWheelZFR = /*TsFR - TBFR*/ + TrackS + TrackD + TcolumnS + TcolumnD; //TODO removed bore torque
	double MWheelYFL = -rDynFL * FxFL + TyFL + TbrakeFL;
	double MWheelYFR = -rDynFR * FxFR + TyFR + TbrakeFR;
	
	//accelerations on body
	double axBody = ForceX / carState.mCar;
	double ayBody = ForceY / carState.mCar;
	double azBody = ForceZ / carState.mCar;
	double aYawBody = MZ / carState.inertiaYaw;
	double aPitchBody = MY / carState.inertiaPitch;
	double aRollBody = MX / carState.inertiaRoll;
	double aSuspZFL = FsuspZFL / carState.mSusFL;
	double aSuspZFR = FsuspZFR / carState.mSusFR;
	double aSuspZRR = FsuspZRR / carState.mSusRR;
	double aSuspZRL = FsuspZRL / carState.mSusRL;
	double aWheelYFL = MWheelYFL / carState.inertiaWheelY;
	double aWheelYFR = MWheelYFR / carState.inertiaWheelY;
	
	if((initialOmegaYFL == 0) && (TbrakeFL >= (-rDynFL * FxFL + TyFL)))
	{
		aWheelYFL == 0;
	}
	if((initialOmegaYFR == 0) && (TbrakeFR >= (-rDynFR * FxFR + TyFR)))
	{
		aWheelYFR == 0;
	}
	if((initialOmegaYRR == 0) && (TbrakeRR >= (-rDynRR * FxRR + TyRR)))
	{
		aWheelYRR == 0;
	}
	if((initialOmegaYRL == 0) && (TbrakeRL >= (-rDynRL * FxRL + TyRL)))
	{
		aWheelYRL == 0;
	}
	
	
	//double aWheelZFL = MWheelZFL / carState.inertiaWheelZ;
	//double aWheelZFR = MWheelZFR / carState.inertiaWheelZ;
	
	//testestestetetestesetstst
	/*double c = 50000;
	double d = 20000;
	double RB = 0.062417;
	double dF0 = 14000;
	double rD = 0.295;
	double FG = 3200;
	
	double cPhi = c * RB * RB;
	double dPhi = d * RB * RB;
	
	double RB2 = RB;
	double TBmax2 = RB2 * FG;
	double TBst2 = cPhi * carState.phiFL2;
	if (abs(TBst2) > TBmax2)
	{
		TBst2 = TBst2 * std::abs(TBmax2 / TBst2);
	}
	double phiADot2 = -(dF0 * RB2 * RB2 * initialOmegaZFL + rD * std::abs(initialOmegaYFL) * TBst2) / (dF0 * RB2 * RB2 + rD * std::abs(initialOmegaYFL) * dPhi);
	double TBD2 = TBst2 + dPhi * phiADot2;
	double phiDot2 = 0;
	if (abs(TBD2) < TBmax2 )
	{
		phiDot2 = phiADot2;
	}*/
  
	FWDState outputAccelerationState;
	//output body speeds
	outputAccelerationState.vX = axBody;
	outputAccelerationState.vY = ayBody;
	outputAccelerationState.vZ = azBody;
	outputAccelerationState.vYaw = aYawBody;
	outputAccelerationState.vPitch = aPitchBody;
	outputAccelerationState.vRoll = aRollBody;
	outputAccelerationState.vSuspZFL = aSuspZFL;
	outputAccelerationState.vSuspZFR = aSuspZFR;
	outputAccelerationState.vSuspZRR = aSuspZRR;
	outputAccelerationState.vSuspZRL = aSuspZRL;
	outputAccelerationState.OmegaYFL = aWheelYFL;
	outputAccelerationState.OmegaYFR = aWheelYFR;
	outputAccelerationState.OmegaYRR = aWheelYRR;
	outputAccelerationState.OmegaYRL = aWheelYRL;
	outputAccelerationState.OmegaZFL = 0;//carState.vSteeringWheel * carState.steeringRatio;aWheelZFL * dT;
	outputAccelerationState.OmegaZFR = 0;//carState.vSteeringWheel * carState.steeringRatio;aWheelZFR * dT;
	outputAccelerationState.phiDotFL1 = phiDotFL1;
	outputAccelerationState.phiDotFL2 = phiDotFL2;
	outputAccelerationState.phiDotFL3 = phiDotFL3;
	outputAccelerationState.phiDotFR1 = phiDotFR1;
	outputAccelerationState.phiDotFR2 = phiDotFR2;
	outputAccelerationState.phiDotFR3 = phiDotFR3;
	outputAccelerationState.phiDotRR1 = phiDotRR1;
	outputAccelerationState.phiDotRR2 = phiDotRR2;
	outputAccelerationState.phiDotRR3 = phiDotRR3;
	outputAccelerationState.phiDotRL1 = phiDotRL1;
	outputAccelerationState.phiDotRL2 = phiDotRL2;
	outputAccelerationState.phiDotRL3 = phiDotRL3;
	outputAccelerationState.engineRPM = omegaEngineDot;
	
	//steering column torque 
	outputAccelerationState.TcolumnCombined = -TsFL - TsFR + TBDFL1 + TBDFL2 + TBDFL3 + TBDFL1 + TBDFR2 + TBDFL3;
	
	//clutch torques
	outputAccelerationState.Tclutch = Tclutch;
	outputAccelerationState.TclutchMax = TclutchMAX;
	
	//slip for sound
	outputAccelerationState.slipFL = sFL;
	outputAccelerationState.slipFR = sFR;
	outputAccelerationState.slipRR = sRR;
	outputAccelerationState.slipRL = sRL;
	
	//for logging
	outputAccelerationState.FweightedFL = FweightedFL;
	outputAccelerationState.FweightedFR = FweightedFR;
	outputAccelerationState.FweightedRR = FweightedRR;
	outputAccelerationState.FweightedRL = FweightedRL;
	outputAccelerationState.FtireFL = FtireFL;
	outputAccelerationState.FtireFR = FtireFR;
	outputAccelerationState.FtireRR = FtireRR;
	outputAccelerationState.FtireRL = FtireRL;
	outputAccelerationState.FxFL = FxFL;
	outputAccelerationState.FxFR = FxFR;
	outputAccelerationState.FxRR = FxRR;
	outputAccelerationState.FxRL = FxRL;
	outputAccelerationState.FyFL = FyFL;
	outputAccelerationState.FyFR = FyFR;
	outputAccelerationState.FyRR = FyRR;
	outputAccelerationState.FyRL = FyRL;
	outputAccelerationState.genericOut1 = FxFLCC;
	outputAccelerationState.genericOut2 = FxFL;
	outputAccelerationState.genericOut3 = FyFL;
	outputAccelerationState.genericOut4 = 0;
	outputAccelerationState.genericOut5 = FxFRCC;
	outputAccelerationState.genericOut6 = FxFR;
	outputAccelerationState.genericOut7 = FyFR;
	outputAccelerationState.genericOut8 = 0;
	outputAccelerationState.genericOut9 = FxRRCC;
	outputAccelerationState.genericOut10 = TyRR;
	outputAccelerationState.genericOut11 = FgravXRR;
	outputAccelerationState.genericOut12 = 0;
	outputAccelerationState.genericOut13 = FxRLCC;
	outputAccelerationState.genericOut14 = TyRL;
	outputAccelerationState.genericOut15 = FgravXRL;
	outputAccelerationState.genericOut16 = 0;
	
	if(carState.timerCounter == 4)
	{
		//std::cout << "initialVSuspZFL: " << initialVSuspZFL <<  " initialVZ: " << initialVZ << " initialVPitch: " << initialVPitch << " initialVRoll: " << initialVRoll << std::endl;
		//std::cout << "FssFL " << FssFL << "FsdFL " << FsdFL << "FtsFL " << FtsFL << "FtdFL " << FtdFL << std::endl;
		//std::cout << "FssRL " << FssRL << "FsdRL " << FsdRL << "FtsRL " << FtsRL << "FtdRL " << FtdRL << std::endl;
		
		//std::cout << "ForceZ " << ForceZ << "FsuspZFL " << FsuspZFL << "vsuspzfl " << outputspeedState.vSuspZFL << std::endl;
		//std::cout << "FweightedFL: " << FweightedFL << " FweightedFR: " << FweightedFR << " FweightedRR: " << FweightedRR << " FweightedRL: " << FweightedRL << std::endl;
		//std::cout << "FgravXFL: " << FgravXFL << " FgravXFR: " << FgravXFR << " FgravXRR: " << FgravXRR << " FgravXRL: " << FgravXRL << std::endl;
		//std::cout << "carState.roadAngleFL: " << carState.roadAngleFL << " carState.roadAngleFR: " << carState.roadAngleFR << " carState.roadAngleRR: " << carState.roadAngleRR << " carState.roadAngleRL: " << carState.roadAngleRL << std::endl;
		//std::cout << "sBFL " << sBFL << " sBFR " << sBFR << " dFx0FL " << dFx0FL << " sxRoofFL " << sxRoofFL << " cosPhiFL " << cosPhiFL << " RBFL " << RBFL << " RBFR " << RBFR << std::endl;
		//std::cout << "#vXWFL " << vXWFL << "#vXWFR " << vXWFR << "vXWRR " << vXWRR << "#vXWRL " << vXWRL << std::endl;
		//std::cout << "#MZ " << MZ << std::endl;
		//std::cout << "#TbrakeFL: " << TbrakeFL << " TbrakeFR: " << TbrakeFR << " TbrakeRR: " << TbrakeRR << " TbrakeRL: " << TbrakeRL << std::endl;
		//std::cout << "initialVX: " << initialVX << "initialVY: " << initialVY << std::endl;
		//std::cout << "vXWFL: " << vXWFL << " vXWFR: " << vXWFR << " vXWRR: " << vXWRR << " vXWRL: " << vXWRL << std::endl;
		//std::cout << "vYWFL: " << vYWFL << " vYWFR: " << vYWFR << " vYWRR: " << vYWRR << " vYWRL: " << vYWRL << std::endl;
		//std::cout << "FxRoadComb: " << FxRoadComb << " FyRoadComb: " << FxRoadComb << std::endl;
		//std::cout << "FxFLCC: " << FxFLCC <<  " FxFRCC: " << FxFRCC << " FxRRCC: " << FxRRCC << " FxRLCC: " << FxRLCC << std::endl;
		//std::cout << "FyFLCC: " << FyFLCC <<  " FyFRCC: " << FyFRCC << " FyRRCC: " << FyRRCC << " FyRLCC: " << FyRLCC << std::endl;
		//std::cout << "FxFL: " << FxFL <<  " FxFR: " << FxFR << " FxRR: " << FxRR << " FxRL: " << FxRL << std::endl;
		//std::cout << "FyFL: " << FyFL <<  " FyFR: " << FyFR << " FyRR: " << FyRR << " FyRL: " << FyRL << std::endl;
	}
	
	return outputAccelerationState;
}
