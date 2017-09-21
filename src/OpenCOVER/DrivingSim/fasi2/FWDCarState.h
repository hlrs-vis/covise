#ifndef __FWDCarState_h
#define __FWDCarState_h

#include <osg/MatrixTransform>

class FWDCarState
{
public:
	FWDCarState();
	
	osg::Vec3d globalPosOddVec;
	
	double
	cogOpencoverRotX1,
	cogOpencoverRotX2,
	cogOpencoverRotX3,
	cogOpencoverRotY1,
	cogOpencoverRotY2,
	cogOpencoverRotY3,
	cogOpencoverRotZ1,
	cogOpencoverRotZ2,
	cogOpencoverRotZ3;
	
	osg::Matrix
	cogOpencoverPos,
	cogOpencoverRot,
	cogOpencoverRotInverse,
	cogOpencoverSpeed,
	globalPosOC,
	globalPosOdd,
	
	globalPosTireFL1,
	globalPosTireFL2,
	globalPosTireFL3,
	
	globalPosTireFR1,
	globalPosTireFR2,
	globalPosTireFR3,
	
	globalPosTireRR1,
	globalPosTireRR2,
	globalPosTireRR3,
	
	globalPosTireRL1,
	globalPosTireRL2,
	globalPosTireRL3,
	
	globalPosSuspFL,
	globalPosSuspFR,
	globalPosSuspRR,
	globalPosSuspRL,
	
	globalPosJointFL,
	globalPosJointFR,
	globalPosJointRR,
	globalPosJointRL;
	
	double 
	acceleratorAngle,
	brakeForce,
	clutch,
	
	globalYaw,
	localRoll,
	localPitch,
	
	localZPosSuspFL,
	localZPosTireFL,
	roadAngleFL,
	
	localZPosSuspFR,
	localZPosTireFR,
	roadAngleFR,
	
	localZPosSuspRR,
	localZPosTireRR,
	roadAngleRR,
	
	localZPosSuspRL,
	localZPosTireRL,
	roadAngleRL,
	
	wheelAngleZFL = toeFL, //rotation of wheel depending on base toe and steering
	wheelAngleZFR = toeFR, //rotation of wheel depending on base toe and steering
	wheelAngleZRR = toeRR, //rotation of wheel depending on base toe and steering
	wheelAngleZRL = toeRL, //rotation of wheel depending on base toe and steering
	
	//distances between center of gravity and origin of car model in open cover
	modelOriginOffsetX,
	modelOriginOffsetY,
	modelOriginOffsetZ;
	osg::Matrix modelOriginOffsetXMatrix;
	
	double inertiaPitch,
	inertiaRoll,
	inertiaYaw,
	cogH, //height of center of gravity
	sRearH, //half of track at rear
	sFrontH, //half of track at front
	lRear, //distance COG to rear axle
	lFront, //distance COG to front axle
	rwF,
	rwR,
	sinawR,
	cosawR,
	sinawF,
	cosawF,
	suspOffsetFL,
	suspOffsetFR,
	suspOffsetRR,
	suspOffsetRL,
	mCar, //mass of car body
	mSusFL,
	mSusFR,
	mSusRR,
	mSusRL,
	mTotal,
	contactPatch,
	tireRadF,
	tireRadR,
	csFL,
	csFR,
	csRR,
	csRL,
	dsFL,
	dsFR,
	dsRR,
	dsRL,
	csFLSport,
	csFRSport,
	csRRSport,
	csRLSport,
	dsFLSport,
	dsFRSport,
	dsRRSport,
	dsRLSport,
	csFLComfort,
	csFRComfort,
	csRRComfort,
	csRLComfort,
	dsFLComfort,
	dsFRComfort,
	dsRRComfort,
	dsRLComfort,
	ctFL,
	ctFR,
	ctRR,
	ctRL,
	dtFL,
	dtFR,
	dtRR,
	dtRL,
	tireDefFL,
	tireDefFR,
	tireDefRR,
	tireDefRL,
	tireDefSpeedFL,
	tireDefSpeedFR,
	tireDefSpeedRR,
	tireDefSpeedRL,
	
	//anti roll bars
	arbStiffnessF,
	arbStiffnessR,
	
	//joint offset
	jointOffsetFL,
	jointOffsetFR,
	jointOffsetRR,
	jointOffsetRL,
	suspNeutralFL,
	suspNeutralFR,
	suspNeutralRR,
	suspNeutralRL,
	
	//spring base tension, was 4000 spring and 4500 tire
	FsFL,
	FtFL,
	FsFR,
	FtFR,
	FsRR,
	FtRR,
	FsRL,
	FtRL,
	
	//road height
	roadHeightFL1,
	roadHeightFL2,
	roadHeightFL3,
	
	roadHeightFR1,
	roadHeightFR2,
	roadHeightFR3,
	
	roadHeightRR1,
	roadHeightRR2,
	roadHeightRR3,
	
	roadHeightRL1,
	roadHeightRL2,
	roadHeightRL3,
	
	//camber
	camberInitFL,
	camberCurrentFL,
	camberInitFR,
	camberCurrentFR,
	camberInitRR,
	camberCurrentRR,
	camberInitRL,
	camberCurrentRL,
	
	//steering
	steeringRatio,
	cRack,
	dRack,
	cInt,
	dInt,
	inertiaWheelZ,
	inertiaWheelY,
	toeFL,
	toeFR,
	toeRR,
	toeRL,
	vSteeringWheel,
	posSteeringWheel,
	posWheelLeftNeutral,
	posWheelRightNeutral,
	posWheelCombined,
	deltaWheel,
	maxDeltaCurrent,
	
	//TMEasy
	cR,
	B,
	fRoll,
	d,
	FzN,
	lambdaN,
	lambda2N,
	czN,
	cz2N,
	vN,
	FxMN,
	FxM2N,
	dFx0N,
	dFx02N,
	FxGN,
	FxG2N,
	FyMN,
	FyM2N,
	dFy0N,
	dFy02N,
	FyGN,
	FyG2N,
	sxMN,
	sxM2N,
	sxGN,
	sxG2N,
	syMN,
	syM2N,
	syGN,
	syG2N,
	nL0,
	syS,
	sy0,
	boreXGN,
	boreXG2N,
	boreYGN,
	boreYG2N,
	phiFL1,
	phiFL2,
	phiFL3,
	phiFR1,
	phiFR2,
	phiFR3,
	phiRR1,
	phiRR2,
	phiRR3,
	phiRL1,
	phiRL2,
	phiRL3,
	cBore,
	dBore,
	inBore,
	slipSoundLimit;
	
	//gearbox
	int gear,
	oldGear;
	double gearRatio,
	gearRatioR,
	gearRatio1,
	gearRatio2,
	gearRatio3,
	gearRatio4,
	gearRatio5,
	finalDrive,
	
	//clutch
	clutchState,
	clutchSwitch,
	clutchSlipBorder,
	clutchTimer,
	frictionCoefStatClutch,
	frictionCoefDynClutch,
	
	//engine
	torqueMap[66] = {	
		0,0,0,0,0,30,
		5,10,10,70,150,200,
		5,10,55,100,280,380,
		5,15,70,100,300,420,
		5,20,75,115,300,420,
		5,20,80,115,300,420,
		5,15,70,120,300,400,
		5,10,55,100,280,380,
		5,5,5,10,30,60,
		0,0,0,0,0,0,
		0,0,0,0,0,0
	},
	idleSpeed,
	revLimiter,
	bEngine,
	lossCoefEngine,
	inertiaEngine,
	Tcomb,				//interpolated torque
	
	//drive train
	bDrive,
	lossCoefDrive,
	inertiaDrive,
	
	//brakes
	bRate,
	Tstat,
	brakeForceAmplification,
	
	//drag
	cDrag,
	
	//environment
	aGrav,
	gravXComponent,
	gravYComponent,
	gravZComponent,
	
	//platform movement
	vX1,
	vX2,
	vX3,
	vXCurrent,
	accX,
	vY1,
	vY2,
	vY3,
	vYCurrent,
	accY,
	mpL,
	mpS,
	mpLZ,
	mpRZ,
	mpBZ,
	motorOffsetXFL,
	motorOffsetYFL,
	motorOffsetZFL,
	motorOffsetXFR,
	motorOffsetYFR,
	motorOffsetZFR,
	motorOffsetXR,
	motorOffsetYR,
	motorOffsetZR,
	carZ,
	mpHeight,
	mpZReturnSpeed,
	
	//for output from integrator
	timerCounter;
};

#endif