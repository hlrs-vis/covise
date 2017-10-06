#include "FWDCarState.h"
#include "iostream"

FWDCarState::FWDCarState()
{
	acceleratorAngle = 0.0;
	brakeForce = 0.0;
	clutch = 0.0;
	
	globalYaw = 0.0;
	localRoll = 0.0;
	localPitch = 0.0;
	
	localZPosSuspFL = 0.0;
	localZPosTireFL = 0.0;
	roadAngleFL = 0.0;
	
	localZPosSuspFR = 0.0;
	localZPosTireFR = 0.0;
	roadAngleFR = 0.0;
	
	localZPosSuspRR = 0.0;
	localZPosTireRR = 0.0;
	roadAngleRR = 0.0;
	
	localZPosSuspRL = 0.0;
	localZPosTireRL = 0.0;
	roadAngleRL = 0.0;
	
	inertiaPitch = 2000.0;
	inertiaRoll = 500.0;
	inertiaYaw = 2200.0;
	cogH = 0.3; //height of center of gravity
	sRearH = 1.0; //half of track at rear
	sFrontH = 1.0; //half of track at front
	lRear = 1.3; //distance COG to rear axle
	lFront = 1.5; //distance COG to front axle
	rwR = sqrt(sRearH * sRearH + lRear + lRear);
	rwF = sqrt(sFrontH * sFrontH + lFront + lFront);
	sinawR = sRearH / rwR;
	cosawR = lRear / rwR;
	sinawF = sFrontH / rwF;
	cosawF = lFront / rwF;
	std::cout << "rwR: " << rwR << "rwF: " << rwF << std::endl;
	std::cout << "sinawR: " << sinawR << "cosawR: " << cosawR << "sinawF: " << sinawF << "cosawF: " << cosawF << std::endl;
	
	suspOffsetSport = -0.25;
	suspOffsetComfort = 0.15;
	suspOffsetFL = suspOffsetComfort;
	suspOffsetFR = suspOffsetComfort;
	suspOffsetRR = suspOffsetComfort;
	suspOffsetRL = suspOffsetComfort;
	
	mCar =1500; //mass of car body
	mSusFL = 40;
	mSusFR = 40;
	mSusRR = 40;
	mSusRL = 40;
	mTotal = mCar + mSusFL + mSusFR + mSusRR + mSusRL;
	contactPatch = 0.1;
	tireRadF = 0.3;
	tireRadR = 0.3;
	csFLSport = 50000;//9000;
	csFRSport = 50000;//9000;
	csRRSport = 60000;//9000;
	csRLSport = 60000;//9000;
	dsFLSport = 7000;//5000;
	dsFRSport = 7000;//5000;
	dsRRSport = 7500;//5000;
	dsRLSport = 7500;//5000;
	csFLComfort = 8000;//9000;
	csFRComfort = 8000;//9000;
	csRRComfort = 9000;//9000;
	csRLComfort = 9000;//9000;
	dsFLComfort = 3000;//5000;
	dsFRComfort = 3000;//5000;
	dsRRComfort = 3000;//5000;
	dsRLComfort = 3000;//5000;
	csFL = csFLComfort;
	csFR = csFRComfort;
	csRR = csRRComfort;
	csRL = csRLComfort;
	dsFL = dsFLComfort;
	dsFR = dsFRComfort;
	dsRR = dsRRComfort;
	dsRL = dsRLComfort;
	ctFL = 120000;//40000;
	ctFR = 120000;//40000;
	ctRR = 120000;//40000;
	ctRL = 120000;//40000;
	dtFL = 45000;//45000;
	dtFR = 45000;//45000;
	dtRR = 45000;//45000;
	dtRL = 45000;//45000;
	tireDefFL = 0;
	tireDefFR = 0;
	tireDefRR = 0;
	tireDefRL = 0;
	tireDefSpeedFL = 0;
	tireDefSpeedFR = 0;
	tireDefSpeedRR = 0;
	tireDefSpeedRL = 0;
	
	//anti roll bars
	arbStiffnessF = 10000;
	arbStiffnessR = 10000;
	
	//joint offset
	jointOffsetFL = 0.2;
	jointOffsetFR = 0.2;
	jointOffsetRR = 0.2;
	jointOffsetRL = 0.2;
	suspNeutralFL = 0.4;
	suspNeutralFR = 0.4;
	suspNeutralRR = 0.4;
	suspNeutralRL = 0.4;
	
	//spring base tension; was 4000 spring and 4500 tire
	FsFL = 0;
	FtFL = 0;
	FsFR = 0;
	FtFR = 0;
	FsRR = 0;
	FtRR = 0;
	FsRL = 0;
	FtRL = 0;
	
	//camber
	camberInitFL=-0.4;
	camberCurrentFL=camberInitFL;
	camberInitFR=-0.4;
	camberCurrentFR=camberInitFR;
	camberInitRR=-0.4;
	camberCurrentRR=camberInitRR;
	camberInitRL=-0.4;
	camberCurrentRL=camberInitRL;
	
	//steering
	steeringRatio = 0.05;
	cRack = 1000000;
	dRack = 10000;
	cInt = 10000;
	dInt = 1000;
	inertiaWheelZ = 10;
	inertiaWheelY = 2;
	toeFL = 0.0;
	toeFR = 0.0;
	toeRR = 0.0;
	toeRL = 0.0;
	wheelAngleZFL = toeFL; //rotation of wheel depending on base toe and steering
	wheelAngleZFR = toeFR; //rotation of wheel depending on base toe and steering
	wheelAngleZRR = toeRR; //rotation of wheel depending on base toe and steering
	wheelAngleZRL = toeRL; //rotation of wheel depending on base toe and steering
	maxDeltaCurrent = 1000;
	
	//distances between center of gravity and origin of car model in open cover
	modelOriginOffsetX = 0;
	modelOriginOffsetY = 0;
	modelOriginOffsetZ = 0;
	
	vSteeringWheel = 0;
	posSteeringWheel = 0;
	posWheelLeftNeutral = 0;
	posWheelRightNeutral = 0;
	posWheelCombined = 0;
	deltaWheel = 0;
	
	//TMEasy
	cR = 80000;//100000;
	B = 0.3;//0.15
	fRoll = 1200.0;
	d = 0.000001;
	FzN = 4000;
	lambdaN = 0.375;
	lambda2N = 0.75;
	czN = 190000;
	cz2N = 206000;
	vN = 0.000000000001;
	FxMN = 5000;//3300
	FxM2N = 10100;//6500
	dFx0N = 70000;//90000
	dFx02N = 140000;//160000
	FxGN = 4800;//3200;
	FxG2N = 9800;//6000
	FyMN = 4500;//3100
	FyM2N = 6800;//5400
	dFy0N = 40000;//70000
	dFy02N = 60000;//100000
	FyGN = 4400;//3200;
	FyG2N = 6700;//5300
	sxMN = 0.13;//0.09
	sxM2N = 0.15;//0.11
	sxGN = 0.5;//0.4
	sxG2N = 0.6;//0.5
	syMN = 0.32;//0.18
	syM2N = 0.38;//0.2
	syGN = 0.7;//0.6
	syG2N = 0.9;//0.8
	nL0 = 0.179;
	syS = 0.6;//0.495;
	sy0 = 0.6;//0.205;
	boreXGN = 2500;
	boreXG2N = 4500;
	boreYGN = 2500;
	boreYG2N = 4500;
	/*phiFL1 = 0;
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
	phiRL3 = 0;*/
	cBore = 350000;//30000;
	dBore = 28000;//10000;
	inBore = 0.001;
	slipSoundLimit = 1.8;
	
	//gearbox
	gear = 0;
	gearRatioR = -4;
	gearRatio1 = 3.2;			//3.2
	gearRatio2 = 2.1;			//2.1
	gearRatio3 = 1.45;			//1.45
	gearRatio4 = 1;				//1
	gearRatio5 = 0.78;			//0.78
	finalDrive = 4;				//4
	
	//clutch
	clutchState = 0;			//1 is connected
	clutchSwitch = 0;			//1 is sticking
	clutchSlipBorder = 1;	//when clutch starts to stick, higher is earlier
	clutchTimer = 300;
	frictionCoefStatClutch = 700;
	frictionCoefDynClutch = 120;
	
	//engine
	idleSpeed = 1000 * 2 * M_PI / 60;
	revLimiter = 8000 * 2 * M_PI / 60;
	bEngine = 0.07;
	lossCoefEngine = 0.02;
	inertiaEngine = 0.2;
	
	//drive train
	bDrive = 0.3;
	lossCoefDrive = 0.02;
	inertiaDrive = 2;
	
	//brakes
	bRate = 40;
	Tstat = 80;
	brakeForceAmplification = 20.0;
	
	//drag
	cDrag = 0.05;
	
	//environment
	aGrav = 9.81;
	
	//platform movement
	vX1 = 0;
	vX2 = 0;
	vX3 = 0;
	vXCurrent = 0;
	accX = 0;
	vY1 = 0;
	vY2 = 0;
	vY3 = 0;
	vYCurrent = 0;
	accY = 0;
	mpL = 1.25;
	mpS = 0.76;
	mpLZ = 0;
	mpRZ = 0;
	mpBZ = 0;
	motorOffsetXFL = -1.2;
	motorOffsetYFL = 0;
	motorOffsetZFL = -0.5;
	motorOffsetXFR = 0.3;
	motorOffsetYFR = 0;
	motorOffsetZFR = -0.5;
	motorOffsetXR = -0.45;
	motorOffsetYR = 0;
	motorOffsetZR = 0.75;
	carZ = 0;
	mpHeight = 0;
	mpZReturnSpeed = 0.5;
}