/*This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FourWheelDynamicsRealtime2.h"
#include "FWDState.h"
#include "FWDAccelerationState.h"
#include "FWDPositionState.h"

#include "GasPedal.h"
#include <fasi2.h>
#include <iostream>
#include <fstream>

FourWheelDynamicsRealtime2::FourWheelDynamicsRealtime2()
#ifdef MERCURY
    : XenomaiTask::XenomaiTask("FourWheelDynamicsRealtime2Task", 0, 99, 0)
#else
    : XenomaiTask::XenomaiTask("FourWheelDynamicsRealtime2Task", 0, 99, T_FPU | T_CPU(5))
#endif
    , dy(cardyn::getExpressionVector())
    , r_i(4)
    , n_i(4)
    , r_n(4)
    , //Start position of hermite
    t_n(4)
    , //Start tangent of hermite
    r_o(4)
    , //End position of hermite
    t_o(4)
    , //End tangent of hermite
    newIntersections(false)
    , i_w(4)
    , startPos(getStartPositionOnRoad())
    , leftRoad(true)
{
    Car2OddlotRotation.makeRotate(M_PI/2,0,0,1);
	Oddlot2CarRotation.makeRotate(-M_PI/2,0,0,1);
	Oddlot2OpencoverRotation.makeRotate(-M_PI/2,1,0,0);
	Opencover2OddlotRotation.makeRotate(M_PI/2,1,0,0);
	Car2OpencoverRotation = Car2OddlotRotation * Oddlot2OpencoverRotation;
	Opencover2CarRotation = Opencover2OddlotRotation * Oddlot2CarRotation;
	
	initState();

    //i_proj[0] = 1.0;

    runTask = true;
    doCenter = false;
    taskFinished = false;
    returningToAction = false;
    movingToGround = false;
    pause = true;
    overruns = 0;
    motPlat = ValidateMotionPlatform::instance();

	steerCon = new CanOpenController("can1");
	steerWheel = new XenomaiSteeringWheel(*steerCon, 1);
	roadPointFinder = new RoadPointFinder();
	
	roadPointFinder->getLongPosMutex().acquire(period);
	
	roadPointFinder->setLongPos(currentLongPosArray[0], 0);
	roadPointFinder->setLongPos(currentLongPosArray[1], 1);
	roadPointFinder->setLongPos(currentLongPosArray[2], 2);
	
	roadPointFinder->setLongPos(currentLongPosArray[3], 3);
	roadPointFinder->setLongPos(currentLongPosArray[4], 4);
	roadPointFinder->setLongPos(currentLongPosArray[5], 5);
	
	roadPointFinder->setLongPos(currentLongPosArray[6], 6);
	roadPointFinder->setLongPos(currentLongPosArray[7], 7);
	roadPointFinder->setLongPos(currentLongPosArray[8], 8);
	
	roadPointFinder->setLongPos(currentLongPosArray[9], 9);
	roadPointFinder->setLongPos(currentLongPosArray[10], 10);
	roadPointFinder->setLongPos(currentLongPosArray[11], 11);
	
	roadPointFinder->getLongPosMutex().release();
	
	roadPointFinder->getRoadMutex().acquire(period);
	
	roadPointFinder->setRoad(currentRoadArray[0], 0);
	roadPointFinder->setRoad(currentRoadArray[1], 1);
	roadPointFinder->setRoad(currentRoadArray[2], 2);
	
	roadPointFinder->setRoad(currentRoadArray[3], 3);
	roadPointFinder->setRoad(currentRoadArray[4], 4);
	roadPointFinder->setRoad(currentRoadArray[5], 5);
	
	roadPointFinder->setRoad(currentRoadArray[6], 6);
	roadPointFinder->setRoad(currentRoadArray[7], 7);
	roadPointFinder->setRoad(currentRoadArray[8], 8);
	
	roadPointFinder->setRoad(currentRoadArray[9], 9);
	roadPointFinder->setRoad(currentRoadArray[10], 10);
	roadPointFinder->setRoad(currentRoadArray[11], 11);

	roadPointFinder->getRoadMutex().release();
	
	roadPointFinder->getPositionMutex().acquire(period);
	
	osg::Matrix tempMatrixFL1 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixFL1.getTrans(), 0);
	osg::Matrix tempMatrixFL2 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixFL2.getTrans(), 1);
	osg::Matrix tempMatrixFL3 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixFL3.getTrans(), 2);
	
	osg::Matrix tempMatrixFR1 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixFR1.getTrans(), 3);
	osg::Matrix tempMatrixFR2 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixFR2.getTrans(), 4);
	osg::Matrix tempMatrixFR3 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixFR3.getTrans(), 5);
	
	osg::Matrix tempMatrixRR1 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixRR1.getTrans(), 6);
	osg::Matrix tempMatrixRR2 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixRR2.getTrans(), 7);
	osg::Matrix tempMatrixRR3 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixRR3.getTrans(), 8);
	
	osg::Matrix tempMatrixRL1 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixRL1.getTrans(), 9);
	osg::Matrix tempMatrixRL2 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixRL2.getTrans(), 10);
	osg::Matrix tempMatrixRL3 = carState.globalPosOC * Opencover2OddlotRotation;
	roadPointFinder->setPoint(tempMatrixRL3.getTrans(), 11);
	
	roadPointFinder->getPositionMutex().release();
	
	roadPointFinder->getCurrentHeightMutex().acquire(period);
	
	for(int i = 0; i < 12; i++)
	{
		roadPointFinder->setHeight(carState.globalPosOC.getTrans().y(),i);
	}
	
	roadPointFinder->getCurrentHeightMutex().release();
	
	roadPointFinder->checkLoadingRoads(false);
    start();
    k_wf_Slider = 17400.0;
    k_wr_Slider = 26100.0;
    d_wf_Slider = 2600.0;
    d_wr_Slider = 2600.0;
    clutchPedal = 0.0;
}

FourWheelDynamicsRealtime2::~FourWheelDynamicsRealtime2()
{
    RT_TASK_INFO info;
    inquire(info);

#ifdef MERCURY
    if (info.stat.status & __THREAD_S_STARTED)
#else
    if (info.status & T_STARTED)
#endif

    {
        runTask = false;
        while (!taskFinished)
        {
            usleep(100000);
        }
    }

    //steering com change
	delete steerWheel;
	delete steerCon;
	
	delete roadPointFinder;

    delete motPlat;
}

void FourWheelDynamicsRealtime2::initState()
{
    y = cardyn::StateVectorType();

    speedState = speedState * 0;
	speedState.vX = 0;
	speedState.vY = 0;
	speedState.vZ = 0;
	speedState.vYaw = 0;
	speedState.vRoll = 0;
	speedState.vPitch = 0;
	speedState.vSuspZFL = 0;
	speedState.vSuspZFR = 0;
	speedState.vSuspZRR = 0;
	speedState.vSuspZRL = 0;
	speedState.OmegaYFL = 0;
	speedState.OmegaYFR = 0;
	speedState.OmegaYRR = 0;
	speedState.OmegaYRL = 0;
	speedState.OmegaZFL = 0;
	speedState.OmegaZFR = 0;
	speedState.phiDotFL1 = 0;
	speedState.phiDotFL2 = 0;
	speedState.phiDotFL3 = 0;
	speedState.phiDotFR1 = 0;
	speedState.phiDotFR2 = 0;
	speedState.phiDotFR3 = 0;
	speedState.phiDotRR1 = 0;
	speedState.phiDotRR2 = 0;
	speedState.phiDotRR3 = 0;
	speedState.phiDotRL1 = 0;
	speedState.phiDotRL2 = 0;
	speedState.phiDotRL3 = 0;
	speedState.engineRPM = 1000 * 2 * M_PI / 60;
	speedState.TcolumnCombined = 0;
	speedState.Tclutch = 0;
	speedState.TclutchMax = 0;
	speedState.FweightedFL = 0;
	speedState.FweightedFR = 0;
	speedState.FweightedRR = 0;
	speedState.FweightedRL = 0;
	speedState.FtireFL = 0;
	speedState.FtireFR = 0;
	speedState.FtireRR = 0;
	speedState.FtireRL = 0;
	speedState.FxFL = 0;
	speedState.FxFR = 0;
	speedState.FxRR = 0;
	speedState.FxRL = 0;
	speedState.FyFL = 0;
	speedState.FyFR = 0;
	speedState.FyRR = 0;
	speedState.FyRL = 0;
	speedState.genericOut1 = 0;
	speedState.genericOut2 = 0;
	speedState.genericOut3 = 0;
	speedState.genericOut4 = 0;
	speedState.genericOut5 = 0;
	speedState.genericOut6 = 0;
	speedState.genericOut7 = 0;
	speedState.genericOut8 = 0;
	std::cout << "speedState.vX at initialization: " << speedState.vX << std::endl;
	std::cout << "speedState.OmegaYRL at initialization: " << speedState.OmegaYRL << std::endl;
	
	carState.cogOpencoverSpeed.makeIdentity();
	
	//initial car position
	carState.cogOpencoverPos.makeTranslate(0,1,0);
    if (startPos.first)
    {
        Transform transform = startPos.first->getRoadTransform(startPos.second.u(), startPos.second.v());
        gealg::mv<4, 0x06050300>::type R_b;
        R_b[0] = transform.q().w();
        R_b[1] = transform.q().z();
        R_b[2] = -transform.q().y();
        R_b[3] = transform.q().x();
        gealg::mv<4, 0x06050300>::type R_xodr = exp(0.5 * (-0.5 * M_PI * cardyn::x * cardyn::y));
        std::get<2>(y) = !(R_b * R_xodr);

        gealg::mv<3, 0x040201>::type p_b_init;
        p_b_init[2] = 0.75;
        gealg::mv<3, 0x040201>::type p_road;
        p_road[0] = transform.v().y();
        p_road[1] = -transform.v().x();
        p_road[2] = transform.v().z();
		std::cout << "p_road: " << p_road << std::endl;
        std::get<0>(y) = p_road + grade<1>((!std::get<2>(y)) * p_b_init * (std::get<2>(y)));

        currentRoad = startPos.first;
				
		currentRoadFL1 = startPos.first;
		currentRoadFL2 = startPos.first;
		currentRoadFL3 = startPos.first;
		
		currentRoadFR1 = startPos.first;
		currentRoadFR2 = startPos.first;
		currentRoadFR3 = startPos.first;
		
		currentRoadRR1 = startPos.first;
		currentRoadRR2 = startPos.first;
		currentRoadRR3 = startPos.first;
		
		currentRoadRL1 = startPos.first;
		currentRoadRL2 = startPos.first;
		currentRoadRL3 = startPos.first;
		
		currentRoadArray[0] = currentRoadFL1;
		currentRoadArray[1] = currentRoadFL2;
		currentRoadArray[2] = currentRoadFL3;
		
		currentRoadArray[3] = currentRoadFR1;
		currentRoadArray[4] = currentRoadFR2;
		currentRoadArray[5] = currentRoadFR3;
		
		currentRoadArray[6] = currentRoadRR1;
		currentRoadArray[7] = currentRoadRR2;
		currentRoadArray[8] = currentRoadRR3;
		
		currentRoadArray[9] = currentRoadRL1;
		currentRoadArray[10] = currentRoadRL2;
		currentRoadArray[11] = currentRoadRL3;
		
        currentLongPos = startPos.second.u();
		
		currentLongPosFL1 = startPos.second.u();
		currentLongPosFL2 = startPos.second.u();
		currentLongPosFL3 = startPos.second.u();
		
		currentLongPosFR1 = startPos.second.u();
		currentLongPosFR2 = startPos.second.u();
		currentLongPosFR3 = startPos.second.u();
		
		currentLongPosRR1 = startPos.second.u();
		currentLongPosRR2 = startPos.second.u();
		currentLongPosRR3 = startPos.second.u();
		
		currentLongPosRL1 = startPos.second.u();
		currentLongPosRL2 = startPos.second.u();
		currentLongPosRL3 = startPos.second.u();
		
		currentLongPosArray[0] = currentLongPosFL1;
		currentLongPosArray[1] = currentLongPosFL2;
		currentLongPosArray[2] = currentLongPosFL3;
		
		currentLongPosArray[3] = currentLongPosFR1;
		currentLongPosArray[4] = currentLongPosFR2;
		currentLongPosArray[5] = currentLongPosFR3;
		
		currentLongPosArray[6] = currentLongPosRR1;
		currentLongPosArray[7] = currentLongPosRR2;
		currentLongPosArray[8] = currentLongPosRR3;
		
		currentLongPosArray[9] = currentLongPosRL1;
		currentLongPosArray[10] = currentLongPosRL2;
		currentLongPosArray[11] = currentLongPosRL3;

        leftRoad = false;
        std::cout << "Found road: " << startPos.first->getId() << ", u: " << startPos.second.u() << ", v: " << startPos.second.v() << std::endl;
        std::cout << "\t p_b: " << std::get<0>(y) << ", R_b: " << std::get<2>(y) << std::endl;
		
		//initial car position
		carState.cogOpencoverPos.makeTranslate(-p_road[1],p_road[2]+1,-p_road[0]);
		carState.globalPosOC.makeTranslate(-p_road[1],p_road[2]+0.5/*0.12346*/,-p_road[0]);
		std::cout << "Road coordinates in" << " 1 " << p_road[0]  << " 2 " << p_road[1]  << " 3 " << p_road[2] << std::endl;
		std::cout << "first road: " << currentRoadFL1 << " end of first road" << std::endl;
		std::cout << "first array road: " << currentRoadArray[0] << std::endl;;
		
		
    }
    else
    {
        std::get<0>(y)[2] = -0.2; //Initial position
        std::get<2>(y)[0] = 1.0; //Initial orientation (Important: magnitude be one!)
        currentRoad = NULL;
		currentRoadFL1 = NULL;
		currentRoadFL2 = NULL;
		currentRoadFL3 = NULL;
		
		currentRoadFR1 = NULL;
		currentRoadFR2 = NULL;
		currentRoadFR3 = NULL;
		
		currentRoadRR1 = NULL;
		currentRoadRR2 = NULL;
		currentRoadRR3 = NULL;
		
		currentRoadRL1 = NULL;
		currentRoadRL2 = NULL;
		currentRoadRL3 = NULL;
		
		currentLongPos = -1.0;
		currentLongPosFL1 = -1.0;
		currentLongPosFL2 = -1.0;
		currentLongPosFL3 = -1.0;
		
		currentLongPosFR1 = -1.0;
		currentLongPosFR2 = -1.0;
		currentLongPosFR3 = -1.0;
		
		currentLongPosRR1 = -1.0;
		currentLongPosRR2 = -1.0;
		currentLongPosRR3 = -1.0;
		
		currentLongPosRL1 = -1.0;
		currentLongPosRL2 = -1.0;
		currentLongPosRL3 = -1.0;
		
        leftRoad = true;
		
    }
    std::get<39>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::get<40>(y)[0] = 1.0; //Initial steering wheel position: magnitude be one!
    std::get<41>(y)[0] = cardyn::i_a; //Initial steering wheel position: magnitude be one!

    r_i[0] = gealg::mv<3, 0x040201>::type();
    r_i[1] = gealg::mv<3, 0x040201>::type();
    r_i[2] = gealg::mv<3, 0x040201>::type();
    r_i[3] = gealg::mv<3, 0x040201>::type();
    n_i[0] = gealg::mv<3, 0x040201>::type();
    n_i[1] = gealg::mv<3, 0x040201>::type();
    n_i[2] = gealg::mv<3, 0x040201>::type();
    n_i[3] = gealg::mv<3, 0x040201>::type();

    r_n[0] = (cardyn::p_b + cardyn::r_wfl - cardyn::z * cardyn::r_w - cardyn::u_wfl)(y);
    r_o[0] = r_n[0];
    r_n[1] = (cardyn::p_b + cardyn::r_wfr - cardyn::z * cardyn::r_w - cardyn::u_wfr)(y);
    r_o[1] = r_n[1];
    r_n[2] = (cardyn::p_b + cardyn::r_wrl - cardyn::z * cardyn::r_w - cardyn::u_wrl)(y);
    r_o[2] = r_n[2];
    r_n[3] = (cardyn::p_b + cardyn::r_wrr - cardyn::z * cardyn::r_w - cardyn::u_wrr)(y);
    r_o[3] = r_n[3];
    t_n[0] = gealg::mv<3, 0x040201>::type();
    t_n[1] = gealg::mv<3, 0x040201>::type();
    t_n[2] = gealg::mv<3, 0x040201>::type();
    t_n[3] = gealg::mv<3, 0x040201>::type();
    t_o[0] = gealg::mv<3, 0x040201>::type();
    t_o[1] = gealg::mv<3, 0x040201>::type();
    t_o[2] = gealg::mv<3, 0x040201>::type();
    t_o[3] = gealg::mv<3, 0x040201>::type();

    cardyn::k_wf.e_(y)[0] = 17400.0;
    cardyn::k_wr.e_(y)[0] = 26100.0;
    cardyn::d_wf.e_(y)[0] = 2600.0;
    cardyn::d_wr.e_(y)[0] = 2600.0;

    newIntersections = false;
    rpms = 0.0;
}

void FourWheelDynamicsRealtime2::setVehicleTransformation(const osg::Matrix &m)
{
    resetState();

    std::get<0>(y)[0] = -m(3, 2);
    std::get<0>(y)[1] = -m(3, 0);
    std::get<0>(y)[2] = m(3, 1);

    std::cout << "Reset: position: " << std::get<0>(y) << std::endl;
}

void FourWheelDynamicsRealtime2::resetState()
{
    if (!pause)
    {
        pause = true;
        initState();
        platformReturnToAction();
    }
    else
    {
        initState();
    }
}

void FourWheelDynamicsRealtime2::move()
{
    y_frame = this->y;

    gealg::mv<3, 0x040201>::type r_bg = (cardyn::r_wfl + cardyn::r_wfr + cardyn::r_wrl + cardyn::r_wrr) * 0.25 - cardyn::z * cardyn::r_w;
    gealg::mv<3, 0x040201>::type p_bg = (cardyn::p_b + grade<1>((!cardyn::q_b) * r_bg * cardyn::q_b))(y_frame);

    /*chassisTrans.setTrans(osg::Vec3(-p_bg[1], p_bg[2], -p_bg[0]));
    chassisTrans.setRotate(osg::Quat(std::get<2>(y_frame)[2],
                                     std::get<2>(y_frame)[1],
                                     -std::get<2>(y_frame)[3],
                                     std::get<2>(y_frame)[0]));*/	

    //vehicle->setVRMLVehicle(chassisTrans);

    cardyn::k_wf.e_(y)[0] = k_wf_Slider;
    cardyn::d_wf.e_(y)[0] = d_wf_Slider;
    cardyn::k_wr.e_(y)[0] = k_wr_Slider;
    cardyn::d_wr.e_(y)[0] = d_wr_Slider;
}

void FourWheelDynamicsRealtime2::setSportDamper(bool sport)
{
    if (sport)
    {
        carState.csFL = carState.csFLSport;
		carState.csFR = carState.csFRSport;
		carState.csRR = carState.csRRSport;
		carState.csRL = carState.csRLSport;
		carState.dsFL = carState.dsFLSport;
		carState.dsFR = carState.dsFRSport;
		carState.dsRR = carState.dsRRSport;
		carState.dsRL = carState.dsRLSport; 
    }
    else
    {
		carState.csFL = carState.csFLComfort;
		carState.csFR = carState.csFRComfort;
		carState.csRR = carState.csRRComfort;
		carState.csRL = carState.csRLComfort;
		carState.dsFL = carState.dsFLComfort;
		carState.dsFR = carState.dsFRComfort;
		carState.dsRR = carState.dsRRComfort;
		carState.dsRL = carState.dsRLComfort; 
    }
}

void FourWheelDynamicsRealtime2::run()
{
    double current = 0.0;
    std::deque<double> currentDeque(10, 0.0);
    std::deque<double>::iterator currentDequeIt;
	
	std::cerr << "--- steerWheel->init(); ---" << std::endl;
    steerWheel->init();
	
	std::cerr << "--- FourWheelDynamicsRealtime2::FourWheelDynamicsRealtime2(): Starting ValidateMotionPlatform task ---" << std::endl;
    //Motion platform
    motPlat->start();
    std::cerr << "--- motPlat->start();  ---" << std::endl;
    while (!motPlat->isInitialized())
    {
        rt_task_sleep(1000000);
        std::cerr << "--- motPlat->waiting for initialization();  ---" << std::endl;
    }
    std::cerr << "--- motPlat->start(); done ---" << std::endl;
    set_periodic(period);
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
    while (!motPlat->isGrounded())
    {
        rt_task_wait_period(&overruns);
    }
    std::cerr << "--- isGrounded(); done ---" << std::endl;
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();

    static bool oldleftRoad = false;
	double initialTicks = 0;
	bool useTest = true;
	/*FILE * outFile;
	outFile = fopen("/mnt/raid/home/hpckgran/Documents/outputLogFile.txt","a");*/
	std::ofstream outfile;
	outfile.open("/mnt/raid/home/hpckgran/Documents/outputLogFile.txt", std::ios::out /*| std::ios::trunc*/);
	//double timerTimer;
	while (runTask)
	{
		double startTicks;
		double currentTicks;
		//startTicks = (double) rt_timer_read();
		
		/*double currentLoopTicks = startTicks - initialTicks;
		initialTicks = startTicks;
		if (timerCounter == 0)
		{
			startTicks = (double) rt_timer_read();
		}*/
		double h = (double)(period * (overruns + 1)) * 1e-9;
		h = 0.001; //to avoid issues with big overruns 
		
		if (overruns != 0)
		{
			std::cerr << "FourWheelDynamicsRealtimeRealtime::run(): overruns: " << overruns << std::endl;
			overruns=0;
		}
				
		bool leftRoadOnce = false;
		if (oldleftRoad != leftRoad)
		{
			oldleftRoad = leftRoad;
			if (leftRoad)
				leftRoadOnce = true;
		}
		
		/*if (leftRoad)
		{
			if (leftRoadOnce)
			{
				std::cout << "Left Road!" << std::endl;
			}
			else
			{
				std::cout << "reset" << std::endl;
			}
			resetState();
		}*/
		
		if (movingToGround)
		{
			steerWheel->setCurrent(0);
			
			if (motPlat->isGrounded())
			{
				movingToGround = false;
				motPlat->getSendMutex().acquire(period);
				motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
				motPlat->getSendMutex().release();
			}
		}
		else if (returningToAction)
		{
			if (motPlat->isLifted())
			{
				returningToAction = false;
				pause = false;

				motPlat->getSendMutex().acquire(period);
				//motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>();
				motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>();
				for (unsigned int motIt = 0; motIt < motPlat->numLinMots; ++motIt)
				{
					motPlat->setVelocitySetpoint(motIt, ValidateMotionPlatform::velMax);
					motPlat->setAccelerationSetpoint(motIt, ValidateMotionPlatform::accMax);
				}
				motPlat->getSendMutex().release();
			}
		}
		else
		{
			if (doCenter)
			{
				doCenter = false;
				//fprintf(stderr, "center\n");
				std::cout << "do center" << std::endl;
				
				//steering com change
				bool centerSuccess = steerWheel->center();
				std::cout << "center succcess? " << centerSuccess << std::endl;
				fprintf(stderr, "center done\n");
			}
			current = 0.0;
		}
		
		steerPosition = steerWheel->getPosition();
		steerWheelAngle = -2 * M_PI * steerPosition;
		
		
		double oldPosSteeringWheel = carState.posSteeringWheel;
		carState.posSteeringWheel = steerWheelAngle;
		carState.vSteeringWheel=(carState.posSteeringWheel-oldPosSteeringWheel)/h;
		
		if (!pause)
		{
			carState.acceleratorAngle = GasPedal::instance()->getActualAngle() / 100.0;
			carState.brakeForce = motPlat->getBrakeForce() * carState.brakeForceAmplification;
			carState.oldGear = carState.gear;
			carState.gear = fasi2::instance()->sharedState.gear;
			
			carState.cogOpencoverRotX1 = carState.cogOpencoverRot(0,0);
			carState.cogOpencoverRotX2 = carState.cogOpencoverRot(0,1);
			carState.cogOpencoverRotX3 = carState.cogOpencoverRot(0,2);
			carState.cogOpencoverRotY1 = carState.cogOpencoverRot(1,0);
			carState.cogOpencoverRotY2 = carState.cogOpencoverRot(1,1);
			carState.cogOpencoverRotY3 = carState.cogOpencoverRot(1,2);
			carState.cogOpencoverRotZ1 = carState.cogOpencoverRot(2,0);
			carState.cogOpencoverRotZ2 = carState.cogOpencoverRot(2,1);
			carState.cogOpencoverRotZ3 = carState.cogOpencoverRot(2,2);
			
			carState.cogOpencoverRotInverse = carState.cogOpencoverRot;
			carState.cogOpencoverRotInverse = carState.cogOpencoverRotInverse.inverse(carState.cogOpencoverRotInverse);
			
			//calculate speeds in car coordinates
			osg::Matrix CCSpeeds = carState.cogOpencoverSpeed * carState.cogOpencoverRotInverse * Opencover2CarRotation;
			speedState.vX = CCSpeeds(3,0);
			speedState.vY = CCSpeeds(3,1);
			speedState.vZ = CCSpeeds(3,2);
			
			
			//calculate gravity direction
			osg::Matrix fGrav;
			fGrav.makeTranslate(0,-carState.mTotal * carState.aGrav,0);
			fGrav = fGrav * carState.cogOpencoverRotInverse * Opencover2CarRotation;
			carState.gravXComponent = fGrav(3,0);
			carState.gravYComponent = fGrav(3,1);
			carState.gravZComponent = fGrav(3,2);
			
			carState.localPitch = -atan(carState.gravXComponent / carState.gravZComponent);
			if(isnan(carState.localPitch))
			{
				carState.localPitch = 0;
			}
			carState.localRoll = atan(carState.gravYComponent / carState.gravZComponent);
			if(isnan(carState.localRoll))
			{
				carState.localRoll = 0;
			}
			
			//calculate geometry
			//current wheel Positions
			osg::Matrix localTrans;
			osg::Matrix frontPoint;
			osg::Matrix rearPoint;
			frontPoint.makeTranslate(carState.contactPatch,0.0,0.0);
			rearPoint.makeTranslate(-carState.contactPatch,0.0,0.0);
			osg::Matrix steerAngle;
			
			steerAngle.makeRotate(carState.wheelAngleZFL,0.0,0.0,1.0);
			localTrans.makeTranslate(carState.lFront, carState.sFrontH, carState.jointOffsetFL - carState.suspNeutralFL + carState.localZPosSuspFL - carState.tireRadF);//TODO: local z pos maybe negative
			carState.globalPosTireFL1 = rearPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireFL2 = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireFL3 = frontPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			steerAngle.makeRotate(carState.wheelAngleZFR,0.0,0.0,1.0);
			localTrans.makeTranslate(carState.lFront, -carState.sFrontH, carState.jointOffsetFR - carState.suspNeutralFR + carState.localZPosSuspFR - carState.tireRadF);
			carState.globalPosTireFR1 = rearPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireFR2 = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireFR3 = frontPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			steerAngle.makeRotate(carState.wheelAngleZRR,0.0,0.0,1.0);
			localTrans.makeTranslate(-carState.lRear, -carState.sRearH, carState.jointOffsetRR - carState.suspNeutralRR + carState.localZPosSuspRR - carState.tireRadR);
			carState.globalPosTireRR1 = rearPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireRR2 = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireRR3 = frontPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			steerAngle.makeRotate(carState.wheelAngleZRL,0.0,0.0,1.0);
			localTrans.makeTranslate(-carState.lRear, carState.sRearH, carState.jointOffsetRL - carState.suspNeutralRL + carState.localZPosSuspRL - carState.tireRadR);
			carState.globalPosTireRL1 = rearPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireRL2 = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			carState.globalPosTireRL3 = frontPoint * steerAngle * localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			//current joint positions
			localTrans.makeTranslate(carState.lFront, carState.sFrontH, carState.jointOffsetFL);
			carState.globalPosJointFL = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			localTrans.makeTranslate(carState.lFront, -carState.sFrontH, carState.jointOffsetFR);
			carState.globalPosJointFR = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			localTrans.makeTranslate(-carState.lRear, -carState.sRearH, carState.jointOffsetRR);
			carState.globalPosJointRR = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			localTrans.makeTranslate(-carState.lRear, carState.sRearH, carState.jointOffsetRL);
			carState.globalPosJointRL = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			//current susp positions
			localTrans.makeIdentity();
			localTrans.makeTranslate(carState.lFront, carState.sFrontH, carState.jointOffsetFL - carState.suspNeutralFL + carState.localZPosSuspFL);
			carState.globalPosSuspFL = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			localTrans.makeIdentity();
			localTrans.makeTranslate(carState.lFront, -carState.sFrontH, carState.jointOffsetFR - carState.suspNeutralFR + carState.localZPosSuspFR);
			carState.globalPosSuspFR = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			localTrans.makeIdentity();
			localTrans.makeTranslate(-carState.lRear, -carState.sRearH, carState.jointOffsetRR - carState.suspNeutralRR + carState.localZPosSuspRR);
			carState.globalPosSuspRR = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			localTrans.makeIdentity();
			localTrans.makeTranslate(-carState.lRear, carState.sRearH, carState.jointOffsetRL - carState.suspNeutralRL + carState.localZPosSuspRL);
			carState.globalPosSuspRL = localTrans * Car2OpencoverRotation * carState.cogOpencoverRot * carState.cogOpencoverPos;
			
			double wheelPosRLX;
			double wheelPosRLY;
			double wheelPosRRX;
			double wheelPosRRY;
			
			startTicks = (double) rt_timer_read();
			
			//get positions of tires on road
			RoadSystem *rSystem = RoadSystem::Instance();
			
			/*roadPointFinder->getLongPosMutex().acquire(period);
			
			currentLongPosArray[0] = roadPointFinder->getLongPos(0);
			currentLongPosArray[1] = roadPointFinder->getLongPos(1);
			currentLongPosArray[2] = roadPointFinder->getLongPos(2);
			
			currentLongPosArray[3] = roadPointFinder->getLongPos(3);
			currentLongPosArray[4] = roadPointFinder->getLongPos(4);
			currentLongPosArray[5] = roadPointFinder->getLongPos(5);
			
			currentLongPosArray[6] = roadPointFinder->getLongPos(6);
			currentLongPosArray[7] = roadPointFinder->getLongPos(7);
			currentLongPosArray[8] = roadPointFinder->getLongPos(8);
			
			currentLongPosArray[9] = roadPointFinder->getLongPos(9);
			currentLongPosArray[10] = roadPointFinder->getLongPos(10);
			currentLongPosArray[11] = roadPointFinder->getLongPos(11);
			
			roadPointFinder->getLongPosMutex().release();*/
			
			/*roadPointFinder->getRoadMutex().acquire(period);
			
			currentRoadArray[0] = roadPointFinder->getRoad(0);
			currentRoadArray[1] = roadPointFinder->getRoad(1);
			currentRoadArray[2] = roadPointFinder->getRoad(2);
			
			currentRoadArray[3] = roadPointFinder->getRoad(3);
			currentRoadArray[4] = roadPointFinder->getRoad(4);
			currentRoadArray[5] = roadPointFinder->getRoad(5);
			
			currentRoadArray[6] = roadPointFinder->getRoad(6);
			currentRoadArray[7] = roadPointFinder->getRoad(7);
			currentRoadArray[8] = roadPointFinder->getRoad(8);
			
			currentRoadArray[9] = roadPointFinder->getRoad(9);
			currentRoadArray[10] = roadPointFinder->getRoad(10);
			currentRoadArray[11] = roadPointFinder->getRoad(11);
			
			roadPointFinder->getRoadMutex().release();*/
			
			roadPointFinder->getPositionMutex().acquire(period);
			
			osg::Matrix tempMatrixFL1 = carState.globalPosTireFL1 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixFL1.getTrans(), 0);
			osg::Matrix tempMatrixFL2 = carState.globalPosTireFL2 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixFL2.getTrans(), 1);
			osg::Matrix tempMatrixFL3 = carState.globalPosTireFL3 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixFL3.getTrans(), 2);
			
			osg::Matrix tempMatrixFR1 = carState.globalPosTireFR1 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixFR1.getTrans(), 3);
			osg::Matrix tempMatrixFR2 = carState.globalPosTireFR2 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixFR2.getTrans(), 4);
			osg::Matrix tempMatrixFR3 = carState.globalPosTireFR3 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixFR3.getTrans(), 5);
			
			osg::Matrix tempMatrixRR1 = carState.globalPosTireRR1 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixRR1.getTrans(), 6);
			osg::Matrix tempMatrixRR2 = carState.globalPosTireRR2 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixRR2.getTrans(), 7);
			osg::Matrix tempMatrixRR3 = carState.globalPosTireRR3 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixRR3.getTrans(), 8);
			
			osg::Matrix tempMatrixRL1 = carState.globalPosTireRL1 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixRL1.getTrans(), 9);
			osg::Matrix tempMatrixRL2 = carState.globalPosTireRL2 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixRL2.getTrans(), 10);
			osg::Matrix tempMatrixRL3 = carState.globalPosTireRL3 * Opencover2OddlotRotation;
			roadPointFinder->setPoint(tempMatrixRL3.getTrans(), 11);
			
			roadPointFinder->getPositionMutex().release();
			
			roadPointFinder->getCurrentHeightMutex().acquire(period);
			
			carState.roadHeightFL1 = roadPointFinder->getHeight(0);
			carState.roadHeightFL2 = roadPointFinder->getHeight(1);
			carState.roadHeightFL3 = roadPointFinder->getHeight(2);
			carState.roadAngleFL = atan(-(carState.roadHeightFL3 - carState.roadHeightFL1) / (cos(carState.wheelAngleZFL) * 2 * carState.contactPatch));
			
			carState.roadHeightFR1 = roadPointFinder->getHeight(3);
			carState.roadHeightFR2 = roadPointFinder->getHeight(4);
			carState.roadHeightFR3 = roadPointFinder->getHeight(5);
			carState.roadAngleFR = atan(-(carState.roadHeightFR3 - carState.roadHeightFR1) / (cos(carState.wheelAngleZFR) * 2 * carState.contactPatch));
			
			carState.roadHeightRR1 = roadPointFinder->getHeight(6);
			carState.roadHeightRR2 = roadPointFinder->getHeight(7);
			carState.roadHeightRR3 = roadPointFinder->getHeight(8);
			carState.roadAngleRR = atan(-(carState.roadHeightRR3 - carState.roadHeightRR1) / (cos(carState.wheelAngleZRR) * 2 * carState.contactPatch));
			
			carState.roadHeightRL1 = roadPointFinder->getHeight(9);
			carState.roadHeightRL2 = roadPointFinder->getHeight(10);
			carState.roadHeightRL3 = roadPointFinder->getHeight(11);
			carState.roadAngleRL = atan(-(carState.roadHeightRL3 - carState.roadHeightRL1) / (cos(carState.wheelAngleZRL) * 2 * carState.contactPatch));
			
			roadPointFinder->getCurrentHeightMutex().release();
			
			/*
			if(currentRoadArray[0])
			{
				Vector3D searchInVec(tempMatrixFL1.getTrans().x(), tempMatrixFL1.getTrans().y(), tempMatrixFL1.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[0]->searchPositionNoBorder(searchInVec, currentLongPosArray[0]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[0] = searchOutVec.x();
					RoadPoint point = currentRoadArray[0]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightFL1 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire fl1 left road!" << std::endl;
						leftRoad = true;
					}
				}
				
			}
			if(currentRoadArray[1])
			{
				Vector3D searchInVec(tempMatrixFL2.getTrans().x(), tempMatrixFL2.getTrans().y(), tempMatrixFL2.getTrans().z());
				//std::cout << "fl2 in: pointx " << searchInVec.x() << " pointy " << searchInVec.y() << " pointz " << searchInVec.z() << std::endl;
				Vector2D searchOutVec = currentRoadArray[1]->searchPositionNoBorder(searchInVec, currentLongPosArray[1]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[1] = searchOutVec.x();
					RoadPoint point = currentRoadArray[1]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightFL2 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire fl2 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			if(currentRoadArray[2])
			{
				Vector3D searchInVec(tempMatrixFL3.getTrans().x(), tempMatrixFL3.getTrans().y(), tempMatrixFL3.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[2]->searchPositionNoBorder(searchInVec, currentLongPosArray[2]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[2] = searchOutVec.x();
					RoadPoint point = currentRoadArray[2]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightFL3 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire fl3 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			carState.roadAngleFL = atan(-(carState.roadHeightFL3 - carState.roadHeightFL1) / (cos(carState.wheelAngleZFL) * 2 * carState.contactPatch));
			
			if(currentRoadArray[3])
			{
				Vector3D searchInVec(tempMatrixFR1.getTrans().x(), tempMatrixFR1.getTrans().y(), tempMatrixFR1.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[3]->searchPositionNoBorder(searchInVec, currentLongPosArray[3]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[3] = searchOutVec.x();
					RoadPoint point = currentRoadArray[3]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightFR1 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire fr1 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			if(currentRoadArray[4])
			{
				Vector3D searchInVec(tempMatrixFR2.getTrans().x(), tempMatrixFR2.getTrans().y(), tempMatrixFR2.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[4]->searchPositionNoBorder(searchInVec, currentLongPosArray[4]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[4] = searchOutVec.x();
					RoadPoint point = currentRoadArray[4]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightFR2 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire fr2 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			if(currentRoadArray[5])
			{
				Vector3D searchInVec(tempMatrixFR3.getTrans().x(), tempMatrixFR3.getTrans().y(), tempMatrixFR3.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[5]->searchPositionNoBorder(searchInVec, currentLongPosArray[5]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[5] = searchOutVec.x();
					RoadPoint point = currentRoadArray[5]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightFR2 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire fr2 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			carState.roadAngleFR = atan(-(carState.roadHeightFR3 - carState.roadHeightFR1) / (cos(carState.wheelAngleZFR) * 2 * carState.contactPatch));
			
			if(currentRoadArray[6])
			{
				Vector3D searchInVec(tempMatrixRR1.getTrans().x(), tempMatrixRR1.getTrans().y(), tempMatrixRR1.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[6]->searchPositionNoBorder(searchInVec, currentLongPosArray[6]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[6] = searchOutVec.x();
					RoadPoint point = currentRoadArray[6]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightRR1 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire rr1 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			if(currentRoadArray[7])
			{
				Vector3D searchInVec(tempMatrixRR2.getTrans().x(), tempMatrixRR2.getTrans().y(), tempMatrixRR2.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[7]->searchPositionNoBorder(searchInVec, currentLongPosArray[7]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[7] = searchOutVec.x();
					RoadPoint point = currentRoadArray[7]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightRR2 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire rr2 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			if(currentRoadArray[8])
			{
				Vector3D searchInVec(tempMatrixRR3.getTrans().x(), tempMatrixRR3.getTrans().y(), tempMatrixRR3.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[8]->searchPositionNoBorder(searchInVec, currentLongPosArray[8]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[8] = searchOutVec.x();
					RoadPoint point = currentRoadArray[8]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightRR3 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire rr3 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			carState.roadAngleRR = atan(-(carState.roadHeightRR3 - carState.roadHeightRR1) / (cos(carState.wheelAngleZRR) * 2 * carState.contactPatch));
			
			if(currentRoadArray[9])
			{
				Vector3D searchInVec(tempMatrixRL1.getTrans().x(), tempMatrixRL1.getTrans().y(), tempMatrixRL1.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[9]->searchPositionNoBorder(searchInVec, currentLongPosArray[9]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[9] = searchOutVec.x();
					RoadPoint point = currentRoadArray[9]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightRL1 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire rl1 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			if(currentRoadArray[10])
			{
				Vector3D searchInVec(tempMatrixRL2.getTrans().x(), tempMatrixRL2.getTrans().y(), tempMatrixRL2.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[10]->searchPositionNoBorder(searchInVec, currentLongPosArray[10]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[10] = searchOutVec.x();
					RoadPoint point = currentRoadArray[10]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightRL2 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire rl2 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			if(currentRoadArray[11])
			{
				Vector3D searchInVec(tempMatrixRL3.getTrans().x(), tempMatrixRL3.getTrans().y(), tempMatrixRL3.getTrans().z());
				
				Vector2D searchOutVec = currentRoadArray[11]->searchPositionNoBorder(searchInVec, currentLongPosArray[11]);
				//std::cout << "search road from scratch" << std::endl;
				if (!searchOutVec.isNaV())
				{
					currentLongPosArray[11] = searchOutVec.x();
					RoadPoint point = currentRoadArray[11]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
					if (!isnan(point.x()))
					{
						carState.roadHeightRL3 = point.z();
						//std::cout << "1: pointx" << point.x() << " pointy" << point.y() << " pointz" << point.z() << std::endl;
					} else 
					{
						std::cout << "tire rl3 left road!" << std::endl;
						leftRoad = true;
					}
				}
			}
			carState.roadAngleRL = atan(-(carState.roadHeightRL3 - carState.roadHeightRL1) / (cos(carState.wheelAngleZRL) * 2 * carState.contactPatch));
			*/
			
			currentTicks = (double) rt_timer_read();
			
			double localZPosTireFLOld = carState.localZPosTireFL;
			double localZPosTireFROld = carState.localZPosTireFR;
			double localZPosTireRROld = carState.localZPosTireRR;
			double localZPosTireRLOld = carState.localZPosTireRL;
			carState.localZPosTireFL = carState.globalPosSuspFL.getTrans().y() - carState.tireRadF - carState.roadHeightFL2;
			carState.localZPosTireFR = carState.globalPosSuspFR.getTrans().y() - carState.tireRadF - carState.roadHeightFR2;
			carState.localZPosTireRR = carState.globalPosSuspRR.getTrans().y() - carState.tireRadR - carState.roadHeightRR2;
			carState.localZPosTireRL = carState.globalPosSuspRL.getTrans().y() - carState.tireRadR - carState.roadHeightRL2;
			
			carState.tireDefSpeedFL = carState.localZPosTireFL - localZPosTireFLOld;
			carState.tireDefSpeedFR = carState.localZPosTireFR - localZPosTireFROld;
			carState.tireDefSpeedRR = carState.localZPosTireRR - localZPosTireRROld;
			carState.tireDefSpeedRL = carState.localZPosTireRL - localZPosTireRLOld;
			
			//steering
			carState.wheelAngleZFL = carState.posWheelLeftNeutral + carState.toeFL;
			carState.wheelAngleZFR = carState.posWheelRightNeutral + carState.toeFR;
			carState.posWheelCombined = (carState.posWheelLeftNeutral + carState.posWheelRightNeutral) / 2;
			carState.deltaWheel = carState.posWheelLeftNeutral - carState.posWheelRightNeutral;
			
			speedState.OmegaZFL = carState.vSteeringWheel * carState.steeringRatio;
			speedState.OmegaZFR = carState.vSteeringWheel * carState.steeringRatio;
			
			carState.posWheelLeftNeutral = carState.posSteeringWheel * carState.steeringRatio;
			carState.posWheelRightNeutral = carState.posSteeringWheel * carState.steeringRatio;
			
			//gear box
			if (carState.gear == -1) 
			{
				carState.gearRatio = carState.gearRatioR;
			}
			else if (carState.gear == 1) 
			{
				carState.gearRatio = carState.gearRatio1;
			}
			else if (carState.gear == 2) 
			{
				carState.gearRatio = carState.gearRatio2;
			}
			else if (carState.gear == 3)
			{
				carState.gearRatio = carState.gearRatio3;
			}
			else if (carState.gear == 4)
			{
				carState.gearRatio = carState.gearRatio4;
			}
			else
			{
				carState.gearRatio = carState.gearRatio5;
			}
			
			if(carState.clutchState != 0 && carState.gear != 0)
			{
				if(carState.clutchSwitch == 0)
				{
					if(speedState.engineRPM > (speedState.OmegaYRR + speedState.OmegaYRL) / 2 * carState.gearRatio * carState.finalDrive)
					{
						carState.acceleratorAngle = 0;
					}
					if(speedState.engineRPM < (speedState.OmegaYRR + speedState.OmegaYRL) / 2 * carState.gearRatio * carState.finalDrive)
					{
						carState.acceleratorAngle = 1;
					}
					
					if(std::abs(speedState.engineRPM - (speedState.OmegaYRR + speedState.OmegaYRL) / 2 * carState.gearRatio * carState.finalDrive) < carState.clutchSlipBorder)
					{
						carState.clutchSwitch = 1;
						//std::cout << "Clutch Switch Set To 1" << std::endl;
						if(!isnan(speedState.engineRPM / (carState.finalDrive * carState.gearRatio)))
						{
							speedState.OmegaYRR = speedState.engineRPM / (carState.finalDrive * carState.gearRatio);
							speedState.OmegaYRL = speedState.engineRPM / (carState.finalDrive * carState.gearRatio);
						} else
						{
							std::cout << "NAN while switching" << std::endl;
						}
						
					}
				}
				else 
				{
					speedState.OmegaYRR = speedState.engineRPM / (carState.finalDrive * carState.gearRatio);
					speedState.OmegaYRL = speedState.engineRPM / (carState.finalDrive * carState.gearRatio);
					if(std::abs(speedState.Tclutch) > speedState.TclutchMax)
					{
						carState.clutchSwitch = 0;
					}
				}
			}
			
			//shifting
			if(carState.gear != carState.oldGear)
			{
				carState.clutchTimer = 0;
				carState.clutchState = 0;
				carState.clutchSwitch = 0;
			}
			
			if(carState.clutchTimer < 400)
			{
				carState.clutchTimer++;
			}
			
			carState.clutchState = (carState.clutchTimer / 400) * (carState.clutchTimer / 400);
			/*if(speedState.engineRPM < carState.idleSpeed)
			{
				carState.clutchState = 0.01;
			}*/
			
			if (carState.gear == 0) 
			{
				carState.clutchState = 0;
				carState.clutchSwitch = 0;
			}
			
			//engine torque interpolation
			if(carState.acceleratorAngle >= 1)
			{
				carState.acceleratorAngle = 0.999999999;
			}
			if (carState.acceleratorAngle < 0)
			{
				carState.acceleratorAngle = 0;
			}
			if(speedState.engineRPM < carState.idleSpeed)
			{
				if (carState.acceleratorAngle < 0.5)
				{
					carState.acceleratorAngle = 0.5;
				}
			}
			if(speedState.engineRPM < 0)
			{
				speedState.engineRPM = 0;
			}
			double engineSpeed = speedState.engineRPM * 60 / (2 * M_PI);
			int interpx1 = (int) (engineSpeed - fmod(engineSpeed,1000.0)) / 1000 + 0.5;
			int interpx2 = (int) (engineSpeed - fmod(engineSpeed,1000.0)) / 1000 + 1 + 0.5;
			int interpy1 = (int) (carState.acceleratorAngle * 5 - fmod(carState.acceleratorAngle * 5,1.0)) + 0.5;
			int interpy2 = (int) (carState.acceleratorAngle * 5 - fmod(carState.acceleratorAngle * 5,1.0)) + 1 + 0.5;
			double interpa = carState.torqueMap[interpx1 * 6 + interpy1];
			double interpb = carState.torqueMap[interpx2 * 6 + interpy1];
			double interpc = carState.torqueMap[interpx1 * 6 + interpy2];
			double interpd = carState.torqueMap[interpx2 * 6 + interpy2];
			double interpremX = fmod(engineSpeed,1000.0) / 1000;
			double interpremY = fmod(carState.acceleratorAngle * 5,1.0);
			//interpolated torque from engine:
			carState.Tcomb = interpa * (1 - interpremX) * (1 - interpremY) + interpb * interpremX * (1 - interpremY) + interpc * (1 - interpremX) * interpremY + interpd * interpremX * interpremY;
			//randomize torque around interpolated value
			double randNumb = 0.3 * (double) std::rand() / RAND_MAX + 0.85;
			carState.Tcomb = carState.Tcomb * randNumb;
			
			
			//engine speed limiting
			if (speedState.engineRPM >= carState.revLimiter)
			{
				speedState.engineRPM = carState.revLimiter - 150 * 2 * M_PI / 60;
			}
			
			FWDState initialSpeeds = speedState;
			initialSpeeds.phiDotFL1 = 0;
			initialSpeeds.phiDotFL2 = 0;
			initialSpeeds.phiDotFL3 = 0;
			initialSpeeds.phiDotFR1 = 0;
			initialSpeeds.phiDotFR2 = 0;
			initialSpeeds.phiDotFR3 = 0;
			initialSpeeds.phiDotRR1 = 0;
			initialSpeeds.phiDotRR2 = 0;
			initialSpeeds.phiDotRR3 = 0;
			initialSpeeds.phiDotRL1 = 0;
			initialSpeeds.phiDotRR2 = 0;
			initialSpeeds.phiDotRR3 = 0;
			
			initialSpeeds.TcolumnCombined = 0;
			initialSpeeds.Tclutch = 0;
			initialSpeeds.TclutchMax = 0;
			initialSpeeds.slipFL = 0;
			initialSpeeds.slipFR = 0;
			initialSpeeds.slipRR = 0;
			initialSpeeds.slipRL = 0;
			initialSpeeds.FweightedFL = 0;
			initialSpeeds.FweightedFR = 0;
			initialSpeeds.FweightedRR = 0;
			initialSpeeds.FweightedRL = 0;
			initialSpeeds.FtireFL = 0;
			initialSpeeds.FtireFR = 0;
			initialSpeeds.FtireRR = 0;
			initialSpeeds.FtireRL = 0;
			initialSpeeds.FxFL = 0;
			initialSpeeds.FxFR = 0;
			initialSpeeds.FxRR = 0;
			initialSpeeds.FxRL = 0;
			initialSpeeds.FyFL = 0;
			initialSpeeds.FyFR = 0;
			initialSpeeds.FyRR = 0;
			initialSpeeds.FyRL = 0;
			initialSpeeds.genericOut1 = 0;
			initialSpeeds.genericOut2 = 0;
			initialSpeeds.genericOut3 = 0;
			initialSpeeds.genericOut4 = 0;
			initialSpeeds.genericOut5 = 0;
			initialSpeeds.genericOut6 = 0;
			initialSpeeds.genericOut7 = 0;
			initialSpeeds.genericOut8 = 0;
			initialSpeeds.genericOut9 = 0;
			initialSpeeds.genericOut10 = 0;
			initialSpeeds.genericOut11 = 0;
			initialSpeeds.genericOut12 = 0;
			initialSpeeds.genericOut13 = 0;
			initialSpeeds.genericOut14 = 0;
			initialSpeeds.genericOut15 = 0;
			initialSpeeds.genericOut16 = 0;
			
			//startTicks = (double) rt_timer_read();
			
			//initial positions
			FWDState initialPos;
			
			initialPos.vSuspZFL = carState.localZPosSuspFL;
			initialPos.vSuspZFR = carState.localZPosSuspFR;
			initialPos.vSuspZRR = carState.localZPosSuspRR;
			initialPos.vSuspZRL = carState.localZPosSuspRL;
			
			initialPos.phiDotFL1 = carState.phiFL1;
			initialPos.phiDotFL2 = carState.phiFL2;
			initialPos.phiDotFL3 = carState.phiFL3;
			initialPos.phiDotFR1 = carState.phiFR1;
			initialPos.phiDotFR2 = carState.phiFR2;
			initialPos.phiDotFR3 = carState.phiFR3;
			initialPos.phiDotRR1 = carState.phiRR1;
			initialPos.phiDotRR2 = carState.phiRR2;
			initialPos.phiDotRR3 = carState.phiRR3;
			initialPos.phiDotRL1 = carState.phiRL1;
			initialPos.phiDotRL2 = carState.phiRL2;
			initialPos.phiDotRL3 = carState.phiRL3;
			
			//first integration
			FWDState K1Acc = integrator.integrate(initialSpeeds, initialPos, carState, h);
			
			FWDState K1Pos = initialSpeeds * (h / 2.0);
			
			FWDState K1Speed = initialSpeeds + K1Acc * (h / 2.0);
			
			if((initialSpeeds.OmegaYFL > 0 && K1Speed.OmegaYFL < 0) || (initialSpeeds.OmegaYFL < 0 && K1Speed.OmegaYFL > 0) || K1Speed.OmegaYFL == 0)
			{
				K1Speed.OmegaYFL = 0;
			}
			if((initialSpeeds.OmegaYFR > 0 && K1Speed.OmegaYFR < 0) || (initialSpeeds.OmegaYFR < 0 && K1Speed.OmegaYFR > 0) || K1Speed.OmegaYFR == 0)
			{
				K1Speed.OmegaYFR = 0;
			}
			if((initialSpeeds.OmegaYRR > 0 && K1Speed.OmegaYRR < 0) || (initialSpeeds.OmegaYRR < 0 && K1Speed.OmegaYRR > 0) || K1Speed.OmegaYRR == 0)
			{
				K1Speed.OmegaYRR = 0;
			}
			if((initialSpeeds.OmegaYRL > 0 && K1Speed.OmegaYRL < 0) || (initialSpeeds.OmegaYRL < 0 && K1Speed.OmegaYRL > 0) || K1Speed.OmegaYRL == 0)
			{
				K1Speed.OmegaYRL = 0;
			}
			if((initialSpeeds.vX > 0 && K1Speed.vX < 0) || (initialSpeeds.vX < 0 && K1Speed.vX > 0) || K1Speed.vX == 0)
			{
				K1Speed.vX = 0;
			}
			
			K1Speed.phiDotFL1 = K1Acc.phiDotFL1;
			K1Speed.phiDotFL2 = K1Acc.phiDotFL2;
			K1Speed.phiDotFL3 = K1Acc.phiDotFL3;
			K1Speed.phiDotFR1 = K1Acc.phiDotFR1;
			K1Speed.phiDotFR2 = K1Acc.phiDotFR2;
			K1Speed.phiDotFR3 = K1Acc.phiDotFR3;
			K1Speed.phiDotRR1 = K1Acc.phiDotRR1;
			K1Speed.phiDotRR2 = K1Acc.phiDotRR2;
			K1Speed.phiDotRR3 = K1Acc.phiDotRR3;
			K1Speed.phiDotRL1 = K1Acc.phiDotRL1;
			K1Speed.phiDotRL2 = K1Acc.phiDotRL2;
			K1Speed.phiDotRL3 = K1Acc.phiDotRL3;
			
			//second integration
			FWDState K2Acc = integrator.integrate(K1Speed, K1Pos, carState, h);
			
			FWDState K2Pos = initialPos + K1Speed * (h / 2.0);
			
			FWDState K2Speed = initialSpeeds + K2Acc * (h / 2.0);
			
			if((initialSpeeds.OmegaYFL > 0 && K2Speed.OmegaYFL < 0) || (initialSpeeds.OmegaYFL < 0 && K2Speed.OmegaYFL > 0) || K2Speed.OmegaYFL == 0)
			{
				K2Speed.OmegaYFL = 0;
			}
			if((initialSpeeds.OmegaYFR > 0 && K2Speed.OmegaYFR < 0) || (initialSpeeds.OmegaYFR < 0 && K2Speed.OmegaYFR > 0) || K2Speed.OmegaYFR == 0)
			{
				K2Speed.OmegaYFR = 0;
			}
			if((initialSpeeds.OmegaYRR > 0 && K2Speed.OmegaYRR < 0) || (initialSpeeds.OmegaYRR < 0 && K2Speed.OmegaYRR > 0) || K2Speed.OmegaYRR == 0)
			{
				K2Speed.OmegaYRR = 0;
			}
			if((initialSpeeds.OmegaYRL > 0 && K2Speed.OmegaYRL < 0) || (initialSpeeds.OmegaYRL < 0 && K2Speed.OmegaYRL > 0) || K2Speed.OmegaYRL == 0)
			{
				K2Speed.OmegaYRL = 0;
			}
			if((initialSpeeds.vX > 0 && K2Speed.vX < 0) || (initialSpeeds.vX < 0 && K2Speed.vX > 0) || K2Speed.vX == 0)
			{
				K2Speed.vX = 0;
			}
			
			K2Speed.phiDotFL1 = K2Acc.phiDotFL1;
			K2Speed.phiDotFL2 = K2Acc.phiDotFL2;
			K2Speed.phiDotFL3 = K2Acc.phiDotFL3;
			K2Speed.phiDotFR1 = K2Acc.phiDotFR1;
			K2Speed.phiDotFR2 = K2Acc.phiDotFR2;
			K2Speed.phiDotFR3 = K2Acc.phiDotFR3;
			K2Speed.phiDotRR1 = K2Acc.phiDotRR1;
			K2Speed.phiDotRR2 = K2Acc.phiDotRR2;
			K2Speed.phiDotRR3 = K2Acc.phiDotRR3;
			K2Speed.phiDotRL1 = K2Acc.phiDotRL1;
			K2Speed.phiDotRL2 = K2Acc.phiDotRL2;
			K2Speed.phiDotRL3 = K2Acc.phiDotRL3;
			
			//third integration
			FWDState K3Acc = integrator.integrate(K2Speed, K2Pos, carState, h);
			
			FWDState K3Pos = initialPos + K2Speed * h;
			
			FWDState K3Speed = initialSpeeds + K3Acc * h;
			
			if((initialSpeeds.OmegaYFL > 0 && K3Speed.OmegaYFL < 0) || (initialSpeeds.OmegaYFL < 0 && K3Speed.OmegaYFL > 0) || K3Speed.OmegaYFL == 0)
			{
				K3Speed.OmegaYFL = 0;
			}
			if((initialSpeeds.OmegaYFR > 0 && K3Speed.OmegaYFR < 0) || (initialSpeeds.OmegaYFR < 0 && K3Speed.OmegaYFR > 0) || K3Speed.OmegaYFR == 0)
			{
				K3Speed.OmegaYFR = 0;
			}
			if((initialSpeeds.OmegaYRR > 0 && K3Speed.OmegaYRR < 0) || (initialSpeeds.OmegaYRR < 0 && K3Speed.OmegaYRR > 0) || K3Speed.OmegaYRR == 0)
			{
				K3Speed.OmegaYRR = 0;
			}
			if((initialSpeeds.OmegaYRL > 0 && K3Speed.OmegaYRL < 0) || (initialSpeeds.OmegaYRL < 0 && K3Speed.OmegaYRL > 0) || K3Speed.OmegaYRL == 0)
			{
				K3Speed.OmegaYRL = 0;
			}
			if((initialSpeeds.vX > 0 && K3Speed.vX < 0) || (initialSpeeds.vX < 0 && K3Speed.vX > 0) || K3Speed.vX == 0)
			{
				K3Speed.vX = 0;
			}
			
			K3Speed.phiDotFL1 = K3Acc.phiDotFL1;
			K3Speed.phiDotFL2 = K3Acc.phiDotFL2;
			K3Speed.phiDotFL3 = K3Acc.phiDotFL3;
			K3Speed.phiDotFR1 = K3Acc.phiDotFR1;
			K3Speed.phiDotFR2 = K3Acc.phiDotFR2;
			K3Speed.phiDotFR3 = K3Acc.phiDotFR3;
			K3Speed.phiDotRR1 = K3Acc.phiDotRR1;
			K3Speed.phiDotRR2 = K3Acc.phiDotRR2;
			K3Speed.phiDotRR3 = K3Acc.phiDotRR3;
			K3Speed.phiDotRL1 = K3Acc.phiDotRL1;
			K3Speed.phiDotRL2 = K3Acc.phiDotRL2;
			K3Speed.phiDotRL3 = K3Acc.phiDotRL3;
			
			//fourth integration
			FWDState K4Acc = integrator.integrate(K3Speed, K3Pos, carState, h);
			
			//currentTicks = (double) rt_timer_read();
			
			speedState = initialSpeeds + (K1Acc + K2Acc * 2.0 + K3Acc * 2.0 + K4Acc) * (h / 6.0);
			
			double angle = (K1Acc.vYaw + K2Acc.vYaw * 2.0 + K3Acc.vYaw * 2.0 + K4Acc.vYaw) * (h / 6.0);
			
			speedState.TcolumnCombined = (K1Acc.TcolumnCombined + 2.0 * K2Acc.TcolumnCombined + 2.0 * K3Acc.TcolumnCombined + K4Acc.TcolumnCombined) / 6.0;
			speedState.slipFL = (K1Acc.slipFL + 2.0 * K2Acc.slipFL + 2.0 * K3Acc.slipFL + K4Acc.slipFL) / 6.0;
			speedState.slipFR = (K1Acc.slipFR + 2.0 * K2Acc.slipFR + 2.0 * K3Acc.slipFR + K4Acc.slipFR) / 6.0;
			speedState.slipRR = (K1Acc.slipRR + 2.0 * K2Acc.slipRR + 2.0 * K3Acc.slipRR + K4Acc.slipRR) / 6.0;
			speedState.slipRL = (K1Acc.slipRL + 2.0 * K2Acc.slipRL + 2.0 * K3Acc.slipRL + K4Acc.slipRL) / 6.0;
			
			speedState.phiDotFL1 = (K1Acc.phiDotFL1 + 2.0 * K2Acc.phiDotFL1 + 2.0 * K3Acc.phiDotFL1 + K4Acc.phiDotFL1) / 6.0;
			speedState.phiDotFL2 = (K1Acc.phiDotFL2 + 2.0 * K2Acc.phiDotFL2 + 2.0 * K3Acc.phiDotFL2 + K4Acc.phiDotFL2) / 6.0;
			speedState.phiDotFL3 = (K1Acc.phiDotFL3 + 2.0 * K2Acc.phiDotFL3 + 2.0 * K3Acc.phiDotFL3 + K4Acc.phiDotFL3) / 6.0;
			speedState.phiDotFR1 = (K1Acc.phiDotFR1 + 2.0 * K2Acc.phiDotFR1 + 2.0 * K3Acc.phiDotFR1 + K4Acc.phiDotFR1) / 6.0;
			speedState.phiDotFR2 = (K1Acc.phiDotFR2 + 2.0 * K2Acc.phiDotFR2 + 2.0 * K3Acc.phiDotFR2 + K4Acc.phiDotFR2) / 6.0;
			speedState.phiDotFR3 = (K1Acc.phiDotFR3 + 2.0 * K2Acc.phiDotFR3 + 2.0 * K3Acc.phiDotFR3 + K4Acc.phiDotFR3) / 6.0;
			speedState.phiDotRR1 = (K1Acc.phiDotRR1 + 2.0 * K2Acc.phiDotRR1 + 2.0 * K3Acc.phiDotRR1 + K4Acc.phiDotRR1) / 6.0;
			speedState.phiDotRR2 = (K1Acc.phiDotRR2 + 2.0 * K2Acc.phiDotRR2 + 2.0 * K3Acc.phiDotRR2 + K4Acc.phiDotRR2) / 6.0;
			speedState.phiDotRR3 = (K1Acc.phiDotRR3 + 2.0 * K2Acc.phiDotRR3 + 2.0 * K3Acc.phiDotRR3 + K4Acc.phiDotRR3) / 6.0;
			speedState.phiDotRL1 = (K1Acc.phiDotRL1 + 2.0 * K2Acc.phiDotRL1 + 2.0 * K3Acc.phiDotRL1 + K4Acc.phiDotRL1) / 6.0;
			speedState.phiDotRL2 = (K1Acc.phiDotRL2 + 2.0 * K2Acc.phiDotRL2 + 2.0 * K3Acc.phiDotRL2 + K4Acc.phiDotRL2) / 6.0;
			speedState.phiDotRL3 = (K1Acc.phiDotRL3 + 2.0 * K2Acc.phiDotRL3 + 2.0 * K3Acc.phiDotRL3 + K4Acc.phiDotRL3) / 6.0;
			
			speedState.Tclutch = (K1Acc.Tclutch + 2.0 * K2Acc.Tclutch + 2.0 * K3Acc.Tclutch + K4Acc.Tclutch) / 6.0;
			speedState.TclutchMax = (K1Acc.TclutchMax + 2.0 * K2Acc.TclutchMax + 2.0 * K3Acc.TclutchMax + K4Acc.TclutchMax) / 6.0;
			
			speedState.FweightedFL = (K1Acc.FweightedFL + 2.0 * K2Acc.FweightedFL + 2.0 * K3Acc.FweightedFL + K4Acc.FweightedFL) / 6.0;
			speedState.FweightedFR = (K1Acc.FweightedFR + 2.0 * K2Acc.FweightedFR + 2.0 * K3Acc.FweightedFR + K4Acc.FweightedFR) / 6.0;
			speedState.FweightedRR = (K1Acc.FweightedRR + 2.0 * K2Acc.FweightedRR + 2.0 * K3Acc.FweightedRR + K4Acc.FweightedRR) / 6.0;
			speedState.FweightedRL = (K1Acc.FweightedRL + 2.0 * K2Acc.FweightedRL + 2.0 * K3Acc.FweightedRL + K4Acc.FweightedRL) / 6.0;
			
			speedState.FtireFL = (K1Acc.FtireFL + 2.0 * K2Acc.FtireFL + 2.0 * K3Acc.FtireFL + K4Acc.FtireFL) / 6.0;
			speedState.FtireFR = (K1Acc.FtireFR + 2.0 * K2Acc.FtireFR + 2.0 * K3Acc.FtireFR + K4Acc.FtireFR) / 6.0;
			speedState.FtireRR = (K1Acc.FtireRR + 2.0 * K2Acc.FtireRR + 2.0 * K3Acc.FtireRR + K4Acc.FtireRR) / 6.0;
			speedState.FtireRL = (K1Acc.FtireRL + 2.0 * K2Acc.FtireRL + 2.0 * K3Acc.FtireRL + K4Acc.FtireRL) / 6.0;
			
			speedState.FxFL = (K1Acc.FxFL + 2.0 * K2Acc.FxFL + 2.0 * K3Acc.FxFL + K4Acc.FxFL) / 6.0;
			speedState.FxFR = (K1Acc.FxFR + 2.0 * K2Acc.FxFR + 2.0 * K3Acc.FxFR + K4Acc.FxFR) / 6.0;
			speedState.FxRR = (K1Acc.FxRR + 2.0 * K2Acc.FxRR + 2.0 * K3Acc.FxRR + K4Acc.FxRR) / 6.0;
			speedState.FxRL = (K1Acc.FxRL + 2.0 * K2Acc.FxRL + 2.0 * K3Acc.FxRL + K4Acc.FxRL) / 6.0;
			
			speedState.FyFL = (K1Acc.FyFL + 2.0 * K2Acc.FyFL + 2.0 * K3Acc.FyFL + K4Acc.FyFL) / 6.0;
			speedState.FyFR = (K1Acc.FyFR + 2.0 * K2Acc.FyFR + 2.0 * K3Acc.FyFR + K4Acc.FyFR) / 6.0;
			speedState.FyRR = (K1Acc.FyRR + 2.0 * K2Acc.FyRR + 2.0 * K3Acc.FyRR + K4Acc.FyRR) / 6.0;
			speedState.FyRL = (K1Acc.FyRL + 2.0 * K2Acc.FyRL + 2.0 * K3Acc.FyRL + K4Acc.FyRL) / 6.0;
			
			speedState.genericOut1 = (K1Acc.genericOut1 + 2.0 * K2Acc.genericOut1 + 2.0 * K3Acc.genericOut1 + K4Acc.genericOut1) / 6.0;
			speedState.genericOut2 = (K1Acc.genericOut2 + 2.0 * K2Acc.genericOut2 + 2.0 * K3Acc.genericOut2 + K4Acc.genericOut2) / 6.0;
			speedState.genericOut3 = (K1Acc.genericOut3 + 2.0 * K2Acc.genericOut3 + 2.0 * K3Acc.genericOut3 + K4Acc.genericOut3) / 6.0;
			speedState.genericOut4 = (K1Acc.genericOut4 + 2.0 * K2Acc.genericOut4 + 2.0 * K3Acc.genericOut4 + K4Acc.genericOut4) / 6.0;
			speedState.genericOut5 = (K1Acc.genericOut5 + 2.0 * K2Acc.genericOut5 + 2.0 * K3Acc.genericOut5 + K4Acc.genericOut5) / 6.0;
			speedState.genericOut6 = (K1Acc.genericOut6 + 2.0 * K2Acc.genericOut6 + 2.0 * K3Acc.genericOut6 + K4Acc.genericOut6) / 6.0;
			speedState.genericOut7 = (K1Acc.genericOut7 + 2.0 * K2Acc.genericOut7 + 2.0 * K3Acc.genericOut7 + K4Acc.genericOut7) / 6.0;
			speedState.genericOut8 = (K1Acc.genericOut8 + 2.0 * K2Acc.genericOut8 + 2.0 * K3Acc.genericOut8 + K4Acc.genericOut8) / 6.0;
			speedState.genericOut9 = (K1Acc.genericOut9 + 2.0 * K2Acc.genericOut9 + 2.0 * K3Acc.genericOut9 + K4Acc.genericOut9) / 6.0;
			speedState.genericOut10 = (K1Acc.genericOut10 + 2.0 * K2Acc.genericOut10 + 2.0 * K3Acc.genericOut10 + K4Acc.genericOut10) / 6.0;
			speedState.genericOut11 = (K1Acc.genericOut11 + 2.0 * K2Acc.genericOut11 + 2.0 * K3Acc.genericOut11 + K4Acc.genericOut11) / 6.0;
			speedState.genericOut12 = (K1Acc.genericOut12 + 2.0 * K2Acc.genericOut12 + 2.0 * K3Acc.genericOut12 + K4Acc.genericOut12) / 6.0;
			speedState.genericOut13 = (K1Acc.genericOut13 + 2.0 * K2Acc.genericOut13 + 2.0 * K3Acc.genericOut13 + K4Acc.genericOut13) / 6.0;
			speedState.genericOut14 = (K1Acc.genericOut14 + 2.0 * K2Acc.genericOut14 + 2.0 * K3Acc.genericOut14 + K4Acc.genericOut14) / 6.0;
			speedState.genericOut15 = (K1Acc.genericOut15 + 2.0 * K2Acc.genericOut15 + 2.0 * K3Acc.genericOut15 + K4Acc.genericOut15) / 6.0;
			speedState.genericOut16 = (K1Acc.genericOut16 + 2.0 * K2Acc.genericOut16 + 2.0 * K3Acc.genericOut16 + K4Acc.genericOut16) / 6.0;
			
			speedState.limitSpeeds();
			speedState.threshold();
			
			//wheel speed brake stick test (at low speeds brake forces makes wheel rotation speed alternate around zero, therefore wheels dont come to stand still)
			
			if(K1Speed.OmegaYFL == 0 && K2Speed.OmegaYFL == 0 && K3Speed.OmegaYFL == 0)
			{
				speedState.OmegaYFL = 0;
			}
			if(K1Speed.OmegaYFR == 0 && K2Speed.OmegaYFR == 0 && K3Speed.OmegaYFR == 0)
			{
				speedState.OmegaYFR = 0;
			}
			if(K1Speed.OmegaYRR == 0 && K2Speed.OmegaYRR == 0 && K3Speed.OmegaYRR == 0)
			{
				speedState.OmegaYRR = 0;
			}
			if(K1Speed.OmegaYRL == 0 && K2Speed.OmegaYRL == 0 && K3Speed.OmegaYRL == 0)
			{
				speedState.OmegaYRL = 0;
			}
			if(K1Speed.vX == 0 && K2Speed.vX == 0 && K3Speed.vX == 0)
			{
				speedState.vX = 0;
			}
			
			rpms = speedState.engineRPM / (2 * M_PI);
			if(rpms > 8000)
			{
				rpms = 8000;
			}
			if(rpms < 0)
			{
				rpms = 0;
			}
			
			double oldCurrent = current;
			current = -speedState.TcolumnCombined;
			if(std::abs(current - oldCurrent) > carState.maxDeltaCurrent)
			{
				if(current > oldCurrent)
				{
					current = oldCurrent + carState.maxDeltaCurrent;
				}
				else 
				{
					current = oldCurrent - carState.maxDeltaCurrent;
				}
			}
			
			
			carState.localZPosSuspFL = carState.localZPosSuspFL + speedState.vSuspZFL * h;
			carState.localZPosSuspFR = carState.localZPosSuspFR + speedState.vSuspZFR * h;
			carState.localZPosSuspRR = carState.localZPosSuspRR + speedState.vSuspZRR * h;
			carState.localZPosSuspRL = carState.localZPosSuspRL + speedState.vSuspZRL * h;
			
			carState.phiFL1 = carState.phiFL1 + (speedState.phiDotFL1) * h;
			carState.phiFL2 = carState.phiFL2 + (speedState.phiDotFL2) * h;
			carState.phiFL3 = carState.phiFL3 + (speedState.phiDotFL3) * h;
			carState.phiFR1 = carState.phiFR1 + (speedState.phiDotFR1) * h;
			carState.phiFR2 = carState.phiFR2 + (speedState.phiDotFR2) * h;
			carState.phiFR3 = carState.phiFR3 + (speedState.phiDotFR3) * h;
			carState.phiRR1 = carState.phiRR1 + (speedState.phiDotRR1) * h;
			carState.phiRR2 = carState.phiRR2 + (speedState.phiDotRR2) * h;
			carState.phiRR3 = carState.phiRR3 + (speedState.phiDotRR3) * h;
			carState.phiRL1 = carState.phiRL1 + (speedState.phiDotRL1) * h;
			carState.phiRL2 = carState.phiRL2 + (speedState.phiDotRL2) * h;
			carState.phiRL3 = carState.phiRL3 + (speedState.phiDotRL3) * h;
			
			//make delta vectors
			CCSpeeds.makeIdentity();
			CCSpeeds.makeTranslate(speedState.vX, speedState.vY, speedState.vZ);
			carState.cogOpencoverSpeed = CCSpeeds * Car2OpencoverRotation * carState.cogOpencoverRot;
			
			osg::Matrix deltaX;
			deltaX.makeTranslate(carState.cogOpencoverSpeed(3,0) * h, 0, 0);
			osg::Matrix deltaY;
			deltaY.makeTranslate(0, carState.cogOpencoverSpeed(3,1) * h, 0);
			osg::Matrix deltaZ;
			deltaZ.makeTranslate(0, 0, carState.cogOpencoverSpeed(3,2) * h);
			
			osg::Matrix deltaYaw;
			deltaYaw.makeRotate(speedState.vYaw * h,carState.cogOpencoverRotY1,carState.cogOpencoverRotY2,carState.cogOpencoverRotY3);
			osg::Matrix deltaPitch;
			deltaPitch.makeRotate(-speedState.vPitch * h,carState.cogOpencoverRotX1,carState.cogOpencoverRotX2,carState.cogOpencoverRotX3);
			osg::Matrix deltaRoll;
			deltaRoll.makeRotate(-speedState.vRoll * h,carState.cogOpencoverRotZ1,carState.cogOpencoverRotZ2,carState.cogOpencoverRotZ3);
			
			carState.cogOpencoverRot = carState.cogOpencoverRot * deltaYaw * deltaPitch * deltaRoll;
			
			carState.cogOpencoverPos = carState.cogOpencoverPos * deltaX * deltaY * deltaZ;
			
			carState.modelOriginOffsetXMatrix.makeTranslate(carState.modelOriginOffsetX * carState.cogOpencoverRotZ1, carState.modelOriginOffsetY * carState.cogOpencoverRotZ1, carState.modelOriginOffsetZ * carState.cogOpencoverRotZ1);
			
			chassisTrans = carState.cogOpencoverRot * carState.cogOpencoverPos * carState.modelOriginOffsetXMatrix;
			
			//motion platform movement
			carState.vX3 = carState.vX2;
			carState.vX2 = carState.vX1;
			carState.vX1 = carState.vXCurrent;
			carState.vXCurrent = speedState.vX;
			double diffVX3 = carState.vX3 - carState.vX2;
			double diffVX2 = carState.vX2 - carState.vX1;
			double diffVX1 = carState.vX1 - carState.vXCurrent;
			carState.accX = (6 * diffVX1 + 3 * diffVX2 + diffVX3) / 10;
			
			carState.vY3 = carState.vY2;
			carState.vY2 = carState.vY1;
			carState.vY1 = carState.vYCurrent;
			carState.vYCurrent = speedState.vY;
			double diffVY3 = carState.vY3 - carState.vY2;
			double diffVY2 = carState.vY2 - carState.vY1;
			double diffVY1 = carState.vY1 - carState.vYCurrent;
			carState.accY = (6 * diffVY1 + 3 * diffVY2 + diffVY3) / 10;
			
			//double yawAngle = atan(carState.cogOpencoverRot(2,2) / carState.cogOpencoverRot(2,0)) - M_PI / 2;
			double l = sqrt(carState.cogOpencoverRot(2,2) * carState.cogOpencoverRot(2,2) + carState.cogOpencoverRot(2,0) * carState.cogOpencoverRot(2,0));
			double yawAngle;
			if(carState.cogOpencoverRot(2,0) > 0)
			{
				yawAngle = asin(carState.cogOpencoverRot(2,2) / l) - M_PI / 2;
			}
			else 
			{
				yawAngle = -(asin(carState.cogOpencoverRot(2,2) / l) - M_PI / 2);
			}
			
			double oldCarZ = carState.carZ;
			carState.carZ = carState.cogOpencoverPos(3,1);
			double carZSpeed = (carState.carZ - oldCarZ) / h;
			if(carZSpeed >= 0)
			{
				carZSpeed = tanh(carZSpeed / 15) * carZSpeed;
			}
			else
			{
				carZSpeed = -tanh(carZSpeed / 15) * carZSpeed;
			}
			
			carState.mpHeight = carState.mpHeight + (carZSpeed + tanh(-carState.mpHeight) * carState.mpZReturnSpeed) * h * 0.01;
			
			if(carState.mpHeight > 0.2)
			{
				carState.mpHeight = 0.2;
			}
			if(carState.mpHeight < -0.2)
			{
				carState.mpHeight = -0.2;
			}
			
			osg::Matrix mpPlane = carState.cogOpencoverRot;
			mpPlane.makeRotate(yawAngle,0,1,0);
			mpPlane = carState.cogOpencoverRot * mpPlane;
			
			
			double motPlatSpeedLim = 0;
			if(time > 5000)
			{
				motPlatSpeedLim = 10;
			}
			else
			{
				motPlatSpeedLim = 0.00002;
			}
			
			double oldMPLZ = carState.mpLZ;
			double oldMPRZ = carState.mpRZ;
			double oldMPBZ = carState.mpBZ;
			
			carState.mpLZ = (-carState.motorOffsetXFL * mpPlane(1,0) - carState.motorOffsetZFL * mpPlane(1,2)) / mpPlane(1,1) + carState.mpHeight;
			carState.mpRZ = (-carState.motorOffsetXFR * mpPlane(1,0) - carState.motorOffsetZFR * mpPlane(1,2)) / mpPlane(1,1) + carState.mpHeight;
			carState.mpBZ = (-carState.motorOffsetXR * mpPlane(1,0) - carState.motorOffsetZR * mpPlane(1,2)) / mpPlane(1,1) + carState.mpHeight;
			
			
			
			if(std::abs(carState.mpLZ - oldMPLZ) > motPlatSpeedLim)
			{
				if(carState.mpLZ > oldMPLZ)
				{
					carState.mpLZ = oldMPLZ + motPlatSpeedLim;
				}
				else
				{
					carState.mpLZ = oldMPLZ - motPlatSpeedLim;
				}
				//std::cout << "if statement reached, mplz: " << carState.mpLZ << " oldmplz: " << oldMPLZ << std::endl;
			}
			
			if(std::abs(carState.mpRZ - oldMPRZ) > motPlatSpeedLim)
			{
				if(carState.mpRZ > oldMPRZ)
				{
					carState.mpRZ = oldMPRZ + motPlatSpeedLim;
				}
				else
				{
					carState.mpRZ = oldMPRZ - motPlatSpeedLim;
				}
			}
			
			if(std::abs(carState.mpBZ - oldMPBZ) > motPlatSpeedLim)
			{
				if(carState.mpBZ > oldMPBZ)
				{
					carState.mpBZ = oldMPBZ + motPlatSpeedLim;
				}
				else
				{
					carState.mpBZ = oldMPBZ - motPlatSpeedLim;
				}
			}
			
			//timers
			timerCounter++;
			time++;
			carState.timerCounter = timerCounter;
			
			if(timerCounter == 500 /*|| speedState.vX != 0 time > 1000 * 2 * M_PI && timerCounter <= 1000 * 2 * M_PI + 1000 * 2 * M_PI*/)
			{
				//outfile << time << " " << carState.mpLZ << " " << carState.mpRZ << " " << carState.mpBZ << " " << carState.mpHeight << " " << carState.cogOpencoverPos(3,1) << " " 
				//<< speedState.vZ << " " << current << std::endl;
				//outfile << time << " " << speedState.vX << " " << speedState.vY << " " << speedState.vYaw << " " << speedState.genericOut9 << " " << speedState.genericOut10 << " "
				//outfile << time << " " << carState.localRoll << " " << carState.localZPosSuspFL << " " << carState.globalPosJointFL.getTrans().y() << " " << 
				//carState.localZPosSuspFR << " " << carState.globalPosJointFR.getTrans().y() << std::endl;
				//<< speedState.genericOut11 << " " << speedState.genericOut12 << std::endl;
				//" " << speedState.FxFL << " " << speedState.FyFL << " " << speedState.genericOut1 << " " << speedState.genericOut5 << std::endl;
				//outfile << time << " " << speedState.vX << " " << speedState.vY << " " << speedState.vYaw << " " <<
				//speedState.slipFL << " " << speedState.slipFR << " " << speedState.slipRR << " " << speedState.slipRL << " " << std::endl;
				//outfile << time << " " << speedState.vX << " " << speedState.vY << " " << speedState.vYaw << " " <<
				//carState.accX << " " << carState.accY << std::endl;
				//carState.globalPosJointFL.getTrans().x() << " " << carState.globalPosJointFL.getTrans().z() << " " <<
				//carState.globalPosJointFR.getTrans().x() << " " << carState.globalPosJointFR.getTrans().z() << " " <<
				//carState.globalPosJointRR.getTrans().x() << " " << carState.globalPosJointRR.getTrans().z() << " " <<
				//carState.globalPosJointRL.getTrans().x() << " " << carState.globalPosJointRL.getTrans().z() << std::endl;
				//carState.globalPosOC.getTrans().x() << " " << carState.globalPosOC.getTrans().z() << " " << std::endl;
				//speedState.genericOut9 << " " << speedState.genericOut10 << " " <<
				//speedState.genericOut11 << " " << speedState.genericOut12 << std::endl;
				//outfile << time << " " << speedState.vX << " " << engineSpeed << " " << carState.gear << " " << carState.clutchSwitch << " " << carState.clutchState << 
				//" " << std::abs(speedState.engineRPM - (speedState.OmegaYRR + speedState.OmegaYRL) / 2 * carState.gearRatio * carState.finalDrive) << std::endl;
				//outfile << time << " " << speedState.vX << " " << speedState.TcolumnCombined << " " << carState.posSteeringWheel << " " << carState.vSteeringWheel << " " 
				//<< speedState.phiDotFL1 << " " << speedState.phiDotFL2 << " " << speedState.phiDotFL3 << std::endl;
				
				//for 2D log view
				//std::cout << time << " " << carState.globalPosJointFL.getTrans().x() << " " << carState.globalPosJointFL.getTrans().y() << " " <<
				//carState.globalPosJointFR.getTrans().x() << " " << carState.globalPosJointFR.getTrans().y() << " " <<
				//carState.globalPosJointRR.getTrans().x() << " " << carState.globalPosJointRR.getTrans().y() << " " <<
				//carState.globalPosJointRL.getTrans().x() << " " << carState.globalPosJointRL.getTrans().y() << " " <<
				//speedState.OmegaYFL << " " << speedState.OmegaYRL << " " << carState.acceleratorAngle << " " << carState.brakeForce << std::endl;*/
				//std::cout << time << " " << current << " " << speedState.vX<< std::endl;
				//std::cout << time << " " << carState.posWheelLeftNeutral << " " << speedState.OmegaZFL << " " << carState.phiFL2 << " " << speedState.phiDotFL2 << " " << speedState.genericOut1 << std::endl;
				
				//double diffTime = rt_timer_ticks2ns(currentTicks - startTicks);
				//std::cerr << "integrator run time:" << diffTime << std::endl;
				//std::cout << "#time period: " << h << std::endl; 
				//std::cout << "#time " << time << std::endl; 
				osg::Vec3d globPosTransVec = carState.globalPosOC.getTrans();
				std::cout << "#globalPosTransVec: x " << globPosTransVec.x() << "; y " << globPosTransVec.y() << "; z " << globPosTransVec.z() << std::endl;
				osg::Vec3d chassisTransVec = chassisTrans.getTrans();
				std::cout << "#chassisTransVec: x " << chassisTransVec.x() << "; y " << chassisTransVec.y() << "; z " << chassisTransVec.z() << std::endl;
				
				//std::cout << "rot:"<< std::endl;
				//std::cout << carState.cogOpencoverRot(0,0) << "," << carState.cogOpencoverRot(0,1) << "," << carState.cogOpencoverRot(0,2) << "," << carState.cogOpencoverRot(0,3) << std::endl;
				//std::cout << carState.cogOpencoverRot(1,0) << "," << carState.cogOpencoverRot(1,1) << "," << carState.cogOpencoverRot(1,2) << "," << carState.cogOpencoverRot(1,3) << std::endl;
				//std::cout << carState.cogOpencoverRot(2,0) << "," << carState.cogOpencoverRot(2,1) << "," << carState.cogOpencoverRot(2,2) << "," << carState.cogOpencoverRot(2,3) << std::endl;
				//std::cout << carState.cogOpencoverRot(3,0) << "," << carState.cogOpencoverRot(3,1) << "," << carState.cogOpencoverRot(3,2) << "," << carState.cogOpencoverRot(3,3) << std::endl;
				//std::cout << "pos:"<< std::endl;
				//std::cout << carState.cogOpencoverPos(0,0) << "," << carState.cogOpencoverPos(0,1) << "," << carState.cogOpencoverPos(0,2) << "," << carState.cogOpencoverPos(0,3) << std::endl;
				//std::cout << carState.cogOpencoverPos(1,0) << "," << carState.cogOpencoverPos(1,1) << "," << carState.cogOpencoverPos(1,2) << "," << carState.cogOpencoverPos(1,3) << std::endl;
				//std::cout << carState.cogOpencoverPos(2,0) << "," << carState.cogOpcarState.mpBZencoverPos(2,1) << "," << carState.cogOpencoverPos(2,2) << "," << carState.cogOpencoverPos(2,3) << std::endl;
				//std::cout << carState.cogOpencoverPos(3,0) << "," << carState.cogOpencoverPos(3,1) << "," << carState.cogOpencoverPos(3,2) << "," << carState.cogOpencoverPos(3,3) << std::endl;
				
				//std::cout << "#temp: x " << temp.getTrans().x() << "; y " << temp.getTrans().y() << "; z " << temp.getTrans().z() << std::endl;
				//std::cout << "#carState.mpLZ: " << carState.mpLZ << " carState.mpRZ: " << carState.mpRZ << 
				//" carState.mpBZ: " << carState.mpBZ << std::endl;
				//std::cout << "pos middle: " << ValidateMotionPlatform::posMiddle << std::endl;
				
				//std::cout << "#accelerator angle " << carState.acceleratorAngle << std::endl;
				//std::cout << "#engine RPM " << speedState.engineRPM << std::endl;
				//std::cout << "#steering wheel angle " << carState.posSteeringWheel << " steering wheel speed " << carState.vSteeringWheel << std::endl;
				//std::cout << "#steering ccolumn torque" << speedState.TcolumnCombined << std::endl;
				//std::cout << "#current " << current << std::endl;
				//std::cout << "#brake " << carState.brakeForce << std::endl;
				//std::cout << "#gear " << carState.gear << std::endl;
				//std::cout << "#clutchState " << carState.clutchState << std::endl;
				//std::cout << "#clutchSwitch " << carState.clutchSwitch << std::endl;
				//std::cout << "#combustion torque " << carState.Tcomb << std::endl;
				//std::cout << "#engine gear box speed diff " << abs(speedState.engineRPM - (speedState.OmegaYRR + speedState.OmegaYRL) / 2 * carState.gearRatio * carState.finalDrive) << std::endl;
				//std::cout << "#Tlossengine " << speedState.genericOut5 << std::endl;
				//std::cout << "#Fxroadcomb  " << speedState.genericOut6 << std::endl;
				//std::cout << "#Forcex      " << speedState.genericOut7 << std::endl;
				//std::cout << "#Fdrag       " << speedState.genericOut4 << std::endl;
				//std::cout << "#accX " << carState.accX << "; accY " << carState.accY << std::endl;
				//std::cout << "motorfl: " << motorFL(3,1) << "motorfr: " << motorFR(3,1) << "motorr: " << motorR(3,1) << std::endl;
				//std::cout << "omegayfl 1: " << initialSpeeds.OmegaYFL << " omegayfl 2: " << K1Speed.OmegaYFL << " omegayfl 3: " << K2Speed.OmegaYFL << " omegayfl 4: " << K3Speed.OmegaYFL << std::endl;
				//std::cout << "vX 1: " << initialSpeeds.vX << " vX 2: " << K1Speed.vX << " vX 3: " << K2Speed.vX << " vX 4: " << K3Speed.vX << std::endl;
				//std::cout << " " << std::endl;
				int length = 13;
				std::cout << "#roadZ    " << std::setw(length) << carState.roadHeightFL2					 	<< "  roadZ     " << std::setw(length) << carState.roadHeightFR2 << std::endl;
				//std::cout << "#tirePosX " << std::setw(length) << carState.globalPosTireFL2.getTrans().x()	<< "  tirePosX  " << std::setw(length) << carState.globalPosTireFR2.getTrans().x() << std::endl;
				std::cout << "#tirePosY " << std::setw(length) << carState.globalPosTireFL2.getTrans().y()	<< "  tirePosY  " << std::setw(length) << carState.globalPosTireFR2.getTrans().y() << std::endl;
				//std::cout << "#tirePosZ " << std::setw(length) << carState.globalPosTireFL2.getTrans().z()	<< "  tirePosZ  " << std::setw(length) << carState.globalPosTireFR2.getTrans().z() << std::endl;
				//std::cout << "#suspX    " << std::setw(length) << carState.globalPosSuspFL.getTrans().x() 	<< "  suspX     " << std::setw(length) << carState.globalPosSuspFR.getTrans().x() << std::endl;
				std::cout << "#suspY    " << std::setw(length) << carState.globalPosSuspFL.getTrans().y() 	<< "  suspY     " << std::setw(length) << carState.globalPosSuspFR.getTrans().y() << std::endl;
				//std::cout << "#suspZ    " << std::setw(length) << carState.globalPosSuspFL.getTrans().z() 	<< "  suspZ     " << std::setw(length) << carState.globalPosSuspFR.getTrans().z() << std::endl;
				//std::cout << "#susSpeed " << std::setw(length) << speedState.vSuspZFL 						<< "  susSpeed  " << std::setw(length) << speedState.vSuspZFR << std::endl;
				//std::cout << "#tireSpd  " << std::setw(length) << carState.tireDefSpeedFL 					<< "  tireSpd   " << std::setw(length) << carState.tireDefSpeedFR << std::endl;
				//std::cout << "#localZS  " << std::setw(length) << carState.localZPosSuspFL 					<< "  localZS   " << std::setw(length) << carState.localZPosSuspFR << std::endl;
				//std::cout << "#localZT  " << std::setw(length) << carState.localZPosTireFL 					<< "  localZT   " << std::setw(length) << carState.localZPosTireFR << std::endl;
				//std::cout << "#joint x  " << std::setw(length) << carState.globalPosJointFL.getTrans().x() 	<< "  joint x   " << std::setw(length) << carState.globalPosJointFR.getTrans().x() << std::endl;
				std::cout << "#joint y  " << std::setw(length) << carState.globalPosJointFL.getTrans().y() 	<< "  joint y   " << std::setw(length) << carState.globalPosJointFR.getTrans().y() << std::endl;
				//std::cout << "#jointz   " << std::setw(length) << carState.globalPosJointFL.getTrans().z() 	<< "  jointz    " << std::setw(length) << carState.globalPosJointFR.getTrans().z() << std::endl;
				//std::cout << "#weighted " << std::setw(length) << speedState.FweightedFL 						<< "  weighted  " << std::setw(length) << speedState.FweightedFR << std::endl;
				//std::cout << "#Ftire    " << std::setw(length) << speedState.FtireFL 							<< "  Ftire     " << std::setw(length) << speedState.FtireFR << std::endl;
				//std::cout << "#Fx       " << std::setw(length) << speedState.FxFL 							<< "  Fx        " << std::setw(length) << speedState.FxFR << std::endl;
				//std::cout << "#Fy       " << std::setw(length) << speedState.FyFL 							<< "  Fy        " << std::setw(length) << speedState.FyFR << std::endl;
				//std::cout << "#slip     " << std::setw(length) << speedState.slipFL 							<< "  slip      " << std::setw(length) << speedState.slipFR << std::endl;
				//std::cout << "#posWheel " << std::setw(length) << carState.posWheelLeftNeutral				<< "  posWheel  " << std::setw(length) << carState.posWheelRightNeutral << std::endl;
				//std::cout << "#realWAng " << std::setw(length) << carState.wheelAngleZFL						<< "  realWAng  " << std::setw(length) << carState.wheelAngleZFR << std::endl;
				//std::cout << "#omegaz   " << std::setw(length) << speedState.OmegaZFL							<< "  omegaz    " << std::setw(length) << speedState.OmegaZFR << std::endl;
				//std::cout << "#omegay   " << std::setw(length) << speedState.OmegaYFL							<< "  omegay    " << std::setw(length) << speedState.OmegaYFR << std::endl;
				//std::cout << "#phi      " << std::setw(length) << carState.phiFL2								<< "  phi       " << std::setw(length) << carState.phiFR2 << std::endl;
				//std::cout << "#phiDot   " << std::setw(length) << speedState.phiDotFL2						<< "  phiDot    " << std::setw(length) << speedState.phiDotFR2	 << std::endl;
				//std::cout << "#genO1    " << std::setw(length) << speedState.genericOut1					<< "  genO5     " << std::setw(length) << speedState.genericOut5 << std::endl;
				//std::cout << "#genO2    " << std::setw(length) << speedState.genericOut2					<< "  genO6     " << std::setw(length) << speedState.genericOut6 << std::endl;
				//std::cout << "#genO3    " << std::setw(length) << speedState.genericOut3					<< "  genO7     " << std::setw(length) << speedState.genericOut7 << std::endl;
				//std::cout << "#genO4    " << std::setw(length) << speedState.genericOut4					<< "  genO8     " << std::setw(length) << speedState.genericOut8 << std::endl;
				//std::cout << "#tb3      " << std::setw(length) << speedState.genericOut5					<< "  tb3       " << std::setw(length) << speedState.genericOut8 << std::endl;
				//std::cout << "#9        " << std::setw(length) << speedState.genericOut9					<< "  10        " << std::setw(length) << speedState.genericOut10 << std::endl;
				//std::cout << "#11       " << std::setw(length) << speedState.genericOut11					<< "  12        " << std::setw(length) << speedState.genericOut12 << std::endl;
				//std::cout << "#13       " << std::setw(length) << speedState.genericOut13					<< "  14        " << std::setw(length) << speedState.genericOut14 << std::endl;
				//std::cout << " " << std::endl;
				std::cout << "#roadZ    " << std::setw(length) << carState.roadHeightRL2					 	<< "  roadZ    " << std::setw(length) << carState.roadHeightRR2 << std::endl;
				//std::cout << "#whlPosX  " << std::setw(length) << wheelPosRLX									<< "  whlPosX  " << std::setw(length) << wheelPosRRX << std::endl;
				//std::cout << "#whlPosY  " << std::setw(length) << wheelPosRLY									<< "  whlPosY  " << std::setw(length) << wheelPosRRY << std::endl;
				
				//std::cout << "#tirePosX " << std::setw(length) << carState.globalPosTireRL2.getTrans().x()	<< "  tirePosX  " << std::setw(length) << carState.globalPosTireRR2.getTrans().x() << std::endl;
				std::cout << "#tirePosY " << std::setw(length) << carState.globalPosTireRL2.getTrans().y()	<< "  tirePosY  " << std::setw(length) << carState.globalPosTireRR2.getTrans().y() << std::endl;
				//std::cout << "#tirePosZ " << std::setw(length) << carState.globalPosTireRL2.getTrans().z()	<< "  tirePosZ  " << std::setw(length) << carState.globalPosTireRR2.getTrans().z() << std::endl;
				//std::cout << "#suspX    " << std::setw(length) << carState.globalPosSuspRL.getTrans().x() 	<< "  suspX     " << std::setw(length) << carState.globalPosSuspRR.getTrans().x() << std::endl;
				std::cout << "#suspY    " << std::setw(length) << carState.globalPosSuspRL.getTrans().y() 	<< "  suspY     " << std::setw(length) << carState.globalPosSuspRR.getTrans().y() << std::endl;
				//std::cout << "#suspZ    " << std::setw(length) << carState.globalPosSuspRL.getTrans().z() 	<< "  suspZ     " << std::setw(length) << carState.globalPosSuspRR.getTrans().z() << std::endl;
				//std::cout << "#susSpeed " << std::setw(length) << speedState.vSuspZRL 						<< "  susSpeed " << std::setw(length) << speedState.vSuspZRR << std::endl;
				//std::cout << "#tireSpd  " << std::setw(length) << carState.tireDefSpeedRL 					<< "  tireSpd   " << std::setw(length) << carState.tireDefSpeedRR << std::endl;
				//std::cout << "#localZS  " << std::setw(length) << carState.localZPosSuspRL 					<< "  localZS   " << std::setw(length) << carState.localZPosSuspRR << std::endl;
				//std::cout << "#localZT  " << std::setw(length) << carState.localZPosTireRL 					<< "  localZT   " << std::setw(length) << carState.localZPosTireRR << std::endl;
				//std::cout << "#joint x  " << std::setw(length) << carState.globalPosJointRL.getTrans().x() 	<< "  joint x  " << std::setw(length) << carState.globalPosJointRR.getTrans().x() << std::endl;
				std::cout << "#joint y  " << std::setw(length) << carState.globalPosJointRL.getTrans().y() 	<< "  joint y  " << std::setw(length) << carState.globalPosJointRR.getTrans().y() << std::endl;
				//std::cout << "#joint z  " << std::setw(length) << carState.globalPosJointRL.getTrans().z() 	<< "  joint z  " << std::setw(length) << carState.globalPosJointRR.getTrans().z() << std::endl;
				//std::cout << "#weighted " << std::setw(length) << speedState.FweightedRL 						<< "  weighted " << std::setw(length) << speedState.FweightedRR << std::endl;
				//std::cout << "#Ftire    " << std::setw(length) << speedState.FtireRL 							<< "  Ftire    " << std::setw(length) << speedState.FtireRR << std::endl;
				//std::cout << "#Fx       " << std::setw(length) << speedState.FxRL 							<< "  Fx        " << std::setw(length) << speedState.FxRR << std::endl;
				//std::cout << "#Fy       " << std::setw(length) << speedState.FyRL 							<< "  Fy        " << std::setw(length) << speedState.FyRR << std::endl;
				//std::cout << "#slip     " << std::setw(length) << speedState.slipRL 							<< "  slip      " << std::setw(length) << speedState.slipRR << std::endl;
				//std::cout << "#omegay   " << std::setw(length) << speedState.OmegaYRL							<< "  omegay   " << std::setw(length) << speedState.OmegaYRR << std::endl;
				//std::cout << "#Fxcc     " << std::setw(length) << speedState.genericOut4					<< "  Fxcc      " << std::setw(length) << speedState.genericOut3 << std::endl;
				//std::cout << "#Fycc     " << std::setw(length) << speedState.genericOut7					<< "  Fycc      " << std::setw(length) << speedState.genericOut8 << std::endl;
				//std::cout << "#genO13   " << std::setw(length) << speedState.genericOut13					<< "  genO9     " << std::setw(length) << speedState.genericOut9 << std::endl;
				//std::cout << "#genO14   " << std::setw(length) << speedState.genericOut14					<< "  genO10    " << std::setw(length) << speedState.genericOut10 << std::endl;
				//std::cout << "#genO15   " << std::setw(length) << speedState.genericOut15					<< "  genO11    " << std::setw(length) << speedState.genericOut11 << std::endl;
				//std::cout << "#genO16   " << std::setw(length) << speedState.genericOut16					<< "  genO12    " << std::setw(length) << speedState.genericOut12 << std::endl;
				
				
				//std::cout << " " << std::endl;
				
				//std::cout << "road used: " << currentRoadFL2 << std::endl;
				//std::cout << "road new: " << currentRoadArray[1] << std::endl;
				
				//std::cout << "#vx " << speedState.vX << std::endl;
				//std::cout << "#vy " << speedState.vY << std::endl;
				//std::cout << "#vz " << speedState.vZ << " carzspeed: " << carZSpeed << std::endl;
				//std::cout << "global speed: x: " << carState.cogOpencoverSpeed(3,0) << " y: " << carState.cogOpencoverSpeed(3,1) << " z: " << carState.cogOpencoverSpeed(3,2) << std::endl;
				//std::cout << "#vYaw " << speedState.vYaw << std::endl;
				//std::cout << "#vPitch " << speedState.vPitch << std::endl;
				//std::cout << "#vRoll " << speedState.vRoll << std::endl;
				
				//std::cout << "#yaw " << carState.globalYaw << std::endl;
				//std::cout << "#roll " << carState.localRoll << std::endl;
				//std::cout << "#pitch " << carState.localPitch << std::endl;
				std::cout << "#=(^-.-^)="<< std::endl;
				
				//timerTimer++;
				//fprintf(outFile, "1  2  3%5g%13g%13g%13g%13g%13g\n",timerTimer,carState.localPitch,carState.localRoll,carState.roadHeightFL2,carState.roadHeightFR2,speedState.vZ);
				
				
				timerCounter = 0;
			}
		} 
		else
		{
			current = 0;
		}
		//std::cout << "steering angle: " << carState.posSteeringWheel << std::endl;
		//std::cout << "current before: " << current << std::endl;
		
		//current = 0;
		
		double currentLimit = 700;
		if(current > currentLimit)
		{
			current = currentLimit;
		}
		if(current < -currentLimit)
		{
			current = -currentLimit;
		}
		if(isnan(current))
		{
			current = 0;
			std::cout << "current is NAN o.O" << std::endl; 
		}
		
		//std::cout << "current after: " << current << std::endl;
		
		steerWheel->setCurrent(current);
		
		//Motion platform handling
		carState.mpRZ = 0;
		carState.mpLZ = 0;
		carState.mpBZ = 0;
		
		double posLimit = 0.2;
		if(carState.mpRZ > posLimit)
		{
			carState.mpRZ = posLimit;
		}
		if(carState.mpRZ < -posLimit)
		{
			carState.mpRZ = -posLimit;
		}
		if(isnan(carState.mpRZ))
		{
			carState.mpRZ = 0;
			std::cout << "mpRZ is NAN o.O" << std::endl; 
		}
		
		if(carState.mpLZ > posLimit)
		{
			carState.mpLZ = posLimit;
		}
		if(carState.mpLZ < -posLimit)
		{
			carState.mpLZ = -posLimit;
		}
		if(isnan(carState.mpLZ))
		{
			carState.mpLZ = 0;
			std::cout << "mpLZ is NAN o.O" << std::endl; 
		}
		
		if(carState.mpBZ > posLimit)
		{
			carState.mpBZ = posLimit;
		}
		if(carState.mpBZ < -posLimit)
		{
			carState.mpBZ = -posLimit;
		}
		if(isnan(carState.mpBZ))
		{
			carState.mpBZ = 0;
			std::cout << "mpBZ is NAN o.O" << std::endl; 
		}
		
		motPlat->getSendMutex().acquire(period);
		//Right
		motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + carState.mpRZ);
		//Left
		motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + carState.mpLZ);
		//Rear
		motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + carState.mpBZ);
		motPlat->getSendMutex().release();
		
		//timerCounter++;
		/*if(timerCounter == 5)
		{
			//currentTicks = (double) rt_timer_read();
			//double diffTime = rt_timer_ticks2ns(currentTicks - startTicks);
			//std::cout << diffTime << std::endl;
			//outfile << time << " " << diffTime << std::endl;
			//std::cout << "globalPosTransVec: x " << globPosTransVec.x() << "; y " << globPosTransVec.y() << "; z " << globPosTransVec.z() << std::endl;
			//osg::Vec3d chassisTransVec = chassisTrans.getTrans();
			//std::cout << "chassisTransVec: x " << chassisTransVec.x() << "; y " << chassisTransVec.y() << "; z " << chassisTransVec.z() << std::endl;
			
			//std::cout << "carState.globalPosSuspFL.getTrans().z() " << carState.globalPosSuspFL.getTrans().z() << std::endl;
			//std::cout << "carState.globalPosJointFL.getTrans().z() " << carState.globalPosJointFL.getTrans().z() << std::endl;
			//
			//std::cout << "carState.globalPosSuspFR.getTrans().z() " << carState.globalPosSuspFR.getTrans().z() << std::endl;
			//std::cout << "carState.globalPosJointFR.getTrans().z() " << carState.globalPosJointFR.getTrans().z() << std::endl;
			
			//std::cout << "carState.globalPosSuspRR.getTrans().z() " << carState.globalPosSuspRR.getTrans().z() << std::endl;
			//std::cout << "carState.globalPosJointRR.getTrans().z() " << carState.globalPosJointRR.getTrans().z() << std::endl;
			
			//std::cout << "carState.globalPosSuspRL.getTrans().z() " << carState.globalPosSuspRL.getTrans().z() << std::endl;
			//std::cout << "carState.globalPosJointRL.getTrans().z() " << carState.globalPosJointRL.getTrans().z() << std::endl;
			
			//std::cout << "yaw " << carState.globalYaw << std::endl;
			//std::cout << "roll " << carState.localRoll << std::endl;
			//std::cout << "pitch " << carState.localPitch << std::endl;
			//std::cout << "=(^-.-^)="<< std::endl;
			
			
			timerCounter = 0;
		}*/

		rt_task_wait_period(&overruns);
	}
	
	
	std::cout << "task not running anymore!" << std::endl;
	
	outfile.close();
    
	//steering com change
	steerWheel->setCurrent(0);
	
	rpms = 0;
	speedState.slipFL = 0;
	speedState.slipFR = 0;
	speedState.slipRR = 0;
	speedState.slipRL = 0;
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
    while (!motPlat->isGrounded())
    {
        rt_task_wait_period(&overruns);
    }
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();

    set_periodic(TM_INFINITE);
    
	//steering com change
	steerWheel->shutdown();
	

    taskFinished = true;
	
	std::cout << "task finished!" << std::endl;
}

void FourWheelDynamicsRealtime2::platformToGround()
{
    pause = true;
    movingToGround = true;
    returningToAction = false;

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
}

void FourWheelDynamicsRealtime2::centerWheel()
{
    std::cout << "fwd center wheel" << std::endl;
	if (pause)
    {
	doCenter = true;
    }
}

void FourWheelDynamicsRealtime2::platformMiddleLift()
{
    pause = true;
    returningToAction = true;
    movingToGround = false;

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlMiddleLift>();
    motPlat->getSendMutex().release();
}

void FourWheelDynamicsRealtime2::platformReturnToAction()
{
    pause = true;
    returningToAction = true;
    movingToGround = false;

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>();
    motPlat->getSendMutex().release();
    gealg::mv<2, 0x0201>::type r_fl;
    r_fl[0] = 0.41;
    r_fl[1] = 1.0;
    gealg::mv<2, 0x0201>::type r_fr;
    r_fr[0] = 0.41;
    r_fr[1] = -0.3;
    gealg::mv<2, 0x0201>::type r_r;
    r_r[0] = -0.73;
    r_r[1] = 0.35;
    gealg::mv<1, 0x07>::type d_Pb = (grade<1>(cardyn::p_b) ^ grade<2>(cardyn::P_xy))(y);
    gealg::mv<1, 0x07>::type d_P_fl = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fl)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
    gealg::mv<1, 0x07>::type d_P_fr = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_fr)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);
    gealg::mv<1, 0x07>::type d_P_r = (((cardyn::p_b + grade<1>((!cardyn::q_b) * (r_r)*cardyn::q_b)) ^ (grade<2>(cardyn::P_xy))) - d_Pb)(y);

    if (d_P_fl[0] == d_P_fl[0] && d_P_fr[0] == d_P_fr[0] && d_P_r[0] == d_P_r[0])
    {
        motPlat->getSendMutex().acquire(period);
        //Right
        motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + d_P_fr[0]);
        //Left
        motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + d_P_fl[0]);
        //Rear
        motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + d_P_r[0]);

        motPlat->setVelocitySetpoint(0, ValidateMotionPlatform::velMax * 0.1);
        motPlat->setVelocitySetpoint(1, ValidateMotionPlatform::velMax * 0.1);
        motPlat->setVelocitySetpoint(2, ValidateMotionPlatform::velMax * 0.1);

        motPlat->setAccelerationSetpoint(0, ValidateMotionPlatform::accMax * 0.1);
        motPlat->setAccelerationSetpoint(1, ValidateMotionPlatform::accMax * 0.1);
        motPlat->setAccelerationSetpoint(2, ValidateMotionPlatform::accMax * 0.1);

        motPlat->getSendMutex().release();
    }
}

std::pair<Road *, Vector2D> FourWheelDynamicsRealtime2::getStartPositionOnRoad()
{
    double targetS = 5.0;

    RoadSystem *system = RoadSystem::Instance();

    for (int roadIt = 0; roadIt < system->getNumRoads(); ++roadIt)
    {
        Road *road = system->getRoad(roadIt);
        if (road->getLength() >= 2.0 * targetS)
        {
            LaneSection *section = road->getLaneSection(targetS);
            if (section)
            {
                for (int laneIt = -1; laneIt >= -section->getNumLanesRight(); --laneIt)
                {
                    Lane *lane = section->getLane(laneIt);
                    if (lane->getLaneType() == Lane::DRIVING)
                    {
                        double t = 0.0;
                        for (int laneWidthIt = -1; laneWidthIt > laneIt; --laneWidthIt)
                        {
                            t -= section->getLane(laneWidthIt)->getWidth(targetS);
                        }
                        t -= 0.5 * lane->getWidth(targetS);
                        return std::make_pair(road, Vector2D(targetS, t));
                    }
                }
            }
        }
    }

    return std::make_pair((Road *)NULL, Vector2D::NaV());
}
