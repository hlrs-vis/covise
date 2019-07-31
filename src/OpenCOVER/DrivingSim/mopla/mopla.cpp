/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <mopla.h>
#include <unistd.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <xenomai/init.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <vrbclient/SharedStateManager.h>

using namespace opencover;
int main(int argc, char* const* argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: mopla file.xodr\n");
        return -1;
    }
    xenomai_init(&argc, &argv); 
    mopla *m = new mopla(argv[1]);
    while(true)
    {
    sleep(1000);
    }
    
    delete m;

    return 1;
}

mopla *mopla::myFasi = NULL;

mopla::mopla(const char *filename)
#ifdef MERCURY
    : XenomaiTask::XenomaiTask("FourWheelDynamicsRealtime2Task", 0, 99, 0)
#else
    : XenomaiTask::XenomaiTask("FourWheelDynamicsRealtime2Task", 0, 99, T_FPU | T_CPU(5))
#endif
{
    myFasi = this;
    new vrb::SharedStateManager(NULL);
    opencover::cover = new opencover::coVRPluginSupport();
    
    
    motPlat = ValidateMotionPlatform::instance();
    
    fum = fasiUpdateManager::instance();
    p_kombi = KI::instance();
    p_klsm = KLSM::instance();
    p_klima = Klima::instance();
    p_beckhoff = Beckhoff::instance();
    //p_brakepedal = BrakePedal::instance();
    p_gaspedal = GasPedal::instance();
    p_ignitionLock = IgnitionLock::instance();
    vehicleUtil = VehicleUtil::instance();

    p_beckhoff->setDigitalOut(0, 0, false);
    p_beckhoff->setDigitalOut(0, 1, false);
    p_beckhoff->setDigitalOut(0, 2, false);
    p_beckhoff->setDigitalOut(0, 3, false);
    p_beckhoff->setDigitalOut(0, 4, false);
    p_beckhoff->setDigitalOut(0, 5, false);
    p_beckhoff->setDigitalOut(0, 6, false);
    p_beckhoff->setDigitalOut(0, 7, false);
    
    
    motPlat->start();
    run();
}

void mopla::run()
{

    std::cout << "motPlat run" << std::endl;
  //==========================================
    enum state { standby, movingUp, running, movingDown, stop };
    state st = stop;
    bool isRunning = false;
    double startTime = 0;

    //====================

    fum->update();
    std::cerr << "--- motPlat->start();  ---" << std::endl;
    while (!motPlat->isInitialized())
    {
        rt_task_sleep(1000000);
        std::cerr << "--- motPlat->waiting for initialization();  ---" << std::endl;
    }
    sleep(10);
    std::cerr << "--- motPlat->start(); done ---" << std::endl;
    set_periodic(period);
    std::cerr << "--- motPlat->start();set_periodic done ---" << std::endl;
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
    while (!motPlat->isGrounded())
    {
        rt_task_wait_period(&overruns);
    }
    
    std::cout << "Plattform on ground" << std::endl;
    
    std::cerr << "--- isGrounded(); done ---" << std::endl;
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();
    //============================
    std::cout << "Platfom control disabled, state == standby" << std::endl;
    state oldState;
    oldState = st = standby;
    
    isRunning = true;
	
	while(isRunning)
	{

	 
	 
		if (st == standby && p_ignitionLock->getLockState() == IgnitionLock::ENGINESTART)
		{
		std::cout << "EngineStart detected: state == movingUp, controlMiddleLift" << std::endl;
			st = movingUp;
			motPlat->getSendMutex().acquire(period);
			motPlat->switchToMode<ValidateMotionPlatform::controlMiddleLift>();
			motPlat->getSendMutex().release();
		}

		if (st == movingUp)
		{
			double relativeSpeed = 0.1;
			double relativeAccel = 0.1;
		
			if(motPlat->isMiddleLifted())
			{
				motPlat->getSendMutex().acquire(period);
				motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>();
				for (unsigned int motIt = 0; motIt < motPlat->numLinMots; ++motIt)
				{
					motPlat->setVelocitySetpoint(motIt, ValidateMotionPlatform::velMax*relativeSpeed);
					motPlat->setAccelerationSetpoint(motIt, ValidateMotionPlatform::accMax*relativeAccel);
				}
				motPlat->getSendMutex().release();
				startTime = opencover::cover->frameTime();
				st = running;
			        std::cout << "Setup done, state = running" << std::endl;
			
			}
			

		}
    
		if (st == running)
		{
			double amplitude = 0.05; //meters
			double wavePeriod = 2.0; //seconds
			double time = (opencover::cover->frameTime()) - startTime;
			double newPosition = amplitude*(sin((2*M_PI/wavePeriod)*time));

			std::cout << "pos: " << newPosition << std::endl;
			motPlat->getSendMutex().acquire(period);
			//Right---- 
			motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + newPosition); 
			//Left
			motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + newPosition);
			//Rear
			motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + newPosition);
			
			motPlat->getSendMutex().release();
		}
     
		if (st != movingDown && p_ignitionLock->getLockState() == IgnitionLock::ENGINESTOP)
		{
			std::cout << "ENGINESTOP detected" << std::endl;
			st = movingDown;
			motPlat->getSendMutex().acquire(period);
			motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
			motPlat->getSendMutex().release();
			
			
		}
		
		if (st == movingDown)
		{
			if (motPlat->isGrounded())
   			 {
        
				motPlat->getSendMutex().acquire(period);
   			        motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
                                motPlat->getSendMutex().release();
                                st = standby;
				std::cout << "Platform down, control disabled" << std::endl;
   			 }
		
		}
		
		if (p_klsm->getHornStat() == true)
		{
			std::cout << "horn detected, isRunning == false" << std::endl;
			isRunning = false;
		}
                if(st != oldState)
                {
			std::cout << "state = " << st << std::endl;
                        oldState = st;
                }
                
                rt_task_wait_period(&overruns);
		
	}


    
}


mopla::~mopla()
{
    delete motPlat;
    delete fum;
}


