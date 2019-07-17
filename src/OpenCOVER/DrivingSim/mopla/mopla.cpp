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
    start();
}

void mopla::run()
{
    fum->update();
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
    /*
    
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlMiddleLift>();
    motPlat->getSendMutex().release();
    
    
				motPlat->getSendMutex().acquire(period);
				//motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>();
				motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>();
				for (unsigned int motIt = 0; motIt < motPlat->numLinMots; ++motIt)
				{
					motPlat->setVelocitySetpoint(motIt, ValidateMotionPlatform::velMax);
					motPlat->setAccelerationSetpoint(motIt, ValidateMotionPlatform::accMax);
				}
				motPlat->getSendMutex().release();
    while (runTask)
	{
		
    
		motPlat->getSendMutex().acquire(period);
		//Right
		motPlat->setPositionSetpoint(0, ValidateMotionPlatform::posMiddle + carState.mpRZ);
		//Left
		motPlat->setPositionSetpoint(1, ValidateMotionPlatform::posMiddle + carState.mpLZ);
		//Rear
		motPlat->setPositionSetpoint(2, ValidateMotionPlatform::posMiddle + carState.mpBZ);
		motPlat->getSendMutex().release();
   }
		*/
    
}


mopla::~mopla()
{
    delete motPlat;
    delete fum;
}


