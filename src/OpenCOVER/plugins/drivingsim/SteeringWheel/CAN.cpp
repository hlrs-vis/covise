/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CAN.h"

#ifdef HAVE_PCAN
#define HANDLE QT_HANDLE
#include <config/CoviseConfig.h>
#undef HANDLE
#include "SteeringWheel.h"

CAN::CAN()
{
    port = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.CAN", 1);
    speed = coCoviseConfig::getInt("speed", "COVER.Plugin.SteeringWheel.CAN", 1000);
    nodeID = coCoviseConfig::getInt("nodeID", "COVER.Plugin.SteeringWheel.CAN", 1);
    origin = coCoviseConfig::getFloat("origin", "COVER.Plugin.SteeringWheel.CAN", 0.0);
    guardtime = coCoviseConfig::getInt("guardtime", "COVER.Plugin.SteeringWheel.CAN", 200);
    doRun = coCoviseConfig::isOn("COVER.Plugin.SteeringWheel.CAN", false);
    carVel = 0;
    actAngle = 0;
    driftAngle = 0;
    tanhCarVel = 0;
    rf = 0;
    maxAngle = 1.5;
    wheel = NULL;
    bus = NULL;
    can = NULL;
    if (doRun && coVRMSController::instance()->isMaster())
    {
        if (speed >= 1000)
            sp = CAN_BAUD_1M;
        else if (speed >= 500)
            sp = CAN_BAUD_500K;
        else if (speed >= 250)
            sp = CAN_BAUD_250K;
        else if (speed >= 125)
            sp = CAN_BAUD_125K;
        else if (speed >= 100)
            sp = CAN_BAUD_100K;
        else if (speed >= 50)
            sp = CAN_BAUD_50K;
        else if (speed >= 20)
            sp = CAN_BAUD_20K;
        else if (speed >= 10)
            sp = CAN_BAUD_10K;
        else if (speed >= 5)
            sp = CAN_BAUD_5K;
#ifdef WIN32
        can = new PcanLight(PCI_1CH, (Baudrates)sp);
#else
        can = new PcanPci(port, sp);
#endif
        bus = new CanOpenBus(can);
        wheel = new PorscheSteeringWheel(bus, nodeID);

        if (initWheel())
        {
            //wheel->niceProcess(1);
            wheel->setAffinity(0);
            Init();
            startThread();
        }
    }
}

CAN::~CAN()
{
    if (coVRMSController::instance()->isMaster())
    {
        fprintf(stderr, "~CAN waiting1\n");
        doRun = false;
        if (wheel)
        {
            fprintf(stderr, "waiting1\n");
            endBarrier.block(2); // wait until communication thread finishes
            fprintf(stderr, "done1\n");
            //wheel->shutdown();
            //wheel->stopNode();
            delete wheel;
            delete bus;
            delete can;
        }
    }
}

bool CAN::initWheel()
{
    std::cerr << "Resetting wheel... ";
    if (!wheel->resetWheel())
        std::cerr << "failed!" << std::endl;
    else
        std::cerr << "done!" << std::endl;

    std::cerr << "Checking wheel... ";
    if (!wheel->checkWheel(100000, 600))
        return false;

    std::cerr << "Calibrating wheel... ";
    if (!wheel->homeWheel())
    {
        std::cerr << "failed!" << std::endl;
        doRun = false;
    }
    else
    {
        std::cerr << "done!" << std::endl;

        std::cerr << "Setting up wheel control... Guardtime: " << guardtime << "ms... ";
        if (!wheel->setupAngleTorqueMode(0x1, guardtime))
        {
            std::cerr << "failed!" << std::endl;
            doRun = false;
        }
        else
        {
            std::cerr << "done!" << std::endl;
            return true;
        }
    }
    return false;
}

double CAN::getAngle() // return steering wheel angle
{
    return appReceiveBuffer.Alpha - origin;
}

double CAN::getfastAngle() // return steering wheel angle
{
    return receiveBuffer.Alpha - origin;
}

void CAN::setRoadFactor(float r) // set roughness
{
    rf = r;
}

void CAN::update()
{

    if (coVRMSController::instance()->isMaster())
    {
        memcpy(&appReceiveBuffer, &receiveBuffer, sizeof(receiveBuffer));
        coVRMSController::instance()->sendSlaves((char *)&appReceiveBuffer, sizeof(receiveBuffer));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&appReceiveBuffer, sizeof(receiveBuffer));
    }
}

void CAN::run() // receiving and sending thread, also does the low level simulation like hard limits
{
    //wheel->setScheduler();
    wheel->setAffinity(1);

    //int direction=1;
    double torque;
    unsigned char count = 0;

    std::cerr << "Starting wheel PDO transfer...";
    if (!wheel->startAngleTorqueMode())
    {
        doRun = false;
        std::cerr << "failed!" << std::endl;
    }
    else
        std::cerr << "done!" << std::endl;

    //unsigned long updates = 0;
    //unsigned long int starttime = wheel->getTimeInMicroseconds();
    while (doRun)
    {
        if ((count % 10) == 0)
            wheel->sendGuardMessage();
        ++count;

        bus->sendSYNC();
        if (wheel->getAngleVelocityInRadians(receiveBuffer.Alpha, receiveBuffer.Speed))
        {

            //if(SteeringWheelPlugin::plugin->vd && SteeringWheelPlugin::plugin->vd->doRun)
            /*
      if(SteeringWheelPlugin::plugin->dynamics)
         //torque = SteeringWheelPlugin::plugin->vd->getTorque()*0.2*SteeringWheelPlugin::plugin->springConstant->getValue();
         torque = SteeringWheelPlugin::plugin->dynamics->getSteeringWheelTorque();
      else {
      */
            tanhCarVel = tanh(SteeringWheelPlugin::plugin->velocityImpactFactor->getValue() * carVel);
            einsMinusTanhCarVel = 1 - tanhCarVel;
            tanhCarVelRumble = tanh(SteeringWheelPlugin::plugin->velocityImpactFactorRumble->getValue() * carVel);

            actAngle = getfastAngle();
            torque = -actAngle * SteeringWheelPlugin::plugin->springConstant->getValue() * tanhCarVel;

            //if((count%100)==0) std::cerr << "Car Velocity: " << carVel << std::endl;

            //torque = (driftAngle-actAngle)*SteeringWheelPlugin::plugin->springConstant->getValue();
            if ((driftAngle - actAngle) > einsMinusTanhCarVel)
                driftAngle = actAngle + einsMinusTanhCarVel;
            else if ((driftAngle - actAngle) < -einsMinusTanhCarVel)
                driftAngle = actAngle - einsMinusTanhCarVel;
            torque += (driftAngle - actAngle) * SteeringWheelPlugin::plugin->drillingFrictionConstant->getValue();

            torque -= receiveBuffer.Speed * SteeringWheelPlugin::plugin->dampingConstant->getValue();
            //}

            torque += ((rand() / ((double)RAND_MAX)) - 0.5) * rf * SteeringWheelPlugin::plugin->rumbleFactor->getValue() * tanhCarVelRumble;
            maxAngle = SteeringWheelPlugin::plugin->blockAngle->getValue();

            if (getfastAngle() > maxAngle)
                torque -= (getfastAngle() - maxAngle) * 100;
            if (getfastAngle() < -maxAngle)
                torque -= (getfastAngle() + maxAngle) * 100;

            wheel->setTorqueInNm(torque);
            //std::cerr << "Angle: " << receiveBuffer.Alpha << "\tVelocity: " << receiveBuffer.Speed << "\tTorque: " << torque << std::endl;

            //microSleep(1000);
            wheel->microsleep(1000);
            //wheel->microwait(1000);

            /*
		++updates;
		if(updates>100) {
			std::cerr << "CAN thread update rate: " << (double)updates/((double)(wheel->getTimeInMicroseconds()-starttime)/1000000) << std::endl;
			updates = 0;
			starttime = wheel->getTimeInMicroseconds();
		}
		*/
        }
    }

    std::cerr << "Stopping PDO transfer, shutdown node, stop life guarding...";
    if (!wheel->stopAngleTorqueMode())
        std::cerr << "failed!" << std::endl;
    else
        std::cerr << "done!" << std::endl;

    fprintf(stderr, "waiting2\n");
    endBarrier.block(2);
    fprintf(stderr, "done2\n");
}

void CAN::softResetWheel()
{
    if (doRun == true)
    {
        doRun = false;
        endBarrier.block(2);
        //cancel();
    }

    actAngle = 0;
    driftAngle = 0;

    if (wheel)
    {
        if (initWheel())
        {
            //wheel->niceProcess(1);
            //wheel->setAffinity(0);
            Init();
            doRun = true;
            startThread();
        }
        else
            doRun = false;
    }
}

void CAN::cruelResetWheel()
{
    cancel();

    delete wheel;
    delete bus;
    delete can;

    actAngle = 0;
    driftAngle = 0;

#ifdef WIN32
    can = new PcanLight(PCI_1CH, (Baudrates)sp);
#else
    can = new PcanPci(port, sp);
#endif
    bus = new CanOpenBus(can);
    wheel = new PorscheSteeringWheel(bus, nodeID);

    if (wheel)
    {
        doRun = true;
        if (initWheel())
        {
            //wheel->niceProcess(1);
            wheel->setAffinity(0);
            Init();
            startThread();
        }
    }
}

void CAN::shutdownWheel()
{
    if (doRun == true)
    {
        doRun = false;
        endBarrier.block(2);
        //cancel();
    }

    if (wheel)
    {
        //wheel->stopNode();
        wheel->shutdown();
    }
}

#endif
