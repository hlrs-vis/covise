/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FKFS.h"

#include <config/CoviseConfig.h>
#include "SteeringWheel.h"

FKFS::FKFS()
{
    std::string remoteHost;
    int remotePort;
    int localPort;
    remoteHost = coCoviseConfig::getEntry("host", "COVER.Plugin.SteeringWheel.FKFS");
    remotePort = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.FKFS", 49030);
    localPort = coCoviseConfig::getInt("localPort", "COVER.Plugin.SteeringWheel.FKFS", 49000);
    origin = coCoviseConfig::getFloat("origin", "COVER.Plugin.SteeringWheel.FKFS", 0.0);
    doRun = coCoviseConfig::isOn("COVER.Plugin.SteeringWheel.FKFS", false);
    toFKFS = NULL;
    if (doRun)
        if (!remoteHost.empty() && coVRMSController::instance()->isMaster())
        {
            toFKFS = new UDPComm(remoteHost.c_str(), remotePort, localPort);
            //fprintf(stderr, "toFKFS: %ld\n", (long)toFKFS);
            sendBuffer.t_req = 1.0;
            sendBuffer.Enable = 1.0;
            sendBuffer.Mode = 0.0;
            sendBuffer.msgcount = 0.0;
            //run();
            //fprintf(stderr, "done: %ld\n", (long)toFKFS);
            startThread();
            maxAngle = 1.5;
        }
}
FKFS::~FKFS()
{
    doRun = false;
    if (toFKFS)
    {
        fprintf(stderr, "waiting1\n");
        endBarrier.block(2); // wait until communication thread finishes
        fprintf(stderr, "done1\n");
        delete toFKFS;
    }
}

double FKFS::getAngle() // return steering wheel angle
{
    return appReceiveBuffer.Alpha - origin;
}

double FKFS::getfastAngle() // return steering wheel angle
{
    return receiveBuffer.Alpha - origin;
}

void FKFS::update()
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

void FKFS::run() // receiving and sending thread, also does the low level simulation like hard limits
{
    int direction = 1;
    while (doRun)
    {
        //fprintf(stderr,"sending: %ld\n",(long)toFKFS);
        sendData();
        // message counter counts up and down from 1 to 10000
        sendBuffer.msgcount += 1.0 * direction;
        if (sendBuffer.msgcount >= 10000.0)
            direction *= -1;
        if (sendBuffer.msgcount <= 0.0)
            direction *= -1;

        if (SteeringWheelPlugin::plugin->dynamics)
            sendBuffer.t_req = SteeringWheelPlugin::plugin->dynamics->getSteeringWheelTorque() * 0.2 * SteeringWheelPlugin::plugin->springConstant->getValue();
        else
            sendBuffer.t_req = -getfastAngle() * SteeringWheelPlugin::plugin->springConstant->getValue();

        sendBuffer.t_req -= receiveBuffer.Speed * SteeringWheelPlugin::plugin->dampingConstant->getValue();
        sendBuffer.t_req += ((rand() / ((double)RAND_MAX)) - 0.5) * SteeringWheelPlugin::plugin->rumbleFactor->getValue();

        maxAngle = SteeringWheelPlugin::plugin->blockAngle->getValue();
        if (getfastAngle() > maxAngle)
            sendBuffer.t_req -= (getfastAngle() - maxAngle) * 100;
        if (getfastAngle() < -maxAngle)
            sendBuffer.t_req -= (getfastAngle() + maxAngle) * 100;
        while (doRun && !readData())
        {
            microSleep(10);
        }
        //fprintf(stderr,"Alpha:%lf Speed:%lf Accel:%lf Brake:%lf Clutch:%lf selectorLever:%lf Ignition:%lf parkBrake:%lf Ready:%lf Simtime:%lf\n",getAngle(),receiveBuffer.Speed,receiveBuffer.Accel,receiveBuffer.Brake,receiveBuffer.Clutch,receiveBuffer.selectorLever,receiveBuffer.Ignition,receiveBuffer.parkBrake,receiveBuffer.Ready,receiveBuffer.Simtime);
    }
    fprintf(stderr, "waiting2\n");
    endBarrier.block(2);
    fprintf(stderr, "done2\n");
}
void FKFS::sendData()
{
    toFKFS->send(&sendBuffer, sizeof(sendBuffer));
}
bool FKFS::readData()
{
    //fprintf(stderr,"reading: %ld\n",(long)toFKFS);
    return (toFKFS->receive(&receiveBuffer, sizeof(receiveBuffer)) == sizeof(receiveBuffer));
} // returns true on success, false if no data has been received.
