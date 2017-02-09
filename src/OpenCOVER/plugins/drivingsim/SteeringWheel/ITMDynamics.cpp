/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ITMDynamics.h"

#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>
#include "SteeringWheel.h"
#include <util/unixcompat.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>

ITMDynamics::ITMDynamics()
{
    serverPort = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.Dynamics.ITMServer", 31022);
    std::string remoteHost = coCoviseConfig::getEntry("host", "COVER.Plugin.SteeringWheel.Dynamics.ITMServer");
    serverHost = NULL;
    haveWheels = true;

    doRun = false;
    if (!remoteHost.empty())
    {
        serverHost = new Host(remoteHost.c_str());
    }
    sendBuffer.angle = 0.0;
    sendBuffer.gas = 0.0;
    sendBuffer.bremse = 0.0;
    //sendBuffer.kupplung=0.0;
    sendBuffer.gear = 0.0;
    receiveBuffer.transform[0] = 1;
    receiveBuffer.transform[1] = 0;
    receiveBuffer.transform[2] = 0;
    receiveBuffer.transform[3] = 0;
    receiveBuffer.transform[4] = 1;
    receiveBuffer.transform[5] = 0;
    receiveBuffer.transform[6] = 0;
    receiveBuffer.transform[7] = 0;
    receiveBuffer.transform[8] = 1;
    receiveBuffer.transform[9] = 0;
    receiveBuffer.transform[10] = 0;
    receiveBuffer.transform[11] = 0;
    for (int i = 0; i < 4; i++)
    {
        receiveBuffer.wheel[i][0] = 1;
        receiveBuffer.wheel[i][1] = 0;
        receiveBuffer.wheel[i][2] = 0;
        receiveBuffer.wheel[i][3] = 0;
        receiveBuffer.wheel[i][4] = 1;
        receiveBuffer.wheel[i][5] = 0;
        receiveBuffer.wheel[i][6] = 0;
        receiveBuffer.wheel[i][7] = 0;
        receiveBuffer.wheel[i][8] = 1;
        receiveBuffer.wheel[i][9] = 0;
        receiveBuffer.wheel[i][10] = 0;
        receiveBuffer.wheel[i][11] = 0;
    }
    for (int i = 0; i < 2; i++)
    {
        receiveBuffer.axle[i][0] = 1;
        receiveBuffer.axle[i][1] = 0;
        receiveBuffer.axle[i][2] = 0;
        receiveBuffer.axle[i][3] = 0;
        receiveBuffer.axle[i][4] = 1;
        receiveBuffer.axle[i][5] = 0;
        receiveBuffer.axle[i][6] = 0;
        receiveBuffer.axle[i][7] = 0;
        receiveBuffer.axle[i][8] = 1;
        receiveBuffer.axle[i][9] = 0;
        receiveBuffer.axle[i][10] = 0;
        receiveBuffer.axle[i][11] = 0;
    }
    receiveBuffer.velocity = 0;
    receiveBuffer.dz = 0;
    receiveBuffer.torque = 0;
    memcpy(&appReceiveBuffer, &receiveBuffer, sizeof(receiveBuffer));

    oldTime = 0;

    itmToOsg.makeIdentity();
    itmToOsg(0, 0) = 0;
    itmToOsg(0, 1) = 0;
    itmToOsg(0, 2) = -1;
    itmToOsg(1, 0) = 1;
    itmToOsg(1, 1) = 0;
    itmToOsg(1, 2) = 0;
    itmToOsg(2, 0) = 0;
    itmToOsg(2, 1) = -1;
    itmToOsg(2, 2) = 0;
    invItmToOsg.invert(itmToOsg);

    conn = NULL;

    carTrans.makeIdentity();
}
ITMDynamics::~ITMDynamics()
{
    doRun = false;
    if (conn)
    {
        fprintf(stderr, "waiting1\n");
        endBarrier.block(2); // wait until communication thread finishes
        fprintf(stderr, "done1\n");
        delete conn;
    }
}
void ITMDynamics::update()
{
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((char *)&doRun, sizeof(doRun));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&doRun, sizeof(doRun));
    }

    if (conn == NULL && (coVRMSController::instance()->isMaster()) && (serverHost != NULL))
    {
        // try to connect to server every 2 secnods
        if ((cover->frameTime() - oldTime) > 2)
        {
            cerr << "trying to connect to Matlab" << endl;
            conn = new SimpleClientConnection(serverHost, serverPort, 0);

            if (conn && conn->is_connected())
            {
                cerr << "connected to Matlab" << endl;
                doRun = true;
                startThread();
            }
            else
            {
                delete conn;
                conn = NULL;
            }
            oldTime = cover->frameTime();
        }
    }
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

void ITMDynamics::run() // receiving and sending thread, also does the low level simulation like hard limits
{
    while (doRun)
    {
        osg::Vec3 carPos = carTrans.getTrans();
        coCoord coord = carTrans;
        sendBuffer.angle = -(InputDevice::instance()->getSteeringWheelAngle() / M_PI) * 180;
        sendBuffer.gas = InputDevice::instance()->getAccelerationPedal();
        sendBuffer.bremse = InputDevice::instance()->getBrakePedal();
        sendBuffer.gear = InputDevice::instance()->getGear();
        sendBuffer.clutch = InputDevice::instance()->getClutchPedal();
        sendBuffer.height = carPos[1];

        sendBuffer.angles[0] = coord.hpr[1];
        sendBuffer.angles[1] = coord.hpr[2];

        while (doRun && !readData()) // should be blocking, but anyway...
        {
            conn = NULL;
            doRun = false;
            return;
        }
        sendData();
        //fprintf(stderr,"velo:%lf\n",receiveBuffer.velocity);
        usleep(1000);
    }
    fprintf(stderr, "waiting2\n");
    endBarrier.block(2);
    fprintf(stderr, "done2\n");
}
void ITMDynamics::sendData()
{
    writeVal(&sendBuffer, sizeof(sendBuffer));
    sendBuffer.reset = 0.0;
}
bool ITMDynamics::readData()
{
    //fprintf(stderr,"reading: %ld\n",(long)toITMDynamics);
    return (readVal(&receiveBuffer, sizeof(receiveBuffer)));
} // returns true on success, false if no data has been received.

bool ITMDynamics::readVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    if (conn == NULL)
        return false;
    while (numRead < numBytes)
    {
        readBytes = conn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes < 0)
        {
            cerr << "error reading data from socket" << endl;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}
bool ITMDynamics::writeVal(void *buf, unsigned int numBytes)
{
    unsigned int toWrite = numBytes;
    unsigned int numWritten = 0;
    int writtenBytes = 0;
    if (conn == NULL)
        return false;
    while (numWritten < numBytes)
    {
        writtenBytes = conn->getSocket()->write(((unsigned char *)buf) + numWritten, toWrite);
        if (writtenBytes < 0)
        {
            cerr << "error reading data from socket" << endl;
            return false;
        }
        numWritten += writtenBytes;
        toWrite = numWritten - numBytes;
    }
    return true;
}

void ITMDynamics::setSteeringWheelAngle(double v)
{
    sendBuffer.angle = v;
}
void ITMDynamics::setGas(double v)
{
    sendBuffer.gas = v;
}
void ITMDynamics::setBrake(double v)
{
    sendBuffer.bremse = v;
}
void ITMDynamics::setGear(double v)
{
    sendBuffer.gear = v;
}
void ITMDynamics::setClutch(double v)
{
    sendBuffer.clutch = v;
}
void ITMDynamics::setHeight(double v)
{
    sendBuffer.height = v;
}
void ITMDynamics::setOrientation(double v1, double v2)
{
    sendBuffer.angles[0] = v1;
    sendBuffer.angles[1] = v2;
}

double ITMDynamics::getVelocity()
{
    return (appReceiveBuffer.velocity);
}
double ITMDynamics::getEngineSpeed()
{
    return (appReceiveBuffer.dz);
}
double ITMDynamics::getSteeringWheelTorque()
{
    return (receiveBuffer.torque);
}

const osg::Matrix &ITMDynamics::getVehicleTransformation()
{
    return carTrans;
}

void ITMDynamics::move(VrmlNodeVehicle *vehicle)
{
    {
        int i, j;
        carTrans.makeIdentity();
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < 4; j++)
            {
                carTrans(j, i) = appReceiveBuffer.transform[j * 3 + i];
            }
        }
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < 3; j++)
            {
                carTrans(i, j) = appReceiveBuffer.transform[j * 3 + i];
            }
        }
        carTrans = invItmToOsg * carTrans * itmToOsg;

        vehicle->moveToStreet();

        osg::Matrix axle1Trans;
        osg::Matrix axle2Trans;
        osg::Matrix wheel1Trans;
        osg::Matrix wheel2Trans;
        osg::Matrix wheel3Trans;
        osg::Matrix wheel4Trans;
        if (haveWheels)
        {
            axle1Trans.makeIdentity();
            axle2Trans.makeIdentity();
            wheel1Trans.makeIdentity();
            wheel2Trans.makeIdentity();
            wheel3Trans.makeIdentity();
            wheel4Trans.makeIdentity();
            for (i = 0; i < 3; i++)
            {
                axle1Trans(3, i) = appReceiveBuffer.axle[0][9 + i];
                axle2Trans(3, i) = appReceiveBuffer.axle[1][9 + i];
                wheel1Trans(3, i) = appReceiveBuffer.wheel[0][9 + i];
                wheel2Trans(3, i) = appReceiveBuffer.wheel[1][9 + i];
                wheel3Trans(3, i) = appReceiveBuffer.wheel[2][9 + i];
                wheel4Trans(3, i) = appReceiveBuffer.wheel[3][9 + i];
            }
            for (i = 0; i < 3; i++)
            {
                for (j = 0; j < 3; j++)
                {
                    axle1Trans(i, j) = appReceiveBuffer.axle[0][j * 3 + i];
                    axle2Trans(i, j) = appReceiveBuffer.axle[1][j * 3 + i];
                    wheel1Trans(i, j) = appReceiveBuffer.wheel[0][j * 3 + i];
                    wheel2Trans(i, j) = appReceiveBuffer.wheel[1][j * 3 + i];
                    wheel3Trans(i, j) = appReceiveBuffer.wheel[2][j * 3 + i];
                    wheel4Trans(i, j) = appReceiveBuffer.wheel[3][j * 3 + i];
                }
            }
            vehicle->setVRMLVehicleAxles(axle1Trans, axle2Trans);
            vehicle->setVRMLVehicleWheels(wheel1Trans, wheel2Trans, wheel3Trans, wheel4Trans);
        }

        vehicle->setVRMLVehicle(carTrans);
    }
}

void ITMDynamics::resetState()
{
    sendBuffer.reset = 1.0;

    sendBuffer.angle = 0.0;
    sendBuffer.gas = 0.0;
    sendBuffer.bremse = 0.0;
    //sendBuffer.kupplung=0.0;
    sendBuffer.gear = 0.0;
    receiveBuffer.transform[0] = 1;
    receiveBuffer.transform[1] = 0;
    receiveBuffer.transform[2] = 0;
    receiveBuffer.transform[3] = 0;
    receiveBuffer.transform[4] = 1;
    receiveBuffer.transform[5] = 0;
    receiveBuffer.transform[6] = 0;
    receiveBuffer.transform[7] = 0;
    receiveBuffer.transform[8] = 1;
    receiveBuffer.transform[9] = 0;
    receiveBuffer.transform[10] = 0;
    receiveBuffer.transform[11] = 0;
    for (int i = 0; i < 4; i++)
    {
        receiveBuffer.wheel[i][0] = 1;
        receiveBuffer.wheel[i][1] = 0;
        receiveBuffer.wheel[i][2] = 0;
        receiveBuffer.wheel[i][3] = 0;
        receiveBuffer.wheel[i][4] = 1;
        receiveBuffer.wheel[i][5] = 0;
        receiveBuffer.wheel[i][6] = 0;
        receiveBuffer.wheel[i][7] = 0;
        receiveBuffer.wheel[i][8] = 1;
        receiveBuffer.wheel[i][9] = 0;
        receiveBuffer.wheel[i][10] = 0;
        receiveBuffer.wheel[i][11] = 0;
    }
    for (int i = 0; i < 2; i++)
    {
        receiveBuffer.axle[i][0] = 1;
        receiveBuffer.axle[i][1] = 0;
        receiveBuffer.axle[i][2] = 0;
        receiveBuffer.axle[i][3] = 0;
        receiveBuffer.axle[i][4] = 1;
        receiveBuffer.axle[i][5] = 0;
        receiveBuffer.axle[i][6] = 0;
        receiveBuffer.axle[i][7] = 0;
        receiveBuffer.axle[i][8] = 1;
        receiveBuffer.axle[i][9] = 0;
        receiveBuffer.axle[i][10] = 0;
        receiveBuffer.axle[i][11] = 0;
    }
    receiveBuffer.velocity = 0;
    receiveBuffer.dz = 0;
    receiveBuffer.torque = 0;
    memcpy(&appReceiveBuffer, &receiveBuffer, sizeof(receiveBuffer));

    oldTime = 0;

    itmToOsg.makeIdentity();
    itmToOsg(0, 0) = 0;
    itmToOsg(0, 1) = 0;
    itmToOsg(0, 2) = -1;
    itmToOsg(1, 0) = 1;
    itmToOsg(1, 1) = 0;
    itmToOsg(1, 2) = 0;
    itmToOsg(2, 0) = 0;
    itmToOsg(2, 1) = -1;
    itmToOsg(2, 2) = 0;
    invItmToOsg.invert(itmToOsg);
}
