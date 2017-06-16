/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FKFSDynamics.h"
#ifdef debug
#undef debug
#endif


#include <config/CoviseConfig.h>
#include "SteeringWheel.h"

FKFSDynamics::FKFSDynamics()
    : VehicleDynamics()
{
    serverPort = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.Dynamics.FKFSServer", 47003);
    localPort = coCoviseConfig::getInt("localPort", "COVER.Plugin.SteeringWheel.Dynamics.FKFSServer", 47002);
    haveWheels = true;
    std::string remoteHost;
    remoteHost = coCoviseConfig::getEntry("host", "COVER.Plugin.SteeringWheel.Dynamics.FKFSServer", "192.168.0.20");
    doRun = false;
    oldTime = 0;
    appNumObjects = receiveNumObjects = 0;

    double startElev = coCoviseConfig::getFloat("startElevation", "COVER.Plugin.SteeringWheel.Dynamics.IntersectionTest", 0.0);
    sendBuffer.wheelElevVL = startElev;
    sendBuffer.wheelElevVR = startElev;
    sendBuffer.wheelElevHL = startElev;
    sendBuffer.wheelElevHR = startElev;

    FKFSDynamicsToOsg.makeIdentity();
    FKFSDynamicsToOsg(0, 0) = 1;
    FKFSDynamicsToOsg(0, 1) = 0;
    FKFSDynamicsToOsg(0, 2) = 0;
    FKFSDynamicsToOsg(1, 0) = 0;
    FKFSDynamicsToOsg(1, 1) = 0;
    FKFSDynamicsToOsg(1, 2) = -1;
    FKFSDynamicsToOsg(2, 0) = 0;
    FKFSDynamicsToOsg(2, 1) = 1;
    FKFSDynamicsToOsg(2, 2) = 0;
    invFKFSDynamicsToOsg.invert(FKFSDynamicsToOsg);

    osg::Matrix rot;
    rot.makeRotate(-M_PI_2, osg::Vec3(0, 1, 0));
    invFKFSDynamicsToOsg = rot * invFKFSDynamicsToOsg;

    toFKFS = NULL;
    if (!remoteHost.empty() && coVRMSController::instance()->isMaster())
    {
        doRun = true;
        toFKFS = new UDPComm(remoteHost.c_str(), serverPort, localPort);
        //fprintf(stderr, "done: %ld\n", (long)toFKFS);
        startThread();
    }

    carTrans.makeIdentity();
}
FKFSDynamics::~FKFSDynamics()
{
    if (doRun)
    {
        doRun = false;
        fprintf(stderr, "waiting for FKFS Dynamics thread\n");
        endBarrier.block(2); // wait until communication thread finishes
        fprintf(stderr, "done waiting\n");
        delete toFKFS;
    }
}
void FKFSDynamics::update()
{
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((char *)&doRun, sizeof(doRun));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&doRun, sizeof(doRun));
    }

    if (coVRMSController::instance()->isMaster())
    {
        memcpy(&appState, &receiveState, sizeof(appState));
        appNumObjects = receiveNumObjects;
        memcpy(&appObjects, &receiveObjects, appNumObjects * sizeof(MotionObjectType));
        coVRMSController::instance()->sendSlaves((char *)&appNumObjects, sizeof(appNumObjects));
        if (appNumObjects > 0)
        {
            coVRMSController::instance()->sendSlaves((char *)&appObjects, appNumObjects * sizeof(MotionObjectType));
        }
        coVRMSController::instance()->sendSlaves((char *)&appState, sizeof(appState));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&appNumObjects, sizeof(appNumObjects));
        if (appNumObjects > 0)
        {
            coVRMSController::instance()->readMaster((char *)&appObjects, appNumObjects * sizeof(MotionObjectType));
        }
        coVRMSController::instance()->readMaster((char *)&appState, sizeof(appState));
    }
}

void FKFSDynamics::run() // receiving and sending thread, also does the low level simulation like hard limits
{
    fprintf(stderr, "starting Thread\n");
    while (doRun)
    {
        sendBuffer.wheelAngle = -(InputDevice::instance()->getSteeringWheelAngle() / M_PI) * 180;
        sendBuffer.gasPedal = InputDevice::instance()->getAccelerationPedal();
        sendBuffer.breakPedal = InputDevice::instance()->getBrakePedal();
        sendBuffer.gear = InputDevice::instance()->getGear();
        sendBuffer.clutchPedal = InputDevice::instance()->getClutchPedal();

        sendData();
        while (doRun && !readData())
        {
            microSleep(10);
        }
    }
    fprintf(stderr, "waiting2\n");
    endBarrier.block(2);
    fprintf(stderr, "done2\n");
}
void FKFSDynamics::sendData()
{
    toFKFS->send(&sendBuffer, sizeof(sendBuffer));
    sendBuffer.reset = 0;
}
bool FKFSDynamics::readData()
{
    int numRecords, i;
    RecordHeader rh;
    MotionObjectType mo;
    CameraObjectType co;
    ControlObjectType cto;
    if (toFKFS->readMessage() == -1)
    {
        sendBuffer.wheelElevVL = 0.0;
        sendBuffer.wheelElevVR = 0.0;
        sendBuffer.wheelElevHL = 0.0;
        sendBuffer.wheelElevHR = 0.0;
    }
    if (toFKFS->getMessagePart(&numRecords, sizeof(numRecords)) != sizeof(numRecords))
        return false;
    for (i = 0; i < numRecords; i++)
    {
        if (toFKFS->getMessagePart(&rh, sizeof(rh)) != sizeof(rh))
            return false;

        if (rh.type == MotionObject)
        {
            if (toFKFS->getMessagePart(&mo, sizeof(mo)) != sizeof(mo))
                return false;
            if (rh.index < MAX_MOTION_OBJECTS)
            {
                if (rh.index > receiveNumObjects)
                    receiveNumObjects = rh.index;
                memcpy(&receiveObjects[rh.index - 1], &mo, sizeof(mo));
            }
        }
        else if (rh.type == CameraObject)
        {
            if (toFKFS->getMessagePart(&co, sizeof(co)) != sizeof(co))
                return false;
            memcpy(&receiveState.co, &co, sizeof(co));
        }
        else if (rh.type == ControlObject)
        {
            if (toFKFS->getMessagePart(&cto, sizeof(cto)) != sizeof(cto))
                return false;
            if (cto.ID == VelocityID)
            {
                receiveState.velocity = cto.val1;
            }
            if (cto.ID == RevolutionID)
            {
                receiveState.revolution = cto.val1;
            }
        }
        else if (rh.type == WheelPositionObject)
        {
            if (toFKFS->getMessagePart(&wheelPosObj, sizeof(wheelPosObj)) != sizeof(wheelPosObj))
            {
                std::cout << "WheelPositionObject error: Received object not of defined object size!" << std::endl;
                return false;
            }
            /*
           //memcpy(&receiveState.co,&co,sizeof(co));

           std::cout << "Wheel pos 1: " << wheelPosObj.transWheelVL[0] << ", " << wheelPosObj.transWheelVL[1] << std::endl;
           std::cout << "Wheel pos 2: " << wheelPosObj.transWheelVR[0] << ", " << wheelPosObj.transWheelVR[1] << std::endl;
           std::cout << "Wheel pos 3: " << wheelPosObj.transWheelHL[0] << ", " << wheelPosObj.transWheelHL[1] << std::endl;
           std::cout << "Wheel pos 4: " << wheelPosObj.transWheelHR[0] << ", " << wheelPosObj.transWheelHR[1] << std::endl;
           */

            osg::Vec2 wheelPosVL(wheelPosObj.transWheelVL[0], wheelPosObj.transWheelVL[1]);
            sendBuffer.wheelElevVL = wheelPosObj.transWheelVL[2];
            osg::Vec2 wheelPosVR(wheelPosObj.transWheelVR[0], wheelPosObj.transWheelVR[1]);
            sendBuffer.wheelElevVR = wheelPosObj.transWheelVR[2];
            osg::Vec2 wheelPosHL(wheelPosObj.transWheelHL[0], wheelPosObj.transWheelHL[1]);
            sendBuffer.wheelElevHL = wheelPosObj.transWheelHL[2];
            osg::Vec2 wheelPosHR(wheelPosObj.transWheelHR[0], wheelPosObj.transWheelHR[1]);
            sendBuffer.wheelElevHR = wheelPosObj.transWheelHR[2];
            sendBuffer.wheelElevVLV = sendBuffer.wheelElevVLL = sendBuffer.wheelElevVL;
            sendBuffer.wheelElevVRV = sendBuffer.wheelElevVRR = sendBuffer.wheelElevVR;
            sendBuffer.wheelElevHLV = sendBuffer.wheelElevHLL = sendBuffer.wheelElevHL;
            sendBuffer.wheelElevHRV = sendBuffer.wheelElevHRR = sendBuffer.wheelElevHR;

            osg::Vec2 wheelOrientLR = wheelPosHR - wheelPosHL;
            wheelOrientLR.normalize();
            osg::Vec2 wheelOrientHV = wheelPosVL - wheelPosHL;
            wheelOrientHV.normalize();

            //contact points
            getWheelElevation(wheelPosVL, wheelPosVR, wheelPosHL, wheelPosHR,
                              sendBuffer.wheelElevVL, sendBuffer.wheelElevVR, sendBuffer.wheelElevHL, sendBuffer.wheelElevHR);
            //side points
            getWheelElevation(wheelPosVL - wheelOrientLR * 0.05, wheelPosVR + wheelOrientLR * 0.05, wheelPosHL - wheelOrientLR * 0.05, wheelPosHR + wheelOrientLR * 0.05,
                              sendBuffer.wheelElevVLL, sendBuffer.wheelElevVRR, sendBuffer.wheelElevHLL, sendBuffer.wheelElevHRR);
            //longitudinal points
            getWheelElevation(wheelPosVL + wheelOrientHV * 0.05, wheelPosVR + wheelOrientHV * 0.05, wheelPosHL + wheelOrientHV * 0.05, wheelPosHR + wheelOrientHV * 0.05,
                              sendBuffer.wheelElevVLV, sendBuffer.wheelElevVRV, sendBuffer.wheelElevHLV, sendBuffer.wheelElevHRV);

            /*std::cerr << "Elevations contact: VL: " << sendBuffer.wheelElevVL << ", VR: " << sendBuffer.wheelElevVR << ", HL: " << sendBuffer.wheelElevHL << ", HR: " << sendBuffer.wheelElevHR << std::endl;
            std::cerr << "Elevations side: VLL: " << sendBuffer.wheelElevVLL << ", VRR: " << sendBuffer.wheelElevVRR << ", HLL: " << sendBuffer.wheelElevHLL << ", HRR: " << sendBuffer.wheelElevHRR << std::endl;
            std::cerr << "Elevations long: VLV: " << sendBuffer.wheelElevVLV << ", VRV: " << sendBuffer.wheelElevVRV << ", HLV: " << sendBuffer.wheelElevHLV << ", HRV: " << sendBuffer.wheelElevHRV << std::endl;*/
        }
    }
    return true;
} // returns true on success, false if no data has been received.

void FKFSDynamics::setHeight(double)
{
    //sendBuffer.height = v;
}
void FKFSDynamics::setOrientation(double, double)
{
    //sendBuffer.angles[0] = v1;
    //sendBuffer.angles[1] = v2;
}

double FKFSDynamics::getVelocity()
{
    return (appState.velocity);
}

double FKFSDynamics::getEngineSpeed()
{
    return (appState.revolution);
}

/*double FKFSDynamics::getTorque()
{
   return(receiveBuffer.torque);
}*/

osg::Matrix FKFSDynamics::getCarTransform()
{
    int i, j;
    osg::Matrix carTrans;
    carTrans.makeIdentity();
    if (appNumObjects > 0)
    {
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < 3; j++)
            {
                carTrans(i, j) = appObjects[0].orientation[j * 3 + i];
            }
            carTrans(3, i) = appObjects[0].translation[i];
        }
    }
    return (invFKFSDynamicsToOsg * carTrans * FKFSDynamicsToOsg);
}

osg::Matrix FKFSDynamics::getBodyTransform(int bodyNum)
{
    int i, j;
    osg::Matrix bodyTrans;
    bodyTrans.makeIdentity();
    if (appNumObjects > bodyNum)
    {
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < 3; j++)
            {
                bodyTrans(i, j) = appObjects[bodyNum].orientation[j * 3 + i];
            }
            bodyTrans(3, i) = appObjects[bodyNum].translation[i];
        }
    }
    return (invFKFSDynamicsToOsg * bodyTrans * FKFSDynamicsToOsg);
}
osg::Matrix FKFSDynamics::getCameraTransform()
{
    int i, j;
    osg::Matrix bodyTrans;
    bodyTrans.makeIdentity();
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            bodyTrans(i, j) = appState.co.orientation[j * 3 + i];
        }
        bodyTrans(3, i) = appState.co.translation[i];
    }
    return (invFKFSDynamicsToOsg * bodyTrans * FKFSDynamicsToOsg);
}

const osg::Matrix &FKFSDynamics::getVehicleTransformation()
{
    return carTrans;
}

void FKFSDynamics::move(VrmlNodeVehicle *vehicle)
{
    osg::Matrix bodyTrans;
    osg::Matrix cameraTrans;

    for (int i = 0; i < appNumObjects; i++)
    {
        bodyTrans = getBodyTransform(i);
        vehicle->setVRMLVehicleBody(i, bodyTrans);
    }

    cameraTrans = getCameraTransform();
    vehicle->setVRMLVehicleCamera(cameraTrans);

    carTrans = getCarTransform();
    vehicle->setVRMLVehicle(carTrans);
}

void FKFSDynamics::resetState()
{
    sendBuffer.reset = 1;
}
