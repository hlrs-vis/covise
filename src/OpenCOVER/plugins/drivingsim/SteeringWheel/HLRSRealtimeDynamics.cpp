/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HLRSRealtimeDynamics.h"


#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>
#include "SteeringWheel.h"

#include <cover/coVRTui.h>

#include <string>
#include <math.h>

#include <fstream>
#include <ctime>
#include <iostream>
#include <string>

// CONSTRUCTOR //
//
HLRSRealtimeDynamics::HLRSRealtimeDynamics()
{
    memset(&recvData, 0, sizeof(recvData));
    memset(&remoteData, 0, sizeof(remoteData));
    remoteData.chassisTransform.makeIdentity();
    recvData.chassisTransform.makeIdentity();
    // config XML //
    serverPort = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.Dynamics", 31880);
    localPort = coCoviseConfig::getInt("localPort", "COVER.Plugin.SteeringWheel.Dynamics", 31880);
    remoteHost = coCoviseConfig::getEntry("host", "COVER.Plugin.SteeringWheel.Dynamics");

    serverHost = new Host(remoteHost.c_str());
    conn = NULL;
    fprintf(stderr, "HLRSRealtimeDynamics::HLRSRealtimeDynamics() %s %d\n", remoteHost.c_str(), serverPort);

    remoteData.chassisTransform.makeIdentity();
}

HLRSRealtimeDynamics::~HLRSRealtimeDynamics()
{
    if (coVRMSController::instance()->isMaster())
    {
        delete conn;
        delete serverHost;
    }
}

void
HLRSRealtimeDynamics::update()
{

    oldTime = 0;
    if (coVRMSController::instance()->isMaster())
    {

        // try to connect to server every 2 secnods
        if (conn == NULL && (cover->frameTime() - oldTime) > 2)
        {
            conn = new SimpleClientConnection(serverHost, serverPort, 0);

            if (!conn->is_connected()) // could not open server port
            {
#ifndef _WIN32
                if (errno != ECONNREFUSED)
                {
                    fprintf(stderr, "Could not connect to fasi on %s; port %d\n", serverHost->getName(), serverPort);
                }
#endif
                fprintf(stderr, "Could not connect to fasi on %s; port %d\n", serverHost->getName(), serverPort);
                delete conn;
                conn = NULL;
            }
            else
            {
                fprintf(stderr, "Connected to fasi on %s; port %d\n", serverHost->getName(), serverPort);
            }
            if (conn && conn->is_connected())
            {
            }
            oldTime = cover->frameTime();
        }
    }
    if (conn && conn->is_connected())
    {
        double currentTime = cover->frameTime();
        int written = conn->getSocket()->write(&currentTime, sizeof(currentTime));
        if (written != sizeof(currentTime))
        {
            delete conn;
            conn = NULL;
            fprintf(stderr, "Connection to fasi closed\n");
        }
        if (!readVal(&recvData, sizeof(recvData)))
        {
            delete conn;
            conn = NULL;
            fprintf(stderr, "Connection to fasi closed\n");
        }
        //fprintf(stderr,"V=%f\n",remoteData.V);
        //fprintf(stderr,"V=%f %f %f\n",remoteData.chassisTransform.getTrans().x(),remoteData.chassisTransform.getTrans().y(),remoteData.chassisTransform.getTrans().z());
    }

    // read data from remote if we have a connection
    if (coVRMSController::instance()->isMaster())
    {
        memcpy(&remoteData, &recvData, sizeof(recvData));
        coVRMSController::instance()->sendSlaves((char *)&remoteData, sizeof(remoteData));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&remoteData, sizeof(remoteData));
    }
}
bool HLRSRealtimeDynamics::readVal(void *buf, unsigned int numBytes)
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
            cout << "error reading data from socket" << endl;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}

void
HLRSRealtimeDynamics::move(VrmlNodeVehicle *vehicle)
{

    vehicle->setVRMLVehicle(remoteData.chassisTransform);

    //vehicle->setVRMLVehicleBody(remoteData.chassisTransform);
    //vehicle->setVRMLVehicleFrontWheels(inertialToWheelTransform[0], inertialToWheelTransform[1]);
    //vehicle->setVRMLVehicleRearWheels(inertialToWheelTransform[2], inertialToWheelTransform[3]);
}

void
HLRSRealtimeDynamics::setVehicleTransformation(const osg::Matrix &m)
{
    //oldHeight = m.getTrans()[2];
    if (conn && conn->is_connected())
    {
        double currentTime = 0.0;
        int written = conn->getSocket()->write(&currentTime, sizeof(currentTime));
        if (written != sizeof(currentTime))
        {
            delete conn;
            conn = NULL;
            fprintf(stderr, "Connection to fasi closed\n");
        }
        written = conn->getSocket()->write(m.ptr(), sizeof(4*4*sizeof(double)));
        if (written != sizeof(4*4*sizeof(double)))
        {
            delete conn;
            conn = NULL;
            fprintf(stderr, "Connection to fasi closed\n");
        }
        osg::Vec3 p = m.getTrans();
        fprintf(stderr, "sentMatrix %f %f %f\n",p[0],p[1],p[2]);
    }
}

void
HLRSRealtimeDynamics::resetState()
{
    //outputData[0]=1;
    //oldHeight=0;
    fprintf(stderr, "V=%f %f %f\n", remoteData.chassisTransform.getTrans().x(), remoteData.chassisTransform.getTrans().y(), remoteData.chassisTransform.getTrans().z());

    remoteData.chassisTransform.makeIdentity();
}
