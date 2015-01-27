/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __HLRSRealtimeDynamics_H
#define __HLRSRealtimeDynamics_H

#include <util/common.h>
#include <fstream>

#include "VehicleDynamics.h"

#include <osg/MatrixTransform>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;

class PLUGINEXPORT HLRSRealtimeDynamics : public VehicleDynamics
{
public:
    enum Mirrors
    {
        RIGHT_MIRROR = 1,
        LEFT_MIRROR = 2,
        MIDDLE_MIRROR = 3
    };
    enum Joystick
    {
        JS_RIGHT = 1,
        JS_LEFT = 2,
        JS_DOWN = 3,
        JS_UP = 4,
        JS_MIDDLE = 5
    };

    HLRSRealtimeDynamics();
    virtual ~HLRSRealtimeDynamics();
    struct RemoteData
    {
        float V;
        float A;
        float rpm;
        float torque;
        osg::Matrix chassisTransform;
        int buttonState;
        int gear;
    };

    virtual double getVelocity()
    {
        return remoteData.V;
    }
    virtual double getAcceleration()
    {
        return remoteData.A;
    }
    virtual double getEngineSpeed()
    {
        return remoteData.rpm;
    }
    virtual int getButtonState()
    {
        return remoteData.buttonState;
    }
    virtual int getLightState()
    {
        return (remoteData.buttonState >> 8) & 0xff;
    }
    virtual int getJoystickState()
    {
        return (remoteData.buttonState >> 16) & 0xff;
    }
    virtual int getGear()
    {
        return remoteData.gear;
    }
    virtual double getEngineTorque()
    {
        return remoteData.torque;
    }
    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }

    virtual const osg::Matrix &getVehicleTransformation()
    {
        return remoteData.chassisTransform;
    }
    virtual void setVehicleTransformation(const osg::Matrix &);

    virtual void move(VrmlNodeVehicle *vehicle);
    virtual void resetState();
    virtual void update();

private:
    SimpleClientConnection *conn;
    Host *serverHost;

    RemoteData recvData;
    RemoteData remoteData;
    std::string remoteHost;

    bool readVal(void *buf, unsigned int numBytes);
    int serverPort;
    int localPort;
    double oldTime;
};

#endif
