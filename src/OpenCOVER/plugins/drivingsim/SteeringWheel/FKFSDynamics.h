/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FKFSDynamics_H
#define __FKFSDynamics_H

#include <util/common.h>
#include <net/covise_connect.h>

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>
#include "VehicleDynamics.h"
#include "UDPComm.h"

#ifndef WIN32
#include <termios.h>
#include <sys/stat.h> /* open */
#include <fcntl.h> /* open */
#include <termios.h> /* tcsetattr */
#include <termio.h> /* tcsetattr */
#include <limits.h> /* sginap */
#endif
#define MAX_MOTION_OBJECTS 100

using namespace covise;

typedef struct
{
    int reset;
    float wheelAngle; // rad left pos.
    float gasPedal; // 0 - 1 0 = - not pressed
    float breakPedal; // 0 - 1 0 = - not pressed
    float clutchPedal; // 0 - 1 0 = - not pressed
    int gear; // 0 = kein Gang, 1 = 1. Gang, ...

    float wheelElevVL; //Rad Aufstand ueber Boden
    float wheelElevVR; //Rad Aufstand ueber Boden
    float wheelElevHL; //Rad Aufstand ueber Boden
    float wheelElevHR; //Rad Aufstand ueber Boden
    float wheelElevVLV;
    float wheelElevVLL;
    float wheelElevVRV;
    float wheelElevVRR;
    float wheelElevHLV;
    float wheelElevHLL;
    float wheelElevHRV;
    float wheelElevHRR;
} toFKFSvd;

typedef struct
{
    float translation[3];
    float orientation[9];
    float reserved1;
    float reserved2;
    float reserved3;
} MotionObjectType;

typedef struct
{
    float translation[3];
    float orientation[9];
    float reserved1;
    float reserved2;
    float reserved3;
} CameraObjectType;

typedef struct
{
    int ID;
    float val1;
    float val2;
} ControlObjectType;

typedef struct
{
    float transWheelVL[3];
    float transWheelVR[3];
    float transWheelHL[3];
    float transWheelHR[3];
} WheelPositionObjectType;

typedef struct
{
    float velocity;
    float revolution;
    CameraObjectType co;
} appStateType;

typedef struct
{
    int index;
    int type;
} RecordHeader;

class PLUGINEXPORT FKFSDynamics : public VehicleDynamics, public OpenThreads::Thread

{
public:
    FKFSDynamics();
    virtual ~FKFSDynamics();
    virtual void run(); // receiving and sending thread, also does the low level simulation like hard limits

    void move(VrmlNodeVehicle *vehicle);
    void resetState();

    virtual void update();
    appStateType appState;
    appStateType receiveState;
    WheelPositionObjectType wheelPosObj;
    SimpleClientConnection *conn;
    OpenThreads::Barrier endBarrier;
    bool haveWheels;
    virtual void setSteeringWheelAngle(double a)
    {
        sendBuffer.wheelAngle = a;
    };
    virtual void setGas(double v)
    {
        sendBuffer.gasPedal = v;
    };
    virtual void setBrake(double v)
    {
        sendBuffer.breakPedal = v;
    };
    virtual void setGear(double v)
    {
        sendBuffer.gear = (int)v;
    };
    virtual void setClutch(double v)
    {
        sendBuffer.clutchPedal = v;
    };
    virtual void setHeight(double v);
    virtual void setOrientation(double v1, double v2);
    virtual double getVelocity();
    virtual double getEngineSpeed();
    virtual void setVehicleTransformation(const osg::Matrix &)
    {
    }
    virtual const osg::Matrix &getVehicleTransformation();

    bool doRun;

private:
    osg::Matrix carTrans;
    virtual osg::Matrix getCarTransform();
    virtual osg::Matrix getBodyTransform(int bodyNum);
    int appNumObjects;
    virtual osg::Matrix getCameraTransform();
    osg::Matrix FKFSDynamicsToOsg;
    osg::Matrix invFKFSDynamicsToOsg;
    enum
    {
        MotionObject = 1,
        CameraObject = 2,
        ControlObject = 3,
        WheelPositionObject = 4
    };
    enum
    {
        VelocityID = 1,
        RevolutionID = 2
    };
    int receiveNumObjects;
    int serverPort;
    int localPort;
    MotionObjectType appObjects[MAX_MOTION_OBJECTS];
    MotionObjectType receiveObjects[MAX_MOTION_OBJECTS];
    UDPComm *toFKFS;
    toFKFSvd sendBuffer;
    double oldTime;
    void sendData();
    bool readData(); // returns true on success, false if no data has been received.

    bool readVal(void *buf, unsigned int numBytes); // blocking read
    bool writeVal(void *buf, unsigned int numBytes); // blocking write
};
#endif
