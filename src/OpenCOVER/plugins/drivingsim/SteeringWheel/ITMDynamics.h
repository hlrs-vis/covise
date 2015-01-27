/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ITMDynamics_H
#define __ITMDynamics_H

#include <util/common.h>
#include <net/covise_connect.h>

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>
#include "VehicleDynamics.h"

#ifndef WIN32
#include <termios.h>
#include <sys/stat.h> /* open */
#include <fcntl.h> /* open */
#include <termios.h> /* tcsetattr */
#include <termio.h> /* tcsetattr */
#include <limits.h> /* sginap */
#endif

#ifndef __ITM_H
typedef struct
{
    double angle; // -1 - 1
    double gas; // 0-1
    double bremse;
    double gear; // 0 -6
    double clutch;
    double reset;
    double height;
    double angles[2];
} tovd;

typedef struct
{
    double transform[12];
    double axle[2][12];
    double wheel[4][12];
    double velocity;
    double dz;
    double torque;
} fromvd;
#endif

using namespace covise;

class PLUGINEXPORT ITMDynamics : public VehicleDynamics, public OpenThreads::Thread
{
public:
    ITMDynamics();
    virtual ~ITMDynamics();
    virtual void run(); // receiving and sending thread, also does the low level simulation like hard limits

    void move(VrmlNodeVehicle *vehicle);
    void resetState();

    virtual void update();
    fromvd receiveBuffer;
    fromvd appReceiveBuffer;
    tovd sendBuffer;
    SimpleClientConnection *conn;
    OpenThreads::Barrier endBarrier;
    bool doRun;
    bool haveWheels;
    virtual void setSteeringWheelAngle(double v);
    virtual void setGas(double v);
    virtual void setBrake(double v);
    virtual void setGear(double v);
    virtual void setClutch(double v);
    virtual void setHeight(double v);
    virtual void setOrientation(double v1, double v2);
    virtual double getVelocity();
    virtual double getEngineSpeed();
    virtual double getSteeringWheelTorque();
    virtual void setVehicleTransformation(const osg::Matrix &)
    {
    }
    virtual const osg::Matrix &getVehicleTransformation();

private:
    osg::Matrix itmToOsg;
    osg::Matrix invItmToOsg;

    osg::Matrix carTrans;

    Host *serverHost;
    int serverPort;
    double oldTime;
    void sendData();
    bool readData(); // returns true on success, false if no data has been received.

    bool readVal(void *buf, unsigned int numBytes); // blocking read
    bool writeVal(void *buf, unsigned int numBytes); // blocking write
};
#endif
