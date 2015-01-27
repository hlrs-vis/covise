/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CAN_H
#define __CAN_H

#ifdef HAVE_PCAN
#include <util/common.h>
#include "FFWheel.h"
// #include "VehicleDynamicsInstitutes.h"
#ifdef WIN32
#include "PcanLight.h"
#else
#include "PcanPci.h"
#endif
#include "CanOpenBus.h"
#include "PorscheSteeringWheel.h"

typedef struct
{
    double Alpha;
    double Speed;
} StateValues;

class PLUGINEXPORT CAN : public FFWheel
{
public:
    CAN();
    virtual ~CAN();
    virtual void run(); // receiving and sending thread, also does the low level simulation like hard limits
    void update();
    double getAngle(); // return steering wheel angle
    double getfastAngle(); // return steering wheel angle
    virtual void setRoadFactor(float); // set roughness 0 == Road 1 == rough
    void cruelResetWheel();
    void softResetWheel();
    void shutdownWheel();

protected:
    bool initWheel();

    StateValues receiveBuffer;
    StateValues appReceiveBuffer;
    float rf;

    double actAngle;
    double driftAngle;
    double tanhCarVel;
    double einsMinusTanhCarVel;
    double tanhCarVelRumble;

    CanInterface *can;
    CanOpenBus *bus;
    PorscheSteeringWheel *wheel;
    int port;
    int speed;
    int nodeID;
    int guardtime;
    int sp;
};
#endif
#endif
