/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#ifdef WIN32
#include <windows.h>
 //#include "lusb0_usb.h"
#include <conio.h>
#else
 //#include <usb.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPlugin.h>
#include <OpenThreads/Thread>



#define MAX_CALLSIGN_LEN        8


class UDPComm;

#ifndef Flux2_H
#define Flux2_H

struct FluxData
{
    float brake;
    float steeringAngle;
    float speed;
    unsigned int button;
};

#pragma pack(push, 1)
struct FluxCtrlData
{
    unsigned char cmd;
    int value;
};
#pragma pack(pop)

class PLUGINEXPORT Flux2 : public opencover::coVRPlugin, public opencover::coVRNavigationProvider, public OpenThreads::Thread
{
public:
    Flux2();
    ~Flux2();
    bool update();
    float getAngle();
    float getBrakeForce();
    float getAccelleration();
    float getSpeed();
    volatile bool running;
    virtual void run();
    bool doStop;

private:
    float stepSizeUp;
    float stepSizeDown;
    const float BrakeThreshold = 1.0;
    bool braking = true;
    bool init();
    float getYAcceleration();
    float resistance;
    osg::Matrix getMatrix();
    osg::Matrix TransformMat;
    osg::Matrix Flux2Pos;
    float speed = 0.0;
    float lastSpeed = 0.0;
    virtual void setEnabled(bool);
    void updateThread();
    UDPComm* udp;
    FluxData fluxData;
    FluxCtrlData fluxControl;
    int ret;
    OpenThreads::Mutex mutex;
    float wheelBase = 0.97;
    void Initialize();
    void setResistance(float f);
    void sendResistance();
};
#endif /* Flux2Plugin_H */
