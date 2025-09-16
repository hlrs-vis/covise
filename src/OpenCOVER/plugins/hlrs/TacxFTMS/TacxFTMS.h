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
#include <cover/input/deviceDiscovery.h>


#define MAX_CALLSIGN_LEN        8


class UDPComm;

#ifndef TacxFTMS_H
#define TacxFTMS_H

#pragma pack(push, 1)
struct FTMSBikeData
{
    float speed;
    float cadence;
    float power;
};

struct FTMSControlData
{
    float wind_speed;
    float grade;
    float crr;
    float cw;
    float weight;
};

struct AlpineData
{
    float steering_angle;
};

#pragma pack(pop)

class PLUGINEXPORT TacxFTMS : public opencover::coVRPlugin, public opencover::coVRNavigationProvider, public OpenThreads::Thread
{
public:
    TacxFTMS();
    ~TacxFTMS();
    bool update();
    float getAngle();
    float getBrakeForce();
    float getAccelleration();
    float getSpeed();
    volatile bool running;
    virtual void run();
    bool doStop;

private:
    bool ftmsfound = false;
    bool alpinefound = false;
    float stepSizeUp;
    float stepSizeDown;
    const float BrakeThreshold = 1.0;
    bool braking = true;
    bool init();
    void addDevice(const opencover::deviceInfo *dev);
    float getYAcceleration();
    float getGrade();
    float resistance;
    osg::Matrix getMatrix();
    osg::Matrix TransformMat;
    osg::Matrix TacxFTMSPos;
    virtual void setEnabled(bool);
    void updateThread();
    UDPComm* udpNeo; 
    UDPComm* udpAlpine;
    UDPComm* udpListen; // for listening to all devices
    FTMSBikeData ftmsData;
    FTMSControlData ftmsControl;
    AlpineData alpineData;
    int ret;
    OpenThreads::Mutex mutex;
    float wheelBase = 0.97;
    void Initialize();
    void sendIndoorBikeSimulationParameters();
};
#endif /* TacxFTMSPlugin_H */
