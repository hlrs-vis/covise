/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Porsche_Controller_h
#define Porsche_Controller_h

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>

#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

struct PorscheControllerInput
{
    double accPedal;
    double brakePedal;
    double clutchPedal;
    double steeringWheelAngle;

    int gearButtonUp;
    int gearButtonDown;

    bool hornButton;
    bool resetButton;

    int mirrorLightLeft;
    int mirrorLightRight;
};

class PorscheController : public OpenThreads::Thread
{
public:
    PorscheController();

    ~PorscheController();

    void update();

    void run();

    double getSteeringWheelAngle();
    double getGas();
    double getBrake();
    double getClutch();

    int getGearButtonUp();
    int getGearButtonDown();

    bool getHorn();
    bool getReset();

    int getMirrorLightLeft();
    int getMirrorLightRight();

    int numFloatsOut;
    int numIntsOut;
    float *floatValuesOut;
    int *intValuesOut;

protected:
    OpenThreads::Barrier endBarrier;
    bool doRun;

    PorscheControllerInput receiveBuffer;
    PorscheControllerInput appReceiveBuffer;

    int fd[MAX_NUMBER_JOYSTICKS];
    Host *serverHost;
    Host *localHost;
    SimpleClientConnection *conn;
    int port;
    int numFloats;
    int numInts;
    int numLocalJoysticks;
    int simulatorJoystick;
    int oldSimulatorJoystick;
    float *floatValues;
    int *intValues;
    float *appFloatValues;
    int *appIntValues;
    double oldTime;

    float updateRate;

    int gearState1;
    int gearState2;

    void connect();
    bool sendValues();
    bool readValues(void *buf, unsigned int numBytes);
};

#endif
