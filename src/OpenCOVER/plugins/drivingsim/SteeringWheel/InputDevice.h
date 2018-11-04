/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _InputDevice_NODE_PLUGIN_H
#define _InputDevice_NODE_PLUGIN_H

#define VEHICLE_INPUT_NONE -1
#define VEHICLE_INPUT_LOGITECH 0
#define VEHICLE_INPUT_SAITEK 1
#define VEHICLE_INPUT_SITZKISTE 2
#define VEHICLE_INPUT_PORSCHE_SIM 3
#define VEHICLE_INPUT_PORSCHE_REALTIME_SIM 4
#define VEHICLE_INPUT_SITZKISTE_LENKRAD 5
#define VEHICLE_INPUT_THRUSTMASTER 6
#define VEHICLE_INPUT_MOMO 7
#define VEHICLE_INPUT_KEYBOARD 8
#define VEHICLE_INPUT_MOTIONPLATFORM 9
#define VEHICLE_INPUT_HLRS_REALTIME_SIM 10

#define GAS_NUMBER 1
#define BRAKE_NUMBER 0

#define USE_CAR_SOUND
#ifdef USE_CAR_SOUND
#include "CarSound.h"
#else

#include <vrml97/vrml/Player.h>
#endif

#include <string>

#ifdef __XENO__
class KI;
class KLSM;
class Klima;
class VehicleUtil;
class Beckhoff;
class GasPedal;
class IgnitionLock;
//#include <VehicleUtil/GasPedal.h>
//#include <VehicleUtil/KI.h>
//#include <VehicleUtil/KLSM.h>
//#include <VehicleUtil/Klima.h>
//#include <VehicleUtil/Beckhoff.h>
//#include <VehicleUtil/IgnitionLock.h>
#ifdef debug
#undef debug
#endif
#endif

class InputDevice
{
public:
#ifndef USE_CAR_SOUND
    static Player *player;
#endif
    static InputDevice *instance();
    static void destroy();
    virtual void updateInputState() = 0;

    virtual ~InputDevice(){};

    virtual float getAccelerationPedal();
    virtual float getBrakePedal();
    virtual float getClutchPedal();
    virtual float getSteeringWheelAngle();
    virtual int getGear();
    virtual bool getHornButton();
    virtual bool getResetButton();
    virtual int getMirrorLightLeft();
    virtual int getMirrorLightRight();

    int getType();
    std::string getName();

    bool getIsAutomatic();
    bool getIsPowerShifting();

protected:
    InputDevice();

    int getAutoGearDiff(float downRPM = 50, float upRPM = 100);

    int type;
    std::string name;

    static InputDevice *autodetect();
    static InputDevice *findInputDevice(std::string name);

    static int autoDetectRetries;
    static InputDevice *inDevice;

    float pedalA; // Acceleration	[0,1]
    float pedalB; // Brake				[0,1]
    float pedalC; // Clutch			[0,1]

    float steeringWheelAngle; // Wheel angle in Radians
    bool resetButton; // Reset button	[true, false]
    int gear; // Present gear	[-1, 0, 1, ...]
    bool hornButton; // Horn button		[true, false]

    float angleRatio;

    bool automatic; //automatic gearbox
    bool powerShift;
    int shiftdirection;

    int mirrorLightLeft;
    int mirrorLightRight;

    double velocitySetpoint;
    bool oldParkState;
#ifndef USE_CAR_SOUND
    Player::Source *anlasserSource;
#endif
};

class InputDeviceSaitek : public InputDevice
{
public:
    InputDeviceSaitek();

    virtual ~InputDeviceSaitek(){};

    void updateInputState();

private:
    int oldButton1;
    int oldButton2;
};

class InputDeviceThrustmaster : public InputDevice
{
public:
    InputDeviceThrustmaster();

    virtual ~InputDeviceThrustmaster(){};

    void updateInputState();

private:
    int oldButton1;
    int oldButton2;
};

class InputDeviceMomo : public InputDevice
{
public:
    InputDeviceMomo();

    virtual ~InputDeviceMomo(){};

    void updateInputState();

private:
    int oldButton1;
    int oldButton2;
};

class InputDeviceLogitech : public InputDevice
{
public:
    InputDeviceLogitech();

    virtual ~InputDeviceLogitech(){};

    void updateInputState();

private:
    int oldButton1;
    int oldButton2;
};

class InputDeviceSitzkiste : public InputDevice
{
public:
    InputDeviceSitzkiste();

    virtual ~InputDeviceSitzkiste(){};

    void updateInputState();

private:
};

class InputDevicePorscheSim : public InputDevice
{
public:
    InputDevicePorscheSim();

    virtual ~InputDevicePorscheSim(){};

    void updateInputState();

private:
    int oldButton1;
    int oldButton2;
};

class InputDevicePorscheRealtimeSim : public InputDevice
{
public:
    InputDevicePorscheRealtimeSim();

    virtual ~InputDevicePorscheRealtimeSim(){};

    void updateInputState();

private:
    int oldButton1;
    int oldButton2;
};

class InputDeviceHLRSRealtimeSim : public InputDevice
{
public:
    InputDeviceHLRSRealtimeSim();

    virtual ~InputDeviceHLRSRealtimeSim(){};

    void updateInputState();

private:
    int oldButton1;
    int oldButton2;
};
class InputDeviceKeyboard : public InputDevice
{
public:
    InputDeviceKeyboard();

    virtual ~InputDeviceKeyboard(){};

    void updateInputState();

private:
};

class InputDeviceMotionPlatform : public InputDevice
{
public:
    InputDeviceMotionPlatform();

    virtual ~InputDeviceMotionPlatform();

    void updateInputState();

protected:
    struct
    {
        float steeringWheelAngle;
        float pedalA;
        float pedalB;
        float pedalC;
        int gear;
        bool SportMode;
        bool PSMState;
        bool SpoilerState;
        bool DamperState;
    } sharedState;
    //XenomaiSocketCan* can0;
    //LinearMotorControlTask* linMot;

    //CanOpenController* con1;
    //XenomaiSteeringWheel* steeringWheel;
    bool ccOn;
    bool ccActive;
    float ccSpeed;
    float iDiff;

#ifdef __XENO__
    //XenomaiSocketCan* can3;

    //CanOpenController* con3;

    //BrakePedal* p_brakepedal;
    KI *p_kombi;
    KLSM *p_klsm;
    Klima *p_klima;
    VehicleUtil *vehicleUtil;
    Beckhoff *p_beckhoff;
    GasPedal *p_gaspedal;
    IgnitionLock *p_ignitionLock;
#endif
    bool oldFanButtonState;
};

#endif
