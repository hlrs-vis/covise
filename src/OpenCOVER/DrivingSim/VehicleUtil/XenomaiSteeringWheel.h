/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __XenomaiSteeringWheel_h
#define __XenomaiSteeringWheel_h

#include "CanOpenDevice.h"
#include "CanOpenController.h"
#include "XenomaiTask.h"
#include <deque>

class XenomaiSteeringWheelHomingTask : public CanOpenDevice, public XenomaiTask
{
public:
    XenomaiSteeringWheelHomingTask(CanOpenController &, uint8_t);

    bool isDone()
    {
        return done;
    }

    bool isSuccess()
    {
        return success;
    }

protected:
    void run();

    bool done;
    bool success;

    static const double speedRes = 1875.0 / 262144.0; // = 1 bit
    static const double homingSpeed = 10; //revs/min
    static const int32_t limitSwitchPosition = -369000; //was -380432;
};

class XenomaiSteeringWheel : public CanOpenDevice, public XenomaiTask
{
public:
    XenomaiSteeringWheel(CanOpenController &, uint8_t);
    ~XenomaiSteeringWheel();

    void init();
    void shutdown();

    bool center();

    unsigned long getPeriodicTaskOverruns();

    int32_t getPosition();
    int32_t getSpeed();
    int32_t getSmoothedSpeed();
    void getPositionSpeed(int32_t &, int32_t &);
    void setCurrent(int32_t);

    void setSpringConstant(double);
    void setDampingConstant(double);
    void setRumbleAmplitude(double);
    void setDrillConstant(double);
    void setDrillElasticity(double);
    double getSpringConstant();
    double getDampingConstant();
    double getRumbleAmplitude();
    double getDrillConstant();
    double getDrillElasticity();

    static const int32_t countsPerTurn = 1048576;

    //static const int32_t peakCurrent = 3280;
    static const int32_t peakCurrent = 1640;
    //static const int32_t peakCurrent = 40;

protected:
    void run();
    bool runTask;
    bool taskFinished;
    unsigned long overruns;

    int32_t position;
    int32_t driftPosition;

    std::deque<int32_t> speedDeque;

    double Kwheel;
    double Dwheel;

    double rumbleAmplitude;
    double Kdrill;
    double drillElasticity;

    uint8_t RPDOData[6]; //enable op
};

inline unsigned long XenomaiSteeringWheel::getPeriodicTaskOverruns()
{
    return overruns;
}

inline int32_t XenomaiSteeringWheel::getPosition()
{
    int32_t position;
    uint8_t *TPDOData = readTPDO(1);
    memcpy(&position, TPDOData, 4);

    return position;
}

inline int32_t XenomaiSteeringWheel::getSpeed()
{
    int32_t speed;
    uint8_t *TPDOData = readTPDO(1);
    //memcpy(&speed, TPDOData+4, 3);
    memcpy(&speed, TPDOData + 3, 4);

    return speed;
}

inline int32_t XenomaiSteeringWheel::getSmoothedSpeed()
{
    int32_t speed;
    uint8_t *TPDOData = readTPDO(1);
    memcpy(&speed, TPDOData + 3, 4);

    int32_t sumSpeed = 0;

    speedDeque.pop_front();
    speedDeque.push_back(speed);

    for (unsigned int i = 0; i < speedDeque.size(); ++i)
    {
        sumSpeed += speedDeque[i];
    }

    speed = (int32_t)((double)sumSpeed / (double)speedDeque.size());

    return speed;
}

inline void XenomaiSteeringWheel::getPositionSpeed(int32_t &position, int32_t &speed)
{
    uint8_t *TPDOData = readTPDO(1);
    memcpy(&position, TPDOData, 4);
    memcpy(&speed, TPDOData + 4, 3);
}

inline void XenomaiSteeringWheel::setCurrent(int32_t current)
{
    if (current > peakCurrent)
    {
        current = peakCurrent;
    }
    else if (current < -peakCurrent)
    {
        current = -peakCurrent;
    }
    *((int32_t *)(RPDOData + 2)) = current;
    writeRPDO(1, RPDOData, 6);
}

inline void XenomaiSteeringWheel::setSpringConstant(double setConst)
{
    Kwheel = setConst;
}

inline void XenomaiSteeringWheel::setDampingConstant(double setConst)
{
    Dwheel = setConst;
}

inline void XenomaiSteeringWheel::setRumbleAmplitude(double setAmp)
{
    rumbleAmplitude = setAmp;
}

inline void XenomaiSteeringWheel::setDrillConstant(double setConst)
{
    Kdrill = setConst;
}

inline void XenomaiSteeringWheel::setDrillElasticity(double setElast)
{
    drillElasticity = setElast;
}

inline double XenomaiSteeringWheel::getSpringConstant()
{
    return Kwheel;
}

inline double XenomaiSteeringWheel::getDampingConstant()
{
    return Dwheel;
}

inline double XenomaiSteeringWheel::getRumbleAmplitude()
{
    return rumbleAmplitude;
}

inline double XenomaiSteeringWheel::getDrillConstant()
{
    return Kdrill;
}

inline double XenomaiSteeringWheel::getDrillElasticity()
{
    return drillElasticity;
}

#endif
