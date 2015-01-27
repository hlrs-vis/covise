/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __LinearMotorControlTask_h
#define __LinearMotorControlTask_h

//
//Forked control task frequently sending and receiving message from gas pedal: Continental TEMIC HLZ FFP
//

#include <iostream>
#include <map>

#include "XenomaiTask.h"
#include "XenomaiSocketCan.h"

class LinearMotorControlTask : public XenomaiTask
{
public:
    LinearMotorControlTask(XenomaiSocketCan &);
    virtual ~LinearMotorControlTask()
    {
    }

    void setPositionOne(uint32_t);
    void setPositionTwo(uint32_t);
    void setPositionThree(uint32_t);
    void setVelocityOne(uint16_t);
    void setVelocityTwo(uint16_t);
    void setVelocityThree(uint16_t);
    void setAccelerationOne(uint16_t);
    void setAccelerationTwo(uint16_t);
    void setAccelerationThree(uint16_t);
    uint32_t getPositionSetpointOne();
    uint32_t getPositionSetpointTwo();
    uint32_t getPositionSetpointThree();
    uint16_t getAccelerationSetpointOne();
    uint16_t getAccelerationSetpointTwo();
    uint16_t getAccelerationSetpointThree();

    void changeStateOne(uint16_t data);
    void changeStateTwo(uint16_t data);
    void changeStateThree(uint16_t data);

    const can_frame &getAnswer(can_id_t);

    void setLongitudinalAngle(double);
    void setLateralAngle(double);
    void setLongitudinalAngularVelocity(double);
    void setLateralAngularVelocity(double);
    void setLongitudinalAngularAcceleration(double);
    void setLateralAngularAcceleration(double);

    int32_t getPositionOne();
    int32_t getVelocityOne();
    int32_t getPositionTwo();
    int32_t getVelocityTwo();
    int32_t getPositionThree();
    int32_t getVelocityThree();

    double getLongitudinalAngle();
    double getLateralAngle();

    unsigned long getPeriodicTaskOverruns();
    void init();
    void sendRecvPDOs();

    //position bounds
    static const uint32_t posLowerBound = 0;
    //static const int posUpperBound = 400000;
    static const uint32_t posUpperBound = 380000;

    //velocity bounds
    //static const uint16_t velLowerBound = 100;
    static const int velLowerBound = 0;
    static const uint16_t velUpperBound = 5000;

    //acceleration bounds
    static const uint16_t accLowerBound = 100;
    //static const int accUpperBound = 25000;
    static const uint16_t accUpperBound = 10000;

    static const int posMiddle = (posUpperBound - posLowerBound) / 2;
    static const int sideMotDist = 650000;
    static const int rearMotDist = 1100000;

    static const can_id_t answerOneId = 0x19;
    static const can_id_t answerTwoId = 0x1a;
    static const can_id_t answerThreeId = 0x1b;

    static const uint16_t statePositioning = (9 << 5) | (4 << 10); //FCB 9, Instance 4
    static const uint16_t stateToGround = (9 << 5) | (3 << 10); //FCB 9, Instance 3
    static const uint16_t stateReset = (13 << 5) | (0 << 10) | 0x0001; //FCB 13, Instance 0, Reset-Bit
    static const uint16_t stateDisable = (1 << 5) | (0 << 10); //FCB 1, Instance 0
    static const uint16_t stateReferenceSet = (12 << 5) | (0 << 10); //FCB 12, Instance 0
    static const uint16_t stateEncoder = (18 << 5) | (0 << 10); //FCB 18, Instance 0, don't use!

protected:
    void run();

    XenomaiSocketCan *can;

    static const can_id_t pdoOneId = 0x11;
    static const can_id_t statusOneId = 0x181;

    static const can_id_t pdoTwoId = 0x12;
    static const can_id_t statusTwoId = 0x281;

    static const can_id_t pdoThreeId = 0x13;
    static const can_id_t statusThreeId = 0x381;

    can_frame pdoOneFrame;
    can_frame stateOneFrame;
    can_frame pdoTwoFrame;
    can_frame stateTwoFrame;
    can_frame pdoThreeFrame;
    can_frame stateThreeFrame;
    bool sendStates;

    std::map<can_id_t, can_frame> answerFrameMap;

    unsigned long overruns;

    double lastAlpha;
    bool dirAlpha;
    bool filterAlpha;
    double lastBeta;
    bool dirBeta;
    bool filterBeta;
};

inline void LinearMotorControlTask::setPositionOne(uint32_t data)
{
    if (data < posLowerBound)
    {
        data = posLowerBound;
    }
    else if (data > posUpperBound)
    {
        data = posUpperBound;
    }
    memcpy(pdoOneFrame.data, &data, 4);
}

inline void LinearMotorControlTask::setPositionTwo(uint32_t data)
{
    if (data < posLowerBound)
    {
        data = posLowerBound;
    }
    else if (data > posUpperBound)
    {
        data = posUpperBound;
    }
    memcpy(pdoTwoFrame.data, &data, 4);
}

inline void LinearMotorControlTask::setPositionThree(uint32_t data)
{
    if (data < posLowerBound)
    {
        data = posLowerBound;
    }
    else if (data > posUpperBound)
    {
        data = posUpperBound;
    }
    memcpy(pdoThreeFrame.data, &data, 4);
}

inline void LinearMotorControlTask::setVelocityOne(uint16_t data)
{
    if ((data < velLowerBound))
    {
        data = velLowerBound;
    }
    else if (data > velUpperBound)
    {
        data = velUpperBound;
    }
    memcpy(pdoOneFrame.data + 4, &data, 2);
}

inline void LinearMotorControlTask::setVelocityTwo(uint16_t data)
{
    if ((data < velLowerBound))
    {
        data = velLowerBound;
    }
    else if (data > velUpperBound)
    {
        data = velUpperBound;
    }
    memcpy(pdoTwoFrame.data + 4, &data, 2);
}

inline void LinearMotorControlTask::setVelocityThree(uint16_t data)
{
    if ((data < velLowerBound))
    {
        data = velLowerBound;
    }
    else if (data > velUpperBound)
    {
        data = velUpperBound;
    }
    memcpy(pdoThreeFrame.data + 4, &data, 2);
}

inline void LinearMotorControlTask::setAccelerationOne(uint16_t data)
{
    if ((data < accLowerBound))
    {
        data = accLowerBound;
    }
    else if (data > accUpperBound)
    {
        data = accUpperBound;
    }
    memcpy(pdoOneFrame.data + 6, &data, 2);
}

inline void LinearMotorControlTask::setAccelerationTwo(uint16_t data)
{
    if ((data < accLowerBound))
    {
        data = accLowerBound;
    }
    else if (data > accUpperBound)
    {
        data = accUpperBound;
    }
    memcpy(pdoTwoFrame.data + 6, &data, 2);
}

inline void LinearMotorControlTask::setAccelerationThree(uint16_t data)
{
    if ((data < accLowerBound))
    {
        data = accLowerBound;
    }
    else if (data > accUpperBound)
    {
        data = accUpperBound;
    }
    memcpy(pdoThreeFrame.data + 6, &data, 2);
}

inline uint32_t LinearMotorControlTask::getPositionSetpointOne()
{
    return *((uint32_t *)(pdoOneFrame.data));
}

inline uint32_t LinearMotorControlTask::getPositionSetpointTwo()
{
    return *((uint32_t *)(pdoTwoFrame.data));
}

inline uint32_t LinearMotorControlTask::getPositionSetpointThree()
{
    return *((uint32_t *)(pdoThreeFrame.data));
}

inline uint16_t LinearMotorControlTask::getAccelerationSetpointOne()
{
    return *((uint16_t *)(pdoOneFrame.data + 6));
}

inline uint16_t LinearMotorControlTask::getAccelerationSetpointTwo()
{
    return *((uint16_t *)(pdoTwoFrame.data + 6));
}

inline uint16_t LinearMotorControlTask::getAccelerationSetpointThree()
{
    return *((uint16_t *)(pdoThreeFrame.data + 6));
}

inline void LinearMotorControlTask::changeStateOne(uint16_t data)
{
    *(uint16_t *)(stateOneFrame.data) = data;
    sendStates = true;
}

inline void LinearMotorControlTask::changeStateTwo(uint16_t data)
{
    *(uint16_t *)stateTwoFrame.data = data;
    sendStates = true;
}

inline void LinearMotorControlTask::changeStateThree(uint16_t data)
{
    *(uint16_t *)stateThreeFrame.data = data;
    sendStates = true;
}

inline unsigned long LinearMotorControlTask::getPeriodicTaskOverruns()
{
    return overruns;
}

#endif
