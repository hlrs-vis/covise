/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LinearMotorControlTask.h"

#include <native/timer.h>
#include <cmath>
#include <cstdlib>

LinearMotorControlTask::LinearMotorControlTask(XenomaiSocketCan &setCan)
    : XenomaiTask("LinearMotorControlTask")
    , can(&setCan)
    , lastAlpha(0.0)
    , lastBeta(0.0)
{
    can->setRecvTimeout(1000);

    pdoOneFrame.can_id = pdoOneId;
    pdoOneFrame.can_dlc = 8;
    pdoTwoFrame.can_id = pdoTwoId;
    pdoTwoFrame.can_dlc = 8;
    pdoThreeFrame.can_id = pdoThreeId;
    pdoThreeFrame.can_dlc = 8;

    stateOneFrame.can_id = statusOneId;
    stateOneFrame.can_dlc = 2;
    *(uint16_t *)(stateOneFrame.data) = 0;
    stateTwoFrame.can_id = statusTwoId;
    stateTwoFrame.can_dlc = 2;
    *(uint16_t *)(stateTwoFrame.data) = 0;
    stateThreeFrame.can_id = statusThreeId;
    stateThreeFrame.can_dlc = 2;
    *(uint16_t *)(stateThreeFrame.data) = 0;

    setPositionOne(posLowerBound);
    setPositionTwo(posLowerBound);
    setPositionThree(posLowerBound);
    setVelocityOne(velLowerBound);
    setVelocityTwo(velLowerBound);
    setVelocityThree(velLowerBound);
    setAccelerationOne(accLowerBound);
    setAccelerationTwo(accLowerBound);
    setAccelerationThree(accLowerBound);

    sendStates = false;
}

const can_frame &LinearMotorControlTask::getAnswer(can_id_t id)
{
    return answerFrameMap[id];
}

void LinearMotorControlTask::setLongitudinalAngle(double alpha)
{
    //if(fabs(lastAlpha-alpha)>1e-5 || getVelocityOne()>0.0001) {
    /*bool currDir;
   if(alpha>lastAlpha)
      currDir = true;
   else
      currDir = false;
   if(dirAlpha!=currDir)
   {
       lastAlpha = alpha;
       filterAlpha = true;
   }
   dirAlpha = currDir;
   int32_t position;
   if(filterAlpha && (fabs(lastAlpha-alpha)>1e-5))
          filterAlpha=false; 
   if(filterAlpha)
      position = posMiddle + (int32_t)(((double)rearMotDist)*tan(lastAlpha));
   else
      position = posMiddle + (int32_t)(((double)rearMotDist)*tan(alpha));*/

    //setPositionThree((position>0)?position:0);
    setPositionThree(posMiddle + (int32_t)(((double)rearMotDist) * tan(alpha)));
}
void LinearMotorControlTask::setLateralAngle(double beta)
{
    /*bool currDir;
   if(beta>lastBeta)
      currDir = true;
   else
      currDir = false;
   if(dirBeta!=currDir)
   {
       lastBeta = beta;
       filterBeta = true;
   }
   dirBeta = currDir;
   int32_t sidePosOff;
   //std::cerr << lastBeta-beta << std::endl;
   if(filterBeta && (fabs(lastBeta-beta)>1e-5))
          filterBeta=false; 
   if(filterBeta)
      sidePosOff = (int32_t)(((double)sideMotDist)*tan(lastBeta));
   else
      sidePosOff = (int32_t)(((double)sideMotDist)*tan(beta));*/

    int32_t sidePosOff = (int32_t)(((double)sideMotDist) * tan(beta));
    int32_t positionOne = posMiddle - sidePosOff;
    int32_t positionTwo = posMiddle + sidePosOff;
    /*setPositionOne((positionOne>0)?positionOne:0);
   setPositionTwo((positionTwo>0)?positionTwo:0);*/
    setPositionOne(positionOne);
    setPositionTwo(positionTwo);
}

void LinearMotorControlTask::setLongitudinalAngularVelocity(double alphaVel)
{
    //int16_t velocity = (int16_t)((double)(rearMotDist/100)*alphaVel*(1+pow(tan(alpha),2)));
    int16_t velocity = (int16_t)((double)(rearMotDist / 100) * alphaVel);
    setVelocityThree(abs(velocity));
}
void LinearMotorControlTask::setLateralAngularVelocity(double betaVel)
{
    //int16_t velocityOne = -(int16_t)((double)(sideMotDist/100)*betaVel*(1+pow(tan(beta),2)));
    //int16_t velocityTwo = (int16_t)((double)(sideMotDist/100)*betaVel*(1+pow(tan(beta),2)));
    int16_t velocityOne = -(int16_t)((double)(sideMotDist / 100) * betaVel);
    int16_t velocityTwo = (int16_t)((double)(sideMotDist / 100) * betaVel);
    setVelocityOne(abs(velocityOne));
    setVelocityTwo(abs(velocityTwo));
}

void LinearMotorControlTask::setLongitudinalAngularAcceleration(double alphaAcc)
{
    int16_t acceleration = (int16_t)((double)(rearMotDist / 1000) * alphaAcc);
    setAccelerationThree(abs(acceleration));
}
void LinearMotorControlTask::setLateralAngularAcceleration(double betaAcc)
{
    int16_t accelerationOne = -(int16_t)((double)(sideMotDist / 1000) * betaAcc);
    int16_t accelerationTwo = (int16_t)((double)(sideMotDist / 1000) * betaAcc);
    setAccelerationOne(abs(accelerationOne));
    setAccelerationTwo(abs(accelerationTwo));
}

int32_t LinearMotorControlTask::getPositionOne()
{
    return (((*((int32_t *)(answerFrameMap[+answerOneId].data))) & 0xffff) << 16) + ((*((int32_t *)(answerFrameMap[+answerOneId].data + 2))) & 0xffff);
    //using unary operator+ to create a copy of static const answerOneId, because vector-operator[] takes a reference.
}
int32_t LinearMotorControlTask::getVelocityOne()
{
    return (((*((int32_t *)(answerFrameMap[+answerOneId].data + 4))) & 0xffff) << 16) + ((*((int32_t *)(answerFrameMap[+answerOneId].data + 6))) & 0xffff);
}

int32_t LinearMotorControlTask::getPositionTwo()
{
    return (((*((int32_t *)(answerFrameMap[+answerTwoId].data))) & 0xffff) << 16) + ((*((int32_t *)(answerFrameMap[+answerTwoId].data + 2))) & 0xffff);
    //using unary operator+ to create a copy of static const answerTwoId, because vector-operator[] takes a reference.
}
int32_t LinearMotorControlTask::getVelocityTwo()
{
    return (((*((int32_t *)(answerFrameMap[+answerTwoId].data + 4))) & 0xffff) << 16) + ((*((int32_t *)(answerFrameMap[+answerTwoId].data + 6))) & 0xffff);
}

int32_t LinearMotorControlTask::getPositionThree()
{
    return (((*((int32_t *)(answerFrameMap[+answerThreeId].data))) & 0xffff) << 16) + ((*((int32_t *)(answerFrameMap[+answerThreeId].data + 2))) & 0xffff);
    //using unary operator+ to create a copy of static const answerThreeId, because vector-operator[] takes a reference.
}
int32_t LinearMotorControlTask::getVelocityThree()
{
    return (((*((int32_t *)(answerFrameMap[+answerThreeId].data + 4))) & 0xffff) << 16) + ((*((int32_t *)(answerFrameMap[+answerThreeId].data + 6))) & 0xffff);
}

double LinearMotorControlTask::getLongitudinalAngle()
{
    return atan((double)(posMiddle - getPositionThree()) / rearMotDist);
}
double LinearMotorControlTask::getLateralAngle()
{
    return atan((double)(getPositionTwo() - getPositionOne()) / (2 * sideMotDist));
}

void LinearMotorControlTask::init()
{
    can_frame startBeckhoffFrame;
    startBeckhoffFrame.can_id = 0x0;
    startBeckhoffFrame.can_dlc = 2;
    startBeckhoffFrame.data[0] = 1;
    startBeckhoffFrame.data[1] = 1;
    can->sendFrame(startBeckhoffFrame);
}

void LinearMotorControlTask::sendRecvPDOs()
{
    can->sendFrame(pdoOneFrame);
    can->sendFrame(pdoTwoFrame);
    can->sendFrame(pdoThreeFrame);

    if (sendStates)
    {
        can->sendFrame(stateOneFrame);
        can->sendFrame(stateTwoFrame);
        can->sendFrame(stateThreeFrame);
        sendStates = false;
    }

    can_frame answerFrame;
    for (int i = 0; i < 3; ++i)
    {
        can->recvFrame(answerFrame);
        answerFrameMap[answerFrame.can_id] = answerFrame;
    }
}

void LinearMotorControlTask::run()
{
    can_frame startBeckhoffFrame;
    startBeckhoffFrame.can_id = 0x0;
    startBeckhoffFrame.can_dlc = 2;
    startBeckhoffFrame.data[0] = 1;
    startBeckhoffFrame.data[1] = 1;
    can->sendFrame(startBeckhoffFrame);

    can_frame answerFrame;
    rt_task_set_periodic(NULL, TM_NOW, rt_timer_ns2ticks(1000000));
    while (1)
    {
        can->sendFrame(pdoOneFrame);
        can->sendFrame(pdoTwoFrame);
        can->sendFrame(pdoThreeFrame);

        if (sendStates)
        {
            can->sendFrame(stateOneFrame);
            can->sendFrame(stateTwoFrame);
            can->sendFrame(stateThreeFrame);
            sendStates = false;
        }

        for (int i = 0; i < 3; ++i)
        {
            can->recvFrame(answerFrame);
            answerFrameMap[answerFrame.can_id] = answerFrame;
        }

        /*setAccelerationOne((uint32_t)((double)getAccelerationSetpointOne()*tanh((double)abs(getPositionSetpointOne()-getPositionOne()))));
      setAccelerationTwo((uint32_t)((double)getAccelerationSetpointTwo()*tanh((double)abs(getPositionSetpointTwo()-getPositionTwo()))));
      setAccelerationThree((uint32_t)((double)getAccelerationSetpointThree()*tanh((double)abs(getPositionSetpointThree()-getPositionThree()))));*/
        /*setAccelerationOne((uint32_t)(accUpperBound*tanh((double)abs(getPositionSetpointOne()-getPositionOne())*0.001)));
      setAccelerationTwo((uint32_t)(accUpperBound*tanh((double)abs(getPositionSetpointTwo()-getPositionTwo())*0.001)));
      setAccelerationThree((uint32_t)(accUpperBound*tanh((double)abs(getPositionSetpointThree()-getPositionThree())*0.001)));*/

        rt_task_wait_period(&overruns);
    }
}
