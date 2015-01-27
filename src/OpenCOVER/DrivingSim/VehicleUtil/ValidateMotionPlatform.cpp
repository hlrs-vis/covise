/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ValidateMotionPlatform.h"

ValidateMotionPlatform *ValidateMotionPlatform::instancePointer = NULL;

ValidateMotionPlatform::ValidateMotionPlatform(const std::string &channelId, int stacksize, int prio, int mode, const can_id_t &syncid,
                                               const can_id_t &pfid1, const can_id_t &cfid1, const can_id_t &sfid1,
                                               const can_id_t &pfid2, const can_id_t &cfid2, const can_id_t &sfid2,
                                               const can_id_t &pfid3, const can_id_t &cfid3, const can_id_t &sfid3,
                                               const can_id_t &pfidRot1, const can_id_t &cfidRot1, const can_id_t &sfidRot1)
    : XenomaiTask("MotionPlatformControl", stacksize, prio, mode)
    , XenomaiSocketCan(channelId)
    , processFrameVector(numLinMots + numRotMots)
    , controlFrameVector(numLinMots + numRotMots)
    , stateFrameVector(numLinMots + numRotMots)
    , posSetVector(numLinMots)
    , trajectoryDequeVector(numLinMots)
    , setpointDequeVector(numLinMots)
    , sendMutex("vmp_send_mutex")
{

    initialized = false; // true after homing of gas pedal
    //send frames
    processFrameVector[0].can_id = pfid1;
    processFrameVector[1].can_id = pfid2;
    processFrameVector[2].can_id = pfid3;
    processFrameVector[3].can_id = pfidRot1;
    processFrameVector[0].can_dlc = 8;
    processFrameVector[1].can_dlc = 8;
    processFrameVector[2].can_dlc = 8;
    processFrameVector[3].can_dlc = 4;

    controlFrameVector[0].can_id = cfid1;
    controlFrameVector[1].can_id = cfid2;
    controlFrameVector[2].can_id = cfid3;
    controlFrameVector[3].can_id = cfidRot1;
    controlFrameVector[0].can_dlc = 2;
    controlFrameVector[1].can_dlc = 2;
    controlFrameVector[2].can_dlc = 2;
    controlFrameVector[3].can_dlc = 2;

    for (unsigned int motIt = 0; motIt < numLinMots; ++motIt)
    {
        setPositionSetpoint(motIt, posMin);
        setVelocitySetpoint(motIt, velMin);
        setAccelerationSetpoint(motIt, accMin);
    }

    //receive frames
    stateIdIndexMap[sfid1] = 0;
    stateIdIndexMap[sfid2] = 1;
    stateIdIndexMap[sfid3] = 2;
    stateIdIndexMap[sfidRot1] = 3;
    //receive filters
    addRecvFilter(sfid1);
    addRecvFilter(sfid2);
    addRecvFilter(sfid3);
    addRecvFilter(sfidRot1);

    //sync frame
    syncFrame.can_dlc = 0;
    syncFrame.can_id = syncid;

    sendControlFrame = true;

    //Position Setpoints
    std::fill(posSetVector.begin(), posSetVector.end(), 0.0);
    brakeForce = 0.0;

    //Trajectory
    trajectoryDequeVector[0].resize(2);
    std::fill(trajectoryDequeVector[0].begin(), trajectoryDequeVector[0].end(), 0.0);
    trajectoryDequeVector[1].resize(2);
    std::fill(trajectoryDequeVector[1].begin(), trajectoryDequeVector[1].end(), 0.0);
    trajectoryDequeVector[2].resize(2);
    std::fill(trajectoryDequeVector[2].begin(), trajectoryDequeVector[2].end(), 0.0);

    //Trajectory error
    setpointDequeVector[0].resize(2);
    std::fill(setpointDequeVector[0].begin(), setpointDequeVector[0].end(), 0.0);
    setpointDequeVector[1].resize(2);
    std::fill(setpointDequeVector[1].begin(), setpointDequeVector[1].end(), 0.0);
    setpointDequeVector[2].resize(2);
    std::fill(setpointDequeVector[2].begin(), setpointDequeVector[2].end(), 0.0);

    runTask = false;
    taskFinished = false;
    overruns = 0;

    updateMisses = 0;
}

ValidateMotionPlatform::~ValidateMotionPlatform()
{
    std::cerr << "ValidateMotionPlatform::~ValidateMotionPlatform()" << std::endl;

    if (runTask)
    {
        runTask = false;
        while (!taskFinished)
        {
            usleep(100000);
        }
    }
}

void ValidateMotionPlatform::applyPositionSetpoint(const unsigned int &mot, double sSet)
{
    //Determine dynamic position limits depending on actual velocity and max acceleration
    double sMaxPos = posMax - 2.0 * 0.5 * (trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1]) * fabs(trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1]) / (accMax * h * h) - 0.001;
    double sMinPos = posMin - 2.0 * 0.5 * (trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1]) * fabs(trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1]) / (accMax * h * h) + 0.001;

    //Apply dynamic position limits
    if (sSet < sMinPos)
    {
        sSet = sMinPos;
    }
    else if (sSet > sMaxPos)
    {
        sSet = sMaxPos;
    }

    //Determine current acceleration limits
    double aMinCur = -1.0 * accMax;
    double aMaxCur = 1.0 * accMax;

    //Determine current acceleration limits
    double vMinCur = -1.0 * velMax;
    double vMaxCur = 1.0 * velMax;

    //Apply velocity/acceleration limits to new position difference
    //Velocity limits to pos diffs:
    double sDiffMinVel = vMinCur * h;
    double sDiffMaxVel = vMaxCur * h;
    //Acceleration limits to pos diff:
    double sDiffMinAcc = aMinCur * h * h + trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
    double sDiffMaxAcc = aMaxCur * h * h + trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
    //Determine limiting position differences:
    double sDiffMin = std::max(sDiffMinVel, sDiffMinAcc);
    double sDiffMax = std::min(sDiffMaxVel, sDiffMaxAcc);
    //Checking consistency
    if (sDiffMin > sDiffMax)
    {
        std::cerr << "ValidateMotionPlatform::applyPositionSetpoint(mot=" << mot << ", sSet=" << sSet << "): Danger: Limits without threshold!" << std::endl;
        sDiffMin = 0;
        sDiffMax = 0;
    }

    //Limiting new position to position difference limits
    //Copying new position setpoint
    double sSetLim = sSet;
    //Computing new position difference:
    double sSetLimDiff = sSetLim - trajectoryDequeVector[mot][0];
    //Limiting due to limiting position differences:
    if (sSetLimDiff < sDiffMin)
    {
        sSetLim = trajectoryDequeVector[mot][0] + sDiffMin;
    }
    else if (sSetLimDiff > sDiffMax)
    {
        sSetLim = trajectoryDequeVector[mot][0] + sDiffMax;
    }

    //Check for setpoint braking action
    //Time for reducing velocity:
    double velCurrent = (trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1]) / h;
    double velCurSign = (velCurrent < 0) ? -1 : 1;
    double velSetpoint = (setpointDequeVector[mot][0] - setpointDequeVector[mot][1]) / h;
    double velDiff = velSetpoint - velCurrent;
    double velDiffSign = (velDiff < 0) ? -1 : 1;
    double diffTimeVel = velDiff / (velDiffSign * accMax);

    double diffTimePos = 1000;
    double discDiffTimePos = velCurrent * velCurrent + 2 * (-1.0 * velCurSign) * accMax * (sSet - trajectoryDequeVector[mot][0]);
    if (discDiffTimePos > 0)
    {
        double diffTimePosFirstExpr = -velCurrent / (-1.0 * velCurSign * accMax);
        double diffTimePosSecondExpr = sqrt(discDiffTimePos) / (-1.0 * velCurSign * accMax);
        double diffTimePosOne = diffTimePosFirstExpr + diffTimePosSecondExpr;
        double diffTimePosTwo = diffTimePosFirstExpr - diffTimePosSecondExpr;
        if (diffTimePosOne <= 0 && diffTimePosTwo > 0)
        {
            diffTimePos = diffTimePosTwo;
        }
        if (diffTimePosTwo <= 0 && diffTimePosOne > 0)
        {
            diffTimePos = diffTimePosOne;
        }
        else if (diffTimePosOne > 0 && diffTimePosTwo > 0)
        {
            diffTimePos = std::min(diffTimePosOne, diffTimePosTwo);
        }
    }
    //std::cerr << "QuadTime: " << diffTimePos << ", LinearTime: " << (sSet-trajectoryDequeVector[mot][0])/velCurrent << std::endl;

    /*double diffTimePos = 1000;
   //Time for braking to position (neglecting acceleration):
   if(velCurrent!=0)
   {
       diffTimePos = (sSet-trajectoryDequeVector[mot][0])/velCurrent;
   }*/

    //std::cerr << "diffTimeVel: " << diffTimeVel << ", diffTimePos: " << diffTimePos << " velCurrent " << velCurrent << " trajectoryDequeVector[mot][0] "<< trajectoryDequeVector[mot][0] << std::endl;
    //When velocity reduction time smaller point brake time, decelerate with max acc
    if (diffTimePos > 0 && (diffTimeVel > diffTimePos))
    {
        //std::cerr << " bremsen " << std::endl;
        sSetLim = -1 * velCurSign * accMax * h * h + 2 * trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
        //std::cerr << "velCurSign: " << velCurSign << "velDiffSign: " << velDiffSign << " trajectoryDequeVector[mot][1]  " << trajectoryDequeVector[mot][1]  << " sSetLim  "<< sSetLim << std::endl;
        //Apply velocity/acceleration limits to new position difference
        //Velocity limits to pos diffs:
        double sDiffMinVel = vMinCur * h;
        double sDiffMaxVel = vMaxCur * h;
        //Acceleration limits to pos diff:
        double sDiffMinAcc = aMinCur * h * h + trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
        double sDiffMaxAcc = aMaxCur * h * h + trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
        //Determine limiting position differences:
        double sDiffMin = std::max(sDiffMinVel, sDiffMinAcc);
        double sDiffMax = std::min(sDiffMaxVel, sDiffMaxAcc);
        //Checking consistency
        if (sDiffMin > sDiffMax)
        {
            std::cerr << "ValidateMotionPlatform::applyPositionSetpoint2(mot=" << mot << ", sSet=" << sSet << "): Danger: Limits without threshold!" << std::endl;
            sDiffMin = 0;
            sDiffMax = 0;
        }
        //Limiting new position to position difference limits
        //Computing new position difference:
        double sSetLimDiff = sSetLim - trajectoryDequeVector[mot][0];
        //Limiting due to limiting position differences:
        if (sSetLimDiff < sDiffMin)
        {
            sSetLim = trajectoryDequeVector[mot][0] + sDiffMin;
        }
        else if (sSetLimDiff > sDiffMax)
        {
            sSetLim = trajectoryDequeVector[mot][0] + sDiffMax;
        }
    }

    //Position hard boundaries check
    if (sSetLim < posMin)
    {
        std::cerr << "sSetLim < posMin!" << std::endl;
        sSetLim = posMin;
    }
    else if (sSetLim > posMax)
    {
        std::cerr << "sSetLim > posMax!" << std::endl;
        sSetLim = posMax;
    }

    //Updateing setpoint deque
    setpointDequeVector[mot].pop_back();
    setpointDequeVector[mot].push_front(sSet);

    //Updateing trajectory deque
    trajectoryDequeVector[mot].pop_back();
    trajectoryDequeVector[mot].push_front(sSetLim);

    *((uint32_t *)(processFrameVector[mot].data)) = (uint32_t)(posFactorIncPerSI * sSetLim);
}
/*
void ValidateMotionPlatform::applyPositionSetpoint(const unsigned int& mot, double sSet)
{
   //std::cerr << "Motor: " << mot << ", applying setpoint: " << sSet;

   //Computing new position difference:
   double sSetDiff = sSet - trajectoryDequeVector[mot][0];

   //Deadtime dependent velocity limit:
   double velMaxLocal = velMax*(1-accMax*h*((double)updateMisses));
   if(velMaxLocal < 0) {
      velMaxLocal = 0;
   }
   //std::cerr << "velMaxLocal: " << velMaxLocal << ", updateMisses: " << updateMisses << std::endl;

   //Computing Boundaries:
   //Velocity max abs diff:
   double sDiffMinVel = (-1.0)*velMaxLocal*h;
   double sDiffMaxVel = velMax*h;
   //Acceleration max abs diff:
   double sDiffMinAcc = (-1.0)*accMax*h*h + trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
   double sDiffMaxAcc = accMax*h*h + trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
   //Determine limiting difference value:
   double sDiffMin = std::max(sDiffMinVel, sDiffMinAcc);
   double sDiffMax = std::min(sDiffMaxVel, sDiffMaxAcc);
   if(sDiffMin>sDiffMax) {
      std::cerr << "ValidateMotionPlatform::applyPositionSetpoint(mot=" << mot << ", sSet=" << sSet << "): Danger: Limits without threshold!" << std::endl;
      sDiffMin=0;
      sDiffMax=0;
   }
   //Limiting due to max vel and max acc
   if(sSetDiff>sDiffMax) {
      sSet = trajectoryDequeVector[mot][0] + sDiffMax;
   }
   else if(sSetDiff<sDiffMin) {
      sSet = trajectoryDequeVector[mot][0] + sDiffMin;
   }
   //std::cerr << "Mot " << mot << ", applied vel: " << (sSet - trajectoryDequeVector[mot][0])/h
   //                << ", acc: " << (sSet - 2*trajectoryDequeVector[mot][0] + trajectoryDequeVector[mot][1])/(h*h) << std::endl;

   //Position Boundaries:
   double sMaxPos = posMax - 2.0*0.5*(trajectoryDequeVector[mot][0]-trajectoryDequeVector[mot][1])*fabs(trajectoryDequeVector[mot][0]-trajectoryDequeVector[mot][1])/(accMax*h*h);
   //double sMaxPos = posMax - 2.0*0.5*(sSet-trajectoryDequeVector[mot][0])*fabs(sSet-trajectoryDequeVector[mot][0])/(accMax*h*h);
   double sMinPos = posMin - 2.0*0.5*(trajectoryDequeVector[mot][0]-trajectoryDequeVector[mot][1])*fabs(trajectoryDequeVector[mot][0]-trajectoryDequeVector[mot][1])/(accMax*h*h);
   //double sMinPos = posMin - 2.0*0.5*(sSet-trajectoryDequeVector[mot][0])*fabs(sSet-trajectoryDequeVector[mot][0])/(accMax*h*h);

   //Position boundaries check including brake threshold
   if(sSet < sMinPos) {
      sSet = accMax*h*h + 2*trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
      //sSet = sMinPos;
   }
   else if(sSet > sMaxPos) {
      sSet = (-1.0)*accMax*h*h + 2*trajectoryDequeVector[mot][0] - trajectoryDequeVector[mot][1];
      //sSet = sMaxPos;
   }
   //std::cerr << ", after bound: " << sSet;

   //Position boundaries hard check
   if(sSet < posMin) {
      std::cerr << "sSet < posMin!" << std::endl;
      sSet = sMinPos;
   }
   if(sSet > posMax) {
      std::cerr << "sSet > posMax!" << std::endl;
      sSet = sMaxPos;
   }
   //std::cerr << ", after 2. bound: " << sSet  << std::endl;
   //std::cerr << "brake threshold: min pos: " << sMinPos << ", max pos: " << sMaxPos << ", vec/acc diff: min: sDiffMin: " << sDiffMin << ", max: " << sDiffMax << std::endl;


   trajectoryDequeVector[mot].pop_back();
   trajectoryDequeVector[mot].push_front(sSet);

   *((uint32_t*)(processFrameVector[mot].data)) = (uint32_t)(posFactorIncPerSI*sSet);
   
   //std::cerr << "Motor: " << mot << ", new trajectory position: " << sSet << std::endl;
} */

void ValidateMotionPlatform::run()
{
    //Starting Beckhoff Emergency Stop
    can_frame startBeckhoffFrame;
    startBeckhoffFrame.can_id = 0x0;
    startBeckhoffFrame.can_dlc = 2;
    startBeckhoffFrame.data[0] = 1;
    startBeckhoffFrame.data[1] = 1;
    sendFrame(startBeckhoffFrame);

    //Applying Receive Filters
    applyRecvFilters();

    //Changing control mode to position interpolation
    setLinMotsControlMode<controlDisabled | controlResetBit>();

    can_frame stateFrame;

    set_periodic(TM_INFINITE);
    std::cerr << "controlDisabled| controlResetBit" << std::endl;
    setControlMode<controlDisabled | controlResetBit>(brakeMot);
    sendMutex.acquire(100000);
    for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
    {
        sendFrame(controlFrameVector[motIt]);
        std::cerr << "control frame: " << controlFrameVector[motIt] << std::endl;
    }
    sendFrame(syncFrame);
    for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
    {
        int numTries = 0;
        do
        {
            recvFrame(stateFrame);
            if (stateFrame.can_id & CAN_ERR_FLAG)
            {
                std::cerr << "ValidateMotionPlatform::run received error frame" << numTries << std::endl;
            }
            numTries++;
        } while (stateFrame.can_id & CAN_ERR_FLAG && (numTries < 10)); //
        std::map<can_id_t, unsigned int>::iterator idIt = stateIdIndexMap.find(stateFrame.can_id);
        if (idIt != stateIdIndexMap.end())
        {
            stateFrameVector[idIt->second] = stateFrame;
        }
    }
    sendMutex.release();
    //rt_task_sleep(10000000);
    sleep(1);
    //Referencing brake pedal
    std::cerr << "controlReference | controlResetBit" << std::endl;
    //setControlMode<controlReference | controlResetBit>(brakeMot);
    setControlMode<controlReference>(brakeMot);
    sendMutex.acquire(100000);
    for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
    {
        sendFrame(controlFrameVector[motIt]);
        std::cerr << "control frame: " << controlFrameVector[motIt] << std::endl;
    }
    sendFrame(syncFrame);
    for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
    {
        int numTries = 0;
        do
        {
            recvFrame(stateFrame);
            if (stateFrame.can_id & CAN_ERR_FLAG)
            {
                std::cerr << "ValidateMotionPlatform::run received error frame" << numTries << std::endl;
            }
            numTries++;
        } while (stateFrame.can_id & CAN_ERR_FLAG && (numTries < 10)); //
        std::map<can_id_t, unsigned int>::iterator idIt = stateIdIndexMap.find(stateFrame.can_id);
        if (idIt != stateIdIndexMap.end())
        {
            stateFrameVector[idIt->second] = stateFrame;
        }
    }
    sendMutex.release();
    rt_task_sleep(1000000);
    /*         std::cerr << "controlDisabled" << std::endl;
   setControlMode<controlDisabled>(brakeMot);
       sendMutex.acquire(100000);
         for(unsigned int motIt = 0; motIt<(numLinMots+numRotMots); ++motIt) {
            sendFrame(controlFrameVector[motIt]);
            std::cerr << "control frame: " << controlFrameVector[motIt] << std::endl;
         }
         sendFrame(syncFrame);
	 for(unsigned int motIt = 0; motIt<(numLinMots + numRotMots); ++motIt) {
          int numTries = 0;
          do {
             recvFrame(stateFrame);
	     if(stateFrame.can_id & CAN_ERR_FLAG)
	     {
	         std::cerr << "ValidateMotionPlatform::run received error frame" << numTries << std::endl;
	     }
	     numTries++;
	  } while (stateFrame.can_id & CAN_ERR_FLAG && (numTries < 10)); // 
         }
      sendMutex.release();
   rt_task_sleep(10000);
            std::cerr << "controlReference | controlResetBit" << std::endl;
   setControlMode<controlReference | controlResetBit>(brakeMot);
       sendMutex.acquire(100000);
         for(unsigned int motIt = 0; motIt<(numLinMots+numRotMots); ++motIt) {
            sendFrame(controlFrameVector[motIt]);
            std::cerr << "control frame: " << controlFrameVector[motIt] << std::endl;
         }
         sendFrame(syncFrame);
	 for(unsigned int motIt = 0; motIt<(numLinMots + numRotMots); ++motIt) {
          int numTries = 0;
          do {
             recvFrame(stateFrame);
	     if(stateFrame.can_id & CAN_ERR_FLAG)
	     {
	         std::cerr << "ValidateMotionPlatform::run received error frame" << numTries << std::endl;
	     }
	     numTries++;
	  } while (stateFrame.can_id & CAN_ERR_FLAG && (numTries < 10)); // 
         }
      sendMutex.release();*/

    set_periodic(h_ns);

    runTask = true;
    while (runTask)
    {
        //Send mutex locking
        sendMutex.acquire(100000);
        //Sending control data
        if (sendControlFrame)
        {
            for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
            {
                sendFrame(controlFrameVector[motIt]);
                std::cerr << "control frame: " << controlFrameVector[motIt] << std::endl;
            }
            sendControlFrame = false;
        }
        //Sending process data
        for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
        {
            if (motIt != brakeMot && getControlMode(motIt) == controlInterpolatedPositioning)
            {
                applyPositionSetpoint(motIt, posSetVector[motIt]);
            }
            else if (motIt == brakeMot)
            {
                //brakeForce =   -(std::max((((double)(*(int16_t*)stateFrameVector[motIt].data))*posFactorSIPerMM-0.02),0.0)*1000.0 + 5.0);
                //- ((double)(*(int32_t*)(stateFrameVector[motIt].data+2)))*velFactorSIPerMM*1.0;
                double s = ((double)(*(int16_t *)stateFrameVector[motIt].data)) * posFactorSIPerMM;
                //brakeForce = -100000.0*s*s*s+5.0*(tanh(200.0*(s-0.01))+1.0);
                //brakeForce = -(200.0*exp(25.486*(s-0.088181))-1.135);
                brakeForce = -(100.0 * exp(25.486 * (s - 0.088181)) - 1.135);

                (*((int32_t *)(processFrameVector[motIt].data))) = (int32_t)(forceFactorPercentPerSI * brakeForce);
            }
            sendFrame(processFrameVector[motIt]);
            //std::cerr << "process frame: " << processFrameVector[motIt] << std::endl;
        }
        //Send mutex unlock
        sendMutex.release();
        //Sending sync frame
        sendFrame(syncFrame);

        //Incrementing misses count
        ++updateMisses;

        //Receiving frames
        for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
        {
            //for(unsigned int motIt = 0; motIt<1; ++motIt) {
            int numTries = 0;
            do
            {
                recvFrame(stateFrame);
                if (stateFrame.can_id & CAN_ERR_FLAG)
                {
                    std::cerr << "ValidateMotionPlatform::run received error frame" << numTries << std::endl;
                }
                numTries++;
            } while (stateFrame.can_id & CAN_ERR_FLAG && (numTries < 10)); //
            std::map<can_id_t, unsigned int>::iterator idIt = stateIdIndexMap.find(stateFrame.can_id);
            if (idIt != stateIdIndexMap.end())
            {
                stateFrameVector[idIt->second] = stateFrame;
                if (idIt->second == brakeMot && (getControlMode(motIt) & controlReference) == controlReference
                    && ((*(uint16_t *)(stateFrameVector[brakeMot].data + 6)) & 0x01) == 0x01)
                {
                    std::cerr << "Control to torque!" << std::endl;
                    setControlMode<controlTorque>(brakeMot);
                    initialized = true;
                }
            }
        }

        if (overruns > 0)
        {
            std::cerr << "ValidateMotionPlatform::run(): overruns: " << overruns << std::endl;
        }
        rt_task_wait_period(&overruns);
    }

    setLinMotsControlMode<controlDisabled>();
    setControlMode<controlDisabled>(brakeMot);
    for (unsigned int motIt = 0; motIt < (numLinMots + numRotMots); ++motIt)
    {
        sendFrame(controlFrameVector[motIt]);
    }
    //Sending sync frame
    sendFrame(syncFrame);

    taskFinished = true;
    set_periodic(TM_INFINITE);
}
