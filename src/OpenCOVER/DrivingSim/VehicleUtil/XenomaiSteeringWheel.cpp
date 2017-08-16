/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "XenomaiSteeringWheel.h"

#ifdef MERCURY
#include <alchemy/timer.h>
#else
#include <native/timer.h>
#endif
#include <unistd.h>

#include <cstdlib>

XenomaiSteeringWheel::XenomaiSteeringWheel(CanOpenController &con, uint8_t id)
    : CanOpenDevice(con, id)
    , XenomaiTask("XenomaiSteeringWheel")
	, currentMutex("xsw_current_mutex")
	, positionMutex("xsw_positioni_mutex")
    , runTask(true)
    , taskFinished(false)
    , homing(false)
    , overruns(0)
    , position(0)
    , driftPosition(0)
    , speedDeque(10, 0)
    , Kwheel(0.005)
    , Dwheel(0.1)
    , rumbleAmplitude(3.0)
    , Kdrill(0.005)
    , drillElasticity(0.0)
{
	
}

XenomaiSteeringWheel::~XenomaiSteeringWheel()
{
    RT_TASK_INFO info;
    inquire(info);

#ifdef MERCURY
    if (info.stat.status & __THREAD_S_STARTED)
#else
    if (info.status & T_STARTED)
#endif
    {
        runTask = false;
        while (!taskFinished)
        {
            sleep(1);
        }
    }
}

bool XenomaiSteeringWheel::center()
{
	homing = true;
    
	fprintf(stderr, "Starting homing...\n");

    uint16_t RPDOData = 0x6; //enable op
    writeRPDO(1, (uint8_t *)&RPDOData, 6);
    controller->sendPDO();

    rt_task_sleep(10000000);

    stopNode();

    //controller->addRecvFilter(0x81 | CAN_INV_FILTER);
    //controller->applyRecvFilters();

    /*fprintf(stderr,"Starting reset...\n");
   resetNode();
   rt_task_sleep(4000000000);
   fprintf(stderr,"done reset...\n");*/
    resetComm();
    rt_task_sleep(100000000);
    enterPreOp();
    rt_task_sleep(100000000);
    
    bool success = false;

    rt_task_set_periodic(NULL, TM_NOW, rt_timer_ns2ticks(1000000));

    uint16_t controlWord = 0x7; //disable operation
    /*uint16_t controlWord = 0x7;     //disable operation
   if(!writeSDO(0x6040, 0, (uint8_t*)&controlWord, 2)) {
       fprintf(stderr,"control word disable operation failed\n");
   }*/

    can_frame frame;
    controller->setRecvTimeout(100000000);

    while (controller->recvFrame(frame) > 0)
    {
        controller->printFrame("waiting until no more messages arrive: ", frame);
    }
    fprintf(stderr, "no more can messages for at least half a second\n");

    uint8_t opMode = 0xf9; //homing mode
    if (!writeSDO(0x6060, 0, &opMode, 1))
    {
        fprintf(stderr, "op mode failed 1\n");
        stopNode();
        //done = true;
        homing = false;
        return success;;
    }

    uint8_t homingType = 0x7; //homing type: mechanical limit + zero mark: 7 or only mechanical limit: 9
    if (!writeSDO(0x2024, 1, &homingType, 1))
    {
        fprintf(stderr, "set homing type failed\n");
        stopNode();
        //done = true;
        homing = false;
        return success;;
    }
    uint8_t homingDir = 0x0; //homing dir
    if (!writeSDO(0x2024, 2, &homingDir, 1))
    {
        fprintf(stderr, "set homing direction failed\n");
        stopNode();
        //done = true;
        homing = false;
        return success;;
    }
    int32_t homingVel = 70000; //homing vel
    if (!writeSDO(0x2024, 3, (uint8_t *)&homingVel, 4))
    {
        fprintf(stderr, "set homing velocity failed\n");
        stopNode();
        //done = true;
        homing = false;
        return success;;
    }
    uint16_t homingAcc = 50; //homing acc
    if (!writeSDO(0x2024, 4, (uint8_t *)&homingAcc, 2))
    {
        fprintf(stderr, "set homing acceleration failed\n");
        stopNode();
        //done = true;
        homing = false;
        return success;;
    }
    uint16_t homingDec = 50; //homing dec
    if (!writeSDO(0x2024, 5, (uint8_t *)&homingDec, 2))
    {
        fprintf(stderr, "set homing decceleration failed\n");
        stopNode();
        //done = true;
        homing = false;
        return success;;
    }
    int32_t homingRef = zeroMarkPosition; //homing reference offset
    if (!writeSDO(0x2024, 6, (uint8_t *)&homingRef, 4))
    {
        fprintf(stderr, "set homing reference offset\n");
        stopNode();
        //done = true;
        homing = false;
        return success;;
    }
    controlWord = 0x1F; //enable operation
    if (!writeSDO(0x6040, 0, (uint8_t *)&controlWord, 2))
    {
        fprintf(stderr, "control word enable starthoming operation failed\n");
    }

    uint8_t pdoSelectData = 0x1; //1. RPDO: Control word
    if (!writeSDO(0x2600, 0, &pdoSelectData, 1))
    {
        fprintf(stderr, "1. rpdo select failed\n");
        stopNode();
        //done = true;
		homing = false;
        return success;
    }
    /* pdoSelectData = 0x0; //2. RPDO: Control word
   if(!writeSDO(0x2601, 0, &pdoSelectData, 1)) {
      std::cerr << "2. rpdo select failed" << std::endl;
   }
   pdoSelectData = 0x0; //3. RPDO: Control word
   if(!writeSDO(0x2602, 0, &pdoSelectData, 1)) {
      std::cerr << "3. rpdo select failed" << std::endl;
   }
   pdoSelectData = 0x0; //4. RPDO: Control word
   if(!writeSDO(0x2603, 0, &pdoSelectData, 1)) {
      std::cerr << "4. rpdo select failed" << std::endl;
   }*/

    pdoSelectData = 0x17; //1. TPDO: enhanced status
    if (!writeSDO(0x2a00, 0, &pdoSelectData, 1))
    {
        fprintf(stderr, "1. rpdo select failed\n");
    }
    /*   pdoSelectData = 0x0; //2. TPDO: enhanced status
   if(!writeSDO(0x2a01, 0, &pdoSelectData, 1)) {
      std::cerr << "2. tpdo select failed" << std::endl;
   }
   pdoSelectData = 0x0; //3. TPDO: enhanced status
   if(!writeSDO(0x2a02, 0, &pdoSelectData, 1)) {
      std::cerr << "3. tpdo select failed" << std::endl;
   }
   pdoSelectData = 0x0; //4. TPDO: enhanced status
   if(!writeSDO(0x2a03, 0, &pdoSelectData, 1)) {
      std::cerr << "4. tpdo select failed" << std::endl;
   }*/
    uint8_t transType = 0xff; //1. TPDO transmission type: asynchronous event, manufacturer specific
    //uint8_t transType = 0x1; //1. TPDO transmission type: synchronous after one SYNC
    if (!writeSDO(0x1800, 2, &transType, 1))
    {
        std::cerr << "1. tpdo transmission type set failed" << std::endl;
    }
    fprintf(stderr, "Running to hardware limit...\n");
    startNode();

    RPDOData = 0x1f; //enable op
    writeRPDO(1, (uint8_t *)&RPDOData, 2);

    unsigned int count = 0;
    unsigned long overruns = 0;
    uint32_t enhStatus = 0;
    bool motionTaskActive = false;
    while (!motionTaskActive)
    {
        controller->sendPDO();
        controller->recvPDO(1);
        uint8_t *TPDOData = readTPDO(1);
        memcpy(&enhStatus, TPDOData + 2, 4);
        motionTaskActive = enhStatus & 0x10000;
		++count;
        if(enhStatus & 0x20000)
		{
        std::cerr << "homing/reference point set "<< std::endl;
		}
		if(enhStatus & 0x40000)
		{
        std::cerr << "home "<< std::endl;
		}
        std::cerr << "first stage: count: " << std::dec << count << ", overruns: " << overruns << ", enhStatus: " << std::hex << enhStatus << std::endl;
		//std::cerr << "first stage: enhStatus: " << std::hex << enhStatus << std::endl;
        rt_task_wait_period(&overruns);
    }

    fprintf(stderr, "Running to hardware limit2...\n");
    count = 0;
    while (motionTaskActive)
    {
        controller->recvPDO(1);
        uint8_t *TPDOData = readTPDO(1);
        memcpy(&enhStatus, TPDOData + 2, 4);
        motionTaskActive = enhStatus & 0x10000;
		++count;
		if(enhStatus & 0x20000)
		{
        std::cerr << "homing/reference point set "<< std::endl;
		}
		if(enhStatus & 0x40000)
		{
        std::cerr << "home "<< std::endl;
		}
        std::cerr << "second stage: count: " << std::dec << count << ", overruns: " << overruns << ", enhStatus: " << std::hex << enhStatus << std::endl;
		//std::cerr << "second stage: enhStatus: " << std::hex << enhStatus << std::endl;
        rt_task_wait_period(&overruns);
    }

    //std::cerr << "status: " << std::hex << enhStatus << std::endl;

    RPDOData = 0x7;
    writeRPDO(1, (uint8_t *)&RPDOData, 2);
    controller->sendPDO();

    enterPreOp();
    rt_task_wait_period(NULL);

    opMode = 0xff; //positioning mode
    if (!writeSDO(0x6060, 0, &opMode, 1))
    {
        std::cerr << "op mode failed" << std::endl;
    }

    int32_t position = 0; //position
    if (!writeSDO(0x2022, 1, (uint8_t *)&position, 4))
    {
        std::cerr << "set target position failed" << std::endl;
    }
    int16_t velocity = 50000; //vel
    if (!writeSDO(0x2022, 2, (uint8_t *)&velocity, 2))
    {
        std::cerr << "set target velocity failed" << std::endl;
    }
    uint16_t motionTaskType = 0;
    if (!writeSDO(0x2022, 3, (uint8_t *)&motionTaskType, 2))
    {
        std::cerr << "set motion task type failed" << std::endl;
    }

    fprintf(stderr, " to center position...\n");
    startNode();

    RPDOData = 0x1f;
    writeRPDO(1, (uint8_t *)&RPDOData, 2);
    controller->sendPDO();

    count = 0;
    motionTaskActive = false;
    while (!motionTaskActive)
    {
        controller->sendPDO();
        controller->recvPDO(1);
        uint8_t *TPDOData = readTPDO(1);
        memcpy(&enhStatus, TPDOData + 2, 4);
        motionTaskActive = enhStatus & 0x10000;
		++count;
		
        if(enhStatus & 0x20000)
		{
        std::cerr << "homing/reference point set "<< std::endl;
		}
		if(enhStatus & 0x40000)
		{
        std::cerr << "home "<< std::endl;
		}
		std::cerr << "first: count: " << std::dec << count << ", overruns: " << overruns << ", enhStatus: " << std::hex << enhStatus << std::endl;
		//std::cerr << "first stage: enhStatus: " << std::hex << enhStatus << std::endl;
        rt_task_wait_period(&overruns);
    }
    
    fprintf(stderr, " to center position2\n");
    count = 0;
    while (motionTaskActive)
    {
        controller->recvPDO(1);
        uint8_t *TPDOData = readTPDO(1);
        memcpy(&enhStatus, TPDOData + 2, 4);
        motionTaskActive = enhStatus & 0x10000;
		++count;
		if(enhStatus & 0x20000)
		{
        std::cerr << "homing/reference point set "<< std::endl;
		}
		if(enhStatus & 0x40000)
		{
        std::cerr << "home "<< std::endl;
		}
        std::cerr << "running: count: " << std::dec << count << ", overruns: " << overruns << ", enhStatus: " << std::hex << enhStatus << std::endl;
		//std::cerr << "second stage: enhStatus: " << std::hex << enhStatus << std::endl;
        rt_task_wait_period(&overruns);
    }

    /*RPDOData = 0x6;
   writeRPDO(1, (uint8_t*)&RPDOData, 2);
   controller->sendPDO();*/

    stopNode();


    fprintf(stderr, "Homing done! \n");

    controlWord = 0x7; //disable operation
    if (!writeSDO(0x6040, 0, (uint8_t *)&controlWord, 2))
    {
        std::cerr << "control word failed" << std::endl;
    }
    rt_task_sleep(10000000);

    opMode = 0xfd; //set current mode
    if (!writeSDO(0x6060, 0, &opMode, 1))
    {
        std::cerr << "op mode failed" << std::endl;
    }

    rt_task_sleep(10000000);
    pdoSelectData = 0x16; //1. RPDO: current setpoint
    if (!writeSDO(0x2600, 0, &pdoSelectData, 1))
    {
        std::cerr << "1. rpdo select failed" << std::endl;
    }

    rt_task_sleep(10000000);
    pdoSelectData = 0x25; //1. TPDO: freely mapable
    if (!writeSDO(0x2a00, 0, &pdoSelectData, 1))
    {
        std::cerr << "1. tpdo select failed" << std::endl;
    }
    rt_task_sleep(10000000);
    uint8_t pdoMapDelete[4] = { 0, 0, 0, 0 }; //Delete mapping
    if (!writeSDO(0x1a00, 0, pdoMapDelete, 4))
    {
        std::cerr << "1. tpdo mapping deletion failed" << std::endl;
    }
    rt_task_sleep(10000000);
    uint8_t pdoMapDataPos[4] = { 0x20, 0x03, 0x70, 0x20 }; //Mapping incremental position
    if (!writeSDO(0x1a00, 1, pdoMapDataPos, 4))
    {
        std::cerr << "1. tpdo mapping pos failed" << std::endl;
    }
    rt_task_sleep(10000000);
    uint8_t pdoMapDataVel[4] = { 0x18, 0x02, 0x70, 0x20 }; //Mapping velocity
    if (!writeSDO(0x1a00, 2, pdoMapDataVel, 4))
    {
        std::cerr << "1. tpdo mapping vel failed" << std::endl;
    }
    rt_task_sleep(10000000);
    //uint8_t transType = 0xff; //1. TPDO transmission type: asynchronous event, manufacturer specific
    transType = 0x1; //1. TPDO transmission type: synchronous after one SYNC
    if (!writeSDO(0x1800, 2, &transType, 1))
    {
        std::cerr << "1. tpdo transmission type set failed" << std::endl;
    }
    /*uint16_t inhibitTime = 1; //1. TPDO inhibit time
   if(!writeSDO(0x1800, 3, (uint8_t*)&inhibitTime, 2)) {
      std::cerr << "1. tpdo inhibit time set failed" << std::endl;
   }*/

    rt_task_sleep(10000000);
    startNode();
    //done = true;
    success = true;

    fprintf(stderr, "back to normal operation! \n");
	
	homing = false;
	
	return success;
	
}

void XenomaiSteeringWheel::run()
{
    std::cerr << "Starting..." << std::endl;

    resetComm();
    rt_task_sleep(100000000);
    enterPreOp();
    rt_task_sleep(100000000);

    rt_task_set_periodic(NULL, TM_NOW, rt_timer_ns2ticks(1000000));

    uint16_t controlWord = 0x7; //disable operation
    if (!writeSDO(0x6040, 0, (uint8_t *)&controlWord, 2))
    {
        std::cerr << "control word failed" << std::endl;
    }
    std::cerr << "Starting2..." << std::endl;
    uint8_t opMode = 0xfd; //digital current
    if (!writeSDO(0x6060, 0, &opMode, 1))
    {
        std::cerr << "op mode failed" << std::endl;
    }

    uint8_t pdoSelectData = 0x16; //1. RPDO: current setpoint
    if (!writeSDO(0x2600, 0, &pdoSelectData, 1))
    {
        std::cerr << "1. rpdo select failed" << std::endl;
    }

    pdoSelectData = 0x25; //1. TPDO: freely mapable
    if (!writeSDO(0x2a00, 0, &pdoSelectData, 1))
    {
        std::cerr << "1. tpdo select failed" << std::endl;
    }
    uint8_t pdoMapDelete[4] = { 0, 0, 0, 0 }; //Delete mapping
    if (!writeSDO(0x1a00, 0, pdoMapDelete, 4))
    {
        std::cerr << "1. tpdo mapping deletion failed" << std::endl;
    }
    uint8_t pdoMapDataPos[4] = { 0x20, 0x03, 0x70, 0x20 }; //Mapping incremental position
    if (!writeSDO(0x1a00, 1, pdoMapDataPos, 4))
    {
        std::cerr << "1. tpdo mapping pos failed" << std::endl;
    }
    uint8_t pdoMapDataVel[4] = { 0x18, 0x02, 0x70, 0x20 }; //Mapping velocity
    if (!writeSDO(0x1a00, 2, pdoMapDataVel, 4))
    {
        std::cerr << "1. tpdo mapping vel failed" << std::endl;
    }
    //uint8_t transType = 0xff; //1. TPDO transmission type: asynchronous event, manufacturer specific
    uint8_t transType = 0x1; //1. TPDO transmission type: synchronous after one SYNC
    if (!writeSDO(0x1800, 2, &transType, 1))
    {
        std::cerr << "1. tpdo transmission type set failed" << std::endl;
    }
    /*uint16_t inhibitTime = 1; //1. TPDO inhibit time
   if(!writeSDO(0x1800, 3, (uint8_t*)&inhibitTime, 2)) {
      std::cerr << "1. tpdo inhibit time set failed" << std::endl;
   }*/

    startNode();

    uint8_t RPDOData[6] = { 0x1f, 0, 0, 0, 0, 0 }; //enable op
    writeRPDO(1, RPDOData, 6);

    unsigned int count = 0;
    std::deque<int32_t> speedDeque(50, 0);
    std::deque<int32_t>::iterator speedDequeIt;
    int32_t speed = 0;
    double lowPassSpeed = 0.0;

    while (runTask)
    {
        //std::cout << "xenomai wheel run task is running" << std::cout;
		if (overruns != 0)
		{
			std::cerr << "FourWheelDynamicsRealtimeRealtime::run(): overruns: " << overruns << std::endl;
			overruns=0;
		}
		if(!homing)
		{
			controller->sendSync();
			controller->recvPDO(1);
			
			uint8_t *TPDOData = readTPDO(1);
			
			positionMutex.acquire(1000000);
			memcpy(&position, TPDOData, 4);
			positionMutex.release();
			
			memcpy(&speed, TPDOData + 4, 3);
			*(((uint8_t *)&speed) + 3) = (*(((uint8_t *)&speed) + 2) & 0x80) ? 0xff : 0x0;
			
			
			bool useSpringDamper =  false;
			
			int32_t springDamperCurrent = 0;
			
			if(useSpringDamper)
			{
				speedDeque.pop_front();
				speedDeque.push_back(speed);
				lowPassSpeed = 0;
				for (speedDequeIt = speedDeque.begin(); speedDequeIt != speedDeque.end(); ++speedDequeIt)
				{
					lowPassSpeed += (*speedDequeIt);
				}
				lowPassSpeed = (double)lowPassSpeed / (double)speedDeque.size();
				
				springDamperCurrent = (int32_t)(-Kwheel * drillElasticity * (double)position - Dwheel * lowPassSpeed); //Spring-Damping-Model

				springDamperCurrent += (int32_t)(((rand() / ((double)RAND_MAX)) - 0.5) * rumbleAmplitude); //Rumbling

				double drillRigidness = 1.0 - drillElasticity;
				if ((driftPosition - position) > 100000 * drillRigidness)
					driftPosition = position + (int32_t)(100000 * drillRigidness);
				else if ((driftPosition - position) < -100000 * drillRigidness)
					driftPosition = position - (int32_t)(100000 * drillRigidness);
				springDamperCurrent += (int32_t)((double)(driftPosition - position) * Kdrill);
				//std::cerr << "drift position - position: " << (int32_t)((double)(driftPosition-position)*0.005) << ", drill current: " << (int32_t)((double)(driftPosition-position)*Kdrill) << ", Kdrill: " << Kdrill << std::endl;
			}
			
			
			if(useSpringDamper)
			{
				current = springDamperCurrent;
			}
			if (current > peakCurrent)
			{
				current = peakCurrent;
			}
			else if (current < -peakCurrent)
			{
				current = -peakCurrent;
			}
			currentMutex.acquire(1000000);
			*((int32_t *)(RPDOData + 2)) = current;
			writeRPDO(1, RPDOData, 6);
			currentMutex.release();
			
			controller->sendPDO();
		}
		
        //std::cerr << std::dec << "count: " << count << ", overruns: " << overruns << ", position: " << position << ", speed: " << speed << ", low pass speed: " << lowPassSpeed << ", deque size: " << speedDeque.size() << ", setpoint: " << *((int32_t*)(RPDOData+2)) << std::endl;
        rt_task_wait_period(&overruns);
    }

    RPDOData[0] = 0x6;
    RPDOData[1] = 0x0;
    RPDOData[2] = 0x0;
    RPDOData[3] = 0x0;
    RPDOData[4] = 0x0;
    RPDOData[5] = 0x0;
    writeRPDO(1, RPDOData, 6);
    controller->sendPDO();

    rt_task_set_periodic(NULL, TM_NOW, TM_INFINITE);

    /*  controller->setRecvTimeout(1000000);   //flushing can socket receive buffer
   int recvErr=0;
   can_frame dummyFrame;
   while(recvErr != -ETIMEDOUT) {
      recvErr = controller->recvFrame(dummyFrame);
   }
   controller->setRecvTimeout(RTDM_TIMEOUT_INFINITE);*/

    stopNode();
    /*rt_task_sleep(10000000);
   
   enterPreOp();

   rt_task_sleep(10000000);

   uint16_t status = 0x6;     
   if(!writeSDO(0x6040, 0, (uint8_t*)&status, 2)) {
      std::cerr << "steering wheel shutdown failed" << std::endl;
   }*/

    taskFinished = true;
}

void XenomaiSteeringWheel::init()
{
    std::cerr << "Starting SteeringWheel..." << std::endl;

    resetComm();
    rt_task_sleep(100000000);
    enterPreOp();
    rt_task_sleep(100000000);

    //rt_task_set_periodic(NULL, TM_NOW, rt_timer_ns2ticks(1000000));

    uint16_t controlWord = 0x7; //disable operation
    if (!writeSDO(0x6040, 0, (uint8_t *)&controlWord, 2))
    {
        std::cerr << "control word failed" << std::endl;
    }
    rt_task_sleep(10000000);
    uint8_t opMode = 0xfd; //set current mode
    if (!writeSDO(0x6060, 0, &opMode, 1))
    {
        std::cerr << "op mode failed" << std::endl;
    }

    rt_task_sleep(10000000);
    uint8_t pdoSelectData = 0x16; //1. RPDO: current setpoint
    if (!writeSDO(0x2600, 0, &pdoSelectData, 1))
    {
        std::cerr << "1. rpdo select failed" << std::endl;
    }

    rt_task_sleep(10000000);
    pdoSelectData = 0x25; //1. TPDO: freely mapable
    if (!writeSDO(0x2a00, 0, &pdoSelectData, 1))
    {
        std::cerr << "1. tpdo select failed" << std::endl;
    }
    rt_task_sleep(10000000);
    uint8_t pdoMapDelete[4] = { 0, 0, 0, 0 }; //Delete mapping
    if (!writeSDO(0x1a00, 0, pdoMapDelete, 4))
    {
        std::cerr << "1. tpdo mapping deletion failed" << std::endl;
    }
    rt_task_sleep(10000000);
    uint8_t pdoMapDataPos[4] = { 0x20, 0x03, 0x70, 0x20 }; //Mapping incremental position
    if (!writeSDO(0x1a00, 1, pdoMapDataPos, 4))
    {
        std::cerr << "1. tpdo mapping pos failed" << std::endl;
    }
    rt_task_sleep(10000000);
    uint8_t pdoMapDataVel[4] = { 0x18, 0x02, 0x70, 0x20 }; //Mapping velocity
    if (!writeSDO(0x1a00, 2, pdoMapDataVel, 4))
    {
        std::cerr << "1. tpdo mapping vel failed" << std::endl;
    }
    rt_task_sleep(10000000);
    //uint8_t transType = 0xff; //1. TPDO transmission type: asynchronous event, manufacturer specific
    uint8_t transType = 0x1; //1. TPDO transmission type: synchronous after one SYNC
    if (!writeSDO(0x1800, 2, &transType, 1))
    {
        std::cerr << "1. tpdo transmission type set failed" << std::endl;
    }
    /*uint16_t inhibitTime = 1; //1. TPDO inhibit time
   if(!writeSDO(0x1800, 3, (uint8_t*)&inhibitTime, 2)) {
      std::cerr << "1. tpdo inhibit time set failed" << std::endl;
   }*/

    rt_task_sleep(10000000);
    startNode();

    RPDOData[0] = 0x1f;
    RPDOData[1] = 0;
    RPDOData[2] = 0;
    RPDOData[3] = 0;
    RPDOData[4] = 0;
    RPDOData[5] = 0;
	
	start();
}

void XenomaiSteeringWheel::shutdown()
{
    std::cerr << "Shutting down steering wheel..." << std::endl;

    RPDOData[0] = 0x6; //enable op
    writeRPDO(1, RPDOData, 6);
    controller->sendPDO();

    /*  controller->setRecvTimeout(1000000);   //flushing can socket receive buffer
   int recvErr=0;
   can_frame dummyFrame;
   while(recvErr != -ETIMEDOUT) {
      recvErr = controller->recvFrame(dummyFrame);
   }
   controller->setRecvTimeout(RTDM_TIMEOUT_INFINITE);*/

    rt_task_sleep(10000000);

    stopNode();
    /*rt_task_sleep(10000000);
   
   enterPreOp();

   rt_task_sleep(10000000);

   uint16_t status = 0x6;     
   if(!writeSDO(0x6040, 0, (uint8_t*)&status, 2)) {
      std::cerr << "steering wheel shutdown failed" << std::endl;
   }*/
}
