/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PorscheSteeringWheel.h"
#include <cstring>

#ifdef HAVE_PCAN

PorscheSteeringWheel::PorscheSteeringWheel(CanOpenBus *bus, unsigned char moduleID)
    : ServoStar(bus, moduleID)
{
    scaleAngle = 2 * PI / 1048576; //[increments to radians]
    scaleVel = 2 * PI * 125 / 1048576; //[incrs/second to radians/second]
    scaleTorque = 3280 / 14.6; //[torque current increments to Nm] linear function assumed

    torqueData[0] = 0xf;
    torqueData[1] = 0;

    for (int i = 0; i < BUFFERLENGTH; ++i)
    {
        angBuffer[i] = 0;
        velBuffer[i] = 0;
    }
    bufferIndex = 0;
};

bool PorscheSteeringWheel::setupAngleTorqueMode(unsigned char tpdott, unsigned short guardTime)
{
    bool success = true;

    setOpMode(0xfd);

    //Setting up RPDO3 (PDO22: control word, current setpoint)
    if (!setRPDO(3, 22))
        success = false;
    if (!setRPDOCom(3, 0xff, 0))
        success = false;

    //Setting up TPDO3 (PDO39: incremental position, velocity)
    if (!setTPDO(3, 39))
        success = false;
    if (!setTPDOCom(3, tpdott, 0))
        success = false;
    if (!setTPDOMap(3, 0x2070, 3, 0x20, 0x2070, 2, 0x18))
        success = false;

    if (!setupLifeGuarding(guardTime))
        success = false;

    return success;
}

bool PorscheSteeringWheel::startAngleTorqueMode()
{
    if (startNode())
    {
        bus->recvPDO();
        bus->recvPDO();
        return true;
    }
    else
        return false;
}

bool PorscheSteeringWheel::stopAngleTorqueMode()
{
    bool success = true;

    if (!stopLifeGuarding())
        success = false;
    if (!sendGuardMessage())
        success = false;
    if (!shutdown())
        success = false;
    if (!stopNode())
        success = false;

    return success;
}

bool PorscheSteeringWheel::resetWheel()
{
    if (resetNode())
    {
        microsleep(2000000);
        while (!bus->recvEmergencyObject(1, NULL))
        {
        }
        return true;
    }
    else
        return false;
}

bool PorscheSteeringWheel::checkWheel(int sleeptime, int cycles)
{
    std::cerr << "Checking output stage..." << std::endl;
    for (int i = 0; i < cycles; ++i)
    {
        if (checkOutputStageEnabled())
        {
            std::cerr << "Output stage enabled!" << std::endl;
            return true;
        }
        std::cerr << "Output stage of ServoStar disabled! Please enable it!" << std::endl;
        microsleep(sleeptime);
    }
    return false;
}

bool PorscheSteeringWheel::homeWheel()
{
    bool run;
    bool success = true;
    unsigned char pdodata[8];

    //std::cerr << "Starting node... ";
    if (!startNode())
        success = false;

    //std::cerr << "Enabling operation... ";
    if (!enableOp())
        success = false;

    //std::cerr << "Setting operation mode... ";
    if (!setOpMode(6))
        success = false;

    //std::cerr << "Setting up homing mode... ";
    if (!setupHoming(0, 33, 100000, 10))
        success = false;
    ;

    //std::cerr << "Starting homing... ";
    if (!enableHoming())
        success = false;

    //std::cerr << "Waiting till homing is finished... ";
    run = true;
    unsigned long starttime = getTimeInSeconds();
    while (run)
    {
        bus->recvPDO(1, 1, pdodata);
        if ((pdodata[1] & 0x14) == 0x14)
            run = false;
        if ((getTimeInSeconds() - starttime) > 10)
        {
            run = false;
            success = false;
        }
        std::cerr << "Elapsed time: " << (getTimeInSeconds() - starttime) << std::endl;
    }
    //std::cerr << "homing finished!" << std::endl;

    //std::cerr << "Shutting down... ";
    if (!shutdown())
        success = false;

    //std::cerr << "Stopping node... ";
    if (!stopNode())
        success = false;

    return success;
}

bool PorscheSteeringWheel::getAngleVelocity(int &angle, int &velocity)
{
    unsigned char data[7];
    if (!bus->recvPDO(1, 3, data))
        return false;

    //int actAngle = -*(int*)(&data[0]);
    angle = -*(int *)(&data[0]);
    int actVelocity = -(*(int *)(&data[3])) >> 8;

    //angBuffer[bufferIndex] = actAngle;
    velBuffer[bufferIndex] = actVelocity;
    ++bufferIndex;
    if (bufferIndex >= BUFFERLENGTH)
        bufferIndex = 0;
    //int sumAngBuffer = 0;
    int sumVelBuffer = 0;
    for (int i = 0; i < BUFFERLENGTH; ++i)
    {
        //sumAngBuffer += angBuffer[i];
        sumVelBuffer += velBuffer[i];
    }
    //angle = (int)((double)sumAngBuffer/BUFFERLENGTH);
    velocity = (int)((double)sumVelBuffer / BUFFERLENGTH);

    return true;
}

bool PorscheSteeringWheel::setTorque(int torque)
{
    torque = -torque;
    if (torque > 3280)
        torque = 3280;
    else if (torque < -3280)
        torque = -3280;
    memcpy(&torqueData[2], &torque, sizeof(int));

    if (!bus->sendPDO(1, 3, 6, torqueData))
        return false;
    else
        return true;
}

bool PorscheSteeringWheel::getAngleVelocityInRadians(double &angleFloat, double &velFloat)
{
    int angleInt, velInt;
    static int oldIntAngle = 0;
    if (!getAngleVelocity(angleInt, velInt))
        return false;

    angleFloat = scaleAngle * (double)angleInt;
    velFloat = scaleVel * (double)velInt;
    //velFloat = scaleAngle*((double)(angleInt-oldIntAngle)*100);
    oldIntAngle = angleInt;

    return true;
}

bool PorscheSteeringWheel::setTorqueInNm(double torque)
{
    return (setTorque((int)(torque * scaleTorque)));
}

bool PorscheSteeringWheel::setScheduler()
{
#ifndef WIN32
#ifdef CPU_SETSIZE
    struct sched_param scheduler;
    scheduler.sched_priority = 1;
    if (sched_setscheduler(0, SCHED_FIFO, &scheduler) != 0)
    {
        std::cerr << "Couldn't set scheduler: ";
        switch (errno)
        {
        case (ESRCH):
            std::cerr << "The process whose ID is pid could not be found." << std::endl;
            break;
        case (EPERM):
            std::cerr << "The calling process does not have appropriate privileges. Only root processes are allowed to activate the SCHED_FIFO and SCHED_RR policies. The process calling sched_setscheduler needs an effective uid equal to the euid or uid of the process identified by pid, or it must be a superuser process." << std::endl;
            break;
        case (EINVAL):
            std::cerr << "The scheduling policy is not one of the recognized policies, or the parameter p does not make sense for the policy." << std::endl;
            break;
        default:
            std::cerr << "Unkown error!" << std::endl;
        }
        return false;
    }
#endif
#endif
    return true;
}

bool PorscheSteeringWheel::niceProcess(int incr)
{
#ifndef WIN32
#ifdef CPU_SETSIZE
    if (nice(incr) != 0)
    {
        std::cerr << "Couldn't set scheduler: ";
        switch (errno)
        {
        case (ESRCH):
            std::cerr << "The combination of class and id does not match any existing process." << std::endl;
            break;
        case (EINVAL):
            std::cerr << "The value of class is not valid." << std::endl;
            break;
        case (EPERM):
            std::cerr << "The call would set the nice value of a process which is owned by a different user than the calling process (i.e. the target process' real or effective uid does not match the calling process' effective uid) and the calling process does not have CAP_SYS_NICE permission." << std::endl;
            break;
        case (EACCES):
            std::cerr << "The call would lower the process' nice value and the process does not have CAP_SYS_NICE permission." << std::endl;
            break;
        default:
            std::cerr << "Unkown error!" << std::endl;
        }
        return false;
    }
#else
    (void)incr;
#endif
#endif
    return true;
}

bool PorscheSteeringWheel::setAffinity(int cpu)
{
#ifndef WIN32
#ifdef CPU_SETSIZE
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

#ifdef CO_rhel3
    if (sched_setaffinity(0, &cpuset) != 0)
    {
#else
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
    {
#endif
        std::cerr << "Couldn't set affinity: ";
        switch (errno)
        {
        case (EFAULT):
            std::cerr << "A supplied memory address was invalid." << std::endl;
            break;
        case (EINVAL):
            std::cerr << "The  affinity  bitmask mask contains no processors that are physically on the system, or the length len is smaller than the size of the affinity mask used by the kernel." << std::endl;
            break;
        case (EPERM):
            std::cerr << "The calling process does not have appropriate privileges.  The process calling  sched_setaffinity  needs  an  effective user ID equal to the user ID or effective user ID of the process identified by pid, or it must possess the CAP_SYS_NICE capability." << std::endl;
            break;
        case (ESRCH):
            std::cerr << "The process whose ID is pid could not be found." << std::endl;
            break;
        default:
            std::cerr << "Unkown error!" << std::endl;
        }
        return false;
    }
#else
    (void)cpu;
#endif
#endif
    return true;
}
#endif
