/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ServoStar.h"
#include <util/unixcompat.h>
#ifdef HAVE_PCAN

ServoStar::ServoStar(CanOpenBus *bus, unsigned char moduleID)
{
    this->bus = bus;

    if ((moduleID < 1) || (moduleID > 0x7f))
        id = 1;
    else
        id = moduleID;
}

bool ServoStar::readObject(unsigned short index, unsigned char subindex, unsigned char &length, void *&data)
{
    TPCANMsg *msg = bus->readObject(id, index, subindex);
    if (msg == NULL)
        return false;

    length = (msg->LEN - 4);
    data = (&(msg->DATA[4]));
    return true;
}

bool ServoStar::writeObject(unsigned short index, unsigned char subindex, unsigned char length, unsigned char *data)
{
    return (bus->writeObject(id, index, subindex, length, data));
}

bool ServoStar::startNode()
{
    return bus->sendNMT(id, 1);
}

bool ServoStar::stopNode()
{
    return bus->sendNMT(id, 2);
}

bool ServoStar::resetNode()
{
    bool success = bus->sendNMT(id, 129);

    return success;
}

bool ServoStar::enableOp()
{
    unsigned char enableOp = 0xf;
    return bus->writeObject(id, 0x6040, 0, 1, &enableOp);
}

bool ServoStar::disableOp()
{
    unsigned char disableOp = 0x2;
    return bus->writeObject(id, 0x6040, 0, 1, &disableOp);
}

bool ServoStar::enableHoming()
{
    unsigned char enableOp = 0x1f;
    return bus->writeObject(id, 0x6040, 0, 1, &enableOp);
}

bool ServoStar::disableHoming()
{
    unsigned char enableOp = 0xf;
    return bus->writeObject(id, 0x6040, 0, 1, &enableOp);
}

bool ServoStar::newSetpoint()
{
    unsigned char enableOp = 0x1f;
    return bus->writeObject(id, 0x6040, 0, 1, &enableOp);
}

bool ServoStar::absolutePos()
{
    unsigned char enableOp = 0x4f;
    return bus->writeObject(id, 0x6040, 0, 1, &enableOp);
}

bool ServoStar::shutdown()
{
    unsigned char shutdown = 0x0;
    return bus->writeObject(id, 0x6040, 0, 1, &shutdown);
    shutdown = 0x6;
    return bus->writeObject(id, 0x6040, 0, 1, &shutdown);
}

bool ServoStar::setOpMode(unsigned char opMode)
{
    return bus->writeObject(id, 0x6060, 0, 1, &opMode);
}

bool ServoStar::setTPDO(unsigned char tpdonum, unsigned char pdo)
{
    if ((tpdonum < 1) || (tpdonum > 4))
        return false;
    return bus->writeObject(id, 0x2a00 + (tpdonum - 1), 0, 1, &pdo);
}
bool ServoStar::setRPDO(unsigned char rpdonum, unsigned char pdo)
{
    if ((rpdonum < 1) || (rpdonum > 4))
        return false;
    return bus->writeObject(id, 0x2600 + (rpdonum - 1), 0, 1, &pdo);
}

bool ServoStar::setTPDOCom(unsigned char tpdonum, unsigned char tt, unsigned char it)
{
    if ((tpdonum < 1) || (tpdonum > 4))
        return false;
    bool success = true;
    if (!bus->writeObject(id, 0x1800 + (tpdonum - 1), 2, 1, &tt))
        success = false;
    if (!bus->writeObject(id, 0x1800 + (tpdonum - 1), 3, 1, &it))
        success = false;

    return success;
}
bool ServoStar::setRPDOCom(unsigned char rpdonum, unsigned char tt, unsigned char it)
{
    if ((rpdonum < 1) || (rpdonum > 4))
        return false;
    bool success = true;
    if (!bus->writeObject(id, 0x1400 + (rpdonum - 1), 2, 1, &tt))
        success = false;
    if (!bus->writeObject(id, 0x1400 + (rpdonum - 1), 3, 1, &it))
        success = false;

    return success;
}

bool ServoStar::setTPDOMap(unsigned char tpdonum, unsigned short index, unsigned char subindex, unsigned char datatype)
{
    if ((tpdonum < 1) || (tpdonum > 4))
        return false;

    if (datatype > 0x40)
    {
        std::cout << "Datatype too big for TD01 Mapping" << std::endl;
        return false;
    }

    bool success = true;

    unsigned char data[] = { 0, 0, 0, 0 };
    if (!bus->writeObject(id, 0x1a00 + (tpdonum - 1), 0, 4, data))
        success = false;

    data[0] = datatype;
    data[1] = subindex;
    data[2] = (index & 0xff);
    data[3] = (index >> 8);
    if (!bus->writeObject(id, 0x1a00 + (tpdonum - 1), 1, 4, data))
        success = false;

    return success;
}

bool ServoStar::setTPDOMap(unsigned char tpdonum,
                           unsigned short index1, unsigned char subindex1, unsigned char datatype1,
                           unsigned short index2, unsigned char subindex2, unsigned char datatype2)
{
    if ((tpdonum < 1) || (tpdonum > 4))
        return false;
    if ((datatype1 > 0x40) || (datatype2 > 0x40))
    {
        std::cout << "Datatype too big for TD01 Mapping" << std::endl;
        return false;
    }

    bool success = true;

    unsigned char data[] = { 0, 0, 0, 0 };
    if (!bus->writeObject(id, 0x1a00 + (tpdonum - 1), 0, 4, data))
        success = false;

    data[0] = datatype1;
    data[1] = subindex1;
    data[2] = (index1 & 0xff);
    data[3] = (index1 >> 8);
    if (!bus->writeObject(id, 0x1a00 + (tpdonum - 1), 1, 4, data))
        success = false;

    data[0] = datatype2;
    data[1] = subindex2;
    data[2] = (index2 & 0xff);
    data[3] = (index2 >> 8);
    if (!bus->writeObject(id, 0x1a00 + (tpdonum - 1), 2, 4, data))
        success = false;

    return success;
}

bool ServoStar::setRPDOMap(unsigned char rpdonum, unsigned short index, unsigned char subindex, unsigned char datatype)
{
    if ((rpdonum < 1) || (rpdonum > 4))
        return false;

    if (datatype > 0x40)
    {
        std::cout << "Datatype too big for TD01 Mapping" << std::endl;
        return false;
    }

    bool success = true;

    unsigned char data[] = { 0, 0, 0, 0 };
    if (!bus->writeObject(id, 0x1600 + (rpdonum - 1), 0, 4, data))
        success = false;

    data[0] = datatype;
    data[1] = subindex;
    data[2] = (index & 0xff);
    data[3] = (index >> 8);
    if (!bus->writeObject(id, 0x1600 + (rpdonum - 1), 1, 4, data))
        success = false;

    return success;
}

bool ServoStar::setRPDOMap(unsigned char rpdonum,
                           unsigned short index1, unsigned char subindex1, unsigned char datatype1,
                           unsigned short index2, unsigned char subindex2, unsigned char datatype2)
{
    if ((rpdonum < 1) || (rpdonum > 4))
        return false;
    if ((datatype1 > 0x40) || (datatype2 > 0x40))
    {
        std::cout << "Datatype too big for TD01 Mapping" << std::endl;
        return false;
    }

    bool success = true;

    unsigned char data[] = { 0, 0, 0, 0 };
    if (!bus->writeObject(id, 0x1600 + (rpdonum - 1), 0, 4, data))
        success = false;

    data[0] = datatype1;
    data[1] = subindex1;
    data[2] = (index1 & 0xff);
    data[3] = (index1 >> 8);
    if (!bus->writeObject(id, 0x1600 + (rpdonum - 1), 1, 4, data))
        success = false;

    data[0] = datatype2;
    data[1] = subindex2;
    data[2] = (index2 & 0xff);
    data[3] = (index2 >> 8);
    if (!bus->writeObject(id, 0x1600 + (rpdonum - 1), 2, 4, data))
        success = false;

    return success;
}

bool ServoStar::setupHoming(int offset, char type, unsigned int vel, unsigned int acc)
{
    bool success = true;
    if (!bus->writeObject(id, 0x607c, 0, 4, (unsigned char *)&offset))
        success = false;
    if (!bus->writeObject(id, 0x6098, 0, 1, (unsigned char *)&type))
        success = false;
    if (!bus->writeObject(id, 0x6099, 1, 4, (unsigned char *)&vel))
        success = false;
    if (!bus->writeObject(id, 0x609a, 0, 4, (unsigned char *)&acc))
        success = false;

    return success;
}

bool ServoStar::setupPositioning(int target, unsigned int vel, unsigned int acc)
{
    bool success = true;
    if (!bus->writeObject(id, 0x607a, 0, 4, (unsigned char *)&target))
        success = false;
    if (!bus->writeObject(id, 0x6081, 0, 4, (unsigned char *)&vel))
        success = false;
    if (!bus->writeObject(id, 0x6083, 0, 4, (unsigned char *)&acc))
        success = false;
    if (!bus->writeObject(id, 0x6084, 0, 4, (unsigned char *)&acc))
        success = false;

    return success;
}

bool ServoStar::setupLifeGuarding(unsigned short guardTime)
{
    bool success = true;

    unsigned char lifeTimeFactor = 1;

    if (!bus->writeObject(id, 0x100c, 0, 2, (unsigned char *)&guardTime))
        success = false;
    if (!bus->writeObject(id, 0x100d, 0, 1, &lifeTimeFactor))
        success = false;

    return success;
}

bool ServoStar::stopLifeGuarding()
{
    bool success = true;

    unsigned short guardTime = 0;
    unsigned char lifeTimeFactor = 0;

    if (!bus->writeObject(id, 0x100c, 0, 2, (unsigned char *)&guardTime))
        success = false;
    if (!bus->writeObject(id, 0x100d, 0, 1, &lifeTimeFactor))
        success = false;

    return success;
}

bool ServoStar::sendGuardMessage()
{
    return bus->sendRTR((0x700 + id), 1);
}

void ServoStar::microwait(unsigned long waittime)
{
#ifndef WIN32
    bool sleep = true;
    timeval starttime, curtime;

    gettimeofday(&starttime, 0);
    while (sleep)
    {
        gettimeofday(&curtime, 0);
        if (((curtime.tv_sec * 1e6 + curtime.tv_usec) - (starttime.tv_sec * 1e6 + starttime.tv_usec)) > waittime)
            sleep = false;
        if (sched_yield() != 0)
            std::cerr << "sched_yield() failed!" << std::endl;
    }
#else
    Sleep(waittime);
#endif
}

void ServoStar::microsleep(unsigned long sleeptime)
{
#ifndef WIN32
    timespec st;
    st.tv_sec = 0;
    st.tv_nsec = 1000 * sleeptime;
    nanosleep(&st, NULL);
#else
    Sleep(sleeptime);
#endif
    //usleep(sleeptime);
}

unsigned long ServoStar::getTimeInSeconds()
{
    timeval time;
    gettimeofday(&time, 0);
    return time.tv_sec;
}

unsigned long ServoStar::getTimeInMicroseconds()
{
    timeval time;
    gettimeofday(&time, 0);
    return (1000000 * time.tv_sec + time.tv_usec);
}

bool ServoStar::checkOutputStageEnabled()
{
    void *status;
    unsigned char len;

    if (!readObject(0x1002, 0, len, status))
        return false;
    else if (len != 4)
        return false;

    //std::cerr << "Status: " << *((unsigned int*)status) << ", Length: " << (int)len << std::endl;
    if (((unsigned char *)(status))[3] & 0x40)
        return true;
    else
        return false;
}
#endif
