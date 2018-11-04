/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CANProviderTasks.h"
#include <cmath>
#include <cstdlib>
#include <unistd.h>

//SendTask////////////////////////////////////////////////////////////

// constructor, destructor, instance ---------------------------------
SendTask::SendTask(CANProvider *provider)
    : XenomaiTask("SendTask")
    , XenomaiMutex("listMutex")
{
    // init
    p_CANProv = provider;
    taskPeriod = 0;
    countMax = 1;
    isRunning = false;
    stop = false;

    // calculate gcd and lcm of CAN Display
    for (CANProvider::CANSimMap::const_iterator it1 = p_CANProv->CANDSimMap.begin(); it1 != p_CANProv->CANDSimMap.end(); ++it1)
    {
        // calculate gcd of all cycle times to determine greates possible
        // send task period. This saves calc time cause send task is
        // executed only as fast as needed for simulated msgs
        taskPeriod = gcd(taskPeriod, it1->first);

        // calculate least common multiple (lcm) to determine when the
        // counter in run method is reset so no message gets simulated
        // with the wrong cycle time
        countMax = (countMax * it1->first) / gcd(countMax, it1->first);
    }

    // calculate gcd and lcm of CAN Komfort in relation to result of
    // above calulation
    for (CANProvider::CANSimMap::const_iterator it1 = p_CANProv->CANKSimMap.begin(); it1 != p_CANProv->CANKSimMap.end(); ++it1)
    {
        taskPeriod = gcd(taskPeriod, it1->first);
        countMax = (countMax * it1->first) / gcd(countMax, it1->first);
    }

    if (taskPeriod != 0)
    {
        std::cerr << "SendTask::SendTask: Send task for CAN simulation runs with a period of "
                  << taskPeriod << " ms, reset after " << countMax << " ms" << std::endl;

        // send task will be executed every gcd_value ms
        set_periodic(taskPeriod * 1000000);
        XenomaiTask::start();
    }
    else
    {
        std::cerr << "SendTask::SendTask: There was a problem calculating the period for the CAN simulation send task!" << std::endl;
    }
}

SendTask::~SendTask()
{
    // should have been done be the application before
    shutdown();

    // do not delete CANProvider - this is done by VehicleUtil class
    //delete p_CANProv;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
// UWE
void SendTask::registerDevice(CanOpenDevice *d)
{
    XenomaiMutex::acquire();
    CanOpenDevices.push_back(d);
    initCanOpenDevices.push_back(d);
    XenomaiMutex::release();
}

// UWE
void SendTask::shutdown()
{
    if (!stop)
    {
        fprintf(stderr, "SendTask::shutdown()\n");
        RT_TASK_INFO info;
        inquire(info);

#ifdef MERCURY
        if (info.stat.status & __THREAD_S_STARTED)
#else
        if (info.status & T_STARTED)
#endif
        {
            stop = true;
            while (isRunning)
            {
                sleep(1);
            }
        }
        // mutex is deleted in destructor of XenomaiMutex class
    }
}
//--------------------------------------------------------------------

// protected methods -------------------------------------------------
void SendTask::run()
{
    unsigned long overrun;
    unsigned int count = taskPeriod;

    isRunning = true;

    while (!stop)
    {
        XenomaiMutex::acquire();

        for (std::list<CanOpenDevice *>::iterator it = initCanOpenDevices.begin(); it != initCanOpenDevices.end(); it++)
        {
            std::cerr << "initCANOpenDevice" << std::endl;
            (*it)->initCANOpenDevice();
            initCanOpenDevices.erase(it);
            break;
        }

        XenomaiMutex::release();
        rt_task_wait_period(&overrun);

        // simulate messages for CAN Display
        for (CANProvider::CANSimMap::const_iterator it1 = p_CANProv->CANDSimMap.begin(); it1 != p_CANProv->CANDSimMap.end(); ++it1)
        {
            if (count % it1->first == 0)
            {
                CANProvider::CANMap tempmap = it1->second;

                for (CANProvider::CANMap::const_iterator it2 = tempmap.begin(); it2 != tempmap.end(); ++it2)
                {
                    // temporary hack for Beckhoff
                    p_CANProv->p_CANOpenDisplay->sendFrame(it2->second->theCANframe());
                    //p_CANProv->pcan_display->sendFrame(it2->second->theCANframe());
                }
            }
        }

        //simulate messages for CAN Komfort
        for (CANProvider::CANSimMap::const_iterator it1 = p_CANProv->CANKSimMap.begin(); it1 != p_CANProv->CANKSimMap.end(); ++it1)
        {
            if (count % it1->first == 0)
            {
                CANProvider::CANMap tempmap = it1->second;

                for (CANProvider::CANMap::const_iterator it2 = tempmap.begin(); it2 != tempmap.end(); ++it2)
                {
                    p_CANProv->p_CANKomfort->sendFrame(it2->second->theCANframe());
                }
            }
        }

        // send CANOpen pdos
        if ((count % 10) == 0)
        {
            p_CANProv->p_CANOpenDisplay->sendPDO();
        }

        if (count >= countMax)
        {
            // reset, new simulation cycle
            count = taskPeriod;
        }
        else
            count = count + taskPeriod;
    }

    for (std::list<CanOpenDevice *>::iterator it = CanOpenDevices.begin(); it != CanOpenDevices.end(); it++)
    {
        (*it)->shutdownCANOpenDevice();
    }

    isRunning = false;
}
//--------------------------------------------------------------------

// private methods ---------------------------------------------------

// Stein's Algorithm: a binary GCD algorithm, computes the greatest
// common divisor of two nonnegative integers
unsigned int SendTask::gcd(unsigned int u, unsigned int v)
{
    int shift;

    /* GCD(0,x) := x */
    if (u == 0 || v == 0)
        return u | v;

    /* Let shift := lg K, where K is the greatest power of 2
      dividing both u and v. */
    for (shift = 0; ((u | v) & 1) == 0; ++shift)
    {
        u >>= 1;
        v >>= 1;
    }

    while ((u & 1) == 0)
        u >>= 1;

    /* From here on, u is always odd. */
    do
    {
        while ((v & 1) == 0) /* Loop X */
            v >>= 1;

        /* Now u and v are both odd, so diff(u, v) is even.
            Let u = min(u, v), v = diff(u, v)/2. */
        if (u < v)
        {
            v -= u;
        }
        else
        {
            unsigned int diff = u - v;
            u = v;
            v = diff;
        }
        v >>= 1;
    } while (v != 0);

    return u << shift;
}
//--------------------------------------------------------------------

//CANDRecvTask////////////////////////////////////////////////////////

// constructor, destructor, instance ---------------------------------
CANDRecvTask::CANDRecvTask(CANProvider *provider)
    : XenomaiTask("CANDRecvTask")
{
    p_CANProv = provider;
    stop = false;
    isRunning = false;

    XenomaiTask::start();
}

CANDRecvTask::~CANDRecvTask()
{
    // should have been done by the application before
    shutdown();

    // do not delete CANProvider - this is done by VehicleUtil class
    //delete p_CANProv;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
// UWE
void CANDRecvTask::shutdown()
{
    if (!stop)
    {
        fprintf(stderr, "CANDRecvTask::shutdown()\n");

        RT_TASK_INFO info;
        inquire(info);
#ifdef MERCURY
        if (info.stat.status & __THREAD_S_STARTED)
#else
        if (info.status & T_STARTED)
#endif
        {
            stop = true;
            while (isRunning)
            {
                sleep(1);
            }
        }
    }
}
//--------------------------------------------------------------------

// protected methods -------------------------------------------------
void CANDRecvTask::run()
{
    can_frame tempframe;
    CANProvider::CANMap::iterator iter = p_CANProv->CANDisplayReceived.begin();

    // set send and receive timeouts
    p_CANProv->p_CANOpenDisplay->setRecvTimeout(1000000000);
    p_CANProv->p_CANOpenDisplay->setSendTimeout(1000000000);

    isRunning = true;

    while (!stop)
    {
        // temporary hack for Beckhoff
        p_CANProv->p_CANOpenDisplay->recvFrame(tempframe);
        //p_CANProv->pcan_display->recvFrame(tempframe);  // blocking read!

        // find will return an iterator to the matching element if it is
        // found or to the end of the map if the key is not found
        // using CANKomfortReceived[tempframe.can_id] = tempframe will
        // add new entries if id cannot found!
        iter = p_CANProv->CANDisplayReceived.find(tempframe.can_id);

        if (iter != p_CANProv->CANDisplayReceived.end())
        {
            // update in list of received messages
            iter->second->theCANframe() = tempframe;

            // pass to software gateway
            p_CANProv->routeCANDMessage(tempframe);
        }

        p_CANProv->p_CANOpenDisplay->handleFrame(tempframe);
    }

    isRunning = false;
}
//--------------------------------------------------------------------

//CANKRecvTask////////////////////////////////////////////////////////

// constructor, destructor, instance ---------------------------------
CANKRecvTask::CANKRecvTask(CANProvider *provider)
    : XenomaiTask("CANKRecvTask")
{
    p_CANProv = provider;
    stop = false;
    isRunning = false;

    XenomaiTask::start();
}

CANKRecvTask::~CANKRecvTask()
{
    // should have been done by the application before
    shutdown();

    // do not delete CANProvider - this is done by VehicleUtil class
    //delete p_CANProv;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
// UWE
void CANKRecvTask::shutdown()
{
    if (!stop)
    {
        fprintf(stderr, "CANKRecvTask::shutdown()\n");

        RT_TASK_INFO info;
        inquire(info);
#ifdef MERCURY
        if (info.stat.status & __THREAD_S_STARTED)
#else
        if (info.status & T_STARTED)
#endif
        {
            stop = true;
            while (isRunning)
            {
                sleep(1);
            }
        }
    }
}
//--------------------------------------------------------------------

// protected methods -------------------------------------------------
void CANKRecvTask::run()
{
    can_frame tempframe;
    CANProvider::CANMap::iterator iter = p_CANProv->CANKomfortReceived.begin();

    // set send and receive timeouts
    p_CANProv->p_CANKomfort->setRecvTimeout(1000000000);
    p_CANProv->p_CANKomfort->setSendTimeout(1000000000);

    isRunning = true;

    while (!stop)
    {
        // blocking read!
        p_CANProv->p_CANKomfort->recvFrame(tempframe);

        iter = p_CANProv->CANKomfortReceived.find(tempframe.can_id);

        if (iter != p_CANProv->CANKomfortReceived.end())
        {
            // update in list of received messages
            iter->second->theCANframe() = tempframe;

            // pass to software gateway
            p_CANProv->routeCANKMessage(tempframe);
        }
    }

    isRunning = false;
}
//--------------------------------------------------------------------
