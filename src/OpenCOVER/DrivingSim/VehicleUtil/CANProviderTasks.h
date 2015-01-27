/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CANProviderTasks_h
#define __CANProviderTasks_h

//--------------------------------------------------------------------
// PROJECT        CANProviderTasks                         © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    BLA BLA BLA BLA BLA BLA
//
// CREATED        15-May-09, S. Franz
// MODIFIED       17-July-09, S. Franz
//                Application of HLRS style guide
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "XenomaiTask.h"
#include "XenomaiMutex.h"
#include "CANProvider.h"
#include "CANMsgDB.h"
#include "CanOpenDevice.h"
#include <stdlib.h>
#include <list>
#include <native/mutex.h>

class CanOpenDevice; // forward class declaration
class CANProvider; // forward class declaration

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class SendTask : public XenomaiTask, public XenomaiMutex
{
public:
    SendTask(CANProvider *provider);
    ~SendTask();
    void registerDevice(CanOpenDevice *d);
    void shutdown();

protected:
    void run(); // override Xenomai Method

private:
    unsigned int gcd(unsigned int, unsigned int);

    std::list<CanOpenDevice *> CanOpenDevices;
    std::list<CanOpenDevice *> initCanOpenDevices;
    //RT_MUTEX listMutex;

    CANProvider *p_CANProv;
    unsigned int taskPeriod;
    unsigned int countMax;
    volatile bool stop;
    volatile bool isRunning;
};
//--------------------------------------------------------------------

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class CANDRecvTask : public XenomaiTask
{
public:
    CANDRecvTask(CANProvider *);
    ~CANDRecvTask();
    void shutdown();

protected:
    void run(); // override Xenomai Method

private:
    CANProvider *p_CANProv;
    volatile bool stop;
    volatile bool isRunning;
};
//--------------------------------------------------------------------

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class CANKRecvTask : public XenomaiTask
{
public:
    CANKRecvTask(CANProvider *);
    ~CANKRecvTask();
    void shutdown();

protected:
    void run(); // override Xenomai Method

private:
    CANProvider *p_CANProv;
    volatile bool stop;
    volatile bool isRunning;
};
//--------------------------------------------------------------------

#endif
