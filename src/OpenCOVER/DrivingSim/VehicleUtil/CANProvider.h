/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CANProvider_h
#define __CANProvider_h

//--------------------------------------------------------------------
// PROJECT        CANProvider                              © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    The CANProvider sets the signals which simulate the
//                clamps (15, S, X, ...) and administrates a send task
//                for simulating CAN messages and two reveice tasks
//                to read messages from CAN Display and CAN Komfort.
//                The necessary structs are inherited from CANMsgDB.
// CREATED        15-May-09, S. Franz
// MODIFIED       17-July-09, S. Franz
//                - Application of HLRS style guide
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "XenomaiSocketCan.h"
#include "CANMsgDB.h"
#include "CANProviderTasks.h"
#include "CanOpenController.h"
#include <config/CoviseConfig.h>

class SendTask; // fwd class declaration
class CANDRecvTask; // fwd class declaration
class CANKRecvTask; // fwd class declaration
class CanOpenDevice; // fwd class declaration

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class VEHICLEUTILEXPORT CANProvider : public CANMsgDB
{
public:
    virtual ~CANProvider();

    void routeCANDMessage(const can_frame &);
    void routeCANKMessage(const can_frame &);
    void registerDevice(CanOpenDevice *d);
    int numUninitializedDevices();
    void shutdown();

    bool keyIsIn();
    bool ignitionIsOn();
    bool engineIsStarted();
    bool engineIsRunning();
    bool keyIsOut();

    static CANProvider *instance(); // singleton

    // temporary hack for Beckhoff
    /* XenomaiSocketCan* pcan_display; */
    // declaration of can socket pointer for CAN Open
    CanOpenController *p_CANOpenDisplay;

    // declaration of can socket pointer for Porsche CAN Komfort
    XenomaiSocketCan *p_CANKomfort;

protected:
    CANProvider();

    static CANProvider *p_CANProv;

    // the send task
    SendTask *p_CANSendTask;

    // two receive tasks necessary due to blocking receive/read of can api
    CANDRecvTask *p_CANDRecvTask;
    CANKRecvTask *p_CANKRecvTask;

private:
    enum m_lockstates
    {
        KEYOUT = 0,
        KEYIN = 1,
        IGNITED = 2,
        STARTED = 3
    };
    int m_state;
};
//--------------------------------------------------------------------

#endif
