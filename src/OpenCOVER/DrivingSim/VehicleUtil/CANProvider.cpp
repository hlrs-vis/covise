/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CANProvider.h"

// set protected static pointer for singleton to NULL
CANProvider *CANProvider::p_CANProv = NULL;

// constructor, destructor, instance ---------------------------------
CANProvider::CANProvider()
    : CANMsgDB()
{
    /* pcan_display = new XenomaiSocketCan (coCoviseConfig::getEntry("channel","COVER.VehicleUtil.CANProvider.CANDisplay","rtcan0")); */

    // temporary hack for Beckhoff
    p_CANOpenDisplay = new CanOpenController(covise::coCoviseConfig::getEntry("channel", "COVER.VehicleUtil.CANProvider.CANDisplay", "rtcan3"));

    p_CANKomfort = new XenomaiSocketCan(covise::coCoviseConfig::getEntry("channel", "COVER.VehicleUtil.CANProvider.CANKomfort", "rtcan2"));

    /*std::cerr << "CANProvider::CANProvider: CAN Display on " << coCoviseConfig::getEntry("channel","COVER.VehicleUtil.CANProvider.CANDisplay","rtcan3")
   << std::endl;*/

    /*std::cerr << "CANProvider::CANProvider: CAN Komfort on " << coCoviseConfig::getEntry("channel","COVER.VehicleUtil.CANProvider.CANKomfort","rtcan2")
   << std::endl;*/

    p_CANDRecvTask = new CANDRecvTask(this);
    p_CANKRecvTask = new CANKRecvTask(this);
    p_CANSendTask = new SendTask(this);

    p_CANDRecvTask->start();
    p_CANKRecvTask->start();
    p_CANSendTask->start();

    m_state = KEYOUT;
}

CANProvider::~CANProvider()
{
    p_CANProv = NULL;

    //delete pcan_display;
    delete p_CANOpenDisplay;
    delete p_CANKomfort;
    delete p_CANDRecvTask;
    delete p_CANKRecvTask;
    delete p_CANSendTask;
}

// singleton
CANProvider *CANProvider::instance()
{
    if (p_CANProv == NULL)
    {
        p_CANProv = new CANProvider();
    }
    return p_CANProv;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------

/**
 * This method routes the data frame of a rtcan frame from CAN Komfort
 * to a rtcan frame which is simulated on CAN Display
 *
 * @param const can_frame&, a rtcan frame to route to another bus
 * @return none
 */
void CANProvider::routeCANKMessage(const can_frame &frame)
{
    switch (frame.can_id)
    {
    // can message for bc control (up, down, set, reset) from klsm
    case 0x361:
    {
        // copy frame data to BC frame data of CAN Display
        // attention: copying the whole frame will also change id and dlc!
        memcpy(CANDisplaySimulated[BC_1_D.values.canmsg.ID]->theCANframe().data,
               frame.data, sizeof(frame.data));

        break;
    }
    }
}

/**
 * This method routes the data frame of a rtcan frame from CAN Display
 * to a rtcan frame which is simulated on CAN Komfort
 *
 * @param const can_frame&, a rtcan frame to route to another bus
 * @return none
 */
void CANProvider::routeCANDMessage(const can_frame &frame)
{
    switch (frame.can_id)
    {
    }
}

// UWE
void CANProvider::registerDevice(CanOpenDevice *d)
{
    p_CANSendTask->registerDevice(d);
}

// UWE
void CANProvider::shutdown()
{
    p_CANSendTask->shutdown();
    p_CANDRecvTask->shutdown();
    p_CANKRecvTask->shutdown();
}

bool CANProvider::keyIsOut()
{
    if (m_state == KEYIN)
    {
        std::cerr << "CANProvider::keyIsOut()" << std::endl;

        // CAN Display
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_S_D = 0;

        // CAN Komfort
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_S = 0;

        m_state = KEYOUT;
        return true;
    }
    else
        return false;
}

bool CANProvider::keyIsIn()
{
    if (m_state == KEYOUT || m_state == IGNITED || m_state == STARTED)
    {
        std::cerr << "CANProvider::keyIsIn()" << std::endl;

        // CAN Display
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_S_D = 1;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_15_D = 0;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_X_D = 0;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_50_D = 0;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_15SV_D = 0;

        p_CANProv->MOTOR_4_D.values.canmsg.cansignals.Motorlauf_D = 0;

        // CAN Komfort
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_S = 1;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_15 = 0;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_X = 0;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_50 = 0;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_15SV = 0;

        m_state = KEYIN;
        return true;
    }
    else
        return false;
}

bool CANProvider::ignitionIsOn()
{
    if (m_state == KEYIN)
    {
        std::cerr << "CANProvider::ignitionIsOn()" << std::endl;

        // CAN Display
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_15SV_D = 1;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_15_D = 1;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_X_D = 1;

        // CAN Komfort
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_15SV = 1;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_15 = 1;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_X = 1;

        m_state = IGNITED;
        return true;
    }
    else
        return false;
}

bool CANProvider::engineIsStarted()
{
    if (m_state == IGNITED)
    {
        std::cerr << "CANProvider::engineIsStarted()" << std::endl;

        // CAN Display
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_X_D = 0;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_50_D = 1;

        // CAN Komfort
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_X = 0;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_50 = 1;

        m_state = STARTED;
        return true;
    }
    else
        return false;
}

bool CANProvider::engineIsRunning()
{
    if (m_state == STARTED)
    {
        std::cerr << "CANProvider::engineIsRunning()" << std::endl;

        // CAN Display
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_X_D = 1;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.Kl_50_D = 0;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.ph1 = 3;
        p_CANProv->FahrID_1k_D.values.canmsg.cansignals.ph2 = 1;

        p_CANProv->MOTOR_4_D.values.canmsg.cansignals.Motorlauf_D = 1;

        // CAN Komfort
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_X = 1;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.Kl_50 = 0;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.ph1 = 3;
        p_CANProv->FahrID_1k.values.canmsg.cansignals.ph2 = 1;

        m_state = KEYOUT;
        return true;
    }
    else
        return false;
}
//--------------------------------------------------------------------
