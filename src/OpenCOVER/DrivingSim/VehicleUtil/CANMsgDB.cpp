/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CANMsgDB.h"
#include <stdio.h>
#include <string.h>

// constructor, destructor, instance ---------------------------------
CANMsgDB::CANMsgDB()
{
    addCANDSimMsgs();
    addCANKSimMsgs();
    addCANDRecvMsgs();
    addCANKRecvMsgs();
    createSimMaps();
}

CANMsgDB::~CANMsgDB()
{
    fprintf(stderr, "CANMsgDB::~CANMsgDB\n");
}
//--------------------------------------------------------------------

// private methods ---------------------------------------------------
void CANMsgDB::addCANDSimMsgs()
{
    // CAN messages to simulate for KI and stopwatch
    CANDisplaySimulated[BC_1_D.values.canmsg.ID] = &BC_1_D;
    CANDisplaySimulated[BREMSE_2_D.values.canmsg.ID] = &BREMSE_2_D;
    CANDisplaySimulated[FahrID_1k_D.values.canmsg.ID] = &FahrID_1k_D;
    CANDisplaySimulated[GW_D_1.values.canmsg.ID] = &GW_D_1;
    CANDisplaySimulated[GW_D_2.values.canmsg.ID] = &GW_D_2;
    CANDisplaySimulated[GW_D_3.values.canmsg.ID] = &GW_D_3;
    CANDisplaySimulated[GW_D_4.values.canmsg.ID] = &GW_D_4;
    CANDisplaySimulated[GW_D_5.values.canmsg.ID] = &GW_D_5;
    CANDisplaySimulated[GW_D_6.values.canmsg.ID] = &GW_D_6;
    CANDisplaySimulated[GW_D_7.values.canmsg.ID] = &GW_D_7;
    CANDisplaySimulated[GW_D_8.values.canmsg.ID] = &GW_D_8;
    CANDisplaySimulated[GW_D_V.values.canmsg.ID] = &GW_D_V;
    CANDisplaySimulated[GW_SVB_D.values.canmsg.ID] = &GW_SVB_D;
    CANDisplaySimulated[KOMBI_6.values.canmsg.ID] = &KOMBI_6;
    CANDisplaySimulated[MOTOR_4_D.values.canmsg.ID] = &MOTOR_4_D;
    CANDisplaySimulated[MOTOR_6_D.values.canmsg.ID] = &MOTOR_6_D;
    CANDisplaySimulated[PCM_1.values.canmsg.ID] = &PCM_1;
    CANDisplaySimulated[RDK_1_D.values.canmsg.ID] = &RDK_1_D;
    CANDisplaySimulated[RDK_2_D.values.canmsg.ID] = &RDK_2_D;
    CANDisplaySimulated[RDK_4_D.values.canmsg.ID] = &RDK_4_D;

    // AVR
    CANDisplaySimulated[LEDState.values.canmsg.ID] = &LEDState;

    // CAN messages to simulate for gas pedal
    CANDisplaySimulated[ANFORDERUNG_FFP.values.canmsg.ID] = &ANFORDERUNG_FFP;
}

void CANMsgDB::addCANKSimMsgs()
{
    // CAN messages to simulate for KLSM and seat
    CANKomfortSimulated[BC_1.values.canmsg.ID] = &BC_1;
    CANKomfortSimulated[BUG_1.values.canmsg.ID] = &BUG_1;
    CANKomfortSimulated[FahrID_1k.values.canmsg.ID] = &FahrID_1k;
    CANKomfortSimulated[FahrID_2.values.canmsg.ID] = &FahrID_2;
    CANKomfortSimulated[GW_K_2.values.canmsg.ID] = &GW_K_2;
    CANKomfortSimulated[KOMBI_1_K.values.canmsg.ID] = &KOMBI_1_K;
    CANKomfortSimulated[KOMBI_7_K.values.canmsg.ID] = &KOMBI_7_K;
    CANKomfortSimulated[LSS_1.values.canmsg.ID] = &LSS_1;
    CANKomfortSimulated[LSS_2.values.canmsg.ID] = &LSS_2;
    CANKomfortSimulated[LSS_3.values.canmsg.ID] = &LSS_3;
    CANKomfortSimulated[NM_SM.values.canmsg.ID] = &NM_SM;
    CANKomfortSimulated[SITZM_V.values.canmsg.ID] = &SITZM_V;
    CANKomfortSimulated[TSG_FT_1.values.canmsg.ID] = &TSG_FT_1;
    CANKomfortSimulated[TSG_FT_2.values.canmsg.ID] = &TSG_FT_2;
}

void CANMsgDB::addCANDRecvMsgs()
{
    // CAN messages received from stopwatch
    CANDisplayReceived[UHR_1.values.canmsg.ID] = &UHR_1;
    CANDisplayReceived[UHR_2.values.canmsg.ID] = &UHR_2;

    // AVR
    CANDisplayReceived[ButtonState.values.canmsg.ID] = &ButtonState;

    // CAN messages received from gas pedal
    CANDisplayReceived[STATUS_FFP.values.canmsg.ID] = &STATUS_FFP;
}

void CANMsgDB::addCANKRecvMsgs()
{
    // CAN messages received from KLSM
    CANKomfortReceived[BC_1.values.canmsg.ID] = &BC_1;
    CANKomfortReceived[LSS_1.values.canmsg.ID] = &LSS_1;
    CANKomfortReceived[LSS_2.values.canmsg.ID] = &LSS_2;
    CANKomfortReceived[LSS_3.values.canmsg.ID] = &LSS_3;
    // CAN messages received from Klima
    CANKomfortReceived[Klima.values.canmsg.ID] = &Klima;
}

/********************************************************************
* FUNCTION: 		createSimMaps()
*	
* DESCRIPTION:		Creates maps with cycle times as key based on above
						maps. There will be a CANMap for every cycle_time
						of the messages in CANDisplaySimulated and 
						CANKomfortSimulated. The appropiate CAN msgs will 
						be added to these maps. Maps are used for an
						accurately timed simulation of CAN msgs by
						CANProvider.
* PARAMETER:		none
* RETURN:			none
* ERROR HANDLING:	none
********************************************************************/
void CANMsgDB::createSimMaps()
{
    // Create a map of maps for CAN Display
    for (CANMap::const_iterator it = CANDisplaySimulated.begin(); it != CANDisplaySimulated.end(); ++it)
    {
        CANMsgIface *pmsg = it->second; // get pointer to CAN msg struct

        // Pay attention here: if there is no map exisiting for a
        // particaluar cycle time, it will be created automatically and
        // subsequently the CAN msg is added to the newly created map!
        (CANDSimMap[pmsg->cycle_time])[pmsg->theCANframe().can_id] = pmsg;
    }

    // Create a map of maps for CAN Komfort
    for (CANMap::const_iterator it = CANKomfortSimulated.begin(); it != CANKomfortSimulated.end(); ++it)
    {
        CANMsgIface *pmsg = it->second;
        (CANKSimMap[pmsg->cycle_time])[pmsg->theCANframe().can_id] = pmsg;
    }
}
//--------------------------------------------------------------------
