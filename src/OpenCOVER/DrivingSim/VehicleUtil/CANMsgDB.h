/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CANMsgDB_h
#define __CANMsgDB_h

//--------------------------------------------------------------------
// PROJECT        CANMsgDB                                 © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    This module creates the CAN msg maps used by
//                CANProvider and acts at the same time as database
//                for all other modules using the CAN message structs.
//                CAN messages for new	modules should be listed here!
//
// CREATED        15-May-09, S. Franz, F. Seybold
// MODIFIED       21-May-09, S. Franz
//                Added declarations of message structs and maps for
//                CAN Komfort defined in CMS_CANKomfort
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "CMS/CMS_CANKomfort.h"
#include "CMS/CMS_CANDisplay.h"
#include <map>
#include "XenomaiSocketCan.h" // needed for use of can_id_t

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class CANMsgDB
{
public:
    CANMsgDB();
    ~CANMsgDB();

    // map with pointers to CAN msg structs, the key is their CAN ID!
    typedef std::map<can_id_t, CANMsgIface *> CANMap;

    // map containing CANMaps shown above, this time key is cycle time!
    typedef std::map<uint, CANMap> CANSimMap;

    // map containing messages received from ecus on CAN Display
    CANMap CANDisplayReceived;

    // map containing messages received from ecus on CAN Komfort
    CANMap CANKomfortReceived;

    // map containing messages simulated for ecus on CAN Display
    CANMap CANDisplaySimulated;

    // map containing messages simulated for ecus on CAN Komfort
    CANMap CANKomfortSimulated;

    // contains all simulated CAN Display msgs
    CANSimMap CANDSimMap;

    // contains all simulated CAN Komfort msgs
    CANSimMap CANKSimMap;

    // declaration of CAN Display messages -------------------------

    // KI and stopwatch messages
    struct BC_1_D_Struct BC_1_D;
    struct BREMSE_2_D_Struct BREMSE_2_D;
    struct FahrID_1k_D_Struct FahrID_1k_D;
    struct GW_D_1_Struct GW_D_1;
    struct GW_D_2_Struct GW_D_2;
    struct GW_D_3_Struct GW_D_3;
    struct GW_D_4_Struct GW_D_4;
    struct GW_D_5_Struct GW_D_5;
    struct GW_D_6_Struct GW_D_6;
    struct GW_D_7_Struct GW_D_7;
    struct GW_D_8_Struct GW_D_8;
    struct GW_D_V_Struct GW_D_V;
    struct GW_SVB_D_Struct GW_SVB_D;
    struct KOMBI_6_Struct KOMBI_6;
    struct MOTOR_4_D_Struct MOTOR_4_D;
    struct MOTOR_6_D_Struct MOTOR_6_D;
    struct PCM_1_Struct PCM_1;
    struct RDK_1_D_Struct RDK_1_D;
    struct RDK_2_D_Struct RDK_2_D;
    struct RDK_4_D_Struct RDK_4_D;
    struct UHR_1_Struct UHR_1;
    struct UHR_2_Struct UHR_2;

    // AVR messages
    struct ButtonState_Struct ButtonState;
    struct LEDState_Struct LEDState;

    // gas pedal messages
    struct ANFORDERUNG_FFP_Struct ANFORDERUNG_FFP;
    struct STATUS_FFP_Struct STATUS_FFP;

    // declaration of CAN Komfort messages -------------------------

    // KLSM and seat messages
    struct BC_1_Struct BC_1;
    struct BUG_1_Struct BUG_1;
    struct FahrID_1k_Struct FahrID_1k;
    struct FahrID_2_Struct FahrID_2;
    struct GW_K_2_Struct GW_K_2;
    struct KOMBI_1_K_Struct KOMBI_1_K;
    struct KOMBI_7_K_Struct KOMBI_7_K;
    struct LSS_1_Struct LSS_1;
    struct LSS_2_Struct LSS_2;
    struct LSS_3_Struct LSS_3;
    struct NM_SM_Struct NM_SM;
    struct SITZM_V_Struct SITZM_V;
    struct TSG_FT_1_Struct TSG_FT_1;
    struct TSG_FT_2_Struct TSG_FT_2;
    struct Klima_Struct Klima;

private:
    void addCANDSimMsgs();
    void addCANKSimMsgs();
    void addCANDRecvMsgs();
    void addCANKRecvMsgs();
    void createSimMaps();
};
//--------------------------------------------------------------------

#endif
